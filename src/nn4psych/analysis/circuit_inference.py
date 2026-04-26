from __future__ import annotations

"""
Circuit data collection for latent circuit inference.

Provides collect_circuit_data() to extract (u, z, y, task_active_mask) tensors
from a trained ActorCritic model running ContextDecisionMaking-v0, for use with
LatentNet.fit().

The four tensors follow the LatentNet convention:
  u: input stimulus at each timestep      (n_trials, T, input_dim)
  z: actor logits at each timestep        (n_trials, T, action_dim)
  y: RNN hidden states                    (n_trials, T, hidden_dim)
  task_active_mask: bool mask over steps  (n_trials, T), True during
    stimulus + decision periods (excludes fixation, delay, and blank padding)

All trials are truncated to the minimum observed trial length (T) so the
arrays are uniform with no NaN values, as required by LatentNet.fit().

Reference:
  Langdon & Engel (2025). Latent circuit inference from heterogeneous neural data.
"""

import copy
import gc
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nn4psych.models.actor_critic import ActorCritic
from nn4psych.models.continuous_rnn import ContinuousActorCritic
from nn4psych.analysis.latent_net import LatentNet
from envs.neurogym_wrapper import SingleContextDecisionMakingWrapper


def collect_circuit_data(
    model: "ActorCritic | ContinuousActorCritic",
    env_class=SingleContextDecisionMakingWrapper,
    modality_contexts: list = None,
    n_trials_per_context: int = 300,
    max_steps: int = 500,
    device: str = 'cpu',
    env_kwargs: dict = None,
) -> dict:
    """
    Collect (u, z, y) tensors from a trained ActorCritic for LatentNet fitting.

    Runs the model through ContextDecisionMaking-v0 for each modality context,
    recording inputs (u), actor logits (z), and hidden states (y) at every
    timestep. All trials are truncated to uniform length.

    Parameters
    ----------
    model : ActorCritic or ContinuousActorCritic
        Trained model. Will be set to eval mode.
    env_class : class, optional
        Environment wrapper class. Default is SingleContextDecisionMakingWrapper.
    modality_contexts : list of int, optional
        Modality contexts to collect. Default is [0, 1].
    n_trials_per_context : int, optional
        Number of trials per modality context. Default is 300.
    max_steps : int, optional
        Maximum steps per trial. Default is 500.
    device : str, optional
        Device for model inference. Default is 'cpu'.
    env_kwargs : dict, optional
        Additional keyword arguments for environment creation.

    Returns
    -------
    dict with keys:
        'u' : np.ndarray, shape (n_trials, T, input_dim), float32
        'z' : np.ndarray, shape (n_trials, T, action_dim), float32
        'y' : np.ndarray, shape (n_trials, T, hidden_dim), float32
        'task_active_mask' : np.ndarray, shape (n_trials, T), bool
            True at each timestep where the environment is in the 'stimulus'
            or 'decision' period. Fixation, delay, and blank/padding timesteps
            are False. Derived from NeuroGym TrialEnv timing (not hard-coded).
            Falls back to all-True with a warning if the env does not expose
            per-period timing (backward-compat safeguard).
        'labels' : dict with per-trial condition arrays:
            'modality_context' : np.ndarray, shape (n_trials,), int
            'coherence_sign'   : np.ndarray, shape (n_trials,), int (+1 or -1)
            'correct_action'   : np.ndarray, shape (n_trials,), int (1 or 2)
        'T' : int, uniform trial length used
        'n_trials' : int, total number of trials
        'modality_contexts' : list, modality contexts used
        'n_trials_per_context' : int
    """
    if modality_contexts is None:
        modality_contexts = [0, 1]
    if env_kwargs is None:
        env_kwargs = {}

    model.eval()
    torch_device = torch.device(device)

    # Per-trial storage (variable-length until we truncate)
    all_u_trials = []      # list of (T_trial, input_dim) arrays
    all_z_trials = []      # list of (T_trial, action_dim) arrays
    all_y_trials = []      # list of (T_trial, hidden_dim) arrays
    all_mask_trials = []   # list of (T_trial,) bool arrays
    all_lengths = []       # int per trial

    # Per-trial label storage
    label_modality_context = []
    label_coherence_sign = []
    label_correct_action = []

    with torch.no_grad():
        for ctx in modality_contexts:
            # Create environment for this modality context
            env = env_class(
                context_id=0,
                modality_context=ctx,
                **env_kwargs,
            )
            env.set_num_tasks(1)

            for trial_idx in range(n_trials_per_context):
                # Reset hidden state per trial (consistent with extract_behavior_with_hidden)
                h = model.reset_hidden(batch_size=1, device=torch_device, preset_value=0.0)

                # Reset environment for new trial
                obs, done = env.reset()

                # Extract trial labels from the environment's internal trial dict
                # env.env is the NeurogymWrapper's underlying ngym env (a gymnasium Wrapper)
                # env.env.unwrapped.trial contains: ground_truth, context, coh_1, coh_2
                try:
                    trial_info = env.env.unwrapped.trial
                    correct_action = int(trial_info['ground_truth'])  # 1 or 2
                    # coherence_sign: +1 when correct action is 1 (choice_1), -1 when 2 (choice_2)
                    coherence_sign = 1 if correct_action == 1 else -1
                except (AttributeError, KeyError):
                    # Fallback if trial info not accessible
                    correct_action = 0
                    coherence_sign = 0
                    warnings.warn(
                        f"Could not extract trial info for modality_context={ctx}, trial={trial_idx}. "
                        "Labels will be 0 for this trial.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                # Per-timestep recordings
                trial_u = []
                trial_z = []
                trial_y = []
                trial_mask = []  # bool per step: True during stimulus or decision

                # Check whether the underlying NeuroGym TrialEnv exposes in_period.
                # Access via env.env.unwrapped (NeurogymWrapper.env is the ngym.make()
                # return value, and .unwrapped is the raw TrialEnv).
                try:
                    _uw = env.env.unwrapped
                    _has_periods = callable(getattr(_uw, 'in_period', None))
                except AttributeError:
                    _uw = None
                    _has_periods = False

                norm_obs = env.normalize_states(obs)
                reward = 0.0
                state = np.concatenate([norm_obs, env.context, [reward]])

                steps = 0
                while not done and steps < max_steps:
                    # Record u BEFORE forward pass (the input that produced next hidden state)
                    u_t = state.astype(np.float32).copy()
                    trial_u.append(u_t)

                    # Record task-active mask at this timestep (before step).
                    # True only during 'stimulus' or 'decision' periods; fixation and
                    # delay are excluded (they carry no discrimination signal).
                    try:
                        if _has_periods:
                            mask_t = bool(
                                _uw.in_period('stimulus') or _uw.in_period('decision')
                            )
                        else:
                            mask_t = True  # fallback: treat all as active
                    except Exception:
                        mask_t = True  # fallback on any period-query failure
                    trial_mask.append(mask_t)

                    # Forward pass
                    x = (
                        torch.tensor(state, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(torch_device)
                    )
                    actor_logits, _, h = model(x, h)

                    # Record z (softmax policy beliefs, bounded [0,1]) and y (hidden state)
                    action_probs = torch.softmax(actor_logits, dim=-1)
                    z_t = action_probs.squeeze().cpu().numpy().copy()  # (action_dim,)
                    y_t = h.squeeze().cpu().numpy().copy()             # (hidden_dim,)
                    trial_z.append(z_t)
                    trial_y.append(y_t)

                    # Select action (deterministic, consistent with extract_behavior_with_hidden)
                    action = torch.argmax(action_probs, dim=-1).item()

                    # Step environment
                    obs, reward, done = env.step(action)
                    norm_obs = env.normalize_states(obs)
                    state = np.concatenate([norm_obs, env.context, [reward]])
                    steps += 1

                if not _has_periods and steps > 0:
                    warnings.warn(
                        f"env.env.unwrapped does not expose in_period() for "
                        f"modality_context={ctx}. task_active_mask defaults to "
                        "all-True for this trial (backward-compat fallback).",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                # Store trial data if at least 1 step was taken
                T_trial = len(trial_u)
                if T_trial > 0:
                    all_u_trials.append(np.stack(trial_u))              # (T_trial, input_dim)
                    all_z_trials.append(np.stack(trial_z))              # (T_trial, action_dim)
                    all_y_trials.append(np.stack(trial_y))              # (T_trial, hidden_dim)
                    all_mask_trials.append(np.array(trial_mask, dtype=bool))  # (T_trial,)
                    all_lengths.append(T_trial)
                    label_modality_context.append(ctx)
                    label_coherence_sign.append(coherence_sign)
                    label_correct_action.append(correct_action)
                else:
                    warnings.warn(
                        f"Trial {trial_idx} for modality_context={ctx} had 0 steps — skipped.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    # Determine uniform trial length by truncating to the minimum observed
    if not all_lengths:
        raise ValueError("No valid trials collected. Check environment and model configuration.")

    T = int(min(all_lengths))
    n_total = len(all_u_trials)

    print(f"Collected {n_total} trials. Trial lengths: min={T}, max={max(all_lengths)}")
    if T != max(all_lengths):
        print(f"  Truncating all trials to T={T} (minimum observed length).")

    # Stack into uniform arrays (truncate to T)
    u = np.stack([arr[:T] for arr in all_u_trials]).astype(np.float32)   # (n_total, T, input_dim)
    z = np.stack([arr[:T] for arr in all_z_trials]).astype(np.float32)   # (n_total, T, action_dim)
    y = np.stack([arr[:T] for arr in all_y_trials]).astype(np.float32)   # (n_total, T, hidden_dim)
    task_active_mask = np.stack(
        [arr[:T] for arr in all_mask_trials]
    )  # (n_total, T), bool

    # Sanity report: mean active steps per trial
    active_per_trial = task_active_mask.sum(axis=1)  # (n_total,)
    print(
        f"task_active_mask: mean_active={float(active_per_trial.mean()):.1f}, "
        f"min={int(active_per_trial.min())}, max={int(active_per_trial.max())} "
        f"steps (out of T={T})"
    )

    labels = {
        'modality_context': np.array(label_modality_context, dtype=np.int32),
        'coherence_sign':   np.array(label_coherence_sign,   dtype=np.int32),
        'correct_action':   np.array(label_correct_action,   dtype=np.int32),
    }

    print(f"Final shapes: u={u.shape}, z={z.shape}, y={y.shape}, "
          f"task_active_mask={task_active_mask.shape}")

    return {
        'u': u,
        'z': z,
        'y': y,
        'task_active_mask': task_active_mask,
        'labels': labels,
        'T': T,
        'n_trials': n_total,
        'modality_contexts': modality_contexts,
        'n_trials_per_context': n_trials_per_context,
    }


def save_circuit_data(data: dict, output_dir: str) -> None:
    """
    Save collected circuit data to disk.

    Saves u, z, y arrays and all label arrays as a single .npz file
    (circuit_data.npz). Label arrays are stored with 'labels_' prefix.
    Also saves a metadata JSON for quick inspection.

    Parameters
    ----------
    data : dict
        Output of collect_circuit_data().
    output_dir : str
        Directory to save outputs. Created if it does not exist.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build save dict: u, z, y, task_active_mask plus flattened label arrays
    save_dict = {
        'u': data['u'],
        'z': data['z'],
        'y': data['y'],
    }
    # Include task_active_mask if present (added in 03-05)
    if 'task_active_mask' in data:
        save_dict['task_active_mask'] = data['task_active_mask']

    labels = data.get('labels', {})
    for key, arr in labels.items():
        save_dict[f'labels_{key}'] = arr

    npz_path = out_dir / 'circuit_data.npz'
    np.savez(npz_path, **save_dict)

    # Save metadata JSON — all values cast to Python builtins for json.dump
    has_mask = 'task_active_mask' in data
    metadata = {
        'n_trials': int(data['n_trials']),
        'T': int(data['T']),
        'modality_contexts': list(data['modality_contexts']),
        'n_trials_per_context': int(data['n_trials_per_context']),
        'u_shape': [int(x) for x in data['u'].shape],
        'z_shape': [int(x) for x in data['z'].shape],
        'y_shape': [int(x) for x in data['y'].shape],
        'label_keys': [f'labels_{k}' for k in labels.keys()],
        'has_task_active_mask': has_mask,
    }
    if has_mask:
        mask = data['task_active_mask']
        active = mask.sum(axis=1)
        metadata['task_active_mask_shape'] = [int(x) for x in mask.shape]
        metadata['task_active_mean_steps'] = float(active.mean())
        metadata['task_active_min_steps'] = int(active.min())
        metadata['task_active_max_steps'] = int(active.max())

    meta_path = out_dir / 'circuit_data_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved circuit data to {out_dir}/:")
    print(
        f"  circuit_data.npz: u={data['u'].shape}, z={data['z'].shape}, "
        f"y={data['y'].shape}"
        + (f", task_active_mask={data['task_active_mask'].shape}" if has_mask else "")
    )
    print(f"  circuit_data_metadata.json")


def fit_latent_circuit_ensemble(
    u: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    n_inits: int = 100,
    n_latent: int = 8,
    epochs: int = 500,
    lr: float = 0.02,
    l_y: float = 1.0,
    weight_decay: float = 0.001,
    sigma_rec: float = 0.15,
    device: str = 'cpu',
    verbose: bool = True,
    include_output_loss: bool = True,
    task_active_mask: np.ndarray | None = None,
) -> dict:
    """
    Fit an ensemble of LatentNet instances and return the best by lowest nmse_y.

    Runs ``n_inits`` independent random initializations of LatentNet, each
    fitted for ``epochs`` epochs on the provided circuit data. The initialization
    with the lowest normalized MSE between projected latent states and full RNN
    hidden states (``nmse_y``) is selected as the best solution.

    When ``task_active_mask`` is provided, the NMSE_y (and mse_z) loss is
    computed over task-active timesteps only (stimulus + decision periods),
    ignoring fixation, delay, and blank/padding steps. This implements Gap 1
    of Phase 3.1: masked-loss fitting to avoid the T=75 padding hypothesis.

    Parameters
    ----------
    u : np.ndarray
        Input stimulus, shape (n_trials, T, input_size=7).
    z : np.ndarray
        Actor logits (target output), shape (n_trials, T, output_size=3).
    y : np.ndarray
        RNN hidden states, shape (n_trials, T, N=64).
    n_inits : int, optional
        Number of random initializations to run. Default is 100.
    n_latent : int, optional
        Latent circuit rank (n in LatentNet). Must be >= max(input_size, output_size).
        Default is 8.
    epochs : int, optional
        Training epochs per initialization. Default is 500.
    lr : float, optional
        Adam learning rate. Default is 0.02.
    l_y : float, optional
        Weight on hidden-state reconstruction term. Default is 1.0.
    weight_decay : float, optional
        Adam weight decay. Default is 0.001.
    sigma_rec : float, optional
        Recurrent noise level during LatentNet forward pass. Default is 0.15.
    device : str, optional
        Device for computation. Default is 'cpu'.
    verbose : bool, optional
        Print per-init progress. Default is True.
    task_active_mask : np.ndarray or None, optional
        Bool mask of shape (n_trials, T). When provided, loss is computed over
        masked timesteps only (NMSE_y and mse_z are both masked). When None,
        behavior is identical to the pre-03-05 unmasked fitting. Default None.

    Returns
    -------
    dict with keys:
        'best_model'     : LatentNet with best weights loaded
        'best_nmse_y'    : float — masked NMSE_y if mask provided, else full
        'best_mse_z'     : float
        'best_nmse_y_full' : float — full (unmasked) NMSE_y (only differs from
                             best_nmse_y when task_active_mask is provided)
        'best_init_idx'  : int (0-indexed)
        'all_nmse_y'     : list of floats (masked if mask provided)
        'all_mse_z'      : list of floats
        'all_nmse_y_full' : list of floats (full NMSE_y per init; equals
                            all_nmse_y when mask is None)
        'n_inits'        : int
        'n_latent'       : int
        'masked'         : bool — True if task_active_mask was applied
    """
    # Convert numpy arrays to float32 tensors on the target device
    torch_device = torch.device(device)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=torch_device).detach()
    z_tensor = torch.tensor(z, dtype=torch.float32, device=torch_device).detach()
    y_tensor = torch.tensor(y, dtype=torch.float32, device=torch_device).detach()

    # Build mask tensor (n_trials, T, 1) once; None when not masking
    mask_t: torch.Tensor | None = None
    if task_active_mask is not None:
        mask_t = torch.tensor(
            task_active_mask, dtype=torch.float32, device=torch_device
        ).unsqueeze(-1)  # (n_trials, T, 1)

    n_trials = u.shape[0]
    T = u.shape[1]
    input_size = u.shape[2]
    output_size = z.shape[2]
    N = y.shape[2]

    all_nmse_y = []       # masked if mask_t provided, else full
    all_nmse_y_full = []  # always full NMSE_y (for cross-comparability with Wave A)
    all_mse_z = []
    all_state_dicts = []

    import time as _time
    t_start = _time.time()

    for i in range(n_inits):
        t_init_start = _time.time()

        # Create fresh LatentNet and move entire module to device
        latent_net = LatentNet(
            n=n_latent,
            N=N,
            input_size=input_size,
            n_trials=n_trials,
            sigma_rec=sigma_rec,
            output_size=output_size,
            device=device,
        )
        latent_net = latent_net.to(torch_device)

        if mask_t is not None:
            # Masked fitting: replicate LatentNet.fit()'s mini-batch loop but
            # substitute masked NMSE_y and mse_z. mask_t is (n_trials, T, 1).
            _batch_size = 128
            optimizer = torch.optim.Adam(
                latent_net.parameters(), lr=lr, weight_decay=weight_decay
            )
            latent_net.train()
            # Re-init q on correct device (mirrors LatentNet.fit() preamble)
            latent_net.q = latent_net.cayley_transform(latent_net.a)
            latent_net.connectivity_masks()

            for _epoch in range(epochs):
                indices = torch.randperm(n_trials, device=torch_device)
                for _start in range(0, n_trials, _batch_size):
                    idx = indices[_start: _start + _batch_size]
                    u_b = u_tensor[idx]
                    z_b = z_tensor[idx]
                    y_b = y_tensor[idx]
                    m_b = mask_t[idx]  # (batch, T, 1)

                    optimizer.zero_grad()
                    x_b = latent_net(u_b)  # (batch, T, n_latent)

                    # Masked NMSE_y: residuals zeroed at non-active timesteps
                    # x_b @ q: (batch, T, N) where q has shape (n_latent, N)
                    y_pred_b = x_b @ latent_net.q
                    resid_sq_y = ((y_pred_b - y_b) ** 2) * m_b
                    denom_sq_y = (y_b ** 2) * m_b
                    nmse_y_loss = (
                        resid_sq_y.sum() / denom_sq_y.sum().clamp_min(1e-12)
                    )

                    if include_output_loss:
                        z_pred_b = latent_net.output_layer(x_b)
                        resid_sq_z = ((z_pred_b - z_b) ** 2) * m_b
                        denom_sq_z = (z_b ** 2) * m_b
                        mse_z_loss = (
                            resid_sq_z.sum() / denom_sq_z.sum().clamp_min(1e-12)
                        )
                        loss = mse_z_loss + l_y * nmse_y_loss
                    else:
                        loss = nmse_y_loss

                    loss.backward()
                    optimizer.step()
                    # Re-orthonormalize Q and re-apply connectivity masks
                    # (mirrors what LatentNet.fit() does after each batch)
                    latent_net.q = latent_net.cayley_transform(latent_net.a)
                    latent_net.connectivity_masks()
            latent_net.eval()
        else:
            # Standard fitting (no mask) — identical to pre-03-05 behavior
            latent_net.fit(
                u_tensor, z_tensor, y_tensor,
                epochs=epochs,
                lr=lr,
                l_y=l_y,
                weight_decay=weight_decay,
                verbose=False,
                include_output_loss=include_output_loss,
            )

        # Compute final metrics
        with torch.no_grad():
            x = latent_net(u_tensor)
            # Full (unmasked) NMSE_y — always computed for cross-wave comparability
            nmse_y_full = latent_net.nmse_y(y_tensor, x).item()
            mse_z = latent_net.mse_z(x, z_tensor).item()

            if mask_t is not None:
                # Masked NMSE_y at inference time (mirrors training loss)
                y_pred_val = x @ latent_net.q
                resid_sq = ((y_pred_val - y_tensor) ** 2) * mask_t
                denom_sq = (y_tensor ** 2) * mask_t
                nmse_y = float(resid_sq.sum() / denom_sq.sum().clamp_min(1e-12))
            else:
                nmse_y = nmse_y_full

        all_nmse_y.append(nmse_y)
        all_nmse_y_full.append(nmse_y_full)
        all_mse_z.append(mse_z)
        # Move state_dict to CPU before storing (saves GPU memory)
        all_state_dicts.append({k: v.cpu() for k, v in latent_net.state_dict().items()})

        t_init_elapsed = _time.time() - t_init_start
        t_total_elapsed = _time.time() - t_start
        eta = t_init_elapsed * (n_inits - i - 1)

        if verbose:
            if mask_t is not None:
                print(
                    f"Init {i + 1}/{n_inits}: nmse_y(masked)={nmse_y:.4f}, "
                    f"nmse_y(full)={nmse_y_full:.4f}, mse_z={mse_z:.4f} "
                    f"({t_init_elapsed:.1f}s, ETA {eta/60:.1f}min)",
                    flush=True,
                )
            else:
                print(
                    f"Init {i + 1}/{n_inits}: nmse_y={nmse_y:.4f}, mse_z={mse_z:.4f} "
                    f"({t_init_elapsed:.1f}s, ETA {eta/60:.1f}min)",
                    flush=True,
                )

        # Free memory between inits
        del latent_net
        torch.cuda.empty_cache()
        gc.collect()

    # Select best initialization by lowest nmse_y
    # (masked nmse_y when mask provided; full otherwise — preserves Wave A
    # selection semantics when mask is None)
    best_init_idx = int(np.argmin(all_nmse_y))
    best_nmse_y = all_nmse_y[best_init_idx]
    best_nmse_y_full = all_nmse_y_full[best_init_idx]
    best_mse_z = all_mse_z[best_init_idx]

    if verbose:
        if mask_t is not None:
            print(
                f"\nBest init: {best_init_idx} "
                f"(nmse_y(masked)={best_nmse_y:.4f}, "
                f"nmse_y(full)={best_nmse_y_full:.4f}, mse_z={best_mse_z:.4f})"
            )
        else:
            print(
                f"\nBest init: {best_init_idx} "
                f"(nmse_y={best_nmse_y:.4f}, mse_z={best_mse_z:.4f})"
            )

    # Reload best state_dict into a fresh LatentNet on the same device
    best_model = LatentNet(
        n=n_latent,
        N=N,
        input_size=input_size,
        n_trials=n_trials,
        sigma_rec=sigma_rec,
        output_size=output_size,
        device=device,
    )
    best_model = best_model.to(torch_device)
    best_model.load_state_dict(all_state_dicts[best_init_idx])
    best_model.eval()

    return {
        'best_model': best_model,
        'best_nmse_y': best_nmse_y,
        'best_nmse_y_full': best_nmse_y_full,
        'best_mse_z': best_mse_z,
        'best_init_idx': best_init_idx,
        'all_nmse_y': all_nmse_y,
        'all_nmse_y_full': all_nmse_y_full,
        'all_mse_z': all_mse_z,
        'n_inits': n_inits,
        'n_latent': n_latent,
        'masked': mask_t is not None,
    }


def validate_latent_circuit(
    latent_net: LatentNet,
    u: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    W_rec: np.ndarray,
    W_in: np.ndarray = None,
    labels: dict = None,
    invariant_threshold: float = 0.85,
    device: str = 'cpu',
) -> dict:
    """
    Validate a fitted LatentNet against the source RNN and task data.

    Computes four validation metrics:
    1. Invariant subspace correlation: corr(q @ W_rec @ q.T, w_rec)  [connectivity level]
    2. Per-trial activity R-squared in full space: R^2(Qx, y)
    3. Per-trial activity R-squared in latent space: R^2(x, Q^T y)
    4. Trial-averaged dynamics R-squared grouped by modality_context condition

    Parameters
    ----------
    latent_net : LatentNet
        Fitted LatentNet instance.
    u : np.ndarray
        Input stimulus, shape (n_trials, T, input_size).
    z : np.ndarray
        Actor logits, shape (n_trials, T, output_size).
    y : np.ndarray
        RNN hidden states, shape (n_trials, T, N).
    W_rec : np.ndarray
        RNN hidden-to-hidden weight matrix (weight_hh_l0), shape (N, N).
    W_in : np.ndarray, optional
        RNN input-to-hidden weight matrix (weight_ih_l0). Currently not used
        in validation but accepted for API completeness.
    labels : dict, optional
        Per-trial condition labels from collect_circuit_data(). Must contain
        'modality_context' key. If None, trial-averaged check is skipped.
    invariant_threshold : float, optional
        Threshold for invariant subspace correlation pass/fail. Default is 0.85.
    device : str, optional
        Device for computation. Default is 'cpu'.

    Returns
    -------
    dict with keys:
        'invariant_subspace_corr'    : float
        'invariant_subspace_pass'    : bool
        'activity_r2_full_space'     : float  (per-trial)
        'activity_r2_latent_space'   : float  (per-trial)
        'trial_avg_r2_full_space'    : float or None
        'trial_avg_r2_latent_space'  : float or None
        'trial_avg_r2_by_condition'  : dict or None  {condition_id: float}
        'nmse_y'                     : float
        'nmse_q'                     : float
        'mse_z'                      : float
        'threshold'                  : float
    """
    latent_net.eval()

    torch_device = torch.device(device)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=torch_device).detach()
    z_tensor = torch.tensor(z, dtype=torch.float32, device=torch_device).detach()
    y_tensor = torch.tensor(y, dtype=torch.float32, device=torch_device).detach()

    with torch.no_grad():
        # Run forward pass to get latent states x: (n_trials, T, n)
        x = latent_net(u_tensor)

        # --- Reconstruction metrics ---
        nmse_y_val = latent_net.nmse_y(y_tensor, x).item()
        nmse_q_val = latent_net.nmse_q(y_tensor).item()
        mse_z_val = latent_net.mse_z(x, z_tensor).item()

        # q has shape (n, N); q.T has shape (N, n)
        q = latent_net.q.detach()   # (n, N)

        # --- Check 1: Invariant subspace (connectivity level) ---
        # Paper Eq. 14: q @ W_rec @ q.T should approximate w_rec (inferred latent recurrent matrix)
        # q: (n, N), W_rec: (N, N), q.T: (N, n) → result: (n, n)
        w_rec_inferred = latent_net.recurrent_layer.weight.data.detach().cpu().numpy()  # (n, n)
        q_np = q.cpu().numpy()     # (n, N)
        Q_W_Q = q_np @ W_rec @ q_np.T  # (n, n)
        inv_corr = float(np.corrcoef(Q_W_Q.flatten(), w_rec_inferred.flatten())[0, 1])
        inv_pass = inv_corr >= invariant_threshold

        # --- Check 2: Per-trial activity R-squared in full space (Qx vs y) ---
        # Qx: x @ q  gives (n_trials, T, N)
        qx = (x @ q).cpu().numpy()       # (n_trials, T, N)
        y_np = y_tensor.cpu().numpy()    # (n_trials, T, N)

        ss_res_full = np.sum((qx - y_np) ** 2)
        y_mean_full = y_np.mean()
        ss_tot_full = np.sum((y_np - y_mean_full) ** 2)
        r2_full = float(1.0 - ss_res_full / ss_tot_full) if ss_tot_full > 0 else float('nan')

        # --- Check 3: Per-trial activity R-squared in latent space (Q^T y vs x) ---
        # Q^T y: y_tensor @ q.T  gives (n_trials, T, n)  (q.T is (N, n))
        x_np = x.cpu().numpy()                       # (n_trials, T, n)
        q_y = (y_tensor @ q.t()).cpu().numpy()      # (n_trials, T, n)

        ss_res_lat = np.sum((q_y - x_np) ** 2)
        x_mean = x_np.mean()
        ss_tot_lat = np.sum((x_np - x_mean) ** 2)
        r2_latent = float(1.0 - ss_res_lat / ss_tot_lat) if ss_tot_lat > 0 else float('nan')

        # --- Check 4: Trial-averaged dynamics R-squared (grouped by condition) ---
        trial_avg_r2_full = None
        trial_avg_r2_latent = None
        trial_avg_r2_by_condition = None

        if labels is not None and 'modality_context' in labels:
            conditions = np.unique(labels['modality_context'])
            condition_r2_full = {}
            condition_r2_latent = {}

            for cond in conditions:
                mask = labels['modality_context'] == cond

                # Average actual RNN hidden states across trials in this condition
                y_avg = y_np[mask].mean(axis=0)    # (T, N)
                # Average projected latent states across trials in this condition
                qx_avg = qx[mask].mean(axis=0)     # (T, N)
                # Average latent states across trials
                x_avg = x_np[mask].mean(axis=0)    # (T, n)
                # Average projected y in latent space
                qy_avg = q_y[mask].mean(axis=0)    # (T, n)

                # R^2 in full space for this condition
                ss_res_c = np.sum((qx_avg - y_avg) ** 2)
                ss_tot_c = np.sum((y_avg - y_avg.mean()) ** 2)
                r2_c_full = float(1.0 - ss_res_c / ss_tot_c) if ss_tot_c > 0 else float('nan')
                condition_r2_full[int(cond)] = r2_c_full

                # R^2 in latent space for this condition
                ss_res_cl = np.sum((qy_avg - x_avg) ** 2)
                ss_tot_cl = np.sum((x_avg - x_avg.mean()) ** 2)
                r2_c_latent = float(1.0 - ss_res_cl / ss_tot_cl) if ss_tot_cl > 0 else float('nan')
                condition_r2_latent[int(cond)] = r2_c_latent

            # Mean across conditions
            valid_full = [v for v in condition_r2_full.values() if not np.isnan(v)]
            valid_latent = [v for v in condition_r2_latent.values() if not np.isnan(v)]
            trial_avg_r2_full = float(np.mean(valid_full)) if valid_full else float('nan')
            trial_avg_r2_latent = float(np.mean(valid_latent)) if valid_latent else float('nan')
            trial_avg_r2_by_condition = condition_r2_full
        else:
            warnings.warn(
                "labels is None or missing 'modality_context' key. "
                "Skipping trial-averaged dynamics R-squared check.",
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        'invariant_subspace_corr': inv_corr,
        'invariant_subspace_pass': inv_pass,
        'activity_r2_full_space': r2_full,
        'activity_r2_latent_space': r2_latent,
        'trial_avg_r2_full_space': trial_avg_r2_full,
        'trial_avg_r2_latent_space': trial_avg_r2_latent,
        'trial_avg_r2_by_condition': trial_avg_r2_by_condition,
        'nmse_y': nmse_y_val,
        'nmse_q': nmse_q_val,
        'mse_z': mse_z_val,
        'threshold': invariant_threshold,
    }


def perturb_and_evaluate(
    model: "ContinuousActorCritic",
    latent_net: LatentNet,
    env_class=SingleContextDecisionMakingWrapper,
    modality_contexts: list[int] | None = None,
    n_eval_trials: int = 200,
    perturbation_strengths: list[float] | None = None,
    n_baseline_runs: int = 5,
    n_top_connections: int = 10,
    significance_k: float = 2.0,
    max_steps: int = 75,
    device: str = "cpu",
    env_kwargs: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Perturb the RNN's recurrent weights via Q-mapped rank-one perturbations.

    Implements Langdon & Engel 2025 Eq. 6/23: a rank-one perturbation delta_ij
    in the n-dim latent circuit corresponds to a rank-one perturbation in the
    full N-dim RNN weight space via W_rec' = W_rec + (q.T @ delta_ij @ q),
    where q has shape (n, N) (code convention; paper's Q is q.T).

    Identifies the top |n_top_connections| largest-magnitude connections in the
    inferred latent w_rec, applies each at every strength in
    perturbation_strengths, re-evaluates the perturbed model on
    ContextDecisionMaking, and reports reward deltas vs an unperturbed
    baseline. Significance is judged against the natural trial-sampling
    variability estimated from |n_baseline_runs| unperturbed runs.

    Parameters
    ----------
    model : ContinuousActorCritic
        Trained RNN (matched to the LatentNet via 03-02's ReLU rewrite).
        Will be mutated in-place (W_hh.weight) per perturbation and restored.
    latent_net : LatentNet
        Wave A's chosen LatentNet (loaded from best_latent_circuit_waveA.pt).
        Provides q (shape (n, N)) and recurrent_layer.weight (shape (n, n)).
    env_class : type, optional
        Environment wrapper class. Default SingleContextDecisionMakingWrapper.
    modality_contexts : list of int, optional
        Contexts to evaluate. Default [0, 1].
    n_eval_trials : int, optional
        Trials per context per evaluation. Default 200.
    perturbation_strengths : list of float, optional
        Multipliers applied to delta_ij. Default [-0.5, -0.2, 0.0, 0.2, 0.5].
        The 0.0 entry serves as a same-conditions sanity check.
    n_baseline_runs : int, optional
        Independent unperturbed runs for baseline_std. Default 5.
    n_top_connections : int, optional
        Number of largest-magnitude w_rec entries to perturb. Default 10
        (top 5 positive + top 5 negative as in Fig. 4 examples).
    significance_k : float, optional
        Multiplier on baseline_std for significance threshold. Default 2.0
        (i.e. |delta| > 2 * baseline_std → significant).
    max_steps : int, optional
        Per-trial step cap inside the env loop. Default 75 (matches T from
        03-02 circuit_data regen). MUST be the per-trial step bound, NOT the
        number of trials. Lower this (e.g. max_steps=20) for smoke tests.
    device : str, optional
        Torch device. Default 'cpu' (perturbation eval is cheap; cluster GPU
        not needed).
    env_kwargs : dict, optional
        Extra kwargs forwarded to env_class. Default {}.
    seed : int, optional
        Optional torch / numpy seed for the baseline runs (perturbation runs
        each seed independently for reproducibility).

    Returns
    -------
    dict
        Keys: wave_a_chosen_rank, n_baseline_runs, n_eval_trials,
        modality_contexts, perturbation_strengths, n_top_connections,
        significance_k, baseline (with significance_threshold, mean/std per
        context and pooled), perturbations (list of per-perturbation dicts with
        reward_delta_by_context and significant flag), summary (n_significant,
        mean/max abs reward delta).

    Notes
    -----
    - W_hh.weight is restored after every perturbation eval. The model is
      semantically unchanged on return.
    - Reward is the per-trial environment reward (already softmax-bounded by
      ContextDecisionMaking-v0); accuracy ≈ mean reward at this task.
    """
    if modality_contexts is None:
        modality_contexts = [0, 1]
    if perturbation_strengths is None:
        perturbation_strengths = [-0.5, -0.2, 0.0, 0.2, 0.5]
    if env_kwargs is None:
        env_kwargs = {}

    torch_device = torch.device(device)
    model.eval()
    latent_net.eval()

    # --- Step 1: Extract Q and w_rec from Wave A's LatentNet ---
    with torch.no_grad():
        q = latent_net.q.detach().to(torch_device)                           # (n, N)
        w_rec_inferred = latent_net.recurrent_layer.weight.data.detach()     # (n, n)
    n = w_rec_inferred.shape[0]

    # --- Step 2: Helper — single-evaluation pass for one (context, model_state) ---
    def _eval_once(eval_seed: int | None) -> dict[int, float]:
        """Run n_eval_trials trials per context, return mean reward per context."""
        if eval_seed is not None:
            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)
        rewards_by_ctx: dict[int, list[float]] = {ctx: [] for ctx in modality_contexts}
        with torch.no_grad():
            for ctx in modality_contexts:
                env = env_class(context_id=0, modality_context=ctx, **env_kwargs)
                # The cluster retraining in 03-02 used collect_circuit_data() which
                # calls set_num_tasks(1), giving state = obs(5) + ctx(1) + reward(1)
                # = input_dim=7. set_num_tasks(2) would give 8-dim input and
                # mismatch the loaded W_ih (64×7). Use 1 here to match training.
                env.set_num_tasks(1)
                for _ in range(n_eval_trials):
                    h = model.reset_hidden(
                        batch_size=1, device=torch_device, preset_value=0.0
                    )
                    obs, done = env.reset()
                    norm_obs = env.normalize_states(obs)
                    reward = 0.0
                    state = np.concatenate([norm_obs, env.context, [reward]])
                    trial_total_reward = 0.0
                    steps = 0
                    while not done and steps < max_steps:
                        x = (
                            torch.tensor(state, dtype=torch.float32)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(torch_device)
                        )
                        actor_logits, _, h = model(x, h)
                        action = torch.argmax(
                            torch.softmax(actor_logits, dim=-1), dim=-1
                        ).item()
                        obs, reward, done = env.step(action)
                        trial_total_reward += float(reward)
                        norm_obs = env.normalize_states(obs)
                        state = np.concatenate([norm_obs, env.context, [reward]])
                        steps += 1
                    rewards_by_ctx[ctx].append(trial_total_reward)
        return {ctx: float(np.mean(rs)) for ctx, rs in rewards_by_ctx.items()}

    # --- Step 3: Establish baseline variability ---
    baseline_per_ctx_per_run: list[dict[int, float]] = []
    for run_idx in range(n_baseline_runs):
        seed_for_run = (seed + run_idx) if seed is not None else None
        baseline_per_ctx_per_run.append(_eval_once(seed_for_run))

    baseline_means_by_ctx = {
        ctx: float(np.mean([r[ctx] for r in baseline_per_ctx_per_run]))
        for ctx in modality_contexts
    }
    baseline_stds_by_ctx = {
        ctx: (
            float(np.std([r[ctx] for r in baseline_per_ctx_per_run], ddof=1))
            if n_baseline_runs > 1
            else 0.0
        )
        for ctx in modality_contexts
    }
    pooled_baseline_runs = [
        float(np.mean([r[ctx] for ctx in modality_contexts]))
        for r in baseline_per_ctx_per_run
    ]
    baseline_mean_pooled = float(np.mean(pooled_baseline_runs))
    baseline_std_pooled = (
        float(np.std(pooled_baseline_runs, ddof=1)) if n_baseline_runs > 1 else 0.0
    )
    significance_threshold = significance_k * baseline_std_pooled

    # --- Step 4: Identify top-magnitude latent w_rec connections ---
    w_rec_np = w_rec_inferred.cpu().numpy()   # (n, n)
    flat_indices = np.argsort(np.abs(w_rec_np).flatten())[::-1]   # descending magnitude
    top_pos: list[tuple[int, int, float]] = []   # (i, j, w_ij) with w_ij > 0
    top_neg: list[tuple[int, int, float]] = []   # (i, j, w_ij) with w_ij < 0
    for idx in flat_indices:
        i, j = int(idx // n), int(idx % n)
        w_ij = float(w_rec_np[i, j])
        if w_ij > 0 and len(top_pos) < n_top_connections // 2:
            top_pos.append((i, j, w_ij))
        elif w_ij < 0 and len(top_neg) < n_top_connections // 2:
            top_neg.append((i, j, w_ij))
        if len(top_pos) + len(top_neg) >= n_top_connections:
            break
    top_connections = top_pos + top_neg

    # --- Step 5: Apply rank-one perturbations and evaluate ---
    W_rec_original = model.W_hh.weight.data.clone()   # (N, N)
    results: list[dict] = []

    try:
        for i, j, w_ij in top_connections:
            for strength in perturbation_strengths:
                # Build delta_ij in latent space
                delta = torch.zeros(
                    n, n, dtype=W_rec_original.dtype, device=torch_device
                )
                delta[i, j] = float(strength)

                # Map to RNN space: q.T @ delta @ q  (paper Eq. 23 in code convention)
                # q: (n, N), delta: (n, n) → (N, n) @ (n, n) @ (n, N) = (N, N)
                W_rec_perturbation = (q.t() @ delta @ q).to(W_rec_original.dtype)

                # Apply perturbation
                model.W_hh.weight.data = W_rec_original + W_rec_perturbation

                # Evaluate (single pass, deterministic given seed)
                eval_seed_pert: int | None = None
                if seed is not None:
                    eval_seed_pert = (
                        seed * 1000 + i * 100 + j * 10 + int(abs(strength) * 10)
                    )
                pert_per_ctx = _eval_once(eval_seed_pert)
                pert_pooled = float(
                    np.mean([pert_per_ctx[ctx] for ctx in modality_contexts])
                )

                reward_delta_pooled = pert_pooled - baseline_mean_pooled
                reward_delta_by_ctx = {
                    int(ctx): float(pert_per_ctx[ctx] - baseline_means_by_ctx[ctx])
                    for ctx in modality_contexts
                }
                significant = bool(abs(reward_delta_pooled) > significance_threshold)

                results.append({
                    "i": int(i),
                    "j": int(j),
                    "w_rec_ij": float(w_ij),
                    "strength": float(strength),
                    "baseline_reward_pooled": float(baseline_mean_pooled),
                    "perturbed_reward_pooled": float(pert_pooled),
                    "reward_delta_pooled": float(reward_delta_pooled),
                    "baseline_reward_by_context": {
                        int(ctx): float(baseline_means_by_ctx[ctx])
                        for ctx in modality_contexts
                    },
                    "perturbed_reward_by_context": {
                        int(ctx): float(pert_per_ctx[ctx])
                        for ctx in modality_contexts
                    },
                    "reward_delta_by_context": reward_delta_by_ctx,
                    "significant": significant,
                })
    finally:
        # ALWAYS restore original weights, even if a perturbation eval raises
        model.W_hh.weight.data = W_rec_original

    # --- Step 6: Compile and return results (all Python builtins — no np scalars) ---
    abs_deltas = [abs(r["reward_delta_pooled"]) for r in results]
    return {
        "wave_a_chosen_rank": int(n),
        "n_baseline_runs": int(n_baseline_runs),
        "n_eval_trials": int(n_eval_trials),
        "modality_contexts": [int(c) for c in modality_contexts],
        "perturbation_strengths": [float(s) for s in perturbation_strengths],
        "n_top_connections": int(n_top_connections),
        "significance_k": float(significance_k),
        "baseline": {
            "mean_reward_pooled": float(baseline_mean_pooled),
            "std_reward_pooled": float(baseline_std_pooled),
            "mean_reward_by_context": {
                int(c): float(baseline_means_by_ctx[c]) for c in modality_contexts
            },
            "std_reward_by_context": {
                int(c): float(baseline_stds_by_ctx[c]) for c in modality_contexts
            },
            "significance_threshold": float(significance_threshold),
        },
        "perturbations": results,
        "summary": {
            "n_perturbations": int(len(results)),
            "n_significant": int(sum(1 for r in results if r["significant"])),
            "mean_abs_reward_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
            "max_abs_reward_delta": float(np.max(abs_deltas)) if abs_deltas else 0.0,
            "max_significant_delta": float(
                max(
                    (r["reward_delta_pooled"] for r in results if r["significant"]),
                    default=0.0,
                    key=abs,
                )
            ),
        },
    }
