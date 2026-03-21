"""
Circuit data collection for latent circuit inference.

Provides collect_circuit_data() to extract (u, z, y) tensors from a trained
ActorCritic model running ContextDecisionMaking-v0, for use with LatentNet.fit().

The three tensors follow the LatentNet convention:
  u: input stimulus at each timestep  (n_trials, T, input_dim)
  z: actor logits at each timestep    (n_trials, T, action_dim)
  y: RNN hidden states                (n_trials, T, hidden_dim)

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
    all_u_trials = []   # list of (T_trial, input_dim) arrays
    all_z_trials = []   # list of (T_trial, action_dim) arrays
    all_y_trials = []   # list of (T_trial, hidden_dim) arrays
    all_lengths = []    # int per trial

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

                norm_obs = env.normalize_states(obs)
                reward = 0.0
                state = np.concatenate([norm_obs, env.context, [reward]])

                steps = 0
                while not done and steps < max_steps:
                    # Record u BEFORE forward pass (the input that produced next hidden state)
                    u_t = state.astype(np.float32).copy()
                    trial_u.append(u_t)

                    # Forward pass
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(torch_device)
                    actor_logits, _, h = model(x, h)

                    # Record z (actor logits, raw — not softmax) and y (hidden state)
                    z_t = actor_logits.squeeze().cpu().numpy().copy()  # (action_dim,)
                    y_t = h.squeeze().cpu().numpy().copy()             # (hidden_dim,)
                    trial_z.append(z_t)
                    trial_y.append(y_t)

                    # Select action (deterministic, consistent with extract_behavior_with_hidden)
                    action_probs = torch.softmax(actor_logits, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).item()

                    # Step environment
                    obs, reward, done = env.step(action)
                    norm_obs = env.normalize_states(obs)
                    state = np.concatenate([norm_obs, env.context, [reward]])
                    steps += 1

                # Store trial data if at least 1 step was taken
                T_trial = len(trial_u)
                if T_trial > 0:
                    all_u_trials.append(np.stack(trial_u))   # (T_trial, input_dim)
                    all_z_trials.append(np.stack(trial_z))   # (T_trial, action_dim)
                    all_y_trials.append(np.stack(trial_y))   # (T_trial, hidden_dim)
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
    u = np.stack([arr[:T] for arr in all_u_trials]).astype(np.float32)  # (n_total, T, input_dim)
    z = np.stack([arr[:T] for arr in all_z_trials]).astype(np.float32)  # (n_total, T, action_dim)
    y = np.stack([arr[:T] for arr in all_y_trials]).astype(np.float32)  # (n_total, T, hidden_dim)

    labels = {
        'modality_context': np.array(label_modality_context, dtype=np.int32),
        'coherence_sign':   np.array(label_coherence_sign,   dtype=np.int32),
        'correct_action':   np.array(label_correct_action,   dtype=np.int32),
    }

    print(f"Final shapes: u={u.shape}, z={z.shape}, y={y.shape}")

    return {
        'u': u,
        'z': z,
        'y': y,
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

    # Build save dict: u, z, y plus flattened label arrays
    save_dict = {
        'u': data['u'],
        'z': data['z'],
        'y': data['y'],
    }
    labels = data.get('labels', {})
    for key, arr in labels.items():
        save_dict[f'labels_{key}'] = arr

    npz_path = out_dir / 'circuit_data.npz'
    np.savez(npz_path, **save_dict)

    # Save metadata JSON
    metadata = {
        'n_trials': int(data['n_trials']),
        'T': int(data['T']),
        'modality_contexts': list(data['modality_contexts']),
        'n_trials_per_context': int(data['n_trials_per_context']),
        'u_shape': list(data['u'].shape),
        'z_shape': list(data['z'].shape),
        'y_shape': list(data['y'].shape),
        'label_keys': [f'labels_{k}' for k in labels.keys()],
    }
    meta_path = out_dir / 'circuit_data_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved circuit data to {out_dir}/:")
    print(f"  circuit_data.npz: u={data['u'].shape}, z={data['z'].shape}, y={data['y'].shape}")
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
) -> dict:
    """
    Fit an ensemble of LatentNet instances and return the best by lowest nmse_y.

    Runs ``n_inits`` independent random initializations of LatentNet, each
    fitted for ``epochs`` epochs on the provided circuit data. The initialization
    with the lowest normalized MSE between projected latent states and full RNN
    hidden states (``nmse_y``) is selected as the best solution.

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

    Returns
    -------
    dict with keys:
        'best_model'    : LatentNet with best weights loaded
        'best_nmse_y'   : float
        'best_mse_z'    : float
        'best_init_idx' : int (0-indexed)
        'all_nmse_y'    : list of floats
        'all_mse_z'     : list of floats
        'n_inits'       : int
        'n_latent'      : int
    """
    # Convert numpy arrays to float32 tensors (detached, no grad)
    u_tensor = torch.tensor(u, dtype=torch.float32).detach()
    z_tensor = torch.tensor(z, dtype=torch.float32).detach()
    y_tensor = torch.tensor(y, dtype=torch.float32).detach()

    n_trials = u.shape[0]
    T = u.shape[1]
    input_size = u.shape[2]
    output_size = z.shape[2]
    N = y.shape[2]

    all_nmse_y = []
    all_mse_z = []
    all_state_dicts = []

    for i in range(n_inits):
        # Create fresh LatentNet for this initialization
        latent_net = LatentNet(
            n=n_latent,
            N=N,
            input_size=input_size,
            n_trials=n_trials,
            sigma_rec=sigma_rec,
            output_size=output_size,
            device=device,
        )

        # Fit (suppress LatentNet's internal epoch prints during ensemble)
        latent_net.fit(
            u_tensor, z_tensor, y_tensor,
            epochs=epochs,
            lr=lr,
            l_y=l_y,
            weight_decay=weight_decay,
            verbose=False,
        )

        # Compute final metrics
        with torch.no_grad():
            x = latent_net(u_tensor)
            nmse_y = latent_net.nmse_y(y_tensor, x).item()
            mse_z = latent_net.mse_z(x, z_tensor).item()

        all_nmse_y.append(nmse_y)
        all_mse_z.append(mse_z)
        all_state_dicts.append(copy.deepcopy(latent_net.state_dict()))

        if verbose:
            print(f"Init {i + 1}/{n_inits}: nmse_y={nmse_y:.4f}, mse_z={mse_z:.4f}")

        # Free memory between inits
        del latent_net
        torch.cuda.empty_cache()
        gc.collect()

    # Select best initialization by lowest nmse_y
    best_init_idx = int(np.argmin(all_nmse_y))
    best_nmse_y = all_nmse_y[best_init_idx]
    best_mse_z = all_mse_z[best_init_idx]

    if verbose:
        print(f"\nBest init: {best_init_idx} (nmse_y={best_nmse_y:.4f}, mse_z={best_mse_z:.4f})")

    # Reload best state_dict into a fresh LatentNet
    best_model = LatentNet(
        n=n_latent,
        N=N,
        input_size=input_size,
        n_trials=n_trials,
        sigma_rec=sigma_rec,
        output_size=output_size,
        device=device,
    )
    best_model.load_state_dict(all_state_dicts[best_init_idx])
    best_model.eval()

    return {
        'best_model': best_model,
        'best_nmse_y': best_nmse_y,
        'best_mse_z': best_mse_z,
        'best_init_idx': best_init_idx,
        'all_nmse_y': all_nmse_y,
        'all_mse_z': all_mse_z,
        'n_inits': n_inits,
        'n_latent': n_latent,
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

    u_tensor = torch.tensor(u, dtype=torch.float32).detach()
    z_tensor = torch.tensor(z, dtype=torch.float32).detach()
    y_tensor = torch.tensor(y, dtype=torch.float32).detach()

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
        w_rec_inferred = latent_net.recurrent_layer.weight.data.detach().numpy()  # (n, n)
        q_np = q.numpy()           # (n, N)
        Q_W_Q = q_np @ W_rec @ q_np.T  # (n, n)
        inv_corr = float(np.corrcoef(Q_W_Q.flatten(), w_rec_inferred.flatten())[0, 1])
        inv_pass = inv_corr >= invariant_threshold

        # --- Check 2: Per-trial activity R-squared in full space (Qx vs y) ---
        # Qx: x @ q  gives (n_trials, T, N)
        qx = (x @ q).numpy()       # (n_trials, T, N)
        y_np = y_tensor.numpy()    # (n_trials, T, N)

        ss_res_full = np.sum((qx - y_np) ** 2)
        y_mean_full = y_np.mean()
        ss_tot_full = np.sum((y_np - y_mean_full) ** 2)
        r2_full = float(1.0 - ss_res_full / ss_tot_full) if ss_tot_full > 0 else float('nan')

        # --- Check 3: Per-trial activity R-squared in latent space (Q^T y vs x) ---
        # Q^T y: y_tensor @ q.T  gives (n_trials, T, n)  (q.T is (N, n))
        x_np = x.numpy()                           # (n_trials, T, n)
        q_y = (y_tensor @ q.t()).numpy()            # (n_trials, T, n)

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
