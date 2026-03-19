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

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nn4psych.models.actor_critic import ActorCritic
from envs.neurogym_wrapper import SingleContextDecisionMakingWrapper


def collect_circuit_data(
    model: ActorCritic,
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
    model : ActorCritic
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
