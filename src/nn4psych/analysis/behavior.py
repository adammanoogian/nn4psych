"""
Behavior extraction and analysis utilities.
"""

from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from nn4psych.models.actor_critic import ActorCritic
from envs import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2


def extract_behavior(
    model: ActorCritic,
    env,  # PIE_CP_OB_v2 or NeurogymWrapper — must have reset_epoch() and get_state_history()
    n_epochs: int = 100,
    n_trials: int = 200,
    reset_memory: bool = True,
    preset_memory: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple:
    """
    Extract behavioral data by running model through task.

    Parameters
    ----------
    model : ActorCritic
        Trained model to evaluate.
    env : PIE_CP_OB_v2
        Task environment.
    n_epochs : int, optional
        Number of epochs to run. Default is 100.
    n_trials : int, optional
        Number of trials per epoch. Default is 200.
    reset_memory : bool, optional
        Whether to reset hidden state between epochs. Default is True.
    preset_memory : float, optional
        Value to preset hidden state to. Default is 0.0.
    device : torch.device, optional
        Device to run model on.

    Returns
    -------
    all_states : list
        List of state tuples for each epoch.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_states = []

    with torch.no_grad():
        for epoch in range(n_epochs):
            # Reset hidden state
            if reset_memory:
                h = model.reset_hidden(batch_size=1, device=device, preset_value=preset_memory)
            else:
                if epoch == 0:
                    h = model.get_initial_hidden(batch_size=1, device=device)

            # Reset environment for epoch
            env.reset_epoch()

            # Run trials
            reward = 0.0  # Initialize reward
            for trial in range(n_trials):
                obs, done = env.reset()
                norm_obs = env.normalize_states(obs)
                state = np.concatenate([norm_obs, env.context, [reward]])

                while not done:
                    # Get model action
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    actor_logits, _, h = model(x, h)
                    action_probs = torch.softmax(actor_logits, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).item()

                    # Take action
                    obs, reward, done = env.step(action)
                    norm_obs = env.normalize_states(obs)
                    state = np.concatenate([norm_obs, env.context, [reward]])

            # Store epoch data
            states = env.get_state_history()
            all_states.append(states)

    return all_states


def extract_behavior_with_hidden(
    model: ActorCritic,
    env,  # PIE_CP_OB_v2 or NeurogymWrapper — must have reset_epoch() and get_state_history()
    n_epochs: int = 1,
    n_trials: int = 200,
    reset_memory: bool = True,
    preset_memory: float = 0.0,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Extract behavioral data AND hidden states for downstream analysis.

    Runs the model through the environment for n_epochs x n_trials, recording
    the RNN hidden state at every timestep. Hidden states are padded to
    max trial length with NaN values.

    Parameters
    ----------
    model : ActorCritic
        Trained model to evaluate.
    env : PIE_CP_OB_v2 or NeurogymWrapper
        Task environment. Must have reset_epoch(), reset(), step(),
        normalize_states(), context, and get_state_history() methods.
    n_epochs : int, optional
        Number of epochs to run. Default is 1.
    n_trials : int, optional
        Number of trials per epoch. Default is 200.
    reset_memory : bool, optional
        Whether to reset hidden state between epochs. Default is True.
    preset_memory : float, optional
        Value to preset hidden state to. Default is 0.0.
    device : torch.device, optional
        Device to run model on.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'states': list of epoch state tuples (from get_state_history())
        - 'hidden': np.ndarray, shape (n_epochs * n_trials, max_T, hidden_dim)
          Hidden states padded with NaN to max trial length.
        - 'trial_lengths': np.ndarray, shape (n_epochs * n_trials,)
          Actual number of timesteps per trial.
        - 'actions': list of lists, one per trial, each containing action ints.
        - 'rewards': list of lists, one per trial, each containing reward floats.
    """
    if device is None:
        device = next(model.parameters()).device

    hidden_dim = model.hidden_dim
    model.eval()

    all_states = []
    all_hidden = []       # list of (T_trial, hidden_dim) arrays
    all_lengths = []      # int per trial
    all_actions = []      # list of action lists per trial
    all_rewards = []      # list of reward lists per trial

    with torch.no_grad():
        for epoch in range(n_epochs):
            # Reset hidden state
            if reset_memory:
                h = model.reset_hidden(batch_size=1, device=device, preset_value=preset_memory)
            else:
                if epoch == 0:
                    h = model.get_initial_hidden(batch_size=1, device=device)

            # Reset environment for epoch
            env.reset_epoch()

            # Run trials
            reward = 0.0
            for trial in range(n_trials):
                obs, done = env.reset()
                norm_obs = env.normalize_states(obs)
                state = np.concatenate([norm_obs, env.context, [reward]])

                trial_hidden = []
                trial_actions = []
                trial_rewards = []

                while not done:
                    # Forward pass
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    actor_logits, _, h = model(x, h)

                    # Record hidden state (squeeze to (hidden_dim,))
                    trial_hidden.append(h.squeeze().cpu().numpy().copy())

                    # Select action
                    action_probs = torch.softmax(actor_logits, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).item()

                    # Step environment
                    obs, reward, done = env.step(action)
                    norm_obs = env.normalize_states(obs)
                    state = np.concatenate([norm_obs, env.context, [reward]])

                    trial_actions.append(action)
                    trial_rewards.append(reward)

                # Store trial data
                if len(trial_hidden) > 0:
                    all_hidden.append(np.stack(trial_hidden))  # (T_trial, hidden_dim)
                else:
                    # Edge case: trial ended immediately (0 steps)
                    all_hidden.append(np.empty((0, hidden_dim)))
                all_lengths.append(len(trial_hidden))
                all_actions.append(trial_actions)
                all_rewards.append(trial_rewards)

            # Store epoch state history
            states = env.get_state_history()
            all_states.append(states)

    # Pad hidden states to (n_total_trials, max_T, hidden_dim)
    n_total = len(all_hidden)
    max_T = max(all_lengths) if all_lengths else 0

    if max_T > 0 and n_total > 0:
        hidden_padded = np.full((n_total, max_T, hidden_dim), np.nan)
        for i, h_trial in enumerate(all_hidden):
            T = h_trial.shape[0]
            if T > 0:
                hidden_padded[i, :T, :] = h_trial
    else:
        hidden_padded = np.empty((n_total, 0, hidden_dim))

    return {
        'states': all_states,
        'hidden': hidden_padded,
        'trial_lengths': np.array(all_lengths, dtype=np.int32),
        'actions': all_actions,
        'rewards': all_rewards,
    }


def get_area(
    model_path: Path,
    epochs: int = 100,
    reset_memory: bool = True,
    model_class: type = ActorCritic,
    input_dim: int = 9,
    hidden_dim: int = 64,
    action_dim: int = 3,
    env_params: Optional[Dict] = None,
) -> float:
    """
    Calculate performance area under learning curve for a trained model.

    Parameters
    ----------
    model_path : Path
        Path to model weights file.
    epochs : int, optional
        Number of evaluation epochs.
    reset_memory : bool, optional
        Whether to reset memory between epochs.
    model_class : type, optional
        Model class to instantiate.
    input_dim : int, optional
        Model input dimension.
    hidden_dim : int, optional
        Model hidden dimension.
    action_dim : int, optional
        Model action dimension.
    env_params : dict, optional
        Parameters for environment initialization.

    Returns
    -------
    float
        Performance metric (mean learning rate area).
    """
    # Load model
    model = model_class(input_dim, hidden_dim, action_dim)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Default environment parameters
    if env_params is None:
        env_params = {
            'total_trials': 200,
            'max_time': 300,
            'train_cond': False,
            'max_displacement': 10,
            'reward_size': 5.0,
        }

    # Create environments
    env_cp = PIE_CP_OB_v2(condition='change-point', **env_params)
    env_ob = PIE_CP_OB_v2(condition='oddball', **env_params)

    # Extract behavior
    states_cp = extract_behavior(
        model, env_cp, n_epochs=epochs, reset_memory=reset_memory
    )
    states_ob = extract_behavior(
        model, env_ob, n_epochs=epochs, reset_memory=reset_memory
    )

    # Calculate learning rates
    lr_areas = []
    for epoch in range(epochs):
        for states in [states_cp[epoch], states_ob[epoch]]:
            pes, lrs = get_lrs_v2(states, threshold=20)
            valid_lrs = lrs[lrs >= 0]
            if len(valid_lrs) > 0:
                lr_areas.append(np.mean(valid_lrs))

    return np.mean(lr_areas) if lr_areas else 0.0


def batch_extract_behavior(
    model_paths: List[Path],
    n_epochs: int = 100,
    reset_memory: bool = True,
    show_progress: bool = True,
    input_dim: int = 9,
    hidden_dim: int = 64,
    action_dim: int = 3,
    env_params: Optional[Dict] = None,
) -> Dict[str, List]:
    """
    Extract behavior from multiple models.

    Parameters
    ----------
    model_paths : list of Path
        List of paths to model weight files.
    n_epochs : int, optional
        Number of epochs per model.
    reset_memory : bool, optional
        Whether to reset memory between epochs.
    show_progress : bool, optional
        Whether to show progress bar.
    input_dim : int, optional
        Model input dimension. Default is 9.
    hidden_dim : int, optional
        Model hidden dimension. Default is 64.
    action_dim : int, optional
        Model action dimension. Default is 3.
    env_params : dict, optional
        Additional keyword arguments passed to PIE_CP_OB_v2 constructor.

    Returns
    -------
    dict
        Dictionary with 'cp' and 'ob' keys containing state lists.
    """
    results = {'cp': [], 'ob': [], 'models': []}

    iterator = tqdm(model_paths, desc="Extracting behavior") if show_progress else model_paths

    _env_params = env_params or {}

    for model_path in iterator:
        # Load model
        model = ActorCritic(input_dim, hidden_dim, action_dim)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)

        # Extract for both conditions
        env_cp = PIE_CP_OB_v2(condition='change-point', **_env_params)
        env_ob = PIE_CP_OB_v2(condition='oddball', **_env_params)

        states_cp = extract_behavior(model, env_cp, n_epochs=n_epochs, reset_memory=reset_memory)
        states_ob = extract_behavior(model, env_ob, n_epochs=n_epochs, reset_memory=reset_memory)

        results['cp'].append(states_cp)
        results['ob'].append(states_ob)
        results['models'].append(str(model_path))

    return results
