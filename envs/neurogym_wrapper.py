"""
NeuroGym Environment Wrappers

This module provides wrappers that adapt NeuroGym environments to work with
the nn4psych multi-task training system. It handles the interface differences
between neurogym's trial-based structure and our step-based training loop.

Supported NeuroGym Tasks:
- DawTwoStep-v0: Two-step decision making task
- SingleContextDecisionMaking-v0: Context-dependent perceptual decision making
- PerceptualDecisionMaking-v0: Random dot motion discrimination
- And other neurogym tasks

References:
- NeuroGym: https://github.com/neurogym/neurogym
- DawTwoStep: "Model-Based Influences on Humans Choices and Striatal Prediction Errors"
- SingleContextDecisionMaking: "Context-dependent computation by recurrent dynamics"
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import neurogym as ngym
    NEUROGYM_AVAILABLE = True
except ImportError:
    NEUROGYM_AVAILABLE = False
    ngym = None


class NeurogymWrapper:
    """
    Wrapper that adapts NeuroGym environments to the nn4psych interface.

    This wrapper provides a consistent interface matching PIE_CP_OB_v2 style,
    enabling neurogym tasks to be used in the multi-task training pipeline.

    Key adaptations:
    - Converts neurogym's (obs, reward, terminated, truncated, info) to (obs, reward, done)
    - Tracks trial history for analysis
    - Provides normalize_states() method
    - Stores context vector for multi-task identification

    Parameters
    ----------
    env_name : str
        NeuroGym environment name (e.g., 'DawTwoStep-v0').
    context_id : int
        Context identifier for multi-task learning.
    total_trials : int, optional
        Number of trials per epoch. Default is 200.
    dt : int, optional
        Timestep in milliseconds. Default is 100.
    **env_kwargs
        Additional keyword arguments passed to the neurogym environment.

    Attributes
    ----------
    env : ngym.TrialEnv
        The underlying neurogym environment.
    action_space : gym.spaces.Discrete
        Action space of the environment.
    observation_space : gym.spaces.Box
        Observation space of the environment.
    context : np.ndarray
        One-hot context vector for multi-task identification.
    obs_dim : int
        Dimension of observation space.
    action_dim : int
        Number of possible actions.

    Examples
    --------
    >>> env = NeurogymWrapper('DawTwoStep-v0', context_id=0)
    >>> obs, done = env.reset()
    >>> next_obs, reward, done = env.step(action=1)
    """

    def __init__(
        self,
        env_name: str,
        context_id: int,
        total_trials: int = 200,
        dt: int = 100,
        **env_kwargs,
    ):
        if not NEUROGYM_AVAILABLE:
            raise ImportError(
                "neurogym is not installed. Install with: pip install neurogym"
            )

        self.env_name = env_name
        self.context_id = context_id
        self.total_trials = total_trials
        self.dt = dt

        # Create neurogym environment
        self.env = ngym.make(env_name, dt=dt, **env_kwargs)

        # Get space dimensions
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.n

        # Context will be set externally based on number of tasks
        self._num_tasks = 2  # Default, will be updated
        self.context = self._make_context(context_id, self._num_tasks)

        # Trial tracking
        self.trial = 0
        self.time = 0
        self._reset_history()

        # Current trial state
        self._current_obs = None
        self._trial_rewards = []
        self._trial_actions = []

    def _make_context(self, context_id: int, num_tasks: int) -> np.ndarray:
        """Create one-hot context vector."""
        context = np.zeros(num_tasks)
        context[context_id] = 1.0
        return context

    def set_num_tasks(self, num_tasks: int) -> None:
        """Update the context vector for the correct number of tasks."""
        self._num_tasks = num_tasks
        self.context = self._make_context(self.context_id, num_tasks)

    def _reset_history(self) -> None:
        """Reset trial history tracking."""
        self.trials = []
        self.rewards_history = []
        self.actions_history = []
        self.observations_history = []
        self.trial_lengths = []

    def normalize_states(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations to approximately [0, 1] range.

        NeuroGym observations are typically already normalized or
        have small ranges, but we apply clipping for safety.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation from environment.

        Returns
        -------
        np.ndarray
            Normalized observation.
        """
        # Neurogym observations are typically in reasonable ranges
        # Apply soft normalization
        return np.clip(obs, -10, 10) / 10.0

    def reset(self) -> Tuple[np.ndarray, bool]:
        """
        Reset environment for new trial.

        Returns
        -------
        obs : np.ndarray
            Initial observation.
        done : bool
            Episode termination flag (always False after reset).
        """
        self.time = 0
        self._trial_rewards = []
        self._trial_actions = []

        # Reset neurogym environment
        obs, info = self.env.reset()
        self._current_obs = obs.astype(np.float32)

        return self._current_obs, False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one time step in the environment.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        obs : np.ndarray
            New observation.
        reward : float
            Reward received.
        done : bool
            Whether trial/episode is complete.
        """
        self.time += 1

        # Step neurogym environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self._current_obs = obs.astype(np.float32)
        self._trial_rewards.append(reward)
        self._trial_actions.append(action)

        # Record trial completion
        if done:
            self.trial += 1
            self.trials.append(self.trial)
            self.rewards_history.append(sum(self._trial_rewards))
            self.actions_history.append(self._trial_actions.copy())
            self.trial_lengths.append(self.time)
            self.observations_history.append(self._current_obs.copy())

        return self._current_obs, float(reward), done

    def get_state_history(self) -> Tuple:
        """
        Get complete state history for analysis.

        Returns
        -------
        tuple
            (trials, rewards, actions, trial_lengths)
        """
        return (
            self.trials,
            self.rewards_history,
            self.actions_history,
            self.trial_lengths,
        )

    def render(self, epoch: int = 0) -> np.ndarray:
        """
        Render trial history (placeholder for compatibility).

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        np.ndarray
            Array with trial history data.
        """
        return np.array([
            self.trials,
            self.rewards_history,
            self.trial_lengths,
        ])


class DawTwoStepWrapper(NeurogymWrapper):
    """
    Wrapper for the Daw Two-Step task.

    The Daw Two-Step task is a paradigm for studying model-based vs model-free
    learning. On each trial:
    1. Initial choice between two options
    2. Transition to one of two second-stage states (probabilistic)
    3. Second-stage choice between two options
    4. Reward based on second-stage choice

    This task tests whether agents use model-based (planning) or model-free
    (habitual) strategies.

    Parameters
    ----------
    context_id : int
        Context identifier for multi-task learning.
    total_trials : int, optional
        Number of trials per epoch. Default is 200.
    dt : int, optional
        Timestep in milliseconds. Default is 100.
    **kwargs
        Additional arguments passed to neurogym.

    Reference
    ---------
    Daw, N. D., et al. (2011). Model-Based Influences on Humans' Choices
    and Striatal Prediction Errors. Neuron, 69(6), 1204-1215.
    """

    def __init__(
        self,
        context_id: int = 0,
        total_trials: int = 200,
        dt: int = 100,
        **kwargs,
    ):
        super().__init__(
            env_name='DawTwoStep-v0',
            context_id=context_id,
            total_trials=total_trials,
            dt=dt,
            **kwargs,
        )

        # Track two-step specific data
        self.first_stage_choices = []
        self.second_stage_states = []
        self.second_stage_choices = []
        self.transition_types = []  # Common vs rare

    def _reset_history(self) -> None:
        """Reset trial history including two-step specific data."""
        super()._reset_history()
        self.first_stage_choices = []
        self.second_stage_states = []
        self.second_stage_choices = []
        self.transition_types = []


class SingleContextDecisionMakingWrapper(NeurogymWrapper):
    """
    Wrapper for the Single Context Decision Making task.

    In this task, the agent receives simultaneous inputs from two modalities
    (e.g., color and motion of a random dot pattern) and must make a decision
    based on only one modality while ignoring the other.

    The relevant modality is determined by a fixed context parameter set
    at initialization (not explicitly signaled during trials).

    Parameters
    ----------
    context_id : int
        Context identifier for multi-task learning.
    modality_context : int, optional
        Which modality to attend to (0 or 1). Default is 0.
    total_trials : int, optional
        Number of trials per epoch. Default is 200.
    dt : int, optional
        Timestep in milliseconds. Default is 100.
    sigma : float, optional
        Noise level for stimuli. Default is 1.0.
    dim_ring : int, optional
        Ring dimension (affects obs/action space). Default is 2.
    **kwargs
        Additional arguments passed to neurogym.

    Reference
    ---------
    Mante, V., et al. (2013). Context-dependent computation by recurrent
    dynamics in prefrontal cortex. Nature, 503(7474), 78-84.
    """

    def __init__(
        self,
        context_id: int = 0,
        modality_context: int = 0,
        total_trials: int = 200,
        dt: int = 100,
        sigma: float = 1.0,
        dim_ring: int = 2,
        **kwargs,
    ):
        self.modality_context = modality_context
        super().__init__(
            env_name='SingleContextDecisionMaking-v0',
            context_id=context_id,
            total_trials=total_trials,
            dt=dt,
            context=modality_context,
            sigma=sigma,
            dim_ring=dim_ring,
            **kwargs,
        )

        # Track task-specific data
        self.correct_choices = []
        self.coherences = []


class PerceptualDecisionMakingWrapper(NeurogymWrapper):
    """
    Wrapper for the Perceptual Decision Making (random dots) task.

    Classic random dot motion discrimination task. The agent must determine
    the direction of coherent motion in a noisy random dot display.

    Parameters
    ----------
    context_id : int
        Context identifier for multi-task learning.
    total_trials : int, optional
        Number of trials per epoch. Default is 200.
    dt : int, optional
        Timestep in milliseconds. Default is 100.
    sigma : float, optional
        Noise level. Default is 1.0.
    dim_ring : int, optional
        Ring dimension. Default is 2.
    **kwargs
        Additional arguments passed to neurogym.

    Reference
    ---------
    Britten, K. H., et al. (1992). The analysis of visual motion: a
    comparison of neuronal and psychophysical performance.
    """

    def __init__(
        self,
        context_id: int = 0,
        total_trials: int = 200,
        dt: int = 100,
        sigma: float = 1.0,
        dim_ring: int = 2,
        **kwargs,
    ):
        super().__init__(
            env_name='PerceptualDecisionMaking-v0',
            context_id=context_id,
            total_trials=total_trials,
            dt=dt,
            sigma=sigma,
            dim_ring=dim_ring,
            **kwargs,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_neurogym_env(
    env_name: str,
    context_id: int = 0,
    **kwargs,
) -> NeurogymWrapper:
    """
    Factory function to create neurogym environment wrappers.

    Parameters
    ----------
    env_name : str
        Name of the neurogym environment.
    context_id : int, optional
        Context ID for multi-task learning. Default is 0.
    **kwargs
        Additional environment-specific arguments.

    Returns
    -------
    NeurogymWrapper
        Wrapped neurogym environment.

    Examples
    --------
    >>> env = create_neurogym_env('DawTwoStep-v0', context_id=0)
    >>> env = create_neurogym_env('SingleContextDecisionMaking-v0', context_id=1, modality_context=0)
    """
    # Map environment names to specialized wrappers
    wrapper_map = {
        'DawTwoStep-v0': DawTwoStepWrapper,
        'SingleContextDecisionMaking-v0': SingleContextDecisionMakingWrapper,
        'PerceptualDecisionMaking-v0': PerceptualDecisionMakingWrapper,
    }

    if env_name in wrapper_map:
        return wrapper_map[env_name](context_id=context_id, **kwargs)
    else:
        # Generic wrapper for other neurogym environments
        return NeurogymWrapper(env_name, context_id=context_id, **kwargs)


def get_neurogym_task_spec(env_name: str, **env_kwargs) -> Dict[str, Any]:
    """
    Get task specification for a neurogym environment.

    Creates a temporary environment to extract observation and action dimensions.

    Parameters
    ----------
    env_name : str
        Name of the neurogym environment.
    **env_kwargs
        Environment-specific keyword arguments.

    Returns
    -------
    Dict[str, Any]
        Task specification including obs_dim, action_dim, and env_kwargs.

    Examples
    --------
    >>> spec = get_neurogym_task_spec('DawTwoStep-v0')
    >>> print(f"Obs dim: {spec['obs_dim']}, Action dim: {spec['action_dim']}")
    """
    if not NEUROGYM_AVAILABLE:
        raise ImportError(
            "neurogym is not installed. Install with: pip install neurogym"
        )

    # Create temporary environment to get dimensions
    env = ngym.make(env_name, **env_kwargs)

    spec = {
        'env_name': env_name,
        'obs_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n,
        'env_kwargs': env_kwargs,
    }

    env.close()
    return spec


def list_available_neurogym_tasks() -> List[str]:
    """
    List all available neurogym tasks.

    Returns
    -------
    List[str]
        List of available neurogym environment names.
    """
    if not NEUROGYM_AVAILABLE:
        return []

    return ngym.envs.ALL_ENVS


# =============================================================================
# Pre-configured Task Specifications
# =============================================================================

# Default specifications for commonly used neurogym tasks
# These are estimates; actual values are determined at runtime
NEUROGYM_TASK_DEFAULTS = {
    'DawTwoStep-v0': {
        'name': 'Daw Two-Step',
        'obs_dim': 4,  # Approximate, determined at runtime
        'action_dim': 3,  # Fixation + 2 choices
        'env_kwargs': {'dt': 100},
        'description': 'Two-step decision making task for model-based vs model-free learning',
    },
    'SingleContextDecisionMaking-v0': {
        'name': 'Single Context Decision Making',
        'obs_dim': 3,  # 1 + dim_ring (default dim_ring=2)
        'action_dim': 3,  # 1 + dim_ring
        'env_kwargs': {'dt': 100, 'sigma': 1.0, 'dim_ring': 2},
        'description': 'Context-dependent perceptual decision making',
    },
    'PerceptualDecisionMaking-v0': {
        'name': 'Perceptual Decision Making',
        'obs_dim': 3,  # 1 + dim_ring
        'action_dim': 3,  # 1 + dim_ring
        'env_kwargs': {'dt': 100, 'sigma': 1.0, 'dim_ring': 2},
        'description': 'Random dot motion discrimination task',
    },
    'GoNogo-v0': {
        'name': 'Go/No-Go',
        'obs_dim': 3,
        'action_dim': 2,
        'env_kwargs': {'dt': 100},
        'description': 'Go/No-Go response inhibition task',
    },
    'DelayMatchSample-v0': {
        'name': 'Delayed Match to Sample',
        'obs_dim': 3,
        'action_dim': 3,
        'env_kwargs': {'dt': 100},
        'description': 'Working memory task with sample and match phases',
    },
}
