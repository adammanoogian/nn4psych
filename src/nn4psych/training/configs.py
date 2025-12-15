"""
Configuration management using dataclasses.

This module provides structured configuration objects that replace hard-coded
parameters throughout the codebase, enabling reproducible experiments.

Includes support for:
- Single-task experiments (ExperimentConfig)
- Multi-task experiments with different obs/action spaces (MultiTaskExperimentConfig)
- Task specification registry for defining new tasks
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Type
from pathlib import Path
from enum import Enum
import yaml
import json


@dataclass
class ModelConfig:
    """
    Configuration for the ActorCritic model.

    Attributes
    ----------
    input_dim : int
        Input feature dimension. Default is 9 (6 obs + 2 context + 1 reward).
    hidden_dim : int
        Number of RNN hidden units. Default is 64.
    action_dim : int
        Number of possible actions. Default is 3.
    gain : float
        RNN weight initialization gain. Default is 1.5.
    noise : float
        Hidden state noise variance. Default is 0.0.
    bias : bool
        Whether to include bias terms. Default is False.
    """

    input_dim: int = 9
    hidden_dim: int = 64
    action_dim: int = 3
    gain: float = 1.5
    noise: float = 0.0
    bias: bool = False


@dataclass
class TaskConfig:
    """
    Configuration for the predictive inference task environment.

    Attributes
    ----------
    condition : str
        Task condition: "change-point" or "oddball".
    total_trials : int
        Number of trials per epoch.
    max_time : int
        Maximum time steps per trial.
    train_cond : bool
        Whether helicopter position is visible.
    max_displacement : float
        Maximum bucket movement per action.
    reward_size : float
        SD for Gaussian reward function.
    step_cost : float
        Per-step penalty.
    alpha : float
        Velocity smoothing factor.
    """

    condition: str = "change-point"
    total_trials: int = 200
    max_time: int = 300
    train_cond: bool = False
    max_displacement: float = 10.0
    reward_size: float = 5.0
    step_cost: float = 0.0
    alpha: float = 1.0


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters.

    Attributes
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor for returns.
    rollout_size : int
        Number of steps to collect before update.
    td_noise : float
        Temporal difference noise.
    preset_memory : float
        Initial hidden state value.
    td_lower_bound : float or None
        Lower bound for TD updates.
    td_upper_bound : float or None
        Upper bound for TD updates.
    td_scale : float
        TD scaling factor.
    seed : int
        Random seed for reproducibility.
    device : str
        PyTorch device (cpu or cuda).
    save_dir : str
        Directory to save model weights.
    save_frequency : int
        Save model every N epochs.
    """

    epochs: int = 100
    learning_rate: float = 5e-4
    gamma: float = 0.95
    rollout_size: int = 100
    td_noise: float = 0.0
    preset_memory: float = 0.0
    td_lower_bound: Optional[float] = None
    td_upper_bound: Optional[float] = None
    td_scale: float = 1.0
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "model_params"
    save_frequency: int = 10


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Attributes
    ----------
    name : str
        Experiment name/identifier.
    model : ModelConfig
        Model configuration.
    task : TaskConfig
        Task configuration.
    training : TrainingConfig
        Training configuration.
    description : str
        Optional experiment description.
    tags : list
        Optional tags for organizing experiments.
    """

    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    description: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, filepath: Optional[Path] = None) -> str:
        """
        Export configuration to YAML format.

        Parameters
        ----------
        filepath : Path, optional
            If provided, save to file.

        Returns
        -------
        str
            YAML string representation.
        """
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        return yaml_str

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """
        Export configuration to JSON format.

        Parameters
        ----------
        filepath : Path, optional
            If provided, save to file.

        Returns
        -------
        str
            JSON string representation.
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_yaml(cls, filepath: Path) -> "ExperimentConfig":
        """
        Load configuration from YAML file.

        Parameters
        ----------
        filepath : Path
            Path to YAML file.

        Returns
        -------
        ExperimentConfig
            Loaded configuration.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, filepath: Path) -> "ExperimentConfig":
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : Path
            Path to JSON file.

        Returns
        -------
        ExperimentConfig
            Loaded configuration.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        return cls(
            name=data.get('name', 'experiment'),
            model=ModelConfig(**data.get('model', {})),
            task=TaskConfig(**data.get('task', {})),
            training=TrainingConfig(**data.get('training', {})),
            description=data.get('description', ''),
            tags=data.get('tags', []),
        )

    def get_filename(self) -> str:
        """
        Generate a descriptive filename based on configuration.

        Returns
        -------
        str
            Filename string encoding key hyperparameters.
        """
        return (
            f"V5_{self.training.gamma}g_{self.training.preset_memory}rm_"
            f"{self.training.rollout_size}bz_{self.training.td_noise}td_"
            f"{self.training.td_scale}tds_{self.training.td_lower_bound}lb_"
            f"{self.training.td_upper_bound}up_{self.model.hidden_dim}n_"
            f"{self.training.epochs}e_{self.task.max_displacement}md_"
            f"{self.task.reward_size}rz_{self.training.seed}s"
        )


# Predefined hyperparameter sweep values
GAMMA_VALUES = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
ROLLOUT_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200]
PRESET_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SCALE_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def create_default_config() -> ExperimentConfig:
    """
    Create default experiment configuration.

    Returns
    -------
    ExperimentConfig
        Default configuration matching pretrain_rnn_with_heli_v5.py defaults.
    """
    return ExperimentConfig(
        name="default_experiment",
        model=ModelConfig(
            input_dim=9,
            hidden_dim=64,
            action_dim=3,
            gain=1.5,
            noise=0.0,
            bias=False,
        ),
        task=TaskConfig(
            condition="change-point",
            total_trials=200,
            max_time=300,
            train_cond=False,
            max_displacement=10.0,
            reward_size=5.0,
            step_cost=0.0,
            alpha=1.0,
        ),
        training=TrainingConfig(
            epochs=100,
            learning_rate=5e-4,
            gamma=0.95,
            rollout_size=100,
            td_noise=0.0,
            preset_memory=0.0,
            td_lower_bound=None,
            td_upper_bound=None,
            td_scale=1.0,
            seed=42,
            device="cpu",
            save_dir="model_params",
            save_frequency=10,
        ),
        description="Default configuration for predictive inference task",
        tags=["default", "baseline"],
    )


# =============================================================================
# Multi-Task Configuration
# =============================================================================

class InterleaveMode(Enum):
    """Task interleaving strategies for multi-task training."""
    EPOCH = "epoch"          # Alternate tasks per epoch (default)
    TRIAL = "trial"          # Random task per trial within epoch
    BLOCK = "block"          # K trials of task A, then K trials of task B
    CURRICULUM = "curriculum"  # Start with easier task, gradually add harder


@dataclass
class TaskSpecConfig:
    """
    Configuration for a single task in multi-task setup.

    This is a serializable version of TaskSpec that doesn't store
    class references directly.

    Attributes
    ----------
    name : str
        Human-readable task name.
    obs_dim : int
        Dimension of observation space.
    action_dim : int
        Number of discrete actions.
    context_id : int
        Unique integer ID for context embedding.
    env_type : str
        Environment type identifier (e.g., "PIE_CP_OB_v2").
    env_kwargs : dict
        Keyword arguments for environment instantiation.
    """
    name: str
    obs_dim: int
    action_dim: int
    context_id: int
    env_type: str = "PIE_CP_OB_v2"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTaskModelConfig:
    """
    Configuration for multi-task ActorCritic model.

    Attributes
    ----------
    hidden_dim : int
        Number of RNN hidden units.
    gain : float
        RNN weight initialization gain.
    bias : bool
        Whether to include bias terms.
    use_task_embedding : bool
        If True, use learned task embeddings instead of one-hot.
    embedding_dim : int
        Dimension of learned task embeddings.
    model_type : str
        Architecture type: "heads" (task-specific) or "padded" (masking).
    """
    hidden_dim: int = 64
    gain: float = 1.5
    bias: bool = False
    use_task_embedding: bool = False
    embedding_dim: int = 8
    model_type: str = "heads"  # "heads" or "padded"


@dataclass
class MultiTaskTrainingConfig:
    """
    Configuration for multi-task training.

    Attributes
    ----------
    epochs : int
        Number of training epochs.
    trials_per_task : int
        Number of trials per task per epoch.
    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor for returns.
    rollout_size : int
        Number of steps to collect before update.
    interleave_mode : str
        Task interleaving strategy.
    block_size : int
        Number of trials per block (for block mode).
    curriculum_warmup : int
        Epochs before adding new tasks (for curriculum mode).
    train_ratio : float
        Fraction of epochs with visible helicopter.
    preset_memory : float
        Probability of random hidden state reset.
    eval_frequency : int
        Evaluate on all tasks every N epochs.
    seed : int
        Random seed for reproducibility.
    device : str
        PyTorch device.
    save_dir : str
        Directory to save model weights.
    """
    epochs: int = 100
    trials_per_task: int = 200
    learning_rate: float = 5e-4
    gamma: float = 0.95
    rollout_size: int = 50
    interleave_mode: str = "epoch"
    block_size: int = 50
    curriculum_warmup: int = 20
    train_ratio: float = 0.5
    preset_memory: float = 0.0
    eval_frequency: int = 10
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "model_params"


@dataclass
class MultiTaskExperimentConfig:
    """
    Complete multi-task experiment configuration.

    Attributes
    ----------
    name : str
        Experiment name/identifier.
    model : MultiTaskModelConfig
        Model configuration.
    training : MultiTaskTrainingConfig
        Training configuration.
    tasks : List[TaskSpecConfig]
        List of task specifications.
    task_params : TaskConfig
        Shared task parameters (max_displacement, reward_size, etc.).
    description : str
        Optional experiment description.
    tags : list
        Optional tags for organizing experiments.
    """
    name: str = "multitask_experiment"
    model: MultiTaskModelConfig = field(default_factory=MultiTaskModelConfig)
    training: MultiTaskTrainingConfig = field(default_factory=MultiTaskTrainingConfig)
    tasks: List[TaskSpecConfig] = field(default_factory=list)
    task_params: TaskConfig = field(default_factory=TaskConfig)
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default tasks if none provided."""
        if not self.tasks:
            self.tasks = [
                TaskSpecConfig(
                    name="Change-Point",
                    obs_dim=6,
                    action_dim=3,
                    context_id=0,
                    env_type="PIE_CP_OB_v2",
                    env_kwargs={"condition": "change-point"},
                ),
                TaskSpecConfig(
                    name="Oddball",
                    obs_dim=6,
                    action_dim=3,
                    context_id=1,
                    env_type="PIE_CP_OB_v2",
                    env_kwargs={"condition": "oddball"},
                ),
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, filepath: Optional[Path] = None) -> str:
        """Export configuration to YAML format."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        return yaml_str

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Export configuration to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_yaml(cls, filepath: Path) -> "MultiTaskExperimentConfig":
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, filepath: Path) -> "MultiTaskExperimentConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "MultiTaskExperimentConfig":
        """Create configuration from dictionary."""
        tasks = [
            TaskSpecConfig(**task_data)
            for task_data in data.get('tasks', [])
        ]
        return cls(
            name=data.get('name', 'multitask_experiment'),
            model=MultiTaskModelConfig(**data.get('model', {})),
            training=MultiTaskTrainingConfig(**data.get('training', {})),
            tasks=tasks,
            task_params=TaskConfig(**data.get('task_params', {})),
            description=data.get('description', ''),
            tags=data.get('tags', []),
        )

    def get_filename(self) -> str:
        """Generate a descriptive filename based on configuration."""
        return (
            f"MT_{self.training.interleave_mode}_{self.training.gamma}g_"
            f"{self.training.rollout_size}bz_{self.model.hidden_dim}n_"
            f"{self.training.epochs}e_{len(self.tasks)}tasks_"
            f"{self.training.seed}s"
        )

    def get_task_summary(self) -> str:
        """Get a summary of task configurations."""
        lines = ["Tasks:"]
        for task in self.tasks:
            lines.append(
                f"  - {task.name}: obs={task.obs_dim}D, act={task.action_dim}, "
                f"ctx_id={task.context_id}"
            )
        return "\n".join(lines)


def create_default_multitask_config() -> MultiTaskExperimentConfig:
    """
    Create default multi-task experiment configuration.

    Returns
    -------
    MultiTaskExperimentConfig
        Default configuration with CP and OB tasks.
    """
    return MultiTaskExperimentConfig(
        name="default_multitask",
        model=MultiTaskModelConfig(
            hidden_dim=64,
            gain=1.5,
            bias=False,
            use_task_embedding=False,
            embedding_dim=8,
            model_type="heads",
        ),
        training=MultiTaskTrainingConfig(
            epochs=100,
            trials_per_task=200,
            learning_rate=5e-4,
            gamma=0.95,
            rollout_size=50,
            interleave_mode="epoch",
            block_size=50,
            train_ratio=0.5,
            preset_memory=0.0,
            seed=42,
            device="cpu",
            save_dir="model_params",
        ),
        tasks=[
            TaskSpecConfig(
                name="Change-Point",
                obs_dim=6,
                action_dim=3,
                context_id=0,
                env_type="PIE_CP_OB_v2",
                env_kwargs={"condition": "change-point"},
            ),
            TaskSpecConfig(
                name="Oddball",
                obs_dim=6,
                action_dim=3,
                context_id=1,
                env_type="PIE_CP_OB_v2",
                env_kwargs={"condition": "oddball"},
            ),
        ],
        task_params=TaskConfig(
            total_trials=200,
            max_time=300,
            train_cond=False,
            max_displacement=10.0,
            reward_size=5.0,
            step_cost=0.0,
            alpha=1.0,
        ),
        description="Default multi-task configuration with CP and OB tasks",
        tags=["multitask", "default", "baseline"],
    )


# =============================================================================
# Task Registry for Easy Extension
# =============================================================================

# Registry of known task types and their default configurations
TASK_REGISTRY: Dict[str, TaskSpecConfig] = {
    # PIE (Predictive Inference Environment) tasks
    "change-point": TaskSpecConfig(
        name="Change-Point",
        obs_dim=6,
        action_dim=3,
        context_id=0,
        env_type="PIE_CP_OB_v2",
        env_kwargs={"condition": "change-point"},
    ),
    "oddball": TaskSpecConfig(
        name="Oddball",
        obs_dim=6,
        action_dim=3,
        context_id=1,
        env_type="PIE_CP_OB_v2",
        env_kwargs={"condition": "oddball"},
    ),
    # NeuroGym tasks (requires neurogym package)
    # Observation/action dims are approximate; actual values determined at runtime
    "daw-two-step": TaskSpecConfig(
        name="Daw Two-Step",
        obs_dim=4,  # Approximate, determined at runtime
        action_dim=3,  # Fixation + 2 choices
        context_id=2,
        env_type="DawTwoStepWrapper",
        env_kwargs={"dt": 100},
    ),
    "single-context-dm": TaskSpecConfig(
        name="Single Context Decision Making",
        obs_dim=3,  # 1 + dim_ring (default dim_ring=2)
        action_dim=3,  # 1 + dim_ring
        context_id=3,
        env_type="SingleContextDecisionMakingWrapper",
        env_kwargs={"dt": 100, "sigma": 1.0, "dim_ring": 2, "modality_context": 0},
    ),
    "perceptual-dm": TaskSpecConfig(
        name="Perceptual Decision Making",
        obs_dim=3,  # 1 + dim_ring
        action_dim=3,  # 1 + dim_ring
        context_id=4,
        env_type="PerceptualDecisionMakingWrapper",
        env_kwargs={"dt": 100, "sigma": 1.0, "dim_ring": 2},
    ),
}


def register_task(
    task_id: str,
    name: str,
    obs_dim: int,
    action_dim: int,
    env_type: str,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> TaskSpecConfig:
    """
    Register a new task type in the task registry.

    Parameters
    ----------
    task_id : str
        Unique identifier for the task.
    name : str
        Human-readable task name.
    obs_dim : int
        Dimension of observation space.
    action_dim : int
        Number of discrete actions.
    env_type : str
        Environment type identifier.
    env_kwargs : dict, optional
        Keyword arguments for environment instantiation.

    Returns
    -------
    TaskSpecConfig
        The registered task configuration.

    Examples
    --------
    >>> register_task(
    ...     task_id="bandit-2arm",
    ...     name="Two-Arm Bandit",
    ...     obs_dim=4,
    ...     action_dim=2,
    ...     env_type="BanditEnv",
    ...     env_kwargs={"num_arms": 2},
    ... )
    """
    context_id = len(TASK_REGISTRY)
    task_spec = TaskSpecConfig(
        name=name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        context_id=context_id,
        env_type=env_type,
        env_kwargs=env_kwargs or {},
    )
    TASK_REGISTRY[task_id] = task_spec
    return task_spec


def get_task_config(task_id: str) -> TaskSpecConfig:
    """
    Get task configuration from registry.

    Parameters
    ----------
    task_id : str
        Task identifier.

    Returns
    -------
    TaskSpecConfig
        Task configuration.

    Raises
    ------
    KeyError
        If task_id is not in registry.
    """
    if task_id not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise KeyError(
            f"Unknown task '{task_id}'. Available tasks: {available}"
        )
    return TASK_REGISTRY[task_id]


def list_available_tasks() -> List[str]:
    """List all registered task IDs."""
    return list(TASK_REGISTRY.keys())
