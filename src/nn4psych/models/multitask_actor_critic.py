"""
Multi-Task Actor-Critic RNN Model

This module provides a multi-task extension of the ActorCritic model that supports
tasks with different observation and action spaces through task-specific encoder
and decoder heads while sharing a common RNN backbone.

Architecture:
    Task A obs (6D) --> Encoder_A --> |
    Task B obs (4D) --> Encoder_B --> |--> Shared RNN (64D) --> Actor_A (3 actions)
    Task C obs (8D) --> Encoder_C --> |                    --> Actor_B (5 actions)
                                                           --> Actor_C (2 actions)
                                                           --> Critic (shared, 1D)
"""

from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


@dataclass
class TaskSpec:
    """
    Specification for a single task.

    Attributes
    ----------
    obs_dim : int
        Dimension of observation space for this task.
    action_dim : int
        Number of discrete actions for this task.
    context_id : int
        Unique integer ID for context embedding.
    env_class : type
        Environment class to instantiate.
    env_kwargs : dict
        Keyword arguments for environment instantiation.
    name : str
        Human-readable task name.
    """
    obs_dim: int
    action_dim: int
    context_id: int
    env_class: type
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = "unnamed_task"


class MultiTaskActorCritic(nn.Module):
    """
    Multi-Task Actor-Critic RNN model with task-specific input/output heads.

    This model uses a shared RNN backbone with:
    - Task-specific input encoders that project different observation spaces
      to a common hidden dimension
    - Task-specific actor heads for different action spaces
    - A shared critic head (value estimation is task-agnostic)

    Parameters
    ----------
    task_specs : Dict[str, TaskSpec]
        Dictionary mapping task names to their specifications.
    hidden_dim : int
        Number of hidden units in the RNN layer.
    context_dim : int, optional
        Dimension of context embedding. If 0, uses one-hot encoding based on
        number of tasks. Default is 0.
    gain : float, optional
        Scaling factor for recurrent weight initialization. Default is 1.5.
    bias : bool, optional
        Whether to include bias terms. Default is False.
    use_task_embedding : bool, optional
        If True, uses learned task embeddings instead of one-hot. Default is False.
    embedding_dim : int, optional
        Dimension of learned task embeddings. Only used if use_task_embedding=True.
        Default is 8.

    Attributes
    ----------
    encoders : nn.ModuleDict
        Task-specific input encoders.
    rnn : nn.RNN
        Shared recurrent neural network layer.
    actors : nn.ModuleDict
        Task-specific actor heads.
    critic : nn.Linear
        Shared critic head.
    task_embeddings : nn.Embedding or None
        Learned task embeddings (if use_task_embedding=True).

    Examples
    --------
    >>> from envs import PIE_CP_OB_v2
    >>> task_specs = {
    ...     'change-point': TaskSpec(
    ...         obs_dim=6, action_dim=3, context_id=0,
    ...         env_class=PIE_CP_OB_v2,
    ...         env_kwargs={'condition': 'change-point'},
    ...         name='change-point'
    ...     ),
    ...     'oddball': TaskSpec(
    ...         obs_dim=6, action_dim=3, context_id=1,
    ...         env_class=PIE_CP_OB_v2,
    ...         env_kwargs={'condition': 'oddball'},
    ...         name='oddball'
    ...     ),
    ... }
    >>> model = MultiTaskActorCritic(task_specs, hidden_dim=64)
    >>> x = torch.randn(1, 1, 6)
    >>> hx = model.get_initial_hidden()
    >>> logits, value, new_hx = model(x, hx, task_id='change-point')
    """

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        hidden_dim: int,
        context_dim: int = 0,
        gain: float = 1.5,
        bias: bool = False,
        use_task_embedding: bool = False,
        embedding_dim: int = 8,
    ):
        super(MultiTaskActorCritic, self).__init__()

        self.task_specs = task_specs
        self.hidden_dim = hidden_dim
        self.gain = gain
        self.bias = bias
        self.use_task_embedding = use_task_embedding
        self.num_tasks = len(task_specs)

        # Determine context dimension
        if use_task_embedding:
            self.context_dim = embedding_dim
            self.task_embeddings = nn.Embedding(self.num_tasks, embedding_dim)
        else:
            # One-hot encoding
            self.context_dim = self.num_tasks
            self.task_embeddings = None

        # Create task ID to index mapping
        self.task_to_idx = {
            task_id: spec.context_id for task_id, spec in task_specs.items()
        }

        # Task-specific input encoders
        # Each encoder projects: [obs + context + reward] -> hidden_dim
        self.encoders = nn.ModuleDict()
        for task_id, spec in task_specs.items():
            # Input: obs_dim + context_dim + 1 (reward)
            encoder_input_dim = spec.obs_dim + self.context_dim + 1
            self.encoders[task_id] = nn.Linear(
                encoder_input_dim, hidden_dim, bias=bias
            )

        # Shared RNN backbone
        self.rnn = nn.RNN(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            nonlinearity='tanh',
            bias=bias,
        )

        # Task-specific actor heads
        self.actors = nn.ModuleDict()
        for task_id, spec in task_specs.items():
            self.actors[task_id] = nn.Linear(hidden_dim, spec.action_dim, bias=bias)

        # Shared critic head (value is task-agnostic)
        self.critic = nn.Linear(hidden_dim, 1, bias=bias)

        # Store max dimensions for padding-based approach (alternative)
        self.max_obs_dim = max(spec.obs_dim for spec in task_specs.values())
        self.max_action_dim = max(spec.action_dim for spec in task_specs.values())

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize model weights using scaled normal distributions.

        Encoders: scaled by 1/sqrt(encoder_input_dim)
        RNN input-to-hidden: scaled by 1/sqrt(hidden_dim)
        RNN hidden-to-hidden: scaled by gain/sqrt(hidden_dim)
        Output weights: scaled by 1/hidden_dim
        """
        # Initialize encoders
        for task_id, encoder in self.encoders.items():
            spec = self.task_specs[task_id]
            encoder_input_dim = spec.obs_dim + self.context_dim + 1
            for name, param in encoder.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1.0 / (encoder_input_dim ** 0.5))
                elif 'bias' in name:
                    init.constant_(param, 0)

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.normal_(param, mean=0, std=1.0 / (self.hidden_dim ** 0.5))
            elif 'weight_hh' in name:
                init.normal_(param, mean=0, std=self.gain / (self.hidden_dim ** 0.5))
            elif 'bias' in name:
                init.constant_(param, 0)

        # Initialize actor heads
        for actor in self.actors.values():
            for name, param in actor.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1.0 / self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # Initialize critic
        for name, param in self.critic.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0, std=1.0 / self.hidden_dim)
            elif 'bias' in name:
                init.constant_(param, 0)

        # Initialize task embeddings if used
        if self.task_embeddings is not None:
            init.normal_(self.task_embeddings.weight, mean=0, std=0.1)

    def get_context(self, task_id: str, device: torch.device) -> torch.Tensor:
        """
        Get context vector for a task.

        Parameters
        ----------
        task_id : str
            Task identifier.
        device : torch.device
            Device to create tensor on.

        Returns
        -------
        torch.Tensor
            Context vector of shape (context_dim,).
        """
        task_idx = self.task_to_idx[task_id]

        if self.use_task_embedding:
            idx_tensor = torch.tensor([task_idx], device=device)
            return self.task_embeddings(idx_tensor).squeeze(0)
        else:
            # One-hot encoding
            context = torch.zeros(self.num_tasks, device=device)
            context[task_idx] = 1.0
            return context

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        task_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Multi-Task Actor-Critic network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, obs_dim + context_dim + 1).
            Should already include context and reward concatenated.
        hx : torch.Tensor
            Hidden state tensor of shape (num_layers, batch_size, hidden_dim).
        task_id : str
            Identifier for the current task.

        Returns
        -------
        actor_logits : torch.Tensor
            Action logits of shape (batch_size, action_dim).
        critic_value : torch.Tensor
            Value estimate of shape (batch_size, 1).
        new_hx : torch.Tensor
            Updated hidden state of shape (num_layers, batch_size, hidden_dim).
        """
        # Encode through task-specific encoder
        encoded = self.encoders[task_id](x)

        # Pass through shared RNN
        rnn_out, new_hx = self.rnn(encoded, hx)

        # Squeeze sequence dimension (assuming seq_len=1)
        rnn_out = rnn_out.squeeze(1)

        # Task-specific actor output
        actor_logits = self.actors[task_id](rnn_out)

        # Shared critic output
        critic_value = self.critic(rnn_out)

        return actor_logits, critic_value, new_hx

    def forward_with_raw_obs(
        self,
        obs: torch.Tensor,
        reward: torch.Tensor,
        hx: torch.Tensor,
        task_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that constructs the full input from components.

        This is a convenience method that handles context embedding internally.

        Parameters
        ----------
        obs : torch.Tensor
            Raw observation of shape (batch_size, obs_dim).
        reward : torch.Tensor
            Reward from previous step of shape (batch_size, 1).
        hx : torch.Tensor
            Hidden state tensor.
        task_id : str
            Identifier for the current task.

        Returns
        -------
        Tuple of (actor_logits, critic_value, new_hx).
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Get context
        context = self.get_context(task_id, device)
        context = context.unsqueeze(0).expand(batch_size, -1)

        # Concatenate: [obs, context, reward]
        x = torch.cat([obs, context, reward], dim=-1)

        # Add sequence dimension
        x = x.unsqueeze(1)

        return self.forward(x, hx, task_id)

    def get_initial_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get initial hidden state (zeros).

        Parameters
        ----------
        batch_size : int, optional
            Batch size. Default is 1.
        device : torch.device, optional
            Device to create tensor on. Default is CPU.

        Returns
        -------
        torch.Tensor
            Initial hidden state of shape (1, batch_size, hidden_dim).
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def get_action_mask(self, task_id: str, device: torch.device) -> torch.Tensor:
        """
        Get action mask for padding-based approaches.

        Parameters
        ----------
        task_id : str
            Task identifier.
        device : torch.device
            Device to create tensor on.

        Returns
        -------
        torch.Tensor
            Binary mask of shape (max_action_dim,) with 1s for valid actions.
        """
        valid_actions = self.task_specs[task_id].action_dim
        mask = torch.zeros(self.max_action_dim, device=device)
        mask[:valid_actions] = 1.0
        return mask

    def get_task_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get summary of task specifications.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary with task info including obs_dim, action_dim, context_id.
        """
        return {
            task_id: {
                'obs_dim': spec.obs_dim,
                'action_dim': spec.action_dim,
                'context_id': spec.context_id,
                'name': spec.name,
            }
            for task_id, spec in self.task_specs.items()
        }

    def freeze_shared_layers(self) -> None:
        """Freeze the shared RNN and critic for fine-tuning task-specific heads."""
        for param in self.rnn.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter counts by component.

        Returns
        -------
        Dict[str, int]
            Dictionary with parameter counts for each component.
        """
        counts = {
            'encoders': sum(
                p.numel() for enc in self.encoders.values() for p in enc.parameters()
            ),
            'rnn': sum(p.numel() for p in self.rnn.parameters()),
            'actors': sum(
                p.numel() for act in self.actors.values() for p in act.parameters()
            ),
            'critic': sum(p.numel() for p in self.critic.parameters()),
        }
        if self.task_embeddings is not None:
            counts['task_embeddings'] = self.task_embeddings.weight.numel()
        counts['total'] = sum(counts.values())
        return counts


class PaddedMultiTaskActorCritic(nn.Module):
    """
    Alternative multi-task model using padding and masking.

    This simpler approach pads all observations to the maximum dimension
    and uses action masking instead of task-specific heads.

    Parameters
    ----------
    task_specs : Dict[str, TaskSpec]
        Dictionary mapping task names to their specifications.
    hidden_dim : int
        Number of hidden units in the RNN layer.
    gain : float, optional
        Scaling factor for recurrent weight initialization. Default is 1.5.
    bias : bool, optional
        Whether to include bias terms. Default is False.
    """

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        hidden_dim: int,
        gain: float = 1.5,
        bias: bool = False,
    ):
        super(PaddedMultiTaskActorCritic, self).__init__()

        self.task_specs = task_specs
        self.hidden_dim = hidden_dim
        self.gain = gain
        self.num_tasks = len(task_specs)

        # Calculate max dimensions
        self.max_obs_dim = max(spec.obs_dim for spec in task_specs.values())
        self.max_action_dim = max(spec.action_dim for spec in task_specs.values())

        # Input: padded_obs + one_hot_context + reward
        self.input_dim = self.max_obs_dim + self.num_tasks + 1

        # Single RNN
        self.rnn = nn.RNN(
            self.input_dim,
            hidden_dim,
            batch_first=True,
            nonlinearity='tanh',
            bias=bias,
        )

        # Single actor (max action dim, with masking)
        self.actor = nn.Linear(hidden_dim, self.max_action_dim, bias=bias)

        # Critic
        self.critic = nn.Linear(hidden_dim, 1, bias=bias)

        # Task ID mapping
        self.task_to_idx = {
            task_id: spec.context_id for task_id, spec in task_specs.items()
        }

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights using scaled normal distributions."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.normal_(param, mean=0, std=1.0 / (self.input_dim ** 0.5))
            elif 'weight_hh' in name:
                init.normal_(param, mean=0, std=self.gain / (self.hidden_dim ** 0.5))
            elif 'bias' in name:
                init.constant_(param, 0)

        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1.0 / self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def pad_observation(
        self,
        obs: torch.Tensor,
        task_id: str,
    ) -> torch.Tensor:
        """
        Pad observation to max_obs_dim.

        Parameters
        ----------
        obs : torch.Tensor
            Original observation of shape (batch_size, obs_dim).
        task_id : str
            Task identifier.

        Returns
        -------
        torch.Tensor
            Padded observation of shape (batch_size, max_obs_dim).
        """
        batch_size = obs.shape[0]
        obs_dim = self.task_specs[task_id].obs_dim
        device = obs.device

        if obs_dim == self.max_obs_dim:
            return obs

        # Pad with zeros
        padding = torch.zeros(batch_size, self.max_obs_dim - obs_dim, device=device)
        return torch.cat([obs, padding], dim=-1)

    def get_action_mask(self, task_id: str, device: torch.device) -> torch.Tensor:
        """Get mask for valid actions (1 for valid, 0 for invalid)."""
        valid_actions = self.task_specs[task_id].action_dim
        mask = torch.zeros(self.max_action_dim, device=device)
        mask[:valid_actions] = 1.0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        task_id: str,
        apply_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic action masking.

        Parameters
        ----------
        x : torch.Tensor
            Padded input of shape (batch_size, seq_len, input_dim).
        hx : torch.Tensor
            Hidden state.
        task_id : str
            Task identifier.
        apply_mask : bool, optional
            Whether to mask invalid actions with -inf. Default is True.

        Returns
        -------
        Tuple of (actor_logits, critic_value, new_hx).
        """
        rnn_out, new_hx = self.rnn(x, hx)
        rnn_out = rnn_out.squeeze(1)

        actor_logits = self.actor(rnn_out)

        # Mask invalid actions
        if apply_mask:
            mask = self.get_action_mask(task_id, x.device)
            actor_logits = actor_logits.masked_fill(mask == 0, float('-inf'))

        critic_value = self.critic(rnn_out)

        return actor_logits, critic_value, new_hx

    def get_initial_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get initial hidden state (zeros)."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
