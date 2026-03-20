"""
Continuous-Time ReLU Actor-Critic RNN

Implements the continuous-time RNN dynamics used in Langdon & Engel 2025:

    h(t) = (1 - alpha) * h(t-1) + alpha * ReLU(W_hh @ h(t-1) + W_ih @ x(t) + noise)

This is a drop-in replacement for ActorCritic that uses ReLU non-negative dynamics
instead of tanh, making it compatible with the LatentNet latent circuit inference
pipeline. The actor and critic heads are identical to ActorCritic.

The RL training algorithm (policy gradient, GAE, rollout buffers) is unchanged —
only the hidden state dynamics differ.

Key differences from ActorCritic:
    - Continuous-time: (1-alpha) leak preserves 80% of previous state per step
    - ReLU: hidden states are non-negative (>= 0), no saturation
    - Explicit noise: Gaussian noise added during forward pass (can be 0)
    - No nn.RNN: custom forward loop for full control over dynamics
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ContinuousActorCritic(nn.Module):
    """
    Continuous-time ReLU Actor-Critic compatible with LatentNet circuit inference.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dim : int
        Number of hidden units.
    action_dim : int
        Number of possible actions.
    alpha : float, optional
        Integration time constant (0 < alpha <= 1). Default is 0.2.
        Controls how much of the previous state persists: (1-alpha).
    sigma_rec : float, optional
        Recurrent noise standard deviation. Default is 0.15.
        Set to 0 for deterministic dynamics during evaluation.
    gain : float, optional
        Scaling factor for recurrent weight initialization. Default is 1.5.
    bias : bool, optional
        Whether to include bias. Default is False.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        alpha: float = 0.2,
        sigma_rec: float = 0.15,
        gain: float = 1.5,
        noise: float = 0.0,  # kept for API compat with ActorCritic; use sigma_rec instead
        bias: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.sigma_rec = sigma_rec
        self.gain = gain
        self.noise = noise

        # Weight matrices (equivalent to nn.RNN but explicit)
        self.W_ih = nn.Linear(input_dim, hidden_dim, bias=bias)   # input -> hidden
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=bias)  # hidden -> hidden

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim, bias=bias)

        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1, bias=bias)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights matching ActorCritic conventions."""
        # Input weights
        init.normal_(self.W_ih.weight, mean=0, std=1.0 / (self.input_dim ** 0.5))
        # Recurrent weights
        init.normal_(self.W_hh.weight, mean=0, std=self.gain / (self.hidden_dim ** 0.5))
        # Output weights
        for layer in [self.actor, self.critic]:
            init.normal_(layer.weight, mean=0, std=1.0 / self.hidden_dim)
        # Biases
        if self.W_ih.bias is not None:
            init.constant_(self.W_ih.bias, 0)
        if self.W_hh.bias is not None:
            init.constant_(self.W_hh.bias, 0)
        for layer in [self.actor, self.critic]:
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with continuous-time ReLU dynamics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).
        hx : torch.Tensor
            Hidden state of shape (1, batch_size, hidden_dim).
            Uses (1, B, H) convention for compatibility with ActorCritic.

        Returns
        -------
        actor_logits : torch.Tensor
            Action logits of shape (batch_size, action_dim).
        critic_value : torch.Tensor
            Value estimate of shape (batch_size, 1).
        new_hx : torch.Tensor
            Updated hidden state of shape (1, batch_size, hidden_dim).
        """
        # hx is (1, batch, hidden) for ActorCritic API compat; squeeze layer dim
        h = hx.squeeze(0)  # (batch, hidden)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)

            # Compute pre-activation
            pre_act = self.W_hh(h) + self.W_ih(x_t)  # (batch, hidden)

            # Add noise during training
            if self.training and self.sigma_rec > 0:
                noise = (
                    torch.sqrt(torch.tensor(2 * self.alpha)) * self.sigma_rec
                    * torch.randn_like(pre_act)
                )
                pre_act = pre_act + noise

            # Continuous-time ReLU update
            h = (1 - self.alpha) * h + self.alpha * F.relu(pre_act)

        # Actor and critic from final hidden state
        actor_logits = self.actor(h)
        critic_value = self.critic(h)

        # Restore (1, batch, hidden) shape for hx
        new_hx = h.unsqueeze(0)

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

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        preset_value: float = 0.0,
    ) -> torch.Tensor:
        """Reset hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.full(
            (1, batch_size, self.hidden_dim),
            preset_value,
            device=device,
        )

    @property
    def rnn_weight_hh(self) -> torch.Tensor:
        """Access recurrent weight matrix (for circuit analysis)."""
        return self.W_hh.weight

    @property
    def rnn_weight_ih(self) -> torch.Tensor:
        """Access input weight matrix (for circuit analysis)."""
        return self.W_ih.weight
