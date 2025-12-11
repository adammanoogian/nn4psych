#!/usr/bin/env python3
"""
Multi-Task RNN Training Script

This script trains an RNN actor-critic model on multiple tasks with different
interleaving strategies. It supports tasks with different observation and action
spaces through task-specific encoder/decoder heads.

Interleaving Strategies:
- epoch: Alternate tasks per epoch (current default approach)
- trial: Random task per trial within epoch
- block: K trials of task A, then K trials of task B
- curriculum: Start with easier task, gradually add harder ones

Usage:
    python train_multitask.py --epochs 100 --interleave_mode epoch
    python train_multitask.py --epochs 100 --interleave_mode trial --seed 42
    python train_multitask.py --epochs 100 --interleave_mode block --block_size 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d

from envs import PIE_CP_OB_v2
from src.nn4psych.models.multitask_actor_critic import (
    MultiTaskActorCritic,
    PaddedMultiTaskActorCritic,
    TaskSpec,
)
from nn4psych.utils.metrics import get_lrs_v2
from nn4psych.utils.plotting import plot_behavior


class InterleaveMode(Enum):
    """Task interleaving strategies."""
    EPOCH = "epoch"      # Alternate tasks per epoch
    TRIAL = "trial"      # Random task per trial
    BLOCK = "block"      # K trials per task in blocks
    CURRICULUM = "curriculum"  # Start easy, add harder tasks


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training."""
    # Training parameters
    epochs: int = 100
    trials_per_task: int = 200
    max_time: int = 300

    # Model parameters
    hidden_dim: int = 64
    gain: float = 1.5
    bias: bool = False
    use_task_embedding: bool = False
    embedding_dim: int = 8

    # Optimization parameters
    gamma: float = 0.95
    rollout_size: int = 50
    learning_rate: float = 5e-4

    # Interleaving parameters
    interleave_mode: str = "epoch"
    block_size: int = 50  # For block mode
    curriculum_warmup: int = 20  # Epochs before adding new tasks

    # Task parameters
    max_displacement: float = 10.0
    reward_size: float = 5.0
    step_cost: float = 0.0
    alpha: float = 1.0

    # Training curriculum
    train_ratio: float = 0.5  # Fraction of epochs with visible helicopter
    preset_memory: float = 0.0

    # Misc
    seed: int = 42
    eval_frequency: int = 10  # Evaluate on all tasks every N epochs
    save_dir: str = "model_params"

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_filename(self) -> str:
        return (
            f"MT_{self.interleave_mode}_{self.gamma}g_{self.rollout_size}bz_"
            f"{self.hidden_dim}n_{self.epochs}e_{self.seed}s"
        )


class RolloutBuffer:
    """Buffer for storing rollout experiences."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.clear()

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        done: bool,
        task_id: str,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.dones.append(done)
        self.task_ids.append(task_id)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.dones = []
        self.task_ids = []

    def __len__(self):
        return len(self.rewards)


@dataclass
class TaskMetrics:
    """Metrics tracked for each task."""
    returns: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    times: List[int] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    prediction_errors: List[float] = field(default_factory=list)

    def add_epoch(
        self,
        returns: np.ndarray,
        loss: float,
        times: np.ndarray,
        distances: np.ndarray,
    ):
        self.returns.append(np.mean(returns))
        self.losses.append(loss)
        self.times.append(np.mean(times))
        self.distances.append(np.mean(distances))


class MultiTaskTrainer:
    """
    Trainer for multi-task RNN models.

    Parameters
    ----------
    config : MultiTaskConfig
        Training configuration.
    task_specs : Dict[str, TaskSpec]
        Task specifications.
    model_type : str, optional
        Model architecture: "heads" or "padded". Default is "heads".
    device : torch.device, optional
        Device to train on.
    """

    def __init__(
        self,
        config: MultiTaskConfig,
        task_specs: Dict[str, TaskSpec],
        model_type: str = "heads",
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.task_specs = task_specs
        self.task_ids = list(task_specs.keys())
        self.num_tasks = len(task_specs)

        # Set device
        if device is None:
            device = torch.device("cpu")
        self.device = device

        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create model
        if model_type == "heads":
            self.model = MultiTaskActorCritic(
                task_specs=task_specs,
                hidden_dim=config.hidden_dim,
                gain=config.gain,
                bias=config.bias,
                use_task_embedding=config.use_task_embedding,
                embedding_dim=config.embedding_dim,
            ).to(device)
        else:
            self.model = PaddedMultiTaskActorCritic(
                task_specs=task_specs,
                hidden_dim=config.hidden_dim,
                gain=config.gain,
                bias=config.bias,
            ).to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Initialize metrics tracking
        self.metrics = {task_id: TaskMetrics() for task_id in task_specs}
        self.epoch_data = {}  # Store detailed epoch data

        # Store model checkpoints
        self.checkpoints = [deepcopy(self.model.state_dict())]

        print(f"Model type: {model_type}")
        print(f"Parameter count: {self.model.get_parameter_count()}")
        print(f"Tasks: {self.task_ids}")

    def create_env(self, task_id: str, train_cond: bool) -> PIE_CP_OB_v2:
        """Create environment for a task."""
        spec = self.task_specs[task_id]
        return spec.env_class(
            total_trials=self.config.trials_per_task,
            max_time=self.config.max_time,
            train_cond=train_cond,
            max_displacement=self.config.max_displacement,
            reward_size=self.config.reward_size,
            step_cost=self.config.step_cost,
            alpha=self.config.alpha,
            **spec.env_kwargs,
        )

    def compute_gae(
        self,
        buffer: RolloutBuffer,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = buffer.rewards
        values = buffer.values
        dones = buffer.dones

        advantages = []
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t]
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = (
                rewards[t]
                + self.config.gamma * next_value * next_non_terminal
                - values[t]
            )
            advantages.insert(0, delta)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(
            [v.item() for v in values], dtype=torch.float32
        ).to(self.device)
        log_probs = torch.stack(buffer.log_probs)
        values_tensor = torch.stack(buffer.values).squeeze()

        return returns, advantages, log_probs, values_tensor

    def train_trial(
        self,
        env: PIE_CP_OB_v2,
        task_id: str,
        hx: torch.Tensor,
        buffer: RolloutBuffer,
    ) -> Tuple[float, int, torch.Tensor]:
        """
        Run a single trial and collect experiences.

        Returns
        -------
        total_return : float
            Total return for this trial.
        time_steps : int
            Number of time steps taken.
        hx : torch.Tensor
            Updated hidden state.
        """
        next_obs, done = env.reset()
        norm_obs = env.normalize_states(next_obs)

        # Get context for this task
        context = self.model.get_context(task_id, self.device)

        # Construct state: [obs, context, reward]
        next_state = torch.cat([
            torch.FloatTensor(norm_obs).to(self.device),
            context,
            torch.tensor([0.0], device=self.device),
        ])
        next_state = next_state.unsqueeze(0).unsqueeze(0)

        hx = hx.detach()
        total_return = 0

        while not done:
            # Random memory reset
            if np.random.random() < self.config.preset_memory:
                hx = (
                    torch.randn(1, 1, self.config.hidden_dim)
                    * 1 / self.config.hidden_dim
                ).to(self.device)

            # Forward pass
            actor_logits, critic_value, hx = self.model(next_state, hx, task_id)

            # Sample action
            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            # Take action
            next_obs, reward, done = env.step(action.item())
            total_return += reward

            # Add to buffer
            buffer.add_experience(
                state=next_state,
                action=action,
                reward=reward,
                value=critic_value,
                log_prob=probs.log_prob(action),
                entropy=probs.entropy(),
                done=done,
                task_id=task_id,
            )

            # Prepare next state
            norm_obs = env.normalize_states(next_obs)
            next_state = torch.cat([
                torch.FloatTensor(norm_obs).to(self.device),
                context,
                torch.tensor([reward], device=self.device),
            ])
            next_state = next_state.unsqueeze(0).unsqueeze(0)

        return total_return, env.time, hx

    def update_model(self, buffer: RolloutBuffer) -> float:
        """Perform a model update from buffer experiences."""
        returns, advantages, log_probs, values = self.compute_gae(buffer)

        # Calculate losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((returns - values) ** 2).mean()
        entropy_loss = -torch.stack(buffer.entropies).mean()

        loss = actor_loss + 0.5 * critic_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        buffer.clear()

        return loss.item()

    def train_epoch_interleaved(
        self,
        epoch: int,
        train_cond: bool,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Train one epoch with epoch-level interleaving (original approach).

        Each task is trained for all trials, tasks are shuffled per epoch.
        """
        results = {}
        task_order = np.random.permutation(self.task_ids)

        for task_id in task_order:
            env = self.create_env(task_id, train_cond)
            buffer = RolloutBuffer(self.config.rollout_size)

            returns = []
            times = []
            losses = []

            hx = (
                torch.randn(1, 1, self.config.hidden_dim)
                * 1 / self.config.hidden_dim ** 0.5
            ).to(self.device)

            for trial in range(self.config.trials_per_task):
                ret, time_steps, hx = self.train_trial(env, task_id, hx, buffer)
                returns.append(ret)
                times.append(time_steps)

                # Update when buffer is full
                if len(buffer) >= self.config.rollout_size:
                    loss = self.update_model(buffer)
                    losses.append(loss)

            # Get environment state history
            states = np.array([
                env.trials,
                env.bucket_positions,
                env.bag_positions,
                env.helicopter_positions,
                env.hazard_triggers,
            ])

            # Calculate distances
            distances = np.abs(states[3] - states[1])  # heli - bucket

            results[task_id] = {
                'returns': np.array(returns),
                'times': np.array(times),
                'losses': np.array(losses) if losses else np.array([0]),
                'distances': distances,
                'states': states,
            }

            # Update metrics
            self.metrics[task_id].add_epoch(
                returns=np.array(returns),
                loss=np.mean(losses) if losses else 0,
                times=np.array(times),
                distances=distances,
            )

        return results

    def train_epoch_trial_interleaved(
        self,
        epoch: int,
        train_cond: bool,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Train one epoch with trial-level interleaving.

        Each trial randomly selects a task.
        """
        # Create environments for all tasks
        envs = {
            task_id: self.create_env(task_id, train_cond)
            for task_id in self.task_ids
        }

        # Initialize tracking
        results = {
            task_id: {
                'returns': [],
                'times': [],
                'losses': [],
            }
            for task_id in self.task_ids
        }

        # Shared buffer (experiences from all tasks)
        buffer = RolloutBuffer(self.config.rollout_size)

        # Hidden states per task
        hx_states = {
            task_id: (
                torch.randn(1, 1, self.config.hidden_dim)
                * 1 / self.config.hidden_dim ** 0.5
            ).to(self.device)
            for task_id in self.task_ids
        }

        # Track trials per task
        trial_counts = {task_id: 0 for task_id in self.task_ids}
        total_trials = self.config.trials_per_task * self.num_tasks

        for _ in range(total_trials):
            # Randomly select task
            task_id = np.random.choice(self.task_ids)

            # Check if task has remaining trials
            if trial_counts[task_id] >= self.config.trials_per_task:
                # Find task with remaining trials
                available = [
                    t for t in self.task_ids
                    if trial_counts[t] < self.config.trials_per_task
                ]
                if not available:
                    break
                task_id = np.random.choice(available)

            env = envs[task_id]
            hx = hx_states[task_id]

            ret, time_steps, hx = self.train_trial(env, task_id, hx, buffer)

            hx_states[task_id] = hx
            results[task_id]['returns'].append(ret)
            results[task_id]['times'].append(time_steps)
            trial_counts[task_id] += 1

            # Update when buffer is full
            if len(buffer) >= self.config.rollout_size:
                loss = self.update_model(buffer)
                # Distribute loss to tasks that contributed
                task_set = set(buffer.task_ids)
                for t in task_set:
                    results[t]['losses'].append(loss)

        # Finalize results
        for task_id in self.task_ids:
            env = envs[task_id]
            states = np.array([
                env.trials,
                env.bucket_positions,
                env.bag_positions,
                env.helicopter_positions,
                env.hazard_triggers,
            ])
            distances = np.abs(states[3] - states[1])

            results[task_id]['returns'] = np.array(results[task_id]['returns'])
            results[task_id]['times'] = np.array(results[task_id]['times'])
            results[task_id]['losses'] = np.array(results[task_id]['losses'])
            results[task_id]['distances'] = distances
            results[task_id]['states'] = states

            self.metrics[task_id].add_epoch(
                returns=results[task_id]['returns'],
                loss=np.mean(results[task_id]['losses']) if results[task_id]['losses'].size > 0 else 0,
                times=results[task_id]['times'],
                distances=distances,
            )

        return results

    def train_epoch_block_interleaved(
        self,
        epoch: int,
        train_cond: bool,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Train one epoch with block-level interleaving.

        Trains K trials on task A, then K trials on task B, etc.
        """
        results = {
            task_id: {
                'returns': [],
                'times': [],
                'losses': [],
            }
            for task_id in self.task_ids
        }

        # Create environments
        envs = {
            task_id: self.create_env(task_id, train_cond)
            for task_id in self.task_ids
        }

        # Hidden states
        hx_states = {
            task_id: (
                torch.randn(1, 1, self.config.hidden_dim)
                * 1 / self.config.hidden_dim ** 0.5
            ).to(self.device)
            for task_id in self.task_ids
        }

        buffer = RolloutBuffer(self.config.rollout_size)

        # Calculate number of blocks
        block_size = self.config.block_size
        trials_per_task = self.config.trials_per_task
        num_blocks = trials_per_task // block_size

        # Interleave blocks
        for block in range(num_blocks):
            # Shuffle task order for this block
            task_order = np.random.permutation(self.task_ids)

            for task_id in task_order:
                env = envs[task_id]
                hx = hx_states[task_id]

                for _ in range(block_size):
                    ret, time_steps, hx = self.train_trial(env, task_id, hx, buffer)
                    results[task_id]['returns'].append(ret)
                    results[task_id]['times'].append(time_steps)

                    if len(buffer) >= self.config.rollout_size:
                        loss = self.update_model(buffer)
                        results[task_id]['losses'].append(loss)

                hx_states[task_id] = hx

        # Finalize
        for task_id in self.task_ids:
            env = envs[task_id]
            states = np.array([
                env.trials,
                env.bucket_positions,
                env.bag_positions,
                env.helicopter_positions,
                env.hazard_triggers,
            ])
            distances = np.abs(states[3] - states[1])

            results[task_id]['returns'] = np.array(results[task_id]['returns'])
            results[task_id]['times'] = np.array(results[task_id]['times'])
            results[task_id]['losses'] = np.array(results[task_id]['losses'])
            results[task_id]['distances'] = distances
            results[task_id]['states'] = states

            self.metrics[task_id].add_epoch(
                returns=results[task_id]['returns'],
                loss=np.mean(results[task_id]['losses']) if results[task_id]['losses'].size > 0 else 0,
                times=results[task_id]['times'],
                distances=distances,
            )

        return results

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns
        -------
        Dict containing training history and final metrics.
        """
        train_epochs = int(self.config.epochs * self.config.train_ratio)
        all_results = []

        print(f"\nStarting training: {self.config.epochs} epochs")
        print(f"Train (visible helicopter): epochs 0-{train_epochs-1}")
        print(f"Test (hidden helicopter): epochs {train_epochs}-{self.config.epochs-1}")
        print(f"Interleave mode: {self.config.interleave_mode}")
        print("-" * 50)

        # Select training function based on interleave mode
        mode = InterleaveMode(self.config.interleave_mode)
        if mode == InterleaveMode.EPOCH:
            train_fn = self.train_epoch_interleaved
        elif mode == InterleaveMode.TRIAL:
            train_fn = self.train_epoch_trial_interleaved
        elif mode == InterleaveMode.BLOCK:
            train_fn = self.train_epoch_block_interleaved
        else:
            train_fn = self.train_epoch_interleaved

        for epoch in range(self.config.epochs):
            # Determine training condition
            train_cond = epoch < train_epochs

            # Reinitialize optimizer at test phase
            if epoch == train_epochs:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                )

            # Train epoch
            results = train_fn(epoch, train_cond)
            all_results.append(results)

            # Log progress
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                log_parts = [f"Epoch {epoch:4d}"]
                for task_id in self.task_ids:
                    mean_ret = np.mean(results[task_id]['returns'])
                    mean_dist = np.mean(results[task_id]['distances'])
                    log_parts.append(f"{task_id[:2].upper()}: G={mean_ret:.3f}, d={mean_dist:.1f}")
                print(" | ".join(log_parts))

            # Checkpoint at phase transitions
            if epoch == train_epochs - 1:
                perf = np.mean([
                    np.mean(results[t]['distances']) for t in self.task_ids
                ])
                if perf < 32:
                    self.checkpoints.append(deepcopy(self.model.state_dict()))
                    print(f"  [Checkpoint saved: end of training phase]")

        # Final checkpoint
        self.checkpoints.append(deepcopy(self.model.state_dict()))

        return {
            'epoch_results': all_results,
            'metrics': self.metrics,
            'config': self.config.to_dict(),
            'checkpoints': self.checkpoints,
        }

    def evaluate(
        self,
        task_id: str,
        num_trials: int = 100,
        train_cond: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate model on a single task without updating weights.

        Parameters
        ----------
        task_id : str
            Task to evaluate on.
        num_trials : int
            Number of evaluation trials.
        train_cond : bool
            Whether helicopter is visible.

        Returns
        -------
        Dict with evaluation metrics.
        """
        self.model.eval()

        env = self.create_env(task_id, train_cond)
        env.total_trials = num_trials

        returns = []
        times = []

        hx = self.model.get_initial_hidden(device=self.device)

        with torch.no_grad():
            for _ in range(num_trials):
                next_obs, done = env.reset()
                norm_obs = env.normalize_states(next_obs)
                context = self.model.get_context(task_id, self.device)

                next_state = torch.cat([
                    torch.FloatTensor(norm_obs).to(self.device),
                    context,
                    torch.tensor([0.0], device=self.device),
                ])
                next_state = next_state.unsqueeze(0).unsqueeze(0)

                total_return = 0

                while not done:
                    actor_logits, _, hx = self.model(next_state, hx, task_id)
                    probs = Categorical(logits=actor_logits)
                    action = probs.sample()

                    next_obs, reward, done = env.step(action.item())
                    total_return += reward

                    norm_obs = env.normalize_states(next_obs)
                    next_state = torch.cat([
                        torch.FloatTensor(norm_obs).to(self.device),
                        context,
                        torch.tensor([reward], device=self.device),
                    ])
                    next_state = next_state.unsqueeze(0).unsqueeze(0)

                returns.append(total_return)
                times.append(env.time)

        self.model.train()

        states = np.array([
            env.trials,
            env.bucket_positions,
            env.bag_positions,
            env.helicopter_positions,
            env.hazard_triggers,
        ])

        return {
            'returns': np.array(returns),
            'times': np.array(times),
            'distances': np.abs(states[3] - states[1]),
            'states': states,
        }

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves for all tasks."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_tasks))
        train_epochs = int(self.config.epochs * self.config.train_ratio)

        for idx, task_id in enumerate(self.task_ids):
            color = colors[idx]
            metrics = self.metrics[task_id]

            # Returns
            axes[0, 0].plot(metrics.returns, color=color, label=task_id, alpha=0.8)

            # Distances
            axes[0, 1].plot(metrics.distances, color=color, label=task_id, alpha=0.8)

            # Times
            axes[1, 0].plot(metrics.times, color=color, label=task_id, alpha=0.8)

            # Losses
            axes[1, 1].plot(metrics.losses, color=color, label=task_id, alpha=0.8)

        # Formatting
        axes[0, 0].set_ylabel('Mean Return')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].axvline(train_epochs, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].legend()

        axes[0, 1].set_ylabel('Mean Distance (Heli-Bucket)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].axhline(32, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(train_epochs, color='gray', linestyle='--', alpha=0.5)

        axes[1, 0].set_ylabel('Mean Time to Confirm')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].axvline(train_epochs, color='gray', linestyle='--', alpha=0.5)

        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].axvline(train_epochs, color='gray', linestyle='--', alpha=0.5)

        plt.suptitle(
            f"Multi-Task Training ({self.config.interleave_mode} interleaving)",
            fontsize=14,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def save_model(self, filepath: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def save_results(self, filepath: str, results: Dict):
        """Save training results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, TaskMetrics):
                return asdict(obj)
            else:
                return obj

        serializable = convert(results)

        # Remove checkpoints (too large for JSON)
        if 'checkpoints' in serializable:
            del serializable['checkpoints']

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {filepath}")


def create_default_task_specs() -> Dict[str, TaskSpec]:
    """Create default task specifications for CP and OB tasks."""
    return {
        'change-point': TaskSpec(
            obs_dim=6,
            action_dim=3,
            context_id=0,
            env_class=PIE_CP_OB_v2,
            env_kwargs={'condition': 'change-point'},
            name='Change-Point',
        ),
        'oddball': TaskSpec(
            obs_dim=6,
            action_dim=3,
            context_id=1,
            env_class=PIE_CP_OB_v2,
            env_kwargs={'condition': 'oddball'},
            name='Oddball',
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-Task RNN Training')

    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--maxt', type=int, default=300)

    # Model params
    parser.add_argument('--nrnn', type=int, default=64)
    parser.add_argument('--gain', type=float, default=1.5)
    parser.add_argument('--use_embedding', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--model_type', type=str, default='heads',
                        choices=['heads', 'padded'])

    # Optimization params
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--rollsz', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)

    # Interleaving params
    parser.add_argument('--interleave_mode', type=str, default='epoch',
                        choices=['epoch', 'trial', 'block', 'curriculum'])
    parser.add_argument('--block_size', type=int, default=50)

    # Task params
    parser.add_argument('--maxdisp', type=float, default=10.0)
    parser.add_argument('--rewardsize', type=float, default=5.0)

    # Training curriculum
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--presetmem', type=float, default=0.0)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='model_params')

    args = parser.parse_args()

    # Create config
    config = MultiTaskConfig(
        epochs=args.epochs,
        trials_per_task=args.trials,
        max_time=args.maxt,
        hidden_dim=args.nrnn,
        gain=args.gain,
        use_task_embedding=args.use_embedding,
        embedding_dim=args.embedding_dim,
        gamma=args.gamma,
        rollout_size=args.rollsz,
        learning_rate=args.lr,
        interleave_mode=args.interleave_mode,
        block_size=args.block_size,
        max_displacement=args.maxdisp,
        reward_size=args.rewardsize,
        train_ratio=args.ratio,
        preset_memory=args.presetmem,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    print("=" * 60)
    print("Multi-Task RNN Training")
    print("=" * 60)
    print(f"Config: {config.to_dict()}")
    print("=" * 60)

    # Create task specs
    task_specs = create_default_task_specs()

    # Create trainer
    trainer = MultiTaskTrainer(
        config=config,
        task_specs=task_specs,
        model_type=args.model_type,
    )

    # Train
    results = trainer.train()

    # Plot
    fig_path = f"figures/model_performance/MT_{config.get_filename()}.png"
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.plot_training_curves(save_path=fig_path)

    # Save model
    model_path = f"{config.save_dir}/MT_{config.get_filename()}.pth"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_path)

    # Save results
    results_path = f"output/MT_{config.get_filename()}_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_results(results_path, results)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model: {model_path}")
    print(f"Figure: {fig_path}")
    print(f"Results: {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
