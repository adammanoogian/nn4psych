#!/usr/bin/env python3
"""
Transfer Learning Script: PIE → NeuroGym

This script tests whether RNN dynamics learned on Predictive Inference tasks
(change-point, oddball) transfer to NeuroGym decision-making tasks.

Transfer Strategy:
1. Load pre-trained single-task ActorCritic weights from PIE training
2. Create MultiTaskActorCritic with both PIE and NeuroGym tasks
3. Transfer RNN hidden→hidden weights (temporal dynamics)
4. Train new encoder/actor heads on NeuroGym tasks
5. Evaluate: (a) NeuroGym performance, (b) PIE performance (catastrophic forgetting)

Usage:
    # Basic transfer with default settings
    python transfer_learning.py --pretrained_path <path_to_pie_model.pth>

    # Transfer and train on specific NeuroGym tasks
    python transfer_learning.py --pretrained_path <path> --target_tasks daw-two-step

    # Compare frozen vs fine-tuned RNN
    python transfer_learning.py --pretrained_path <path> --freeze_rnn

    # Full experiment with forgetting analysis
    python transfer_learning.py --pretrained_path <path> --eval_pie_forgetting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Project imports
from envs import PIE_CP_OB_v2
from src.nn4psych.models.multitask_actor_critic import (
    MultiTaskActorCritic,
    TaskSpec,
)
from nn4psych.models.actor_critic import ActorCritic

# Optional neurogym import
try:
    from envs.neurogym_wrapper import (
        DawTwoStepWrapper,
        SingleContextDecisionMakingWrapper,
        PerceptualDecisionMakingWrapper,
        NEUROGYM_AVAILABLE,
    )
except ImportError:
    NEUROGYM_AVAILABLE = False
    DawTwoStepWrapper = None
    SingleContextDecisionMakingWrapper = None
    PerceptualDecisionMakingWrapper = None


@dataclass
class TransferConfig:
    """Configuration for transfer learning experiment."""
    # Pre-trained model
    pretrained_path: str = ""

    # Target tasks to transfer to
    target_tasks: List[str] = field(default_factory=lambda: ["daw-two-step"])

    # Whether to also keep training on source tasks (prevent forgetting)
    include_source_tasks: bool = True
    source_tasks: List[str] = field(default_factory=lambda: ["change-point", "oddball"])

    # Transfer settings
    freeze_rnn: bool = False  # If True, only train encoder/actor heads
    transfer_input_weights: bool = False  # Transfer input→hidden for matching dims

    # Training parameters
    adaptation_epochs: int = 50
    trials_per_task: int = 100
    max_time: int = 300

    # Model parameters (should match pretrained)
    hidden_dim: int = 64

    # Optimizer
    learning_rate: float = 0.0005
    gamma: float = 0.95
    rollout_size: int = 50

    # Evaluation
    eval_pie_forgetting: bool = True
    eval_frequency: int = 10

    # Output
    seed: int = 42
    save_dir: str = "trained_models/transfer"


# Task specifications
def get_task_specs() -> Dict[str, TaskSpec]:
    """Get all available task specifications."""
    specs = {}

    # PIE tasks (with shorter episodes for faster training)
    specs["change-point"] = TaskSpec(
        obs_dim=6,
        action_dim=3,
        context_id=0,
        env_class=PIE_CP_OB_v2,
        env_kwargs={"condition": "change-point", "total_trials": 20, "max_time": 100},
        name="change-point",
    )
    specs["oddball"] = TaskSpec(
        obs_dim=6,
        action_dim=3,
        context_id=1,
        env_class=PIE_CP_OB_v2,
        env_kwargs={"condition": "oddball", "total_trials": 20, "max_time": 100},
        name="oddball",
    )

    # NeuroGym tasks (if available)
    if NEUROGYM_AVAILABLE:
        specs["daw-two-step"] = TaskSpec(
            obs_dim=3,  # DawTwoStep observation dim
            action_dim=3,  # DawTwoStep action dim
            context_id=2,
            env_class=DawTwoStepWrapper,
            env_kwargs={"total_trials": 100},
            name="daw-two-step",
        )
        specs["context-dm"] = TaskSpec(
            obs_dim=33,  # ContextDM with default dim_ring
            action_dim=33,
            context_id=3,
            env_class=SingleContextDecisionMakingWrapper,
            env_kwargs={"total_trials": 100},
            name="context-dm",
        )
        specs["perceptual-dm"] = TaskSpec(
            obs_dim=33,  # PerceptualDM with default dim_ring
            action_dim=33,
            context_id=4,
            env_class=PerceptualDecisionMakingWrapper,
            env_kwargs={"total_trials": 100},
            name="perceptual-dm",
        )

    return specs


def load_pretrained_weights(path: str) -> Dict[str, torch.Tensor]:
    """Load pre-trained single-task ActorCritic weights."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"Loaded pretrained weights from: {path}")
    print(f"  Layers: {list(state_dict.keys())}")

    return state_dict


def transfer_weights(
    multitask_model: MultiTaskActorCritic,
    pretrained_state: Dict[str, torch.Tensor],
    transfer_hh: bool = True,
    transfer_ih: bool = False,
) -> Dict[str, bool]:
    """
    Transfer weights from single-task to multi-task model.

    Returns dict indicating which weights were transferred.
    """
    transferred = {}

    # Get current model state
    model_state = multitask_model.state_dict()

    # Transfer RNN hidden→hidden weights (always compatible if hidden_dim matches)
    if transfer_hh and 'rnn.weight_hh_l0' in pretrained_state:
        pretrained_hh = pretrained_state['rnn.weight_hh_l0']
        model_hh = model_state['rnn.weight_hh_l0']

        if pretrained_hh.shape == model_hh.shape:
            model_state['rnn.weight_hh_l0'] = pretrained_hh.clone()
            transferred['rnn.weight_hh_l0'] = True
            print(f"  [OK] Transferred rnn.weight_hh_l0: {pretrained_hh.shape}")
        else:
            print(f"  [SKIP] Shape mismatch rnn.weight_hh_l0: {pretrained_hh.shape} vs {model_hh.shape}")
            transferred['rnn.weight_hh_l0'] = False

    # Transfer critic weights (task-agnostic value function)
    if 'critic.weight' in pretrained_state:
        pretrained_critic = pretrained_state['critic.weight']
        model_critic = model_state['critic.weight']

        if pretrained_critic.shape == model_critic.shape:
            model_state['critic.weight'] = pretrained_critic.clone()
            transferred['critic.weight'] = True
            print(f"  [OK] Transferred critic.weight: {pretrained_critic.shape}")
        else:
            print(f"  [SKIP] Shape mismatch critic.weight: {pretrained_critic.shape} vs {model_critic.shape}")
            transferred['critic.weight'] = False

    # Optionally transfer input weights for PIE tasks (if dimensions match)
    if transfer_ih and 'rnn.weight_ih_l0' in pretrained_state:
        # This is trickier - only works if we have a PIE encoder that matches
        # For now, skip this as MultiTaskActorCritic uses task-specific encoders
        print(f"  - Skipping rnn.weight_ih_l0 (task-specific encoders used)")
        transferred['rnn.weight_ih_l0'] = False

    # Load the modified state dict
    multitask_model.load_state_dict(model_state)

    return transferred


class TransferTrainer:
    """Trainer for transfer learning experiments."""

    def __init__(self, config: TransferConfig, device: torch.device):
        self.config = config
        self.device = device

        # Get task specs
        all_specs = get_task_specs()

        # Build task list: source + target
        self.task_ids = []
        self.task_specs = {}

        if config.include_source_tasks:
            for task_id in config.source_tasks:
                if task_id in all_specs:
                    self.task_ids.append(task_id)
                    self.task_specs[task_id] = all_specs[task_id]

        for task_id in config.target_tasks:
            if task_id in all_specs and task_id not in self.task_ids:
                self.task_ids.append(task_id)
                self.task_specs[task_id] = all_specs[task_id]

        print(f"Tasks for transfer: {self.task_ids}")

        # Create model
        self.model = MultiTaskActorCritic(
            self.task_specs,
            hidden_dim=config.hidden_dim,
        ).to(device)

        # Load and transfer pretrained weights
        if config.pretrained_path:
            pretrained_state = load_pretrained_weights(config.pretrained_path)
            self.transferred = transfer_weights(
                self.model,
                pretrained_state,
                transfer_hh=True,
                transfer_ih=config.transfer_input_weights,
            )
        else:
            self.transferred = {}
            print("No pretrained weights - training from scratch (baseline)")

        # Freeze RNN if requested
        if config.freeze_rnn:
            self.model.freeze_shared_layers()
            print("Froze shared RNN layers - only training task-specific heads")

        # Create environments with shorter episodes for faster training
        self.envs = {}
        for task_id, spec in self.task_specs.items():
            env_kwargs = {**spec.env_kwargs}
            # Override with config settings
            if 'condition' in env_kwargs:
                # PIE environment - has max_time parameter
                env_kwargs['total_trials'] = min(config.trials_per_task, 20)
                env_kwargs['max_time'] = min(config.max_time, 100)
            else:
                # NeuroGym environment - only total_trials
                env_kwargs['total_trials'] = min(config.trials_per_task, 20)
            self.envs[task_id] = spec.env_class(**env_kwargs)
            print(f"  Created env {task_id}: {env_kwargs}", flush=True)

        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
        )

        # Metrics storage
        self.metrics = {task_id: {'returns': [], 'distances': [], 'losses': []}
                       for task_id in self.task_ids}
        self.source_metrics = {task_id: {'returns': [], 'distances': []}
                              for task_id in config.source_tasks}

    def run_episode(self, task_id: str, train: bool = True) -> Dict[str, float]:
        """Run one episode on a task."""
        env = self.envs[task_id]
        obs, done = env.reset()

        hx = self.model.get_initial_hidden(1, self.device)

        episode_reward = 0
        episode_steps = 0
        distances = []

        log_probs = []
        values = []
        rewards = []

        max_steps = 500  # Safety limit
        while not done and episode_steps < max_steps:
            # Normalize observation
            obs_norm = env.normalize_states(obs)
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get context
            context = self.model.get_context(task_id, self.device).unsqueeze(0)

            # Concatenate obs + context + reward placeholder
            reward_tensor = torch.zeros(1, 1, device=self.device)
            x = torch.cat([obs_tensor, context, reward_tensor], dim=-1)

            # Add sequence dimension: (batch, seq, features)
            x = x.unsqueeze(1)

            # Forward pass
            with torch.set_grad_enabled(train):
                logits, value, hx = self.model(x, hx, task_id)

            # Sample action
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            if train:
                log_probs.append(dist.log_prob(action))
                values.append(value)

            # Step environment
            obs, reward, done = env.step(action.item())
            rewards.append(reward)
            episode_reward += reward
            episode_steps += 1

            # Track distance for PIE tasks
            if hasattr(env, 'bucket_pos') and hasattr(env, 'helicopter_pos'):
                distances.append(abs(env.bucket_pos - env.helicopter_pos))

        # Training update
        loss = 0
        if train and len(log_probs) > 0:
            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.config.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Compute loss
            log_probs_t = torch.stack(log_probs).view(-1)
            values_t = torch.stack(values).view(-1)

            advantages = returns - values_t.detach()

            actor_loss = -(log_probs_t * advantages).mean()
            critic_loss = nn.functional.mse_loss(values_t, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss = loss.item()

        return {
            'return': episode_reward,
            'steps': episode_steps,
            'distance': np.mean(distances) if distances else 0,
            'loss': loss,
        }

    def evaluate_task(self, task_id: str, n_episodes: int = 3) -> Dict[str, float]:
        """Evaluate performance on a task without training."""
        self.model.eval()

        returns = []
        distances = []

        for _ in range(n_episodes):
            result = self.run_episode(task_id, train=False)
            returns.append(result['return'])
            distances.append(result['distance'])

        self.model.train()

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_distance': np.mean(distances),
        }

    def train(self) -> Dict[str, Any]:
        """Run the transfer learning training loop."""
        print(f"\n{'='*60}", flush=True)
        print("Starting Transfer Learning", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Source tasks: {self.config.source_tasks}", flush=True)
        print(f"Target tasks: {self.config.target_tasks}", flush=True)
        print(f"Adaptation epochs: {self.config.adaptation_epochs}", flush=True)
        print(f"Freeze RNN: {self.config.freeze_rnn}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Initial evaluation on source tasks (before adaptation)
        initial_source_perf = {}
        if self.config.eval_pie_forgetting:
            print("Initial evaluation on source tasks:", flush=True)
            for task_id in self.config.source_tasks:
                if task_id in self.envs:
                    print(f"  Evaluating {task_id}...", flush=True)
                    perf = self.evaluate_task(task_id, n_episodes=2)
                    initial_source_perf[task_id] = perf
                    print(f"  {task_id}: return={perf['mean_return']:.2f}, dist={perf['mean_distance']:.1f}", flush=True)

        # Training loop
        for epoch in range(self.config.adaptation_epochs):
            epoch_metrics = {task_id: {'returns': [], 'distances': [], 'losses': []}
                           for task_id in self.task_ids}

            # Interleave tasks within epoch
            for task_id in self.task_ids:
                # Run a few episodes per task per epoch
                n_episodes = max(1, self.config.trials_per_task // 20)
                for trial in range(n_episodes):
                    result = self.run_episode(task_id, train=True)
                    epoch_metrics[task_id]['returns'].append(result['return'])
                    epoch_metrics[task_id]['distances'].append(result['distance'])
                    epoch_metrics[task_id]['losses'].append(result['loss'])

            # Store metrics
            for task_id in self.task_ids:
                self.metrics[task_id]['returns'].append(np.mean(epoch_metrics[task_id]['returns']))
                self.metrics[task_id]['distances'].append(np.mean(epoch_metrics[task_id]['distances']))
                self.metrics[task_id]['losses'].append(np.mean(epoch_metrics[task_id]['losses']))

            # Periodic logging
            if epoch % self.config.eval_frequency == 0 or epoch == self.config.adaptation_epochs - 1:
                msg = f"Epoch {epoch:4d} |"
                for task_id in self.task_ids:
                    r = self.metrics[task_id]['returns'][-1]
                    d = self.metrics[task_id]['distances'][-1]
                    abbrev = task_id[:2].upper()
                    msg += f" {abbrev}: R={r:.1f}, d={d:.1f} |"
                print(msg, flush=True)

        # Final evaluation
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")

        final_perf = {}
        for task_id in self.task_ids:
            perf = self.evaluate_task(task_id)
            final_perf[task_id] = perf
            print(f"  {task_id}: return={perf['mean_return']:.2f} ± {perf['std_return']:.2f}, dist={perf['mean_distance']:.1f}")

        # Catastrophic forgetting analysis
        forgetting = {}
        if self.config.eval_pie_forgetting and initial_source_perf:
            print(f"\n{'='*60}")
            print("Catastrophic Forgetting Analysis")
            print(f"{'='*60}")
            for task_id in self.config.source_tasks:
                if task_id in initial_source_perf and task_id in final_perf:
                    initial = initial_source_perf[task_id]['mean_return']
                    final = final_perf[task_id]['mean_return']
                    delta = final - initial
                    pct = (delta / abs(initial) * 100) if initial != 0 else 0
                    forgetting[task_id] = {
                        'initial': initial,
                        'final': final,
                        'delta': delta,
                        'percent': pct,
                    }
                    print(f"  {task_id}: {initial:.2f} -> {final:.2f} (delta={delta:+.2f}, {pct:+.1f}%)")

        return {
            'config': asdict(self.config),
            'transferred_weights': self.transferred,
            'metrics': self.metrics,
            'final_performance': final_perf,
            'forgetting': forgetting,
            'initial_source_performance': initial_source_perf if self.config.eval_pie_forgetting else {},
        }

    def save_results(self, results: Dict[str, Any], suffix: str = ""):
        """Save model and results."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"transfer_{timestamp}{suffix}"

        # Save model
        model_path = save_dir / f"{base_name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'task_specs': {k: asdict(v) for k, v in self.task_specs.items()},
        }, model_path)
        print(f"\nSaved model: {model_path}")

        # Save results JSON
        results_path = save_dir / f"{base_name}_results.json"
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Saved results: {results_path}")

        # Save learning curves plot
        self.plot_learning_curves(save_dir / f"{base_name}_curves.png")

        return model_path, results_path

    def plot_learning_curves(self, save_path: Path):
        """Plot learning curves for all tasks."""
        n_tasks = len(self.task_ids)
        fig, axes = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 8))

        if n_tasks == 1:
            axes = axes.reshape(2, 1)

        for i, task_id in enumerate(self.task_ids):
            # Returns
            axes[0, i].plot(self.metrics[task_id]['returns'], label='Return')
            axes[0, i].set_title(f'{task_id} - Returns')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Mean Return')
            axes[0, i].grid(True, alpha=0.3)

            # Distances (for PIE tasks)
            if any(self.metrics[task_id]['distances']):
                axes[1, i].plot(self.metrics[task_id]['distances'], label='Distance', color='orange')
                axes[1, i].set_title(f'{task_id} - Distance')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Mean Distance')
                axes[1, i].grid(True, alpha=0.3)
            else:
                axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{task_id} - Distance (N/A)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved learning curves: {save_path}")


def main():
    from nn4psych.training.resources import configure_cpu_threads
    n_threads = configure_cpu_threads()
    if n_threads:
        print(f"CPU threads limited to {n_threads}")

    parser = argparse.ArgumentParser(description="Transfer Learning: PIE → NeuroGym")

    # Required
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained PIE model weights')

    # Target tasks
    parser.add_argument('--target_tasks', nargs='+', default=['daw-two-step'],
                       help='NeuroGym tasks to transfer to')

    # Transfer settings
    parser.add_argument('--freeze_rnn', action='store_true',
                       help='Freeze RNN weights, only train task heads')
    parser.add_argument('--no_source_tasks', action='store_true',
                       help='Do not include source tasks during adaptation')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Adaptation epochs')
    parser.add_argument('--trials', type=int, default=100,
                       help='Trials per task per epoch')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor')

    # Evaluation
    parser.add_argument('--eval_pie_forgetting', action='store_true', default=True,
                       help='Evaluate catastrophic forgetting on PIE tasks')

    # Output
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='trained_models/transfer')

    args = parser.parse_args()

    # Check neurogym availability
    if not NEUROGYM_AVAILABLE:
        print("ERROR: NeuroGym not available. Install with: pip install neurogym")
        sys.exit(1)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create config
    config = TransferConfig(
        pretrained_path=args.pretrained_path,
        target_tasks=args.target_tasks,
        include_source_tasks=not args.no_source_tasks,
        freeze_rnn=args.freeze_rnn,
        adaptation_epochs=args.epochs,
        trials_per_task=args.trials,
        learning_rate=args.lr,
        gamma=args.gamma,
        eval_pie_forgetting=args.eval_pie_forgetting,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    # Run transfer learning
    trainer = TransferTrainer(config, device)
    results = trainer.train()

    # Save
    suffix = f"_{'frozen' if args.freeze_rnn else 'finetuned'}"
    trainer.save_results(results, suffix)

    print("\nTransfer learning complete!")


if __name__ == "__main__":
    main()
