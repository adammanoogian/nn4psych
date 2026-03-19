#!/usr/bin/env python3
"""
Train RNN ActorCritic on ContextDecisionMaking-v0 and extract hidden states.

This script trains a single-task ActorCritic model on the neurogym
ContextDecisionMaking task, then extracts behavioral data and hidden
state trajectories for downstream latent circuit inference (Phase 3).

Usage:
    python scripts/training/train_context_dm.py --epochs 200 --trials 200
    python scripts/training/train_context_dm.py --epochs 50 --trials 100 --seed 42
    python scripts/training/train_context_dm.py --both_modalities --epochs 10 --trials 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from nn4psych.models.actor_critic import ActorCritic
from nn4psych.analysis.behavior import extract_behavior_with_hidden
from nn4psych.training.resources import configure_cpu_threads
from envs.neurogym_wrapper import (
    SingleContextDecisionMakingWrapper,
    NEUROGYM_AVAILABLE,
)


def _run_trial_block(model, env, optimizer, args, hx, device, n_trials):
    """Run n_trials on the given environment, updating model via REINFORCE/A2C.

    Returns updated hx and per-trial rewards. Modifies model in-place via optimizer.
    """
    buffer_states = []
    buffer_actions = []
    buffer_rewards = []
    buffer_values = []
    buffer_log_probs = []
    buffer_dones = []

    epoch_rewards = []
    reward = 0.0

    for trial in range(n_trials):
        obs, done = env.reset()
        norm_obs = env.normalize_states(obs)
        state = np.concatenate([norm_obs, env.context, [reward]])
        hx = hx.detach()
        trial_reward = 0.0

        step_count = 0
        while not done and step_count < args.max_steps:
            x = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            actor_logits, critic_value, hx = model(x, hx)

            dist = Categorical(logits=actor_logits)
            action = dist.sample()

            obs, reward, done = env.step(action.item())
            trial_reward += reward

            buffer_states.append(x)
            buffer_actions.append(action)
            buffer_rewards.append(reward)
            buffer_values.append(critic_value)
            buffer_log_probs.append(dist.log_prob(action))
            buffer_dones.append(done)

            norm_obs = env.normalize_states(obs)
            state = np.concatenate([norm_obs, env.context, [reward]])
            step_count += 1

            # Update when buffer is full
            if len(buffer_rewards) >= args.rollout_size:
                values_detached = [v.item() for v in buffer_values]
                advantages = []
                next_value = 0.0
                for t in reversed(range(len(buffer_rewards))):
                    if t == len(buffer_rewards) - 1:
                        next_non_terminal = 1.0 - float(buffer_dones[t])
                        next_value = values_detached[t]
                    else:
                        next_non_terminal = 1.0 - float(buffer_dones[t + 1])
                        next_value = values_detached[t + 1]
                    delta = (
                        buffer_rewards[t]
                        + args.gamma * next_value * next_non_terminal
                        - values_detached[t]
                    )
                    advantages.insert(0, delta)

                advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)
                returns = advantages_t + torch.tensor(
                    values_detached, dtype=torch.float32
                ).to(device)
                log_probs_t = torch.stack(buffer_log_probs)
                values_t = torch.stack(buffer_values).squeeze()

                actor_loss = -(log_probs_t * advantages_t).mean()
                critic_loss = ((returns.detach() - values_t) ** 2).mean()
                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hx = hx.detach()

                buffer_states.clear()
                buffer_actions.clear()
                buffer_rewards.clear()
                buffer_values.clear()
                buffer_log_probs.clear()
                buffer_dones.clear()

        epoch_rewards.append(trial_reward)

    return hx, epoch_rewards


def train_context_dm(args):
    """Train ActorCritic on ContextDecisionMaking-v0 (single modality)."""
    if not NEUROGYM_AVAILABLE:
        raise ImportError(
            "neurogym is not installed. Install with: "
            "pip install git+https://github.com/neurogym/neurogym"
        )

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Create environment
    env = SingleContextDecisionMakingWrapper(
        context_id=0,
        modality_context=args.modality_context,
        total_trials=args.trials,
        dt=args.dt,
        sigma=args.sigma,
        dim_ring=args.dim_ring,
    )
    env.set_num_tasks(1)  # Single-task: context is just [1.0]

    # Build model — input_dim = obs_dim + context_dim + 1 (reward)
    obs_dim = env.obs_dim
    context_dim = 1  # Single task
    input_dim = obs_dim + context_dim + 1
    hidden_dim = args.hidden_dim
    action_dim = env.action_dim

    print(f"Task: ContextDecisionMaking-v0")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  input_dim={input_dim} (obs={obs_dim} + ctx={context_dim} + reward=1)")
    print(f"  hidden_dim={hidden_dim}")
    print(f"  modality_context={args.modality_context}")
    print(f"  epochs={args.epochs}, trials={args.trials}")

    model = ActorCritic(input_dim, hidden_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    reward_history = []

    for epoch in range(args.epochs):
        env.reset_epoch()
        hx = (torch.randn(1, 1, hidden_dim) * 1 / hidden_dim ** 0.5).to(device)

        hx, epoch_rewards = _run_trial_block(model, env, optimizer, args, hx, device, args.trials)

        mean_reward = np.mean(epoch_rewards)
        reward_history.append(mean_reward)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d} | mean_reward={mean_reward:.4f}")

    return model, reward_history


def train_context_dm_dual(args):
    """Train ActorCritic on both modality contexts (0 and 1) alternating within each epoch.

    Each epoch runs trials/2 trials on modality_context=0 then trials/2 on modality_context=1.
    The trained model is saved as model_context_dm_dual.pth.

    This enables latent circuit analysis that reveals context-dependent routing
    (Langdon & Engel 2025).
    """
    if not NEUROGYM_AVAILABLE:
        raise ImportError(
            "neurogym is not installed. Install with: "
            "pip install git+https://github.com/neurogym/neurogym"
        )

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Create two environments, one per modality context
    env_0 = SingleContextDecisionMakingWrapper(
        context_id=0,
        modality_context=0,
        total_trials=args.trials,
        dt=args.dt,
        sigma=args.sigma,
        dim_ring=args.dim_ring,
    )
    env_0.set_num_tasks(1)

    env_1 = SingleContextDecisionMakingWrapper(
        context_id=0,
        modality_context=1,
        total_trials=args.trials,
        dt=args.dt,
        sigma=args.sigma,
        dim_ring=args.dim_ring,
    )
    env_1.set_num_tasks(1)

    # Build model — input_dim = obs_dim + context_dim + 1 (reward)
    obs_dim = env_0.obs_dim
    context_dim = 1  # Single task (set_num_tasks(1) for both)
    input_dim = obs_dim + context_dim + 1
    hidden_dim = args.hidden_dim
    action_dim = env_0.action_dim

    print(f"Task: ContextDecisionMaking-v0 (DUAL modality)")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  input_dim={input_dim} (obs={obs_dim} + ctx={context_dim} + reward=1)")
    print(f"  hidden_dim={hidden_dim}")
    print(f"  modality_contexts=[0, 1], alternating half-epoch blocks")
    print(f"  epochs={args.epochs}, trials_per_epoch={args.trials} ({args.trials//2} per context)")

    model = ActorCritic(input_dim, hidden_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_per_context = max(1, args.trials // 2)

    reward_history = []

    for epoch in range(args.epochs):
        env_0.reset_epoch()
        env_1.reset_epoch()
        hx = (torch.randn(1, 1, hidden_dim) * 1 / hidden_dim ** 0.5).to(device)

        # First half-epoch: modality_context=0
        hx, rewards_0 = _run_trial_block(model, env_0, optimizer, args, hx, device, n_per_context)
        # Second half-epoch: modality_context=1
        hx, rewards_1 = _run_trial_block(model, env_1, optimizer, args, hx, device, n_per_context)

        mean_reward = np.mean(rewards_0 + rewards_1)
        reward_history.append(mean_reward)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            r0 = np.mean(rewards_0) if rewards_0 else 0.0
            r1 = np.mean(rewards_1) if rewards_1 else 0.0
            print(f"Epoch {epoch:4d} | mean_reward={mean_reward:.4f} (ctx0={r0:.4f}, ctx1={r1:.4f})")

    return model, reward_history


def extract_and_save(model, args):
    """Extract hidden states and save to disk."""
    device = torch.device(args.device)

    # Create fresh environment for extraction
    env = SingleContextDecisionMakingWrapper(
        context_id=0,
        modality_context=args.modality_context,
        total_trials=args.extract_trials,
        dt=args.dt,
        sigma=args.sigma,
        dim_ring=args.dim_ring,
    )
    env.set_num_tasks(1)

    print(f"\nExtracting hidden states: {args.extract_epochs} epochs x {args.extract_trials} trials")

    result = extract_behavior_with_hidden(
        model=model,
        env=env,
        n_epochs=args.extract_epochs,
        n_trials=args.extract_trials,
        reset_memory=True,
        preset_memory=0.0,
        device=device,
    )

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(out_dir / 'hidden_context_dm.npy', result['hidden'])
    np.save(out_dir / 'trial_lengths_context_dm.npy', result['trial_lengths'])

    # Determine model filename based on training mode
    both_modalities = getattr(args, 'both_modalities', False)
    model_filename = 'model_context_dm_dual.pth' if both_modalities else 'model_context_dm.pth'
    model_path = out_dir / model_filename
    torch.save(model.state_dict(), model_path)

    # Save metadata (convert numpy ints to Python ints for JSON serialization)
    metadata = {
        'task': 'ContextDecisionMaking-v0',
        'modality_context': int(args.modality_context) if not both_modalities else 'both',
        'both_modalities': bool(both_modalities),
        'hidden_dim': int(args.hidden_dim),
        'obs_dim': int(env.obs_dim),
        'action_dim': int(env.action_dim),
        'input_dim': int(env.obs_dim + 1 + 1),
        'n_trials': int(result['hidden'].shape[0]),
        'max_T': int(result['hidden'].shape[1]),
        'extract_epochs': args.extract_epochs,
        'extract_trials': args.extract_trials,
        'training_epochs': args.epochs,
        'training_trials': args.trials,
        'seed': args.seed,
        'model_path': str(model_path),
        'dt': args.dt,
        'sigma': args.sigma,
        'dim_ring': args.dim_ring,
        'gamma': args.gamma,
        'lr': args.lr,
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {out_dir}/:")
    print(f"  hidden_context_dm.npy: shape {result['hidden'].shape}")
    print(f"  trial_lengths_context_dm.npy: shape {result['trial_lengths'].shape}")
    print(f"  {model_filename}")
    print(f"  metadata.json")

    return result


def main():
    configure_cpu_threads()

    parser = argparse.ArgumentParser(description="Train RNN on ContextDecisionMaking-v0")

    # Training args
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--trials', type=int, default=200, help='Trials per epoch')
    parser.add_argument('--hidden_dim', type=int, default=64, help='RNN hidden dimension')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--rollout_size', type=int, default=50, help='Rollout buffer size')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per trial')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device')

    # Task args
    parser.add_argument('--modality_context', type=int, default=0, help='Which modality (0 or 1)')
    parser.add_argument('--dt', type=int, default=100, help='Timestep ms')
    parser.add_argument('--sigma', type=float, default=1.0, help='Noise level')
    parser.add_argument('--dim_ring', type=int, default=2, help='Ring dimension')

    # Extraction args
    parser.add_argument('--extract_epochs', type=int, default=5, help='Epochs for extraction')
    parser.add_argument('--extract_trials', type=int, default=200, help='Trials per extraction epoch')
    parser.add_argument('--output_dir', type=str, default='data/processed/rnn_behav', help='Output directory')

    # Flags
    parser.add_argument('--skip_training', action='store_true', help='Skip training, load model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--both_modalities', action='store_true',
                        help='Train on both modality_context=0 and 1 (alternating half-epoch blocks)')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip hidden state extraction after training')

    args = parser.parse_args()

    if args.skip_training and args.model_path:
        # Load existing model
        env = SingleContextDecisionMakingWrapper(
            context_id=0, modality_context=args.modality_context,
            dt=args.dt, sigma=args.sigma, dim_ring=args.dim_ring,
        )
        env.set_num_tasks(1)
        input_dim = env.obs_dim + 1 + 1
        model = ActorCritic(input_dim, args.hidden_dim, env.action_dim)
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print(f"Loaded model from {args.model_path}")
    elif args.both_modalities:
        model, reward_history = train_context_dm_dual(args)
    else:
        model, reward_history = train_context_dm(args)

    if not getattr(args, 'skip_extraction', False):
        extract_and_save(model, args)
    else:
        # Even when skipping extraction, save the trained model
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        both_modalities = getattr(args, 'both_modalities', False)
        model_filename = 'model_context_dm_dual.pth' if both_modalities else 'model_context_dm.pth'
        model_path = out_dir / model_filename
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()
