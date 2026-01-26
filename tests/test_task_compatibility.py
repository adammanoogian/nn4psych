"""
Task Compatibility Tests for Multi-Task Actor-Critic Architecture.

Tests that each task (PIE and NeuroGym) works correctly with the
MultiTaskActorCritic model using a tiny model (hidden_dim=16) for fast execution.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project paths to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "envs"))
sys.path.insert(0, str(project_root))

from nn4psych.models.multitask_actor_critic import MultiTaskActorCritic, TaskSpec
from envs import PIE_CP_OB_v2

# Check if neurogym is available
try:
    from envs import (
        NEUROGYM_AVAILABLE,
        DawTwoStepWrapper,
        SingleContextDecisionMakingWrapper,
        PerceptualDecisionMakingWrapper,
    )
except ImportError:
    NEUROGYM_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_hidden_dim():
    """Tiny hidden dimension for fast tests."""
    return 16


# =============================================================================
# PIE Task Tests
# =============================================================================

class TestPIEChangepoint:
    """Tests for PIE change-point task."""

    @pytest.fixture
    def env(self):
        """Create change-point environment."""
        return PIE_CP_OB_v2(condition="change-point", total_trials=10)

    @pytest.fixture
    def task_spec(self):
        """Create task spec for change-point."""
        return TaskSpec(
            obs_dim=6,
            action_dim=3,
            context_id=0,
            env_class=PIE_CP_OB_v2,
            env_kwargs={"condition": "change-point"},
            name="change-point",
        )

    @pytest.fixture
    def model(self, task_spec, tiny_hidden_dim, device):
        """Create model with single task."""
        task_specs = {"change-point": task_spec}
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_env_dimensions(self, env):
        """Test that environment has correct dimensions."""
        assert env.observation_space.shape[0] == 6
        assert env.action_space.n == 3

    def test_env_reset(self, env):
        """Test environment reset returns correct shapes."""
        obs, done = env.reset()

        assert obs.shape == (6,)
        assert done is False
        assert obs.dtype == np.float32

    def test_env_normalize(self, env):
        """Test environment normalization."""
        obs, _ = env.reset()
        norm_obs = env.normalize_states(obs)

        # After normalization, values should be in reasonable range
        assert norm_obs.max() <= 2.0
        assert norm_obs.min() >= -2.0

    def test_model_shapes(self, model, task_spec, tiny_hidden_dim, device):
        """Test model forward pass shapes."""
        task_id = "change-point"

        # Verify encoder input dimension
        expected_input = task_spec.obs_dim + model.context_dim + 1  # obs + context + reward
        assert model.encoders[task_id].in_features == expected_input

        # Verify actor output dimension
        assert model.actors[task_id].out_features == task_spec.action_dim

        # Create input tensor: [obs (6), context (1), reward (1)] = 8
        obs = torch.randn(1, 1, task_spec.obs_dim + model.context_dim + 1, device=device)
        hx = model.get_initial_hidden(batch_size=1, device=device)

        # Forward pass
        actor_logits, critic_value, new_hx = model(obs, hx, task_id)

        assert actor_logits.shape == (1, task_spec.action_dim)
        assert critic_value.shape == (1, 1)
        assert new_hx.shape == (1, 1, tiny_hidden_dim)

    def test_complete_trial(self, env, model, tiny_hidden_dim, device):
        """Test running a complete trial with model."""
        task_id = "change-point"
        num_tasks = 1

        # Reset environment
        obs, done = env.reset()
        assert done is False

        # Initialize hidden state
        hx = model.get_initial_hidden(batch_size=1, device=device)

        # Run trial
        total_reward = 0.0
        step = 0
        max_steps = 100

        while not done and step < max_steps:
            # Normalize observation
            norm_obs = env.normalize_states(obs)

            # Create context (one-hot for single task)
            context = np.zeros(num_tasks)
            context[0] = 1.0

            # Concatenate: [obs, context, reward]
            prev_reward = 0.0 if step == 0 else total_reward
            state = np.concatenate([norm_obs, context, [prev_reward / 10.0]])

            # Convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)

            # Forward pass
            actor_logits, critic_value, hx = model(state_tensor, hx, task_id)

            # Get action (greedy)
            action = actor_logits.argmax(dim=1).item()

            # Step environment
            obs, reward, done = env.step(action)
            total_reward += reward
            step += 1

        # Verify trial completed
        assert done is True or step == max_steps
        assert isinstance(total_reward, (float, np.floating))


class TestPIEOddball:
    """Tests for PIE oddball task."""

    @pytest.fixture
    def env(self):
        """Create oddball environment."""
        return PIE_CP_OB_v2(condition="oddball", total_trials=10)

    @pytest.fixture
    def task_spec(self):
        """Create task spec for oddball."""
        return TaskSpec(
            obs_dim=6,
            action_dim=3,
            context_id=1,
            env_class=PIE_CP_OB_v2,
            env_kwargs={"condition": "oddball"},
            name="oddball",
        )

    @pytest.fixture
    def model(self, task_spec, tiny_hidden_dim, device):
        """Create model with single task."""
        task_specs = {"oddball": task_spec}
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_env_dimensions(self, env):
        """Test that environment has correct dimensions."""
        assert env.observation_space.shape[0] == 6
        assert env.action_space.n == 3

    def test_env_reset(self, env):
        """Test environment reset returns correct shapes."""
        obs, done = env.reset()

        assert obs.shape == (6,)
        assert done is False

    def test_model_shapes(self, model, task_spec, tiny_hidden_dim, device):
        """Test model forward pass shapes."""
        task_id = "oddball"

        # Create input tensor
        input_dim = task_spec.obs_dim + model.context_dim + 1
        obs = torch.randn(1, 1, input_dim, device=device)
        hx = model.get_initial_hidden(batch_size=1, device=device)

        # Forward pass
        actor_logits, critic_value, new_hx = model(obs, hx, task_id)

        assert actor_logits.shape == (1, task_spec.action_dim)
        assert critic_value.shape == (1, 1)
        assert new_hx.shape == (1, 1, tiny_hidden_dim)

    def test_complete_trial(self, env, model, tiny_hidden_dim, device):
        """Test running a complete trial with model."""
        task_id = "oddball"

        obs, done = env.reset()
        hx = model.get_initial_hidden(batch_size=1, device=device)

        step = 0
        max_steps = 100

        while not done and step < max_steps:
            norm_obs = env.normalize_states(obs)
            context = np.array([1.0])  # Single task
            state = np.concatenate([norm_obs, context, [0.0]])

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

            actor_logits, _, hx = model(state_tensor, hx, task_id)
            action = actor_logits.argmax(dim=1).item()

            obs, reward, done = env.step(action)
            step += 1

        assert done is True or step == max_steps


# =============================================================================
# NeuroGym Task Tests
# =============================================================================

@pytest.mark.skipif(not NEUROGYM_AVAILABLE, reason="neurogym not installed")
class TestNeurogymDawTwoStep:
    """Tests for Daw Two-Step task."""

    @pytest.fixture
    def env(self):
        """Create Daw Two-Step environment."""
        return DawTwoStepWrapper(context_id=0)

    @pytest.fixture
    def task_spec(self, env):
        """Create task spec from environment."""
        return TaskSpec(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            context_id=0,
            env_class=DawTwoStepWrapper,
            env_kwargs={},
            name="daw-two-step",
        )

    @pytest.fixture
    def model(self, task_spec, tiny_hidden_dim, device):
        """Create model with single task."""
        task_specs = {"daw-two-step": task_spec}
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_env_dimensions(self, env):
        """Test that environment has expected dimensions."""
        assert env.obs_dim > 0
        assert env.action_dim > 0
        print(f"DawTwoStep: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    def test_env_reset(self, env):
        """Test environment reset returns correct shapes."""
        obs, done = env.reset()

        assert obs.shape == (env.obs_dim,)
        assert done is False

    def test_env_normalize(self, env):
        """Test environment normalization."""
        obs, _ = env.reset()
        norm_obs = env.normalize_states(obs)

        # Neurogym normalization clips to [-1, 1]
        assert norm_obs.max() <= 1.0
        assert norm_obs.min() >= -1.0

    def test_model_shapes(self, model, task_spec, tiny_hidden_dim, device):
        """Test model forward pass shapes."""
        task_id = "daw-two-step"

        input_dim = task_spec.obs_dim + model.context_dim + 1
        obs = torch.randn(1, 1, input_dim, device=device)
        hx = model.get_initial_hidden(batch_size=1, device=device)

        actor_logits, critic_value, new_hx = model(obs, hx, task_id)

        assert actor_logits.shape == (1, task_spec.action_dim)
        assert critic_value.shape == (1, 1)
        assert new_hx.shape == (1, 1, tiny_hidden_dim)

    def test_complete_trial(self, env, model, tiny_hidden_dim, device):
        """Test running a complete trial with model."""
        task_id = "daw-two-step"

        obs, done = env.reset()
        hx = model.get_initial_hidden(batch_size=1, device=device)

        step = 0
        max_steps = 100

        while not done and step < max_steps:
            norm_obs = env.normalize_states(obs)
            context = np.array([1.0])
            state = np.concatenate([norm_obs, context, [0.0]])

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

            actor_logits, _, hx = model(state_tensor, hx, task_id)
            action = actor_logits.argmax(dim=1).item()

            obs, reward, done = env.step(action)
            step += 1

        assert done is True or step == max_steps


@pytest.mark.skipif(not NEUROGYM_AVAILABLE, reason="neurogym not installed")
class TestNeurogymContextDM:
    """Tests for Context Decision Making task."""

    @pytest.fixture
    def env(self):
        """Create Context DM environment."""
        return SingleContextDecisionMakingWrapper(context_id=0, modality_context=0)

    @pytest.fixture
    def task_spec(self, env):
        """Create task spec from environment."""
        return TaskSpec(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            context_id=0,
            env_class=SingleContextDecisionMakingWrapper,
            env_kwargs={"modality_context": 0},
            name="context-dm",
        )

    @pytest.fixture
    def model(self, task_spec, tiny_hidden_dim, device):
        """Create model with single task."""
        task_specs = {"context-dm": task_spec}
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_env_dimensions(self, env):
        """Test that environment has expected dimensions."""
        assert env.obs_dim > 0
        assert env.action_dim > 0
        print(f"ContextDM: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    def test_env_reset(self, env):
        """Test environment reset returns correct shapes."""
        obs, done = env.reset()

        assert obs.shape == (env.obs_dim,)
        assert done is False

    def test_model_shapes(self, model, task_spec, tiny_hidden_dim, device):
        """Test model forward pass shapes."""
        task_id = "context-dm"

        input_dim = task_spec.obs_dim + model.context_dim + 1
        obs = torch.randn(1, 1, input_dim, device=device)
        hx = model.get_initial_hidden(batch_size=1, device=device)

        actor_logits, critic_value, new_hx = model(obs, hx, task_id)

        assert actor_logits.shape == (1, task_spec.action_dim)
        assert critic_value.shape == (1, 1)

    def test_complete_trial(self, env, model, tiny_hidden_dim, device):
        """Test running a complete trial with model."""
        task_id = "context-dm"

        obs, done = env.reset()
        hx = model.get_initial_hidden(batch_size=1, device=device)

        step = 0
        max_steps = 100

        while not done and step < max_steps:
            norm_obs = env.normalize_states(obs)
            context = np.array([1.0])
            state = np.concatenate([norm_obs, context, [0.0]])

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

            actor_logits, _, hx = model(state_tensor, hx, task_id)
            action = actor_logits.argmax(dim=1).item()

            obs, reward, done = env.step(action)
            step += 1

        assert done is True or step == max_steps


@pytest.mark.skipif(not NEUROGYM_AVAILABLE, reason="neurogym not installed")
class TestNeurogymPerceptualDM:
    """Tests for Perceptual Decision Making task."""

    @pytest.fixture
    def env(self):
        """Create Perceptual DM environment."""
        return PerceptualDecisionMakingWrapper(context_id=0)

    @pytest.fixture
    def task_spec(self, env):
        """Create task spec from environment."""
        return TaskSpec(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            context_id=0,
            env_class=PerceptualDecisionMakingWrapper,
            env_kwargs={},
            name="perceptual-dm",
        )

    @pytest.fixture
    def model(self, task_spec, tiny_hidden_dim, device):
        """Create model with single task."""
        task_specs = {"perceptual-dm": task_spec}
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_env_dimensions(self, env):
        """Test that environment has expected dimensions."""
        assert env.obs_dim > 0
        assert env.action_dim > 0
        print(f"PerceptualDM: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    def test_env_reset(self, env):
        """Test environment reset returns correct shapes."""
        obs, done = env.reset()

        assert obs.shape == (env.obs_dim,)
        assert done is False

    def test_model_shapes(self, model, task_spec, tiny_hidden_dim, device):
        """Test model forward pass shapes."""
        task_id = "perceptual-dm"

        input_dim = task_spec.obs_dim + model.context_dim + 1
        obs = torch.randn(1, 1, input_dim, device=device)
        hx = model.get_initial_hidden(batch_size=1, device=device)

        actor_logits, critic_value, new_hx = model(obs, hx, task_id)

        assert actor_logits.shape == (1, task_spec.action_dim)
        assert critic_value.shape == (1, 1)

    def test_complete_trial(self, env, model, tiny_hidden_dim, device):
        """Test running a complete trial with model."""
        task_id = "perceptual-dm"

        obs, done = env.reset()
        hx = model.get_initial_hidden(batch_size=1, device=device)

        step = 0
        max_steps = 100

        while not done and step < max_steps:
            norm_obs = env.normalize_states(obs)
            context = np.array([1.0])
            state = np.concatenate([norm_obs, context, [0.0]])

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

            actor_logits, _, hx = model(state_tensor, hx, task_id)
            action = actor_logits.argmax(dim=1).item()

            obs, reward, done = env.step(action)
            step += 1

        assert done is True or step == max_steps


# =============================================================================
# Multi-Task Combination Tests
# =============================================================================

class TestMultiTaskPIEOnly:
    """Tests for multi-task model with both PIE tasks."""

    @pytest.fixture
    def task_specs(self):
        """Create task specs for both PIE tasks."""
        return {
            "change-point": TaskSpec(
                obs_dim=6,
                action_dim=3,
                context_id=0,
                env_class=PIE_CP_OB_v2,
                env_kwargs={"condition": "change-point"},
                name="change-point",
            ),
            "oddball": TaskSpec(
                obs_dim=6,
                action_dim=3,
                context_id=1,
                env_class=PIE_CP_OB_v2,
                env_kwargs={"condition": "oddball"},
                name="oddball",
            ),
        }

    @pytest.fixture
    def model(self, task_specs, tiny_hidden_dim, device):
        """Create multi-task model."""
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_model_structure(self, model, task_specs):
        """Test that model has correct structure for both tasks."""
        assert len(model.encoders) == 2
        assert len(model.actors) == 2
        assert "change-point" in model.encoders
        assert "oddball" in model.encoders
        assert model.context_dim == 2  # One-hot for 2 tasks

    def test_context_vectors(self, model, device):
        """Test that context vectors are correct."""
        ctx_cp = model.get_context("change-point", device)
        ctx_ob = model.get_context("oddball", device)

        assert ctx_cp.shape == (2,)
        assert ctx_ob.shape == (2,)
        assert torch.allclose(ctx_cp, torch.tensor([1.0, 0.0], device=device))
        assert torch.allclose(ctx_ob, torch.tensor([0.0, 1.0], device=device))

    def test_forward_both_tasks(self, model, tiny_hidden_dim, device):
        """Test forward pass works for both tasks."""
        for task_id in ["change-point", "oddball"]:
            # Input: obs (6) + context (2) + reward (1) = 9
            x = torch.randn(1, 1, 9, device=device)
            hx = model.get_initial_hidden(batch_size=1, device=device)

            actor_logits, critic_value, new_hx = model(x, hx, task_id)

            assert actor_logits.shape == (1, 3)
            assert critic_value.shape == (1, 1)
            assert new_hx.shape == (1, 1, tiny_hidden_dim)

    def test_run_trials_both_tasks(self, model, task_specs, tiny_hidden_dim, device):
        """Test running trials on both tasks alternately."""
        envs = {
            "change-point": PIE_CP_OB_v2(condition="change-point", total_trials=5),
            "oddball": PIE_CP_OB_v2(condition="oddball", total_trials=5),
        }

        # Run one trial on each task
        for task_id, env in envs.items():
            obs, done = env.reset()
            hx = model.get_initial_hidden(batch_size=1, device=device)

            step = 0
            while not done and step < 50:
                norm_obs = env.normalize_states(obs)
                context = model.get_context(task_id, device).cpu().numpy()
                state = np.concatenate([norm_obs, context, [0.0]])

                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

                actor_logits, _, hx = model(state_tensor, hx, task_id)
                action = actor_logits.argmax(dim=1).item()

                obs, reward, done = env.step(action)
                step += 1

            assert done is True or step == 50, f"Task {task_id} didn't complete"


@pytest.mark.skipif(not NEUROGYM_AVAILABLE, reason="neurogym not installed")
class TestMultiTaskMixed:
    """Tests for multi-task model with PIE and NeuroGym tasks."""

    @pytest.fixture
    def task_specs(self):
        """Create task specs for mixed tasks."""
        # Create environments to get actual dimensions
        daw_env = DawTwoStepWrapper(context_id=1)

        return {
            "change-point": TaskSpec(
                obs_dim=6,
                action_dim=3,
                context_id=0,
                env_class=PIE_CP_OB_v2,
                env_kwargs={"condition": "change-point"},
                name="change-point",
            ),
            "daw-two-step": TaskSpec(
                obs_dim=daw_env.obs_dim,
                action_dim=daw_env.action_dim,
                context_id=1,
                env_class=DawTwoStepWrapper,
                env_kwargs={},
                name="daw-two-step",
            ),
        }

    @pytest.fixture
    def model(self, task_specs, tiny_hidden_dim, device):
        """Create multi-task model."""
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_model_structure(self, model, task_specs):
        """Test that model has correct structure."""
        assert len(model.encoders) == 2
        assert len(model.actors) == 2
        assert model.context_dim == 2

    def test_different_input_dims(self, model, task_specs, device):
        """Test that model handles different observation dimensions."""
        for task_id, spec in task_specs.items():
            input_dim = spec.obs_dim + model.context_dim + 1
            x = torch.randn(1, 1, input_dim, device=device)
            hx = model.get_initial_hidden(batch_size=1, device=device)

            actor_logits, critic_value, new_hx = model(x, hx, task_id)

            assert actor_logits.shape == (1, spec.action_dim)
            assert critic_value.shape == (1, 1)

    def test_forward_with_raw_obs(self, model, task_specs, tiny_hidden_dim, device):
        """Test forward_with_raw_obs convenience method."""
        for task_id, spec in task_specs.items():
            obs = torch.randn(1, spec.obs_dim, device=device)
            reward = torch.zeros(1, 1, device=device)
            hx = model.get_initial_hidden(batch_size=1, device=device)

            actor_logits, critic_value, new_hx = model.forward_with_raw_obs(
                obs, reward, hx, task_id
            )

            assert actor_logits.shape == (1, spec.action_dim)
            assert critic_value.shape == (1, 1)


# =============================================================================
# Utility Tests
# =============================================================================

class TestModelUtilities:
    """Tests for model utility functions."""

    @pytest.fixture
    def model(self, tiny_hidden_dim, device):
        """Create a simple model."""
        task_specs = {
            "task1": TaskSpec(obs_dim=4, action_dim=2, context_id=0,
                             env_class=PIE_CP_OB_v2, name="task1"),
            "task2": TaskSpec(obs_dim=6, action_dim=4, context_id=1,
                             env_class=PIE_CP_OB_v2, name="task2"),
        }
        model = MultiTaskActorCritic(task_specs, hidden_dim=tiny_hidden_dim)
        return model.to(device)

    def test_get_task_info(self, model):
        """Test get_task_info returns correct information."""
        info = model.get_task_info()

        assert len(info) == 2
        assert info["task1"]["obs_dim"] == 4
        assert info["task1"]["action_dim"] == 2
        assert info["task2"]["obs_dim"] == 6
        assert info["task2"]["action_dim"] == 4

    def test_get_parameter_count(self, model):
        """Test parameter counting."""
        counts = model.get_parameter_count()

        assert "encoders" in counts
        assert "rnn" in counts
        assert "actors" in counts
        assert "critic" in counts
        assert "total" in counts
        assert counts["total"] > 0

    def test_freeze_unfreeze(self, model):
        """Test freezing and unfreezing layers."""
        # Initially all trainable
        assert all(p.requires_grad for p in model.rnn.parameters())

        # Freeze shared layers
        model.freeze_shared_layers()
        assert all(not p.requires_grad for p in model.rnn.parameters())
        assert all(not p.requires_grad for p in model.critic.parameters())

        # Task-specific heads should still be trainable
        for encoder in model.encoders.values():
            assert all(p.requires_grad for p in encoder.parameters())
        for actor in model.actors.values():
            assert all(p.requires_grad for p in actor.parameters())

        # Unfreeze all
        model.unfreeze_all()
        assert all(p.requires_grad for p in model.parameters())

    def test_initial_hidden_state(self, model, tiny_hidden_dim, device):
        """Test initial hidden state generation."""
        hx = model.get_initial_hidden(batch_size=4, device=device)

        assert hx.shape == (1, 4, tiny_hidden_dim)
        assert hx.device == device
        assert torch.all(hx == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
