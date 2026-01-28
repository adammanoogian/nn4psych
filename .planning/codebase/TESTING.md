# Testing Patterns

**Analysis Date:** 2026-01-28

## Test Framework

**Runner:**
- pytest (version 6.0+)
- Config: `pytest.ini` at project root
- Alternative config: `pyproject.toml` `[tool.pytest.ini_options]` section

**Assertion Library:**
- pytest's built-in assertions and `assert` statements
- torch-specific assertions: `torch.allclose()`, `torch.all()`, `assert tensor.shape == expected_shape`
- NumPy assertions: `np.array_equal()`, `np.testing.assert_array_almost_equal()`

**Run Commands:**
```bash
# Run all tests with coverage
pytest --cov=nn4psych --cov-report=term-missing

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching pattern
pytest -k "test_initialization" -v

# Run only non-slow tests
pytest -m "not slow" -v

# Run validation tests
pytest validation/ -v

# Run with short traceback
pytest --tb=short
```

## Test File Organization

**Location:**
- Primary: `tests/` directory at project root
- Validation: `validation/` directory at project root
- Pattern: Co-located with source in same structure as needed, but separate in own directory

**Naming:**
- Test files: `test_*.py` (e.g., `test_models.py`, `test_task_compatibility.py`)
- Test functions: `test_*` (e.g., `test_initialization`, `test_forward_pass`)
- Test classes: `Test*` (e.g., `TestActorCritic`, `TestPIEChangepoint`)

**Structure:**
```
tests/
├── __init__.py                    # Empty or shared fixtures
├── test_imports.py                # Import and package structure tests
├── test_models.py                 # Core model tests
└── test_task_compatibility.py     # Multi-task and environment tests

validation/
├── conftest.py                    # Shared pytest configuration
├── test_reorganization.py         # Package structure validation
├── test_parameter_recovery.py     # Parameter recovery validation
└── test_nassar2021.py             # Domain-specific validation
```

## Test Structure

**Suite Organization (Class-based):**
```python
# From tests/test_models.py
class TestActorCritic:
    """Test ActorCritic model functionality."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
        assert model.input_dim == 9
        assert model.hidden_dim == 64

    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
        x = torch.randn(1, 1, 9)
        hx = torch.zeros(1, 1, 64)
        actor_logits, critic_value, new_hx = model(x, hx)
        assert actor_logits.shape == (1, 3)
```

**Patterns:**
- One test class per major component
- One assertion per test (or closely related assertions)
- Fixture-based setup for complex objects
- Clear test names describing what is tested

**Setup/Teardown:**
- No explicit setup/teardown; use pytest fixtures instead
- Fixtures defined in `conftest.py` or at class level
- Automatic cleanup via scope management

**Assertion Pattern:**
- Direct `assert` statements: `assert model.input_dim == 9`
- Shape validation: `assert tensor.shape == (batch_size, expected_dim)`
- Tensor comparison: `torch.allclose(tensor1, tensor2)` for floating point
- All-equal checks: `torch.all(tensor == value)`
- Example from `tests/test_models.py`:
  ```python
  assert actor_logits.shape == (batch_size, 3)
  assert critic_value.shape == (batch_size, 1)
  assert new_hx.shape == (1, batch_size, 64)
  assert torch.all(h == 0)
  assert torch.allclose(h, torch.full_like(h, preset_val))
  ```

## Mocking

**Framework:** pytest built-in monkeypatch for fixtures; torch's capabilities used directly

**Patterns:**
- Minimal mocking; prefer real object creation
- Use real PyTorch tensors and models rather than mocks
- Environment objects created fresh per test
- Example from `tests/test_task_compatibility.py`:
  ```python
  # Real environment created via fixture
  @pytest.fixture
  def env(self):
      """Create change-point environment."""
      return PIE_CP_OB_v2(condition="change-point", total_trials=10)
  ```

**What to Mock:**
- Optional dependencies (neurogym integration via try-except at module level)
- External file I/O (if needed, use temporary directories)
- Time-dependent behavior (rarely needed)

**What NOT to Mock:**
- Model forward passes (test with real tensors)
- Environment steps (test with real environment)
- Core PyTorch operations (rely on PyTorch's own testing)
- Configuration objects (test with real configs)

## Fixtures and Factories

**Test Data/Fixtures:**
Located in `validation/conftest.py` and `tests/conftest.py`:

```python
# From validation/conftest.py
@pytest.fixture
def sample_model():
    """Create a sample ActorCritic model for testing."""
    from nn4psych.models import ActorCritic
    return ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

@pytest.fixture
def sample_env():
    """Create a sample environment for testing."""
    from envs import PIE_CP_OB_v2
    return PIE_CP_OB_v2(condition="change-point", total_trials=10)

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from nn4psych.training.configs import create_default_config
    config = create_default_config()
    config.training.epochs = 2  # Reduce for testing
    config.task.total_trials = 10
    return config

@pytest.fixture(scope="session")
def validation_seeds():
    """Standard seeds for validation testing."""
    return [42, 123, 456, 789, 1011]
```

**Fixture Scopes:**
- `function` (default): Fresh fixture per test
- `class`: Fresh fixture per test class
- `session`: Single fixture for entire test session (validation_seeds)

**Device Fixture (from test_task_compatibility.py):**
```python
@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Environment Fixtures:**
- Created per test/class as needed
- Include environment-specific validation
- Example from `tests/test_task_compatibility.py`:
  ```python
  @pytest.fixture
  def env(self):
      """Create change-point environment."""
      return PIE_CP_OB_v2(condition="change-point", total_trials=10)
  ```

**Location:**
- Shared fixtures: `validation/conftest.py`, `tests/conftest.py`
- Test-specific fixtures: Defined within test file or test class
- Conftest.py automatically discovered by pytest

## Coverage

**Requirements:**
- Target: High coverage for core modules (models, utils, training)
- Not enforced (no minimum percentage in config)
- Measured for: `nn4psych` package only (not tests or scripts)

**View Coverage:**
```bash
# Generate coverage report
pytest --cov=nn4psych --cov-report=term-missing

# Generate HTML report
pytest --cov=nn4psych --cov-report=html

# View specific module coverage
pytest --cov=nn4psych.models --cov-report=term-missing
```

**Coverage Configuration:**
From `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=nn4psych --cov-report=term-missing"
```

## Test Types

**Unit Tests:**
- Scope: Individual functions/methods
- Location: `tests/test_models.py`, `tests/test_task_compatibility.py`
- Approach: Isolated from dependencies, test single behavior
- Examples:
  - `test_initialization()` - Model initialization
  - `test_forward_pass()` - Single forward pass through network
  - `test_weight_initialization()` - Proper weight initialization
  - `test_gradient_flow()` - Gradient propagation through model

**Integration Tests:**
- Scope: Multiple components working together
- Location: `tests/test_task_compatibility.py` (marked with fixture dependencies)
- Approach: Test real environments with real models
- Examples:
  - `test_complete_trial()` - Full trial execution (model + environment)
  - `test_forward_both_tasks()` - Multi-task model with different tasks
  - `test_run_trials_both_tasks()` - Alternating task execution

**Validation Tests:**
- Scope: System-level correctness and reproducibility
- Location: `validation/` directory
- Approach: Test against known results or mathematical properties
- Examples:
  - `test_parameter_recovery.py` - Recover parameters from synthetic data
  - `test_reorganization.py` - Verify package structure after refactoring
  - `test_nassar2021.py` - Validate against published benchmark

**System Tests:**
- Located in: `tests/test_imports.py`, `tests/test_task_compatibility.py`
- Tests: Full import chains, environment setup, model instantiation

## Test Markers

**Available markers (from pytest.ini):**
```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    parameter_recovery: marks parameter recovery tests
```

**Usage:**
```python
# Mark slow test
@pytest.mark.slow
def test_complete_training():
    """Test full training pipeline."""
    pass

# Mark integration test
@pytest.mark.integration
def test_multi_task_training():
    """Test multi-task training."""
    pass

# Conditional skip for optional dependencies
@pytest.mark.skipif(not NEUROGYM_AVAILABLE, reason="neurogym not installed")
class TestNeurogymDawTwoStep:
    """Tests for Daw Two-Step task."""
    pass
```

## Common Patterns

**Async Testing:**
Not used in this codebase (synchronous PyTorch operations only).

**Error Testing:**
```python
# Test that exceptions are raised with correct messages
def test_invalid_parameter():
    """Test that invalid parameter raises ValueError."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        # Code that should raise
        function_with_validation(invalid_param)

# Example from actual codebase pattern (from src/nn4psych/training/configs.py):
# Test would verify:
# raise ValueError(f"Unknown parameter: {param_name}")
```

**Shape Testing Pattern (common throughout):**
```python
def test_forward_pass(self):
    """Test forward pass produces correct shapes."""
    model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

    batch_size = 1
    x = torch.randn(batch_size, 1, 9)  # (batch, seq_len, features)
    hx = torch.zeros(1, batch_size, 64)  # (num_layers, batch, hidden)

    actor_logits, critic_value, new_hx = model(x, hx)

    # Validate output shapes
    assert actor_logits.shape == (batch_size, 3)      # (batch, actions)
    assert critic_value.shape == (batch_size, 1)      # (batch, value)
    assert new_hx.shape == (1, batch_size, 64)        # (num_layers, batch, hidden)
```

**Device Testing Pattern:**
```python
def test_complete_trial(self, env, model, tiny_hidden_dim, device):
    """Test running a complete trial with model."""
    obs, done = env.reset()
    hx = model.get_initial_hidden(batch_size=1, device=device)

    # All tensors created on specified device
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

    # Forward pass on correct device
    actor_logits, _, hx = model(state_tensor, hx, task_id)
    action = actor_logits.argmax(dim=1).item()
```

**Multi-Task Testing Pattern (from test_task_compatibility.py):**
```python
class TestMultiTaskPIEOnly:
    """Tests for multi-task model with both PIE tasks."""

    @pytest.fixture
    def task_specs(self):
        """Create task specs for both PIE tasks."""
        return {
            "change-point": TaskSpec(...),
            "oddball": TaskSpec(...),
        }

    def test_forward_both_tasks(self, model, tiny_hidden_dim, device):
        """Test forward pass works for both tasks."""
        for task_id in ["change-point", "oddball"]:
            x = torch.randn(1, 1, 9, device=device)
            hx = model.get_initial_hidden(batch_size=1, device=device)

            actor_logits, critic_value, new_hx = model(x, hx, task_id)

            # Verify output shapes
            assert actor_logits.shape == (1, 3)
            assert critic_value.shape == (1, 1)
```

**Conditional Testing (NeuroGym Optional):**
```python
# At module level - skip entire class if dependency missing
@pytest.mark.skipif(not NEUROGYM_AVAILABLE, reason="neurogym not installed")
class TestNeurogymDawTwoStep:
    """Tests for Daw Two-Step task."""

    def test_env_dimensions(self, env):
        """Test that environment has expected dimensions."""
        assert env.obs_dim > 0
        assert env.action_dim > 0
```

---

*Testing analysis: 2026-01-28*
