# Developer Guide

**Last Updated:** 2025-11-19
**Target Audience:** Contributors and maintainers

---

## Overview

This guide helps developers understand the NN4Psych architecture and provides instructions for extending the project.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Extending the Project](#extending-the-project)
- [Development Workflow](#development-workflow)
- [Testing Strategy](#testing-strategy)
- [Common Development Tasks](#common-development-tasks)

---

## Architecture Overview

NN4Psych follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Scripts                            │
│  (scripts/training/, scripts/analysis/, scripts/fitting/)   │
└────────────────┬──────────────────┬────────────────────────┘
                 │                  │
                 ▼                  ▼
    ┌────────────────────┐  ┌──────────────────┐
    │   nn4psych Package │  │  envs/ Module    │
    │  (src/nn4psych/)   │  │  (standalone)    │
    │                    │  │                  │
    │  ├─ models/        │  │  PIE_CP_OB_v2   │
    │  ├─ training/      │  └──────────────────┘
    │  ├─ analysis/      │
    │  └─ utils/         │
    └────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │   Configuration    │
    │    (config.py)     │
    └────────────────────┘
```

### Design Principles

1. **Modularity**: Each component has a single, well-defined purpose
2. **Reusability**: Core components can be used independently
3. **Configurability**: Behavior controlled via config.py and YAML files
4. **Testability**: All components can be tested in isolation
5. **Documentation**: Every module includes comprehensive docstrings

---

## Project Structure

### src/ Layout

We use the **src/ layout** pattern (PEP 517/518):

**Benefits:**
- Prevents accidental imports from development directory
- Clear distinction between package code and project code
- Better for package distribution

**Structure:**
```
nn4psych/                     # Project root
├── src/                      # Source code
│   └── nn4psych/            # Actual package
│       ├── models/
│       ├── training/
│       ├── analysis/
│       └── utils/
├── envs/                     # Standalone (not in package)
├── scripts/                  # Executable scripts
└── tests/                    # Tests (not in package)
```

### Directory Purposes

| Directory | Purpose | In Package? |
|-----------|---------|-------------|
| `src/nn4psych/` | Core package code | ✅ Yes |
| `envs/` | Task environments | ❌ No (standalone) |
| `scripts/` | Executable scripts | ❌ No |
| `tests/` | Unit tests | ❌ No |
| `validation/` | Integration tests | ❌ No |
| `trained_models/` | Model weights | ❌ No (gitignored) |
| `data/` | Data files | ❌ No (gitignored) |
| `output/` | Analysis outputs | ❌ No (gitignored) |
| `figures/` | Plot outputs | ❌ No (gitignored) |

---

## Core Components

### 1. Models (`src/nn4psych/models/`)

**Purpose:** Neural network model definitions

**Key File:** `actor_critic.py`
- `ActorCritic` class: RNN-based actor-critic model
- Configurable via `ModelConfig` dataclass
- Supports both training and inference modes

**Example:**
```python
from nn4psych.models import ActorCritic

model = ActorCritic(
    input_dim=9,      # 6 obs + 2 context + 1 reward
    hidden_dim=64,    # RNN hidden units
    action_dim=3,     # left, right, confirm
    gain=1.5,         # Weight initialization
    noise=0.0,        # Hidden state noise
)
```

### 2. Environments (`envs/`)

**Purpose:** Task environment implementations

**Key File:** `pie_environment.py`
- `PIE_CP_OB_v2` class: Predictive inference environment
- Gym-style interface (reset, step, render)
- Supports change-point and oddball conditions

**Why separate from package?**
- Used by both RNN training AND Bayesian models
- Can be imported by external projects
- Standalone = clear dependencies

**Example:**
```python
from envs import PIE_CP_OB_v2

env = PIE_CP_OB_v2(
    condition="change-point",
    total_trials=200,
    max_time=300,
)
```

### 3. Training (`src/nn4psych/training/`)

**Purpose:** Training infrastructure

**Key File:** `configs.py`
- `ExperimentConfig`: Complete experiment configuration
- `ModelConfig`, `TaskConfig`, `TrainingConfig`: Component configs
- YAML serialization support

### 4. Analysis (`src/nn4psych/analysis/`)

**Purpose:** Behavioral analysis utilities

**Key Files:**
- `behavior.py`: Extract behavior from trained models
- `hyperparams.py`: Unified hyperparameter analysis

### 5. Utilities (`src/nn4psych/utils/`)

**Purpose:** Shared utility functions

**Key Files:**
- `io.py`: Model loading/saving
- `metrics.py`: Learning rate calculations, Bayesian utilities
- `plotting.py`: Visualization functions

### 6. Configuration (`config.py`)

**Purpose:** Centralized project configuration

**Contents:**
- All directory paths (as Path objects)
- Model/task/training parameters
- Hyperparameter sweep values
- Column naming conventions
- Versioning constants

**Why centralized?**
- Single source of truth
- Easy to update paths
- Prevents scattered magic numbers
- Supports reproducibility

---

## Extending the Project

### Adding a New Model

1. **Create model file** in `src/nn4psych/models/`
```python
# src/nn4psych/models/lstm_actor_critic.py
import torch.nn as nn

class LSTMActorCritic(nn.Module):
    """LSTM-based actor-critic model."""

    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # ... rest of model
```

2. **Add to package init** in `src/nn4psych/__init__.py`
```python
from nn4psych.models.actor_critic import ActorCritic
from nn4psych.models.lstm_actor_critic import LSTMActorCritic  # NEW

__all__ = ["ActorCritic", "LSTMActorCritic"]
```

3. **Add tests** in `tests/test_models.py`
```python
def test_lstm_actor_critic_forward():
    model = LSTMActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
    # ... test code
```

4. **Update documentation**
   - Add docstring to model class
   - Update `docs/API_REFERENCE.md`
   - Add example to README if user-facing

### Adding a New Environment

1. **Create environment** in `envs/`
```python
# envs/new_task.py
class NewTaskEnv:
    """New task environment."""

    def reset(self):
        # Reset logic
        return observation, done

    def step(self, action):
        # Step logic
        return observation, reward, done
```

2. **Add to envs init** in `envs/__init__.py`
```python
from envs.pie_environment import PIE_CP_OB_v2
from envs.new_task import NewTaskEnv  # NEW

__all__ = ["PIE_CP_OB_v2", "NewTaskEnv"]
```

3. **Add configuration support**
   - Add to `TASK_PARAMS` in `config.py`
   - Create `TaskConfig` in `training/configs.py`

### Adding a Pipeline Stage

1. **Create numbered script** in `scripts/data_pipeline/`
```python
# scripts/data_pipeline/04_new_analysis.py
"""
Stage 04: New Analysis Description
"""

from config import OUTPUT_DIR

def run_analysis():
    # Analysis code
    pass

if __name__ == "__main__":
    run_analysis()
```

2. **Update master runner** in `scripts/data_pipeline/00_run_full_pipeline.py`
```python
STAGES = {
    1: "01_extract_model_behavior.py",
    2: "02_compute_learning_metrics.py",
    3: "03_analyze_hyperparameter_sweeps.py",
    4: "04_new_analysis.py",  # NEW
}
```

3. **Update documentation** in `docs/ANALYSIS_PIPELINE.md`

### Adding Analysis Functions

1. **Add to appropriate module**:
   - Behavioral metrics → `src/nn4psych/utils/metrics.py`
   - Plotting functions → `src/nn4psych/utils/plotting.py`
   - Analysis workflows → `src/nn4psych/analysis/`

2. **Follow existing patterns**:
```python
def new_metric(data: np.ndarray, param: float = 1.0) -> float:
    """
    Calculate new metric.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    param : float, optional
        Parameter for calculation. Default is 1.0.

    Returns
    -------
    float
        Calculated metric value.

    Examples
    --------
    >>> result = new_metric(np.array([1, 2, 3]))
    """
    # Implementation
    pass
```

---

## Development Workflow

### Daily Development Cycle

1. **Pull latest changes**
```bash
git pull origin main
```

2. **Create feature branch**
```bash
git checkout -b feature/my-feature
```

3. **Make changes**
   - Edit code
   - Add tests
   - Update docs

4. **Run tests locally**
```bash
pytest
black --check src/
flake8 src/
```

5. **Commit and push**
```bash
git add .
git commit -m "Add feature: description"
git push origin feature/my-feature
```

6. **Create pull request**

### Before Committing

Always run:
```bash
# Format code
black src/ scripts/ tests/

# Run tests
pytest

# Check coverage
pytest --cov=nn4psych --cov-report=term-missing

# Lint
flake8 src/
```

---

## Testing Strategy

### Test Organization

```
tests/              # Unit tests
├── test_models.py
├── test_envs.py
└── test_utils.py

validation/         # Integration tests
├── test_parameter_recovery.py
└── test_full_pipeline.py
```

### Unit Tests (`tests/`)

- Test individual functions/classes in isolation
- Fast execution (<1s per test)
- No external dependencies
- Use mocks/fixtures for complex dependencies

**Example:**
```python
def test_actor_critic_forward_pass():
    """Test ActorCritic forward pass."""
    model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
    x = torch.randn(1, 1, 9)  # batch, seq, features
    h = model.reset_hidden(batch_size=1)

    actor_logits, value, h_new = model(x, h)

    assert actor_logits.shape == (1, 1, 3)
    assert value.shape == (1, 1, 1)
    assert h_new.shape == h.shape
```

### Integration Tests (`validation/`)

- Test complete workflows
- May be slower (mark with `@pytest.mark.slow`)
- Test interactions between components

**Example:**
```python
@pytest.mark.integration
def test_full_training_pipeline():
    """Test complete training workflow."""
    config = create_default_config()
    config.training.epochs = 2  # Minimal for testing

    model = ActorCritic.from_config(config.model)
    env = PIE_CP_OB_v2.from_config(config.task)

    # Run training
    # ... training loop

    # Verify model was trained
    assert model is not None
```

### Test Fixtures (`validation/conftest.py`)

Shared test setup:
```python
@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

@pytest.fixture
def sample_env():
    """Create a sample environment."""
    return PIE_CP_OB_v2(condition="change-point", total_trials=10)
```

---

## Common Development Tasks

### Running Specific Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_actor_critic_forward_pass

# Run tests matching pattern
pytest -k "actor_critic"

# Run with verbose output
pytest -v

# Skip slow tests
pytest -m "not slow"
```

### Code Coverage

```bash
# Generate coverage report
pytest --cov=nn4psych --cov-report=html

# Open report
open htmlcov/index.html  # Mac/Linux
start htmlcov\index.html  # Windows
```

### Building Documentation

```bash
# Future: Sphinx documentation
cd docs
make html
```

### Debugging

**Interactive debugging:**
```python
import pdb; pdb.set_trace()  # Set breakpoint
```

**VS Code:** Add to `launch.json`:
```json
{
    "name": "Python: Current File",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
}
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = some_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10
```

---

## Questions & Support

- **Documentation**: Check `docs/` directory first
- **Issues**: [GitHub Issues](https://github.com/adammanoogian/nn4psych/issues)
- **Contributing**: See `CONTRIBUTING.md`

---

## Additional Resources

- [Architecture Decisions](./ARCHITECTURE.md) (future)
- [API Reference](./API_REFERENCE.md) (future)
- [Troubleshooting Guide](./TROUBLESHOOTING.md) (future)
