# Coding Conventions

**Analysis Date:** 2026-01-28

## Naming Patterns

**Files:**
- Module files: `lowercase_with_underscores.py`
- Examples: `actor_critic.py`, `behavior.py`, `hyperparams.py`, `io.py`, `metrics.py`
- Test files: `test_*.py` (e.g., `test_models.py`, `test_task_compatibility.py`)
- Configuration files: `configs.py`

**Functions:**
- Snake case for all function names
- Descriptive names indicating purpose and return type
- Examples: `extract_behavior()`, `get_lrs()`, `save_model()`, `reset_hidden()`
- Verb-first for action functions: `extract_`, `get_`, `calculate_`, `filter_`

**Variables:**
- Snake case for local variables and attributes
- Examples: `prediction_error`, `hidden_dim`, `action_dim`, `learning_rate`
- Private attributes: Prefix with underscore (rare in this codebase)
- Constants: UPPERCASE_WITH_UNDERSCORES in config files
- Examples: `PROJECT_ROOT`, `MODEL_PARAMS`, `VALIDATION_SEEDS`

**Classes:**
- PascalCase for all class names
- Examples: `ActorCritic`, `MultiTaskActorCritic`, `TaskSpec`, `ExperimentConfig`, `ModelConfig`
- Descriptive names indicating purpose

**Types:**
- Full type hints in function signatures using `typing` module
- Examples: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`, `Optional[torch.device]`, `Dict[str, TaskSpec]`

## Code Style

**Formatting:**
- Tool: Black (version 23.12.1)
- Line length: 100 characters
- Configuration: `pyproject.toml` `[tool.black]` section
- Run via pre-commit hooks automatically

**Linting:**
- Tool: flake8 (version 7.0.0)
- Configuration: `.pre-commit-config.yaml`
- Max line length: 100 characters
- Ignored rules: E203 (whitespace before ':'), W503 (line break before binary operator)
- Use: `flake8 . --max-line-length=100 --extend-ignore=E203,W503`

**Import Sorting:**
- Tool: isort (version 5.13.2)
- Profile: black (compatible with Black)
- Line length: 100 characters
- Configuration: `.pre-commit-config.yaml`

## Import Organization

**Order:**
1. Standard library imports (e.g., `import sys`, `import os`, `from pathlib import Path`)
2. Third-party imports (e.g., `import numpy`, `import torch`, `import matplotlib`)
3. Local project imports (e.g., `from nn4psych.models import ActorCritic`, `from envs import PIE_CP_OB_v2`)

**Examples from codebase:**
```python
# From src/nn4psych/models/actor_critic.py
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import init

# From tests/test_task_compatibility.py
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from nn4psych.models.multitask_actor_critic import MultiTaskActorCritic, TaskSpec
from envs import PIE_CP_OB_v2
```

**Path Aliases:**
- Direct imports from package root: `from nn4psych.models import ActorCritic`
- Avoid relative imports; use absolute imports from package root
- Environment module imported as: `from envs import PIE_CP_OB_v2`

## Error Handling

**Patterns:**
- Explicit exception types: Catch specific exceptions, not generic `Exception`
- Example from `src/nn4psych/analysis/hyperparams.py`:
  ```python
  try:
      # Code that might fail
  except (ValueError, IndexError):
      # Handle expected errors
  except Exception as e:
      # Only catch all as last resort with context
  ```

- Raise with descriptive messages using f-strings
- Example from `src/nn4psych/training/configs.py`:
  ```python
  raise KeyError(f"Task '{task_id}' not found in registry. Available tasks: {list(TASK_REGISTRY.keys())}")
  raise ValueError(f"Unknown parameter: {param_name}")
  ```

- Use try-except blocks in data processing pipelines
- Silent failures only for optional features (neurogym integration)
- Example from `tests/test_task_compatibility.py`:
  ```python
  try:
      from envs import NEUROGYM_AVAILABLE, DawTwoStepWrapper
  except ImportError:
      NEUROGYM_AVAILABLE = False
  ```

**Error Categories:**
- `ValueError`: Invalid parameter values
- `KeyError`: Missing keys in registries/configs
- `ImportError`: Missing optional dependencies
- `IndexError`: Array indexing errors
- `TypeError`: Type mismatches (rely on type hints)

## Logging

**Framework:** Python's `print()` for user-facing messages; not using logging module

**Patterns:**
- Print for confirmation messages: `print(f'File saved: {filepath}')`
- Use f-strings for formatting
- Print progress information from tqdm progress bars in loops
- Example from `src/nn4psych/utils/io.py`:
  ```python
  print(f'Model saved: {filepath}')
  print(f'File saved: {filepath}')
  ```

- No structured logging; warnings go to stderr implicitly via exceptions

## Comments

**When to Comment:**
- Explain non-obvious algorithmic choices (e.g., weight initialization scaling)
- Document complex mathematical operations (e.g., learning rate calculation)
- Clarify task-specific configuration parameters
- Explain workarounds or temporary solutions
- Rarely needed in well-named functions

**JSDoc/TSDoc:**
- Use NumPy-style docstrings (Google-compatible variant)
- Mandatory for all public functions, classes, and methods
- Three-part structure: summary, Parameters, Returns

## Docstring Format

**Standard structure (NumPy style):**

```python
def function_name(param1: type, param2: type) -> ReturnType:
    """
    Brief one-line summary of what the function does.

    Extended description if needed, explaining the purpose,
    algorithm, or important notes about usage.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    ReturnType
        Description of return value.

    Examples
    --------
    >>> result = function_name(value1, value2)
    >>> result
    ExpectedOutput

    Notes
    -----
    Additional technical notes or caveats.
    """
```

**Examples from codebase:**
- `src/nn4psych/models/actor_critic.py` - Complete docstrings with Parameters, Returns, Examples
- `src/nn4psych/utils/metrics.py` - Docstrings with Notes sections explaining algorithm details
- `src/nn4psych/utils/io.py` - Examples section showing usage patterns

## Function Design

**Size:**
- Keep functions focused on single responsibility (typically 10-40 lines for core logic)
- Longer functions (50+ lines) should be broken into helper functions
- Examples: `extract_behavior()` is ~40 lines for main logic, `get_lrs_v2()` is ~50 lines

**Parameters:**
- Maximum 5-6 positional parameters; use beyond that requires config object
- Use type hints for all parameters
- Default values for optional parameters should be documented
- Example from `src/nn4psych/analysis/behavior.py`:
  ```python
  def extract_behavior(
      model: ActorCritic,
      env: PIE_CP_OB_v2,
      n_epochs: int = 100,
      n_trials: int = 200,
      reset_memory: bool = True,
      preset_memory: float = 0.0,
      device: Optional[torch.device] = None,
  ) -> Tuple:
  ```

**Return Values:**
- Single meaningful value or named tuple for multiple returns
- Use `Tuple[type1, type2, ...]` for ordered tuples
- Example: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` for (actor_logits, critic_value, new_hidden_state)
- Document return types in docstrings

## Module Design

**Exports:**
- Define `__all__` in `__init__.py` files for public API
- Example from `src/nn4psych/models/__init__.py`:
  ```python
  from nn4psych.models.actor_critic import ActorCritic
  __all__ = ['ActorCritic']
  ```

**Barrel Files:**
- Used in `src/nn4psych/__init__.py` to expose main classes
- Enables clean imports: `from nn4psych.models import ActorCritic`
- Reduces internal import complexity

**Module Organization:**
- One main class per module (e.g., `ActorCritic` in `actor_critic.py`)
- Related utilities in `utils/` submodule
- Config objects in `training/configs.py`
- Tests in separate `tests/` directory with mirrored structure

## Type Hints

**Usage:**
- Required on all function signatures
- Required on class attributes (in `__init__` or via dataclass fields)
- Optional on local variables (use when it aids readability)

**Examples:**
```python
# Function signature with full type hints
def extract_behavior(
    model: ActorCritic,
    env: PIE_CP_OB_v2,
    n_epochs: int = 100,
    device: Optional[torch.device] = None,
) -> Tuple:

# Class with type hints (via dataclass)
@dataclass
class ModelConfig:
    input_dim: int = 9
    hidden_dim: int = 64
    action_dim: int = 3
    gain: float = 1.5
```

## Special Patterns

**Configuration Objects:**
- Use dataclasses from `dataclasses` module
- Located in `src/nn4psych/training/configs.py`
- Examples: `ModelConfig`, `TaskConfig`, `TrainingConfig`, `ExperimentConfig`
- Enables type-safe configuration with defaults

**Context Managers:**
- Use `torch.no_grad()` for inference: `with torch.no_grad(): ...`
- Pattern in `src/nn4psych/analysis/behavior.py` for evaluation mode

**Model Device Handling:**
- Get device from model parameters: `device = next(model.parameters()).device`
- Allows device-agnostic code (CPU/GPU)
- Pattern in `src/nn4psych/models/actor_critic.py` and `src/nn4psych/analysis/behavior.py`

---

*Convention analysis: 2026-01-28*
