# Archived Legacy Code

This directory contains legacy implementations and scripts that have been superseded by the new modular `nn4psych` package.

## Structure

```
archive/
├── legacy_scripts/          # Old training and utility scripts
│   ├── pretrain_rnn_with_heli_v1-4.py  # Superseded by v5
│   ├── train_rnn_without_heli_server.py
│   ├── utils.py              # Now in nn4psych/utils/
│   ├── utils_funcs.py        # Now in nn4psych/utils/
│   ├── tasks.py              # Now in nn4psych/envs/
│   └── get_behavior.py       # Now in nn4psych/analysis/behavior.py
├── legacy_analysis/          # Old analysis scripts
│   ├── analyze_hyperparams_*.py  # 5 duplicates → 1 unified script
│   ├── analysis.py           # Wang et al. incomplete
│   ├── analyze_compiled.py   # Simple utility
│   └── compile.py            # Data compilation
├── v0/                       # Early JAX implementations
├── v1/                       # JAX actor-critic variants
├── v2/                       # PyTorch variants (simpler models)
├── toy/                      # Simplified prototype
├── normative_model/          # Standalone Bayesian models
├── readme.txt                # Old readme (replaced by README.md)
└── code_figures_behaviour    # Old figure documentation
```

## Migration Guide

### Old → New Locations

| Old File | New Location | Notes |
|----------|--------------|-------|
| `utils_funcs.py` → `ActorCritic` | `nn4psych/models/actor_critic.py` | Consolidated from 8 copies |
| `utils_funcs.py` → `get_lrs_v2` | `nn4psych/utils/metrics.py` | No more duplicates |
| `utils.py` → functions | `nn4psych/utils/metrics.py` | State extraction utils |
| `tasks.py` → `PIE_CP_OB_v2` | `nn4psych/envs/predictive_inference.py` | Canonical environment |
| `bayesian_models.py` | `nn4psych/models/bayesian/` | With proper imports |
| `analyze_hyperparams_*.py` | `nn4psych/analysis/hyperparams.py` | Single unified class |
| `get_behavior.py` | `nn4psych/analysis/behavior.py` | Modular extraction |
| `pretrain_rnn_with_heli_v5.py` | `scripts/training/train_rnn_canonical.py` | Canonical training |

### Import Changes

```python
# OLD
from utils_funcs import ActorCritic, get_lrs_v2, saveload
from tasks import PIE_CP_OB_v2
from utils import extract_states

# NEW
from nn4psych.models import ActorCritic
from nn4psych.envs import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2, extract_states
from nn4psych.utils.io import saveload
```

## Contents Detail

### legacy_scripts/
- **pretrain_rnn_with_heli_v1-4.py** - Earlier versions of training script
- **train_rnn_without_heli_server.py** - Alternative training approach
- **utils.py** - State extraction and Bayesian utility functions
- **utils_funcs.py** - ActorCritic model and learning rate functions
- **tasks.py** - Environment definitions (PIE_CP_OB, PIE_CP_OB_v2)
- **get_behavior.py** - Behavior extraction from trained models

### legacy_analysis/
- **analyze_hyperparams_*.py** - 5 nearly identical scripts (now unified)
- **analysis.py** - Wang et al. 2018 reproduction (incomplete)
- **analyze_compiled.py** - Data compilation utility
- **compile.py** - State data compilation

### v0/ - Early JAX Implementations (Outdated)
- `context_bandits.py` - JAX-based actor-critic for context bandits
- `rnn_helicopter.py` - JAX RNN implementation
- `play_helicopter_discrete.py` - Pygame visualization

### v1/ - JAX Actor-Critic Variants
- `main.py` - Two-arm bandit without context
- `main_healthy.py` - Change-point task
- `main_healthy_ff.py` - Feed-forward variant

### v2/ - PyTorch Variants
- `pt_rnn_basic.py` - Basic PyTorch RNN
- `pt_rnn_context.py` - With context input
- `pt_rnn_nocontext.py` - Without context
- `pt_rnn_context_td.py` - With temporal difference

### toy/ - Simplified Prototype
- `model.py` - Reservoir-style actor-critic (NumPy)
- `main.py` - Minimal training loop
- `task.py` - Simplified task environment

### normative_model/
- `simple_normative_model.py` - Bayesian change-point model
- `tasks.py` - Duplicate of root tasks.py (name collision risk)

## Why Archived?

1. **Code Duplication**: ActorCritic was duplicated 8 times across files
2. **Inconsistent APIs**: Different versions had slightly different interfaces
3. **Hard-coded Parameters**: Magic numbers scattered throughout
4. **No Package Structure**: Could not be imported as a proper Python package
5. **Mixed Frameworks**: JAX, PyTorch, and NumPy implementations mixed together

## When to Reference

- Understanding historical evolution of the codebase
- Checking original implementation details
- Comparing JAX vs PyTorch approaches
- Debugging compatibility issues with old model weights

## Not Recommended For

- New experiments (use `nn4psych` package instead)
- Production use
- Active development
- New model training

## Notes

- Files in `legacy_scripts/` were the active codebase before reorganization
- Files in `v0/`, `v1/`, `v2/`, `toy/` were already deprecated
- Most functionality is now available through the `nn4psych` package with improved APIs
- The canonical training script is now at `scripts/training/train_rnn_canonical.py`
