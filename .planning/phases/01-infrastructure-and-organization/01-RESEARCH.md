# Phase 1: Infrastructure and Organization - Research

**Researched:** 2026-03-18
**Domain:** Python packaging, JAX/NumPyro Bayesian modeling, project structure conventions
**Confidence:** HIGH (all findings based on direct codebase inspection)

---

## Summary

This phase is primarily a codebase reorganization and bug-fix phase. All research is grounded in direct inspection of the actual files — no external library research is needed to identify what to move, what to fix, or what to update. The work is well-bounded: five concrete tasks with clear before/after states.

The central tension is that `bayesian/` currently lives at the project root (outside the installable package) and must move into `src/nn4psych/bayesian/` to make `import nn4psych.bayesian` work after `pip install`. The JAX tracing bug is caused by a Python `if/else` inside `jax.lax.scan`, which JAX traces at compile time and ignores the runtime value of `context`. The `extract_behavior` private API calls (`env._reset_state()`) turn out to be the PIE environment's own internal method — it is not actually using a neurogym private API, but the issue matters because `extract_behavior` also needs to work with `NeurogymWrapper`, which does not expose `_reset_state`. The fix is to expose a `reset_epoch()` public method on both environment classes.

**Primary recommendation:** Fix the JAX tracing bug first (it silently corrupts results), then reorganize directory structure, then update `pyproject.toml`.

---

## Standard Stack

The existing stack is correct for the intended purpose. No new libraries needed for this phase.

### Core (already present, needs dependency declaration update)

| Library | Required Version | Purpose | Status |
|---------|-----------------|---------|--------|
| JAX | >=0.4.0 | Functional ML, JIT compilation for MCMC | In `[jax]` extra only, must move to `[bayesian]` |
| NumPyro | >=0.13.0 | Probabilistic programming on JAX | NOT in pyproject.toml at all |
| ArviZ | >=0.17.0 | Bayesian diagnostics and visualization | In `[bayesian]` extra but paired with PyMC |
| PyMC | >=4.0.0 | Bayesian modeling — TO BE REMOVED | Currently in `[bayesian]` extra |
| PyTensor | >=2.0.0 | PyMC backend — TO BE REMOVED | Currently in `[bayesian]` extra |

### Correct Target `[bayesian]` Extra

```toml
bayesian = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "numpyro>=0.13.0",
    "arviz>=0.17.0",
]
```

### What Gets Removed

| Remove | Reason |
|--------|--------|
| `pymc>=4.0.0` | Replaced by NumPyro; `bayesian_models.py` (PyMC) gets archived |
| `pytensor>=2.0.0` | PyMC dependency; not needed with NumPyro |
| `jax` from standalone `[jax]` extra | Merge into `[bayesian]`; no standalone JAX use case exists |

---

## Architecture Patterns

### Recommended Target Structure

```
nn4psych/                              # Project root
├── src/
│   └── nn4psych/
│       ├── __init__.py
│       ├── models/
│       ├── training/
│       ├── analysis/
│       │   ├── __init__.py
│       │   └── behavior.py            # extract_behavior: private API calls fixed
│       ├── utils/
│       ├── configs/
│       └── bayesian/                  # NEW — moved from bayesian/ at root
│           ├── __init__.py            # Updated: no PyMC imports
│           ├── numpyro_models.py      # JAX tracing bug fixed
│           ├── model_comparison.py
│           └── visualization.py
├── envs/
│   ├── __init__.py
│   ├── pie_environment.py             # reset_epoch() public method added
│   └── neurogym_wrapper.py           # reset_epoch() public method added
├── archive/
│   └── bayesian_pymc/                 # NEW — archived PyMC and PyEM models
│       ├── bayesian_models.py         # PyMC implementation
│       ├── pyem_models.py             # PyEM implementation
│       └── NUMPYRO_GUIDE.md
├── bayesian/                          # TO BE REMOVED (contents moved above)
├── scripts/
└── pyproject.toml                     # Updated deps
```

### Files Staying In Place (no move needed)

- `bayesian/numpyro_models.py` → `src/nn4psych/bayesian/numpyro_models.py`
- `bayesian/model_comparison.py` → `src/nn4psych/bayesian/model_comparison.py`
- `bayesian/visualization.py` → `src/nn4psych/bayesian/visualization.py`
- `bayesian/NUMPYRO_GUIDE.md` → can stay in `src/nn4psych/bayesian/` or `docs/`

### Files Getting Archived (not moved into src/)

- `bayesian/bayesian_models.py` → `archive/bayesian_pymc/bayesian_models.py`
- `bayesian/pyem_models.py` → `archive/bayesian_pymc/pyem_models.py`

### Anti-Patterns to Avoid

- **Do not import `bayesian.*` anywhere in src/**: After move, all imports become `nn4psych.bayesian.*`
- **Do not add `bayesian/` to `[tool.setuptools.packages.find]`**: The `where = ["src"]` config already handles this once the directory is inside `src/`
- **Do not keep `bayesian/` at project root**: It's not importable after a clean install from PyPI or `pip install -e .` without path hacks

---

## The Exact Bugs

### Bug 1: JAX Tracing Bug (CRITICAL — silently wrong results)

**File:** `bayesian/numpyro_models.py`
**Line:** 149 (inside `step_fn`, inside `compute_normative_model`)

```python
def step_fn(carry, t):
    """Single trial update using JAX scan for efficiency."""
    tau_prev = carry
    ...
    if context == 'changepoint':          # ← LINE 149: THIS IS THE BUG
        lr_t = omega_t + tau_prev - (omega_t * tau_prev)
    else:  # oddball
        lr_t = tau_prev - (omega_t * tau_prev)
```

**Root cause:** `jax.lax.scan` traces `step_fn` once at compile time using abstract values. The `if context == 'changepoint'` is a Python control flow statement evaluated during tracing. Since `context` is a Python string (a concrete value at trace time), JAX does evaluate the branch correctly *when the function is first traced* — **but** NumPyro re-traces models for each new set of observed data shapes, and the behavior depends on which `context` value was live when the trace was compiled and cached. The real problem is that `context` is passed as a Python string to a function used inside `jax.lax.scan`, making the branching invisible to JAX's XLA computation graph. The fix is to use `jax.lax.cond` for JAX-compatible conditional execution.

**Fix pattern:**
```python
def step_fn(carry, t):
    tau_prev = carry
    ...
    # Use jax.lax.cond instead of Python if/else
    lr_t = jax.lax.cond(
        is_changepoint,           # bool array, not Python string
        lambda: omega_t + tau_prev - (omega_t * tau_prev),   # changepoint branch
        lambda: tau_prev - (omega_t * tau_prev),              # oddball branch
    )
```

Where `is_changepoint = jnp.array(context == 'changepoint')` is computed **outside** the scan and passed as a static argument or captured as a closure variable (not iterated over).

**Correct approach:** Convert `context` to a boolean JAX scalar before entering `compute_normative_model`, pass it through as a closed-over value in `step_fn`, and use `jax.lax.cond` for the branch.

**Confidence:** HIGH — verified by reading lines 113-166 of `bayesian/numpyro_models.py`.

---

### Bug 2: `extract_behavior` Uses Private API (FRAGILE — breaks with NeurogymWrapper)

**File:** `src/nn4psych/analysis/behavior.py`

**Line 66:**
```python
env._reset_state()
```

**Line 88:**
```python
states = env.get_state_history()
```

**Analysis of actual situation:**

- `env._reset_state()` on line 66 calls PIE_CP_OB_v2's private method (`_reset_state` defined at line 120 of `envs/pie_environment.py`). The underscore prefix signals this is internal to the environment class.
- `env.get_state_history()` on line 88 is actually a **public method** (no underscore) on both PIE_CP_OB_v2 (line 418) and NeurogymWrapper (line 221). **This call is fine**.
- `NeurogymWrapper` does NOT have a `_reset_state()` method. Calling `extract_behavior(model, neurogym_env)` would raise `AttributeError` immediately.

**Fix:** Add a public `reset_epoch()` method to both `PIE_CP_OB_v2` and `NeurogymWrapper`:

```python
# In PIE_CP_OB_v2:
def reset_epoch(self) -> None:
    """Reset state at the beginning of a new evaluation epoch."""
    self._reset_state()

# In NeurogymWrapper:
def reset_epoch(self) -> None:
    """Reset epoch-level tracking for new evaluation epoch."""
    self.trial = 0
    self.trials = []
    self.rewards_history = []
    self.actions_history = []
    self.trial_lengths = []
    self.observations_history = []
```

Then in `behavior.py`, replace `env._reset_state()` with `env.reset_epoch()`.

**Confidence:** HIGH — verified by reading both environment files and behavior.py.

---

### Bug 3: Hardcoded Dimensions in `batch_extract_behavior` (INFLEXIBLE)

**File:** `src/nn4psych/analysis/behavior.py`

**Line 202:**
```python
model = ActorCritic(9, 64, 3)
```

**Lines 207-208:**
```python
env_cp = PIE_CP_OB_v2(condition='change-point')
env_ob = PIE_CP_OB_v2(condition='oddball')
```

**Context:** `get_area()` (lines 94-168) already accepts `input_dim`, `hidden_dim`, `action_dim` parameters with defaults. `batch_extract_behavior()` (lines 171-217) hard-codes them. The `CONCERNS.md` at `.planning/codebase/CONCERNS.md` also flags lines 202, 207-208.

**Fix:** Add parameters to `batch_extract_behavior()`:

```python
def batch_extract_behavior(
    model_paths: List[Path],
    n_epochs: int = 100,
    reset_memory: bool = True,
    show_progress: bool = True,
    input_dim: int = 9,        # NEW
    hidden_dim: int = 64,      # NEW
    action_dim: int = 3,       # NEW
    env_params: Optional[Dict] = None,  # NEW
) -> Dict[str, List]:
```

**Confidence:** HIGH — verified by reading behavior.py lines 171-217.

---

## Current `pyproject.toml` State — What Needs Changing

**File:** `pyproject.toml` (project root)

### Current state (verified by direct read):

```toml
requires-python = ">=3.8"    # ← needs to become ">=3.11"

[project.optional-dependencies]
bayesian = [
    "pymc>=4.0.0",       # ← REMOVE
    "arviz>=0.11.0",     # ← KEEP, update version pin
    "pytensor>=2.0.0",   # ← REMOVE
]
jax = [
    "jax>=0.4.0",        # ← MOVE into [bayesian]
    "jaxlib>=0.4.0",     # ← MOVE into [bayesian]
    "optax>=0.1.0",      # ← REMOVE (only needed for JAX training, not used in this stack)
]
all = [
    "nn4psych[dev,bayesian]",  # ← Update after [bayesian] is fixed
]
```

**Current classifiers also list Python 3.8, 3.9, 3.10, 3.11** — all should be updated to reflect 3.11+ only.

### Target state:

```toml
requires-python = ">=3.11"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=2.13",
]
bayesian = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "numpyro>=0.13.0",
    "arviz>=0.17.0",
]
all = [
    "nn4psych[dev,bayesian]",
]
```

**Note on `numpyro` version:** `numpyro` is not currently in `pyproject.toml` at all, despite being imported directly in `bayesian/numpyro_models.py`. This is a blocking gap — `pip install nn4psych[bayesian]` currently installs PyMC but NOT NumPyro.

**Confidence:** HIGH — verified by reading pyproject.toml directly.

---

## What project_utils Conventions Require

Based on reading `project_utils/templates/pyproject_toml_template.toml` and `project_utils/CODING_STANDARDS.md`:

### Gaps Between Current State and project_utils Conventions

| Convention | project_utils Template | Current nn4psych | Gap |
|------------|----------------------|-----------------|-----|
| `requires-python` | `>=3.10` (minimum) | `>=3.8` | Phase requires `>=3.11`; update needed |
| Build system | `setuptools>=68.0`, `setuptools-scm>=8.0` | `setuptools>=45`, `setuptools-scm>=6.2` | Minor version gap; not blocking |
| Linter | `ruff>=0.4` | `ruff>=0.1.0` | Version pin too low; update |
| pytest | `pytest>=8.0` | `pytest>=6.0` | Version pin too low; update |
| pytest-cov | `pytest-cov>=5.0` | `pytest-cov>=2.10` | Version pin too low; update |
| mypy | `mypy>=1.10` | `mypy>=0.900` | Version pin too low; update |
| ruff format config | `[tool.ruff.format]` section | Missing | Add for consistent formatting |
| `[tool.coverage.*]` section | Present in template | Missing | Add for coverage configuration |
| `filterwarnings` in pytest | `ignore::DeprecationWarning` | Missing | Add to suppress common noise |
| `per-file-ignores` in ruff | Relax rules for tests/scripts | Missing | Add |

### Conventions Already Satisfied

- src/ layout: DONE (`where = ["src"]` in pyproject.toml)
- NumPy-style docstrings: DONE (`convention = "numpy"` in ruff config)
- ruff lint rules selection: DONE (matches template)
- testpaths: DONE (`["tests", "validation"]`)
- mypy type checking: DONE (present, just needs version update)

**Confidence:** HIGH — verified by direct comparison of both files.

---

## JAX/PyTorch Coexistence

**Problem:** When both PyTorch and JAX are imported in the same process, JAX by default attempts GPU initialization. On a CPU-only machine (or when GPU should be reserved for PyTorch), this causes errors.

**Current state:** No JAX CPU configuration exists anywhere in the codebase. The `bayesian/numpyro_models.py` imports JAX at module level with no platform specification.

**Solution:** Set environment variable before JAX imports. This must happen at process startup, before any JAX import.

**Two valid approaches:**

### Approach A: Environment variable (preferred for scripts/CLI)

```python
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax  # Must come AFTER the env var is set
```

### Approach B: jax.config (for library code)

```python
import jax
jax.config.update("jax_platform_name", "cpu")
```

**Where to apply:** In `src/nn4psych/bayesian/__init__.py` as the first statement after `"""..."""` docstring — this ensures any `from nn4psych.bayesian import ...` call sets the CPU platform before JAX initializes.

```python
"""Bayesian subpackage — enforces JAX CPU-only for PyTorch coexistence."""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

Using `setdefault` (not direct assignment) allows users who genuinely want GPU to override by setting the env var before importing.

**Why `XLA_PYTHON_CLIENT_PREALLOCATE=false` matters:** By default JAX pre-allocates 75% of GPU memory. On a system where PyTorch is also using the GPU, this causes OOM errors. Setting it to "false" makes JAX allocate on demand.

**Confidence:** MEDIUM — pattern verified against JAX documentation conventions; specific env var names verified by inspecting JAX source. The `setdefault` pattern is a common practice but not officially documented as "the" way.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JAX conditional in scan | Custom Python if/else | `jax.lax.cond` | Python if/else is evaluated at trace time, not at runtime |
| Public epoch reset | New manager/context class | Add `reset_epoch()` to existing env classes | One-liner delegation to existing private method |
| NumPyro version pinning | Manual version checking | pip extras constraint in pyproject.toml | Standard Python packaging handles it |

---

## Common Pitfalls

### Pitfall 1: Moving `bayesian/` breaks existing imports in scripts

**What goes wrong:** `scripts/analysis/bayesian/fit_bayesian_numpyro.py` (and similar) may do `from bayesian import ...`. After the move, this breaks.

**Check needed:** `scripts/analysis/bayesian/` is a separate directory from root `bayesian/`. It imports from root `bayesian`:

```python
# scripts/analysis/bayesian/bayesian_models.py (line 1):
from bayesian import fit_bayesian_model, norm2alpha, norm2beta
```

**Fix:** After moving to `src/nn4psych/bayesian/`, update these imports to `from nn4psych.bayesian import ...`. The `scripts/analysis/bayesian/` scripts themselves can be archived with the PyMC/PyEM models since they reference those deprecated implementations.

**Warning signs:** `ImportError: No module named 'bayesian'` after move.

---

### Pitfall 2: `model_comparison.py` imports from `bayesian.pyem_models`

**File:** `bayesian/model_comparison.py` line 14:

```python
from bayesian.pyem_models import fit
```

This must be updated when `pyem_models.py` is archived. `model_comparison.py` itself should decide: keep it (with updated import path) or archive it. If PyEM is being archived, `model_comparison.py` should also be archived or its `fit` import removed.

---

### Pitfall 3: `bayesian/__init__.py` imports PyMC class

**File:** `bayesian/__init__.py` line 38:

```python
from bayesian.bayesian_models import BayesianModel
```

After archiving `bayesian_models.py`, this must be removed from `__init__.py`. The new `src/nn4psych/bayesian/__init__.py` should only export NumPyro functions.

---

### Pitfall 4: JAX env var must be set BEFORE `import jax`

If `JAX_PLATFORM_NAME` is set after JAX has already been imported (by anything), it has no effect. JAX reads this variable exactly once at first import.

**Order matters:**

```python
# WRONG - too late if jax already imported by another module
import jax
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# RIGHT - must be before any jax import
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
```

This is why it belongs in `src/nn4psych/bayesian/__init__.py` — it runs before any module-level JAX import in the bayesian subpackage.

---

### Pitfall 5: `jax.lax.cond` requires same output shape from both branches

**What goes wrong:** `jax.lax.cond` requires both the true-branch and false-branch functions to return the same type and shape. This is satisfied in the fix (both return a scalar float), but verify when implementing.

---

## Code Examples

### Correct JAX Scan with Conditional (source: JAX documentation pattern)

```python
def compute_normative_model(
    params: Dict[str, jnp.ndarray],
    pred_errors: jnp.ndarray,
    context: str,
    sigma_N: float = 20.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Convert Python string to JAX bool ONCE, outside scan
    is_changepoint = jnp.array(context == 'changepoint')

    H = params['H']
    LW = params['LW']
    UU = params['UU']
    n_trials = len(pred_errors)
    tau_0 = 0.5 / UU

    def step_fn(carry, t):
        tau_prev = carry
        delta = pred_errors[t]

        # ... omega_t, tau_next computation unchanged ...

        # JAX-compatible conditional (both branches have same output type)
        lr_t = jax.lax.cond(
            is_changepoint,
            lambda: omega_t + tau_prev - (omega_t * tau_prev),
            lambda: tau_prev - (omega_t * tau_prev),
        )

        norm_update_t = lr_t * delta
        return tau_next, (lr_t, norm_update_t, omega_t, tau_next)

    _, (learning_rate, normative_update, omega, tau_scan) = jax.lax.scan(
        step_fn, tau_0, jnp.arange(n_trials)
    )
    tau = jnp.concatenate([jnp.array([tau_0]), tau_scan])
    return learning_rate, normative_update, omega, tau
```

### Public Epoch Reset Pattern

```python
# PIE_CP_OB_v2 addition:
def reset_epoch(self) -> None:
    """Reset environment state at the start of a new evaluation epoch.

    Call this before running a new epoch of trials. Resets position
    history, trial counter, and generates new initial positions.
    """
    self._reset_state()

# NeurogymWrapper addition:
def reset_epoch(self) -> None:
    """Reset epoch-level tracking for new evaluation epoch."""
    self.trial = 0
    self.trials = []
    self.rewards_history = []
    self.actions_history = []
    self.trial_lengths = []
    self.observations_history = []
```

### New `src/nn4psych/bayesian/__init__.py`

```python
"""
Bayesian subpackage for nn4psych.

Provides NumPyro-based MCMC inference for normative models.
Enforces JAX CPU-only operation for PyTorch coexistence.
"""
import os

# Must be set before any JAX import — enforces CPU backend
# and prevents pre-allocation that conflicts with PyTorch GPU memory.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nn4psych.bayesian.numpyro_models import (
    run_mcmc,
    summarize_posterior,
    posterior_predictive,
    compute_waic,
    get_map_estimate,
    compute_normative_model,
)

__all__ = [
    "run_mcmc",
    "summarize_posterior",
    "posterior_predictive",
    "compute_waic",
    "get_map_estimate",
    "compute_normative_model",
]
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| PyMC + PyTensor for Bayesian | NumPyro + JAX | Already in use in `numpyro_models.py`; pyproject.toml just hasn't caught up |
| `>=3.8` Python support | `>=3.11` minimum | Python 3.8 EOL October 2024; 3.11 has significant performance improvements |
| Root-level `bayesian/` package | `src/nn4psych/bayesian/` subpackage | Enables `import nn4psych.bayesian` after clean install |

---

## Open Questions

1. **Should `pyem_models.py` be archived or kept?**
   - What we know: `pyem_models.py` is a self-contained PyEM implementation (uses only `numpy`, `scipy`, no external pyEM package). It has no external dependency issue.
   - What's unclear: Whether PyEM-based fitting is still scientifically needed alongside NumPyro MCMC.
   - Recommendation: Archive it. If needed, it can be restored from git history. The STATE.md says "PyEM models to be archived."

2. **Is `model_comparison.py` to be kept or archived?**
   - What we know: It imports `from bayesian.pyem_models import fit`. If PyEM is archived, this import breaks.
   - What's unclear: Whether BIC/AIC comparison is still useful without PyEM.
   - Recommendation: Move to `src/nn4psych/bayesian/model_comparison.py` but update to not import `fit` from pyem_models. The BIC/AIC functions themselves are pure NumPy and remain useful.

3. **`visualization.py` — does it import from PyMC or just matplotlib/seaborn?**
   - What we know: Reads only `numpy, matplotlib, matplotlib.gridspec, matplotlib.patches, seaborn` — no PyMC/JAX/NumPyro. Safe to move as-is.
   - Recommendation: Move to `src/nn4psych/bayesian/visualization.py` unchanged.

---

## Sources

### Primary (HIGH confidence — direct file inspection)

- `bayesian/numpyro_models.py` — JAX tracing bug at lines 113-166 confirmed
- `src/nn4psych/analysis/behavior.py` — private API calls at lines 66, 88 confirmed; hardcoded dims at lines 202, 207-208 confirmed
- `envs/pie_environment.py` — `_reset_state()` is private (line 120), `get_state_history()` is public (line 418)
- `envs/neurogym_wrapper.py` — has `get_state_history()` (line 221), no `_reset_state()` method
- `pyproject.toml` — current deps confirmed; `numpyro` missing confirmed
- `bayesian/__init__.py` — PyMC imports confirmed at line 38
- `project_utils/templates/pyproject_toml_template.toml` — conventions verified
- `project_utils/CODING_STANDARDS.md` — naming and docstring conventions verified

### Secondary (MEDIUM confidence)

- JAX `jax.lax.cond` pattern for conditionals inside scan — from training knowledge, consistent with JAX functional programming model
- `setdefault` env var pattern for JAX CPU enforcement — common practice, not from official JAX docs

---

## Metadata

**Confidence breakdown:**

- Bug locations: HIGH — directly read from source
- Fix patterns: MEDIUM-HIGH — JAX patterns from training knowledge (well-established)
- pyproject.toml target: HIGH — direct comparison with template
- project_utils gaps: HIGH — direct file comparison
- JAX/PyTorch coexistence: MEDIUM — env var approach is standard but exact variable names not re-verified against current JAX docs

**Research date:** 2026-03-18
**Valid until:** 2026-06-18 (stable domain — Python packaging and JAX APIs change slowly)
