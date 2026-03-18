# Architecture Patterns

**Domain:** Computational neuroscience RNN analysis pipeline
**Researched:** 2026-03-18
**Milestone context:** Subsequent — adding latent circuit inference, NumPyro Bayesian fitting, and new tasks to existing PyTorch RNN-RL codebase

---

## Recommended Architecture

The pipeline is a five-stage linear DAG with a hard runtime boundary between PyTorch
(training and hidden-state extraction) and JAX (Bayesian inference and circuit fitting).
All data crossing that boundary passes as NumPy arrays stored in `data/processed/`.

```
┌───────────────────────────────────────────────────────────────────┐
│  Stage 1 — Training (PyTorch)                                     │
│  envs/  →  src/nn4psych/models/  →  scripts/training/            │
│  Outputs: trained_models/  +  ExperimentConfig YAML              │
└────────────────────────┬──────────────────────────────────────────┘
                         │  checkpoint (.pt)
┌────────────────────────▼──────────────────────────────────────────┐
│  Stage 2 — Behavioral Extraction (PyTorch)                        │
│  src/nn4psych/analysis/behavior.py                                │
│  Outputs: data/processed/rnn_behav/  (hidden states, actions)    │
└────────────────────────┬──────────────────────────────────────────┘
                         │  NumPy .npy  (bucket, bag, hx arrays)
┌────────────────────────▼──────────────────────────────────────────┐
│  Stage 3a — Latent Circuit Inference (NumPy / SciPy)              │
│  src/nn4psych/analysis/circuit_inference.py  [NEW]                │
│  Fits Q, w_rec, w_in, w_out from hidden states                   │
│  Outputs: data/processed/circuit_params/  (.npy)                 │
│                                                                   │
│  Stage 3b — Fixed Point Analysis (NumPy / SciPy)                 │
│  src/nn4psych/analysis/fixed_points.py  [NEW — promote script]   │
│  Finds attractors, Jacobian eigenvalues, PCA projections         │
│  Outputs: data/processed/fixed_points/  +  figures/dynamical/   │
└────────────────────────┬──────────────────────────────────────────┘
                         │  NumPy arrays (behavioral sequences)
┌────────────────────────▼──────────────────────────────────────────┐
│  Stage 4 — Bayesian Model Fitting (JAX / NumPyro)                 │
│  src/nn4psych/bayesian/  [NEW — consolidate from bayesian/]       │
│  numpyro_models.py, model_comparison.py                          │
│  Inputs: bucket_positions, bag_positions as np.ndarray           │
│  Outputs: data/processed/bayesian_fits/  (posterior .npy, CSV)   │
└────────────────────────┬──────────────────────────────────────────┘
                         │  fitted params  +  posterior samples
┌────────────────────────▼──────────────────────────────────────────┐
│  Stage 5 — Behavioral Comparison (NumPy / Pandas / Matplotlib)   │
│  src/nn4psych/analysis/comparison.py  [NEW]                      │
│  RNN vs human (Nassar 2021) on learning rate, prediction error   │
│  Outputs: output/behavioral_summary/  +  figures/comparison/    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component Boundaries

### Existing Components (do not restructure)

| Component | Location | Responsibility | Communicates With |
|-----------|----------|----------------|-------------------|
| Environments | `envs/` | Gym-compatible task loops (PIE_CP_OB_v2, NeuroGym wrappers) | Training scripts, behavior extraction |
| Models | `src/nn4psych/models/` | ActorCritic, MultiTaskActorCritic forward pass | Training scripts, behavior extraction |
| Training configs | `src/nn4psych/training/configs.py` | Dataclass config system, YAML serialization | Training scripts, io utilities |
| Behavior extraction | `src/nn4psych/analysis/behavior.py` | Run trained model, collect hidden states + actions | Models, environments, metrics |
| Metrics | `src/nn4psych/utils/metrics.py` | get_lrs_v2, prediction errors | Analysis scripts, hyperparams |
| IO utilities | `src/nn4psych/utils/io.py` | save_model, load_model | All scripts |
| Legacy Bayesian | `bayesian/` | PyEM + NumPyro models (top-level, outside package) | Fitting scripts only |

### New Components

| Component | Proposed Location | Responsibility | Communicates With |
|-----------|-------------------|----------------|-------------------|
| Latent circuit inference | `src/nn4psych/analysis/circuit_inference.py` | Fit Q, w_rec, w_in, w_out from hidden state sequences via gradient descent or least squares | Behavior extraction (inputs hidden states), fixed point analysis (outputs weight matrices) |
| Fixed point analysis | `src/nn4psych/analysis/fixed_points.py` | Find fixed points, Jacobian eigenvalue analysis, PCA trajectories — promoted from script to reusable module | Circuit inference (receives w_rec), training scripts, pipeline scripts |
| NumPyro Bayesian | `src/nn4psych/bayesian/` (new subpackage) | Full posterior MCMC fitting, WAIC, posterior predictive checks — consolidates `bayesian/numpyro_models.py` into package | Behavior extraction outputs (np.ndarray), comparison module |
| Context-dependent task env | `envs/context_decision.py` | New gym-compatible task for context-dependent decision-making | Training scripts, behavior extraction |
| Behavioral comparison | `src/nn4psych/analysis/comparison.py` | Load human data (Nassar 2021), align with RNN behavioral metrics, compute statistics | Behavior extraction, Bayesian fitting outputs, plotting utilities |

---

## Data Flow

### PyTorch to JAX Handoff (Critical Boundary)

PyTorch tensors cannot be passed directly to JAX. The handoff always goes through NumPy:

```
# In behavior extraction (src/nn4psych/analysis/behavior.py)
hx_tensor: torch.Tensor  →  hx_numpy: np.ndarray  →  save to .npy

# In circuit inference (src/nn4psych/analysis/circuit_inference.py)
hidden_states: np.ndarray  →  scipy.optimize.minimize  →  circuit_params: np.ndarray

# In Bayesian fitting (src/nn4psych/bayesian/numpyro_models.py)
bucket_positions: np.ndarray  →  jnp.array(bucket_positions)  →  NUTS MCMC
posterior_samples: dict[str, jnp.ndarray]  →  np.array(v)  →  save to .npy
```

The existing `numpyro_models.py` already follows this pattern correctly at the `run_mcmc` boundary
(accepts `np.ndarray`, converts to `jnp.array` internally). The new subpackage should preserve this.

### Full Data Flow Sequence

```
data/raw/nassar2021/          →  (immutable human data)
trained_models/checkpoints/   →  Stage 2: behavior extraction
                              →  data/processed/rnn_behav/*.npy
                                 shape: (n_epochs, n_trials, [bucket, bag, hx_dim, hazard])

data/processed/rnn_behav/     →  Stage 3a: circuit inference
                              →  data/processed/circuit_params/*.npy
                                 keys: {Q, w_rec, w_in, w_out, fit_error}

data/processed/rnn_behav/     →  Stage 3b: fixed point analysis
                              →  data/processed/fixed_points/*.npy
                                 keys: {fixed_points, jacobians, eigenvalues, pca_basis}
                              →  figures/dynamical/

data/processed/rnn_behav/     →  Stage 4: Bayesian fitting
                              →  data/processed/bayesian_fits/*.npy
                                 keys: {H, LW, UU, sigma_motor, sigma_LR} posteriors
                              →  output/bayesian_fits/*.csv

data/processed/rnn_behav/ +
data/processed/bayesian_fits/ +
data/raw/nassar2021/          →  Stage 5: comparison
                              →  output/behavioral_summary/*.csv
                              →  figures/comparison/
```

---

## Where New Modules Fit in src/nn4psych/

### Promote existing script to analysis module

`scripts/analysis/analyze_fixed_points.py` contains a well-structured `FixedPointAnalyzer` class
that already follows the codebase conventions. It should be promoted to a proper module:

```
src/nn4psych/analysis/fixed_points.py   ← move class here
```

The script at `scripts/analysis/analyze_fixed_points.py` becomes a thin entry-point wrapper
calling `from nn4psych.analysis.fixed_points import FixedPointAnalyzer`.

### New subpackage: src/nn4psych/bayesian/

The `bayesian/` top-level directory exists outside the installable package. Consolidation into
`src/nn4psych/bayesian/` makes the Bayesian models importable as `from nn4psych.bayesian import run_mcmc`.

```
src/nn4psych/bayesian/
├── __init__.py               # exports: run_mcmc, normative_model, summarize_posterior
├── numpyro_models.py         # moved + cleaned from bayesian/numpyro_models.py
├── model_comparison.py       # moved + cleaned from bayesian/model_comparison.py
└── pyem_models.py            # optional: keep PyEM as legacy fast-path
```

The top-level `bayesian/` directory should be retained for backward compatibility with existing
scripts but marked as deprecated in its `__init__.py`.

### New file: src/nn4psych/analysis/circuit_inference.py

Latent circuit inference fits a low-dimensional linear RNN circuit:
```
h_{t+1} ≈ tanh(w_rec @ h_t + w_in @ x_t + b)
y_t = w_out @ h_t
```
from extracted hidden state sequences. This is pure NumPy/SciPy (no JAX required), so it sits
naturally in the analysis layer alongside `fixed_points.py`.

### New file: src/nn4psych/analysis/comparison.py

Human vs RNN comparison is a post-processing step that reads from `data/processed/` and
`data/raw/nassar2021/`. It depends on Bayesian fitting outputs and behavior extraction outputs
but not on PyTorch or JAX at runtime.

### New file: envs/context_decision.py

New task environments follow the same pattern as `envs/pie_environment.py`: gym-compatible
class with `reset()` and `step()`, exported from `envs/__init__.py`.

---

## Patterns to Follow

### Pattern 1: NumPy Array at Module Boundaries

All public functions in `analysis/` and `bayesian/` accept and return `np.ndarray`, never
`torch.Tensor` or `jnp.ndarray`. Conversion happens inside the module, not at the call site.

**Why:** Keeps callers framework-agnostic. A pipeline script can call circuit inference and
then Bayesian fitting without knowing about JAX.

### Pattern 2: Dataclass Config for New Components

New analysis components that have configurable parameters should use dataclasses following
the pattern in `src/nn4psych/training/configs.py`:

```python
@dataclass
class CircuitInferenceConfig:
    n_components: int = 10
    max_iter: int = 5000
    tolerance: float = 1e-6
    method: str = "gradient"
```

### Pattern 3: Numbered Pipeline Script per Stage

Each new analysis stage gets a numbered pipeline script:

```
scripts/data_pipeline/08_infer_latent_circuits.py    ← Stage 3a
scripts/data_pipeline/09_fit_bayesian_numpyro.py     ← Stage 4 (replaces scripts/fitting/)
scripts/data_pipeline/10_compare_rnn_human.py        ← Stage 5
```

Scripts are thin wrappers: load data, call library function, save outputs. Logic lives in
`src/nn4psych/`.

### Pattern 4: Optional JAX Dependency Guard

NumPyro/JAX are heavy optional dependencies. Guard imports at the module level:

```python
try:
    import jax.numpy as jnp
    import numpyro
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
```

Public functions in `src/nn4psych/bayesian/` should raise a clear `ImportError` with install
instructions if JAX is unavailable, rather than failing cryptically.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Mixed Framework Objects Across Boundaries

**What:** Passing `torch.Tensor` into JAX functions or `jnp.ndarray` into PyTorch modules.

**Why bad:** Silent type coercion failures, device mismatch errors, and gradient tracking bugs
that are difficult to debug. JAX and PyTorch use different memory layouts on GPU.

**Instead:** Convert to `np.ndarray` at the boundary and let each framework import from NumPy.

### Anti-Pattern 2: Inline JAX in Training Scripts

**What:** Adding JAX computation inside `scripts/training/` to "speed up" analysis done during
training.

**Why bad:** Training scripts are already complex; JAX requires careful random key management
and JIT compilation semantics that clash with PyTorch's autograd graph.

**Instead:** Keep JAX in the separate `bayesian/` subpackage. Run Bayesian fitting as a
post-training pipeline stage.

### Anti-Pattern 3: Putting Analysis Logic in Pipeline Scripts

**What:** Writing the `FixedPointAnalyzer` or circuit fitting math directly in
`scripts/data_pipeline/07_analyze_fixed_points.py`.

**Why bad:** Not testable, not reusable from notebooks, duplicated if needed in multiple scripts.

**Instead:** Logic in `src/nn4psych/analysis/`; scripts are one-liners that call library code.
(The existing `analyze_fixed_points.py` script already moved logic into a class — now finish the
job by moving the class into the package.)

### Anti-Pattern 4: Keeping Two Bayesian Module Locations Active

**What:** Maintaining both `bayesian/` (top-level) and `src/nn4psych/bayesian/` as active sources
of truth for the same models.

**Why bad:** Import path confusion, version drift between the two copies.

**Instead:** Consolidate into `src/nn4psych/bayesian/`. Leave `bayesian/__init__.py` as a
deprecated shim that re-exports from the new location with a deprecation warning.

---

## Suggested Build Order

Dependencies determine what can be built independently versus sequentially.

```
Phase A (no cross-dependencies):
  ├── envs/context_decision.py          (depends on: envs pattern only)
  └── src/nn4psych/analysis/fixed_points.py  (depends on: existing models/io)

Phase B (depends on Phase A or existing code):
  ├── src/nn4psych/analysis/circuit_inference.py  (depends on: behavior.py outputs)
  └── src/nn4psych/bayesian/  subpackage          (depends on: existing bayesian/ code)

Phase C (depends on Phase B):
  └── src/nn4psych/analysis/comparison.py   (depends on: bayesian/ outputs + human data)

Phase D (pipeline wiring — depends on Phases A–C):
  ├── scripts/data_pipeline/08_infer_latent_circuits.py
  ├── scripts/data_pipeline/09_fit_bayesian_numpyro.py
  └── scripts/data_pipeline/10_compare_rnn_human.py
```

**Rationale:**
- The context-dependent task and fixed point module have no dependency on JAX, so they can
  be built first and tested immediately with existing test infrastructure.
- Circuit inference depends on having hidden state data available (behavior extraction output),
  but the module itself only depends on NumPy/SciPy.
- The Bayesian subpackage consolidation is independent of circuit inference but should happen
  before the comparison module, which needs both fitted parameters and behavioral sequences.
- Pipeline scripts are always last because they wire together components that must already exist.

---

## Scalability Considerations

The current pipeline runs serially on CPU. For hyperparameter sweeps (8 gamma × 9 rollout × 8
preset × 6 scale = 3,456 model variants), the stages most affected by scale are:

| Stage | Current | At 3,456 models | Mitigation |
|-------|---------|-----------------|------------|
| Behavioral extraction | Serial, CPU | ~hours | `batch_extract_behavior` already exists; add joblib parallelism |
| Circuit inference | Not yet built | Linear in n_models | Pure NumPy: trivially parallelizable with `multiprocessing` |
| Fixed point finding | Random restarts are slow | Linear in n_models | Vectorize with JAX `vmap` if needed; otherwise serial is fine |
| Bayesian MCMC | 4 chains × ~minutes each | Prohibitive if per-model | Use PyEM (fast MAP) for sweep; MCMC only for final best models |
| Human comparison | Fast (aggregation only) | Negligible | No change needed |

The existing `batch_extract_behavior` function in `src/nn4psych/analysis/behavior.py` is the
right pattern to extend for the new stages. Each stage should accept either a single input or a
list of inputs and return results in a dict keyed by model identifier.

---

## Sources

- Existing codebase: `src/nn4psych/`, `bayesian/`, `scripts/`, `envs/` (HIGH confidence —
  direct code inspection)
- `bayesian/numpyro_models.py`: JAX/NumPy boundary pattern confirmed at `run_mcmc` signature
  (HIGH confidence — direct code inspection)
- `scripts/analysis/analyze_fixed_points.py`: `FixedPointAnalyzer` class confirmed as suitable
  for promotion to package module (HIGH confidence — direct code inspection)
- PyTorch/JAX interop pattern (np.ndarray as interchange format): standard practice, confirmed
  in existing `posterior_predictive` function which calls `np.array(v)` on all JAX outputs
  (HIGH confidence — direct code inspection)
- Build order: derived from import dependency analysis of module files (HIGH confidence)
