# Technology Stack

**Project:** nn4psych — Computational Neuroscience RNN Analysis Pipeline
**Researched:** 2026-03-18
**Scope:** Subsequent milestone additions — latent circuit inference, NumPyro/JAX Bayesian fitting, fixed point analysis, context-dependent DM task

---

## Summary Recommendation

The new capabilities require three distinct sub-stacks that must coexist with the existing PyTorch codebase. The key constraint is that JAX and PyTorch must share the same Python process (for analysis pipelines), which is achievable via DLPack zero-copy tensor sharing. Do not run them in separate processes unless GPU memory contention becomes a real problem — that adds unnecessary complexity.

---

## Existing Stack (Do Not Change)

| Technology | Version | Role |
|------------|---------|------|
| PyTorch | 2.10.0 (latest stable as of 2026-01) | RNN actor-critic training |
| Gymnasium | >=0.28.0 | RL environment interface |
| NumPy | >=1.20.0 | Shared array substrate |
| SciPy | >=1.7.0 | Scientific computing utilities |
| NeuroGym | 2.3.0 | DawTwoStep, SingleContextDecisionMaking |

---

## New Stack: Area 1 — Latent Circuit Inference (Langdon & Engel 2025)

### Recommendation: Reimplement from the engellab/latentcircuit reference codebase in PyTorch

**Confidence: HIGH** — Verified from official repo (github.com/engellab/latentcircuit).

| Technology | Version | Purpose | Rationale |
|------------|---------|---------|-----------|
| PyTorch | 2.10.0 (already in project) | LatentNet optimization (gradient-based fitting of Q, w_rec, w_in, w_out) | The Langdon & Engel official code is pure PyTorch. No new framework needed. Gradient descent on latent circuit parameters uses standard torch.optim |
| SciPy | >=1.7.0 (already in project) | Eigenvalue decomposition for circuit analysis | Already a dependency |
| Pandas | 2.0.x | Results DataFrames for connectivity statistics | The reference code uses pandas for tabular outputs |
| Seaborn | 0.13.2 | Visualization of latent circuit structure | Used in reference code; aligns with matplotlib already in project |

**Key architectural insight:** The LatentNet class in the reference repo (latent_net.py) fits a low-dimensional circuit model `y = Qx` by gradient descent using PyTorch autograd — it is NOT a JAX computation. The Net class (RNN trainer) is also pure PyTorch. No framework boundary crossing required for this capability.

**What NOT to use:**
- Do not attempt to implement latent circuit inference in JAX/Flax. The reference implementation is PyTorch, and reimplementing in JAX adds no value while requiring re-verification.
- Do not use scikit-learn PCA or UMAP as a substitute. Latent circuit inference produces interpretable circuit parameters (recurrent weights, input/output projections) not just a reduced-dimensional embedding.

**Reference repo dependencies (pinned by authors):**
```
torch==2.4.1
pandas==2.0.3
scipy==1.10.1
seaborn==0.13.2
```

Sources: https://github.com/engellab/latentcircuit (HIGH confidence — official repo)

---

## New Stack: Area 2 — NumPyro/JAX Bayesian Model Fitting with HMC/NUTS

### Recommendation: NumPyro 0.20.0 + JAX 0.9.1 + ArviZ 1.0.0

**Confidence: HIGH** — Verified from PyPI release pages and existing project code.

| Technology | Version | Purpose | Rationale |
|------------|---------|---------|-----------|
| JAX | 0.9.1 | Autodiff + JIT compilation backend for NumPyro | Required by NumPyro. Provides `jax.lax.scan` for efficient trial-by-trial loop in normative model (already used in existing numpyro_models.py). Python >=3.11 required. |
| jaxlib | 0.9.1 | JAX C++ backend (CPU build for this project) | Must match JAX version exactly. Use CPU build — project does not require GPU for Bayesian fitting at current model/data scale. |
| NumPyro | 0.20.0 | Probabilistic programming with NUTS/HMC | NUTS sampler already implemented in existing bayesian/numpyro_models.py. Provides `MCMC`, `NUTS`, `Predictive`, `hpdi`. Latest release Jan 30, 2026. |
| ArviZ | 1.0.0 | MCMC diagnostics, posterior visualization, WAIC | `az.from_numpyro()` already used in existing code. Provides trace plots, R-hat convergence, WAIC/LOO model comparison. Latest release Mar 2, 2026. |
| Optax | 0.2.7 | Not needed for Bayesian fitting (NUTS handles it) | Only needed if switching to SVI (variational inference) — defer unless NUTS is too slow. Latest release Feb 5, 2026. |

**Implementation status:** The core NumPyro model is already implemented in `bayesian/numpyro_models.py`. The model implements the Nassar 2021 normative learning equations using `jax.lax.scan` for efficiency. The stack is essentially already correct — the task is to verify it works end-to-end on actual data and extend it to RNN agent outputs.

**What NOT to use:**
- Do not use PyMC. Per project requirements, this is being archived. PyMC uses PyTensor as backend (not JAX) and creates a third framework dependency with no benefit over NumPyro for this use case.
- Do not use PyEM. Being archived — legacy point-estimate approach, lacks full posterior, no longer maintained as active codebase here.
- Do not use BlackJAX as a standalone replacement. NumPyro already wraps JAX's NUTS cleanly; BlackJAX is lower-level and would require reimplementing the probabilistic model layer. NumPyro's `numpyro.sample`, `numpyro.plate`, and effect handlers are already used in the codebase.
- Do not use Stan/PyStan. Adds another language dependency. JAX is already in the project.

**Installation (new dependencies to add to pyproject.toml):**
```toml
[project.optional-dependencies]
bayesian = [
    "jax>=0.9.1",
    "jaxlib>=0.9.1",
    "numpyro>=0.20.0",
    "arviz>=1.0.0",
]
```
Note: The current pyproject.toml has a `bayesian` extra with PyMC/PyTensor dependencies — this must be replaced.

Sources:
- PyPI: https://pypi.org/project/numpyro/ (v0.20.0, Jan 30 2026) — HIGH confidence
- PyPI: https://pypi.org/project/jax/ (v0.9.1, Mar 2 2026) — HIGH confidence
- PyPI: https://pypi.org/project/arviz/ (v1.0.0, Mar 2 2026) — HIGH confidence
- Existing code in bayesian/numpyro_models.py confirms ArviZ-NumPyro integration pattern — HIGH confidence

---

## New Stack: Area 3 — Fixed Point Analysis of Trained RNNs

### Recommendation: Implement custom fixed-point finder in PyTorch using scipy.optimize + torch.autograd.functional.jacobian

**Confidence: MEDIUM** — Based on verified ecosystem survey; no single dominant maintained library for PyTorch as of 2026.

| Technology | Version | Purpose | Rationale |
|------------|---------|---------|-----------|
| PyTorch (torch.autograd.functional) | 2.10.0 | Compute Jacobian of RNN transition at fixed points | `torch.autograd.functional.jacobian` provides the exact Jacobian needed for linearized dynamics analysis. No extra library. |
| SciPy (scipy.optimize) | >=1.7.0 | Minimize `||h* - F(h*)||^2` to find fixed points via L-BFGS-B | Standard approach from Sussillo & Barak 2013. SciPy already in project. |
| NumPy | >=1.20.0 | Convert between PyTorch tensors and scipy arrays | Bridge layer: `.detach().numpy()` to SciPy, back to tensor for Jacobian. |

**Ecosystem assessment:**
- **FixedPointFinder (mattgolub)**: Supports PyTorch but last released December 2022, tests are broken as of 2022-2023 due to package upgrades. Low maintenance risk. Can be used as inspiration but not as a direct dependency.
- **pytorch-fixed-point-analysis (tripdancer0916)**: Last commit 2019. Abandoned. Do not use.
- **rnn-fxpts (garrettkatz)**: Niche, low adoption. Do not use.

**Recommended implementation pattern:**
```python
# Pseudocode for custom fixed-point finder
def find_fixed_points(rnn_cell, initial_states, inputs):
    """Find h* such that F(h*, x) = h* via L-BFGS-B."""
    for h0 in initial_states:
        result = scipy.optimize.minimize(
            fun=lambda h: ||rnn_cell(h, inputs) - h||^2,
            x0=h0,
            method='L-BFGS-B',
            jac=True  # autograd provides gradient
        )
    return fixed_points

def linearize_at_fixed_point(rnn_cell, h_star, inputs):
    """Eigendecompose Jacobian at fixed point."""
    J = torch.autograd.functional.jacobian(
        lambda h: rnn_cell(h, inputs), h_star
    )
    eigenvalues, eigenvectors = torch.linalg.eig(J)
    return eigenvalues, eigenvectors
```

**What NOT to use:**
- Do not add FixedPointFinder as a pip dependency — its tests are broken and the API may have compatibility issues with PyTorch 2.x. Implement the minimal required functionality directly.
- Do not use JAX for fixed point analysis of PyTorch RNNs — requires converting model weights to JAX, which adds significant complexity for no benefit when torch.autograd.functional.jacobian exists.

Sources:
- https://github.com/mattgolub/fixed-point-finder (inspected: PyTorch supported, last release Dec 2022, tests broken) — MEDIUM confidence
- PyTorch 2.10 docs: `torch.autograd.functional.jacobian` — HIGH confidence (built-in)
- Sussillo & Barak 2013 method: SciPy L-BFGS-B + Jacobian eigendecomposition — standard practice, MEDIUM confidence

---

## New Stack: Area 4 — Context-Dependent Decision-Making Task

### Recommendation: NeuroGym ContextDecisionMaking-v0

**Confidence: HIGH** — Verified from official NeuroGym docs and GitHub.

| Technology | Version | Purpose | Rationale |
|------------|---------|---------|-----------|
| NeuroGym | 2.3.0 | `ContextDecisionMaking-v0` environment | Implements Mante et al. 2013 context-DM task. NeuroGym inherits from Gymnasium — compatible with existing Gymnasium-based workflow. This is the primary task used in Langdon & Engel 2025 for latent circuit validation. Latest release Dec 12 (year confirmed 2025). |
| Gymnasium | >=0.28.0 (already in project) | Environment interface | NeuroGym inherits from Gymnasium directly — no adapter needed. |

**Task description (ContextDecisionMaking-v0):**
- Agent receives simultaneous inputs from two modalities (e.g., color + motion)
- A context/rule signal indicates which modality is relevant
- Agent must discriminate the relevant modality and ignore the other
- Directly analogous to Mante et al. 2013 and used as-is in Langdon & Engel 2025

**Important NeuroGym nuance:** NeuroGym also provides `SingleContextDecisionMaking-v0`, which is already integrated in the project (per PROJECT.md). `ContextDecisionMaking-v0` is the dual-modality version needed for full Mante 2013 / Langdon & Engel 2025 replication. Both are in NeuroGym 2.3.0.

**What NOT to use:**
- Do not implement a custom context-DM task from scratch. NeuroGym's `ContextDecisionMaking-v0` is the standard, community-maintained implementation with correct trial structure (fixation, stimulus, decision periods with configurable timing).
- Do not use the OpenAI Gym version of NeuroGym (pre-Gymnasium). The current NeuroGym 2.3.0 uses Gymnasium natively.

Sources:
- https://neurogym.github.io/envs/ContextDecisionMaking-v0.html — HIGH confidence
- https://github.com/neurogym/neurogym/releases (v2.3.0) — HIGH confidence

---

## JAX and PyTorch Coexistence Strategy

**Confidence: MEDIUM** — Verified from official JAX discussions and community sources. CPU-only confirmed safe; GPU coexistence has known issues.

### The Problem

JAX pre-allocates GPU memory (typically 75% of available GPU RAM) on import. PyTorch uses a different allocator. Running both in the same Python process on GPU can cause out-of-memory errors. On CPU, there is no memory contention and coexistence is straightforward.

### Recommended Strategy: Sequential execution with CPU-side NumPy handoff

For this project, JAX and PyTorch serve different pipeline stages:
- **PyTorch stage:** RNN training, hidden state extraction, latent circuit fitting — outputs are NumPy arrays (`.detach().numpy()`)
- **JAX/NumPyro stage:** Bayesian model fitting on behavioral data (bucket/bag positions, RNN behavioral outputs) — inputs are NumPy arrays, converted with `jnp.array()`

Since the pipeline runs sequentially (train RNN → extract behavior → fit Bayesian model), the frameworks are not active simultaneously. The handoff is NumPy arrays, which both frameworks read without ownership conflict.

### Implementation

```python
# Stage 1 (PyTorch): Extract RNN behavioral outputs
import torch
hidden_states = rnn_model(inputs)  # torch.Tensor
bucket_positions = behavior['bucket'].detach().numpy()  # numpy.ndarray
bag_positions = behavior['bag'].detach().numpy()        # numpy.ndarray

# Stage 2 (NumPyro/JAX): Bayesian fitting on NumPy arrays
import jax.numpy as jnp
import numpyro
bucket_jax = jnp.array(bucket_positions)  # JAX array from NumPy
bag_jax = jnp.array(bag_positions)
mcmc = run_mcmc(bucket_positions, bag_positions, context='changepoint')
```

### GPU memory: Do not enable JAX GPU for this project

```python
# Add to project initialization / analysis scripts:
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force JAX to CPU
# Or equivalently:
import jax
jax.config.update("jax_platform_name", "cpu")
```

**Rationale:** The Bayesian model (Nassar 2021 normative model) operates on ~200-400 trial sequences per subject. JAX's `jax.lax.scan` vectorizes this efficiently on CPU. GPU is unnecessary and would create allocator conflicts with PyTorch. Fixed this at project level — not per-script.

### DLPack (for future use only)

If a future use case requires passing tensors between PyTorch and JAX within the same computation (e.g., differentiating through both), DLPack provides zero-copy tensor sharing:

```python
# PyTorch → JAX (same device)
import jax.dlpack
jax_array = jax.dlpack.from_dlpack(torch_tensor.to_dlpack())

# JAX → PyTorch
import torch.utils.dlpack
torch_tensor = torch.utils.dlpack.from_dlpack(jax_array.__dlpack__())
```

**Do not use DLPack for current pipeline.** NumPy handoff is simpler and equally fast for the current use case (no GPU involved, no within-computation tensor sharing needed).

### Environment setup

```bash
# Single conda environment — both frameworks coexist on CPU
conda create -n nn4psych python=3.11
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install "jax[cpu]>=0.9.1"
pip install numpyro>=0.20.0 arviz>=1.0.0

# JAX_PLATFORM_NAME=cpu in project .env or analysis script header
```

**Note on Python version:** JAX 0.9.1 requires Python >=3.11. The existing pyproject.toml supports Python 3.8+. The Python minimum version constraint in pyproject.toml should be updated to `>=3.11` for the new stack to be consistent, or JAX should be kept as an optional dependency with a separate environment note.

Sources:
- JAX GitHub issues #8362, #15268 (GPU coexistence conflict reports) — MEDIUM confidence
- https://github.com/mila-iqia/torch_jax_interop (DLPack approach) — MEDIUM confidence
- Existing bayesian/numpyro_models.py (NumPy handoff pattern already in use) — HIGH confidence

---

## Alternatives Considered

| Category | Recommended | Alternatives Considered | Why Not |
|----------|-------------|------------------------|---------|
| Latent circuit inference | Reimplement from engellab/latentcircuit (PyTorch) | Custom JAX implementation | Official code is PyTorch; no benefit to porting |
| Bayesian fitting | NumPyro 0.20.0 + JAX | PyMC 5.x + PyTensor | Project decision to archive PyMC; PyTensor adds 3rd framework |
| Bayesian fitting | NumPyro + NUTS | BlackJAX | NumPyro's model layer (numpyro.sample, plates) is already implemented; BlackJAX is lower-level, requires more code |
| Bayesian fitting | NumPyro + NUTS | Stan/PyStan | Another language dependency; JAX is already present |
| Fixed point analysis | Custom implementation (SciPy + torch.autograd.functional) | FixedPointFinder (mattgolub) | Tests broken since 2022; PyTorch 2.x compatibility unverified |
| Context-DM task | NeuroGym ContextDecisionMaking-v0 | Custom implementation | NeuroGym is community-maintained, already used in project (SingleContextDM) |
| JAX/PyTorch bridge | NumPy handoff | DLPack | NumPy handoff is sufficient; DLPack adds complexity for no gain on CPU |

---

## pyproject.toml Changes Required

The current `pyproject.toml` has incorrect optional dependencies for the new stack:

```toml
# CURRENT (to be replaced):
[project.optional-dependencies]
bayesian = [
    "pymc>=4.0.0",        # REMOVE — archiving
    "arviz>=0.11.0",      # UPDATE version
    "pytensor>=2.0.0",    # REMOVE — archiving
]
jax = [
    "jax>=0.4.0",         # UPDATE version
    "jaxlib>=0.4.0",      # UPDATE version
    "optax>=0.1.0",       # NOT needed for Bayesian (HMC, not SVI)
]

# REPLACEMENT:
[project.optional-dependencies]
bayesian = [
    "jax[cpu]>=0.9.1",
    "numpyro>=0.20.0",
    "arviz>=1.0.0",
]
analysis = [
    "pandas>=2.0.0",
    "seaborn>=0.13.0",
]
```

Also update `requires-python = ">=3.11"` (from `>=3.8`) if JAX is in the default installation path, since JAX 0.9.1 requires Python 3.11.

---

## Version Summary Table

| Package | Current in Project | Recommended | Source | Confidence |
|---------|-------------------|-------------|--------|------------|
| PyTorch | >=1.9.0 | >=2.10.0 | PyPI releases | HIGH |
| JAX | >=0.4.0 (jax extra) | >=0.9.1 | PyPI (Mar 2 2026) | HIGH |
| jaxlib | >=0.4.0 (jax extra) | >=0.9.1 | PyPI (Mar 2 2026) | HIGH |
| NumPyro | not listed (used in code) | >=0.20.0 | PyPI (Jan 30 2026) | HIGH |
| ArviZ | >=0.11.0 (bayesian extra) | >=1.0.0 | PyPI (Mar 2 2026) | HIGH |
| NeuroGym | not listed (used in code) | >=2.3.0 | GitHub releases | HIGH |
| Pandas | not listed | >=2.0.0 | Required by latentcircuit | HIGH |
| Seaborn | not listed | >=0.13.0 | Required by latentcircuit | HIGH |
| PyMC | >=4.0.0 (bayesian extra) | REMOVE | Project decision | HIGH |
| PyTensor | >=2.0.0 (bayesian extra) | REMOVE | Project decision | HIGH |
| Optax | >=0.1.0 (jax extra) | Defer — not needed now | Not needed for NUTS | MEDIUM |

---

## Sources

- engellab/latentcircuit GitHub repo: https://github.com/engellab/latentcircuit
- Langdon & Engel 2025, Nature Neuroscience: https://www.nature.com/articles/s41593-025-01869-7
- NumPyro PyPI: https://pypi.org/project/numpyro/ (v0.20.0, 2026-01-30)
- JAX PyPI: https://pypi.org/project/jax/ (v0.9.1, 2026-03-02)
- ArviZ PyPI: https://pypi.org/project/arviz/ (v1.0.0, 2026-03-02)
- Optax PyPI: https://pypi.org/project/optax/ (v0.2.7, 2026-02-05)
- NeuroGym GitHub releases: https://github.com/neurogym/neurogym/releases (v2.3.0)
- NeuroGym ContextDecisionMaking-v0 docs: https://neurogym.github.io/envs/ContextDecisionMaking-v0.html
- FixedPointFinder GitHub: https://github.com/mattgolub/fixed-point-finder (last release Dec 2022)
- JAX/PyTorch GPU coexistence issue: https://github.com/jax-ml/jax/issues/8362
- torch_jax_interop: https://github.com/mila-iqia/torch_jax_interop
