# Phase 3: Latent Circuit Inference - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the Langdon & Engel 2025 latent circuit inference pipeline: fit Q, w_rec, w_in, w_out from context-DM RNN hidden states, validate with invariant subspace and activity-level checks, and run perturbation analysis. Uses the engellab/latentcircuit codebase. Does NOT include new task training or Bayesian model fitting.

</domain>

<decisions>
## Implementation Decisions

### Latent rank & dimensions
- Follow Langdon & Engel 2025 exactly for rank selection (researcher looks up their choice for context-DM, likely 3-5 dims)
- Timestep windowing: follow the paper's approach — researcher determines if they use full trial or task-relevant window
- Trial averaging: follow the paper — likely condition-averaged responses grouped by context x stimulus
- **Both modality contexts needed:** Retrain the model on both modality_context=0 and modality_context=1 interleaved, then extract hidden states for both conditions. Single-modality extraction from Phase 2 is insufficient for circuit inference because the paper requires both contexts to reveal context-dependent routing

### Optimization approach
- **Use engellab/latentcircuit as a pip dependency** — wrap their LatentNet class rather than reimplementing
  - Reference repo: C:\Users\aman0087\Documents\Github\latentcircuit (already cloned)
  - LatentNet is a PyTorch nn.Module with Cayley transform for orthonormal Q
  - Loss = mse_z (task output) + l_y * nmse_y (hidden state reconstruction)
  - Adam optimizer with weight decay, batch_size=128
- Write a thin adapter layer to convert ActorCritic hidden states → LatentNet expected format (u, z, y tensors)
- **100-init ensemble runs sequentially** — no parallelism, to stay safe on 16GB RAM
- Best solution selected by lowest reconstruction loss

### Validation thresholds
- Invariant subspace correlation (Q^T W_rec Q vs inferred w_rec): **soft threshold at 0.85** — report the value, don't block downstream analysis if below
- Activity-level validation: **quantitative R^2 threshold** between projected RNN states and latent model states
- Use LatentNet's built-in metrics: nmse_y (reconstruction) and nmse_q (variance unexplained by Q), plus the invariant subspace correlation
- **Output as JSON report** — save validation_results.json with all metrics, threshold comparisons, pass/soft-fail status

### Perturbation analysis
- Follow the paper's perturbation methodology exactly (researcher reads their approach)
- Behavioral metric after perturbation: **task accuracy** on context-DM trials (accuracy delta between perturbed and unperturbed RNN)
- Threshold: **soft — report the delta**, don't set a hard pass/fail cutoff
- Results saved to **output/circuit_analysis/**

### Claude's Discretion
- Exact adapter code structure for converting ActorCritic hidden states to LatentNet format
- Number of epochs for LatentNet fitting (follow paper's recommendation or use their tutorial defaults)
- How to install latentcircuit as dependency (pip install from local path or git URL)

</decisions>

<specifics>
## Specific Ideas

- The engellab/latentcircuit repo is already cloned at C:\Users\aman0087\Documents\Github\latentcircuit
- Their LatentNet uses ReLU activation internally (even though our ActorCritic uses tanh) — this is fine because LatentNet fits a *latent* model, not matching the exact activation
- Their Q is parameterized via Cayley transform of a free matrix `a` — guarantees orthonormality without explicit constraints
- Connectivity masks enforce diagonal input/output structure — must be preserved in adapter
- Their `Net` class (net.py) is their own RNN — we do NOT use it; we only use `LatentNet` (latent_net.py) and `connectivity.py`

</specifics>

<deferred>
## Deferred Ideas

- Multi-task latent circuit comparison (PIE vs context-DM circuits) — v2 requirement ADV-02
- dPCA / demixed PCA analysis — v2 requirement ADV-01
- Input-conditioned fixed points from latent circuit — v2 requirement FP-02

</deferred>

---

*Phase: 03-latent-circuit-inference*
*Context gathered: 2026-03-19*
