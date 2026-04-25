# Phase 3: Latent Circuit Inference - Context

**Gathered:** 2026-03-19
**Last updated:** 2026-04-25 (post 03-02; 03-03 decisions added)
**Status:** Plans 03-01, 03-02 complete; 03-03 ready for planning

<domain>
## Phase Boundary

Implement the Langdon & Engel 2025 latent circuit inference pipeline: fit Q, w_rec, w_in, w_out from context-DM RNN hidden states, validate with invariant subspace and activity-level checks, and run perturbation analysis. Uses the engellab/latentcircuit codebase. Does NOT include new task training or Bayesian model fitting.

</domain>

<decisions>
## Implementation Decisions

### Locked from 03-01 + 03-02 (executed)

These were originally open and have now been resolved by execution:

- **Latent rank for benchmark run:** n=16 (cluster default 8, plan-spec 8 — overridden for benchmark). Subject to revision in 03-03 Wave A.
- **Timestep window:** T=75 (regenerated from initial T=500 to address blank-padding concern). Trial structure for ContextDecisionMaking-v0 at dt=100ms uses ~15–30 active steps; T=75 is ~20–40% task-relevant.
- **Trial count:** n_trials=1000 (500 per modality_context). Resolves 03-01 batch-size concern.
- **Both modality contexts:** dual-modality alternating training (half-epoch blocks); ContinuousActorCritic ReLU RNN at hidden_dim=64.
- **Optimization:** vendored LatentNet, sequential 100-init ensemble, Adam + weight decay, batch_size=128. Now runs on Monash M3 GPU via `cluster/run_circuit_ensemble.sh` with env-var parameter sweep support.
- **z signal:** softmax(actor_logits), bounded [0,1], sums to 1 — naturally balances mse_z (~0.3) with nmse_y (~0.25) at l_y=1.0. Replaces 03-01's raw-logits choice.
- **Validation thresholds:** invariant subspace ≥ 0.85 (soft, originally), activity-level R² reported quantitatively. JSON report at `output/circuit_analysis/validation_results.json`.
- **Soft-fail observed:** 03-02 final invariant subspace corr = 0.703 < 0.85. Trial-avg R² = 0.98 (exceeds paper's 0.96). Per-trial R² = 0.85.

### Plan 03-03 Decisions (added 2026-04-25)

03-03 splits into two waves rather than going straight to perturbation analysis:

#### Wave A — Diagnose Q before perturbing
- **Run n_latent sweep at {4, 8, 12, 16}** on the cluster, reusing `run_circuit_ensemble.sh` with `N_LATENT` env var sweep (1 ensemble per rank, ~3.5h GPU each).
- **Output:** Pareto curve of (rank vs NMSE_y vs invariant subspace corr vs trial-avg R²).
- **Decision rule:** select Q from the rank with the **best invariant subspace correlation**, with the rank-vs-NMSE_y curve reported alongside as the rank-knee diagnostic.
- **What this tests:** the user's working hypothesis that the RNN is not strongly low-rank. If invariant subspace corr is essentially flat across ranks (e.g., all 0.65–0.75), and NMSE_y also doesn't improve much with rank, that's positive evidence for the method-limit interpretation. If corr at rank=8 is much higher than rank=16, it's over-parameterisation.
- **Acceptance threshold:** **informational, not gating.** Phase 3 SC-2 (CIRC-04, ≥0.85) is reported as a value with provenance, not a pass/fail gate. The chosen Q is whatever Wave A's sweep produced — no fixed minimum bar.

#### Wave B — Perturbation analysis on Wave A's chosen Q
- **Q is locked** by Wave A's best-corr selection.
- **Methodology, behavioral readout, pipeline shape** — Claude's discretion (see below).
- **Reporting:** writeup explicitly distinguishes two scientific stories:
  1. *"We couldn't reach 0.85 because of method/data limits"* — the RNN doesn't admit a clean low-rank embedding at this hidden_dim/task combo.
  2. *"We couldn't reach 0.85 because we ran out of fixes to try"* — Wave A wasn't enough; deferred ideas (masking, shorter T) remain on the table.

  These are substantively different findings; the writeup must commit to one based on Wave A's evidence.

### Claude's Discretion

Areas the user explicitly did not flag as gray — researcher and planner resolve from the paper + repo conventions:

- **Perturbation methodology:** which weight matrix (w_rec, w_in, w_out, all), magnitude regime (small linear vs large), direction selection (random / Q's columns / task-aligned axes). Default: follow Langdon & Engel 2025 exactly; deviate only if their recipe is incompatible with the soft-fail Q.
- **Behavioral readout:** how to measure "behavioral change" from perturbation (per-context accuracy delta, per-coherence delta, trajectory divergence in hidden space, or a composite). Default: task accuracy delta on context-DM, soft threshold (report value, no hard cutoff per existing CONTEXT.md decision).
- **Pipeline structure:** `08_infer_latent_circuits.py` shape (single end-to-end vs modular). Default: end-to-end with command-line flags so it can run as one script locally and via SLURM on cluster (mirrors `run_circuit_ensemble.sh` UX).
- **Execution venue for Wave B:** local CPU vs cluster. Default: local if a single perturbation analysis is cheap; cluster if the methodology requires sweeps over perturbation magnitude/direction.
- **Number of LatentNet fitting epochs, exact adapter structures, etc.** — already covered in 03-CONTEXT's original `Claude's Discretion` block.

</decisions>

<specifics>
## Specific Ideas

- The engellab/latentcircuit repo is at `C:\Users\aman0087\Documents\Github\latentcircuit` (already vendored as `src/nn4psych/analysis/latent_net.py`).
- LatentNet uses ReLU activation internally. The context-DM RNN was retrained as `ContinuousActorCritic` (ReLU) for activation-match in 03-02.
- Q parameterised via Cayley transform of free matrix `a` — guarantees orthonormality.
- Connectivity masks enforce diagonal input/output structure.
- Their `Net` class is NOT used; only `LatentNet` (latent_net.py) is.
- **Cluster pipeline reusable for Wave A:** `cluster/run_circuit_ensemble.sh` already accepts `N_LATENT` env var; sweeping is one-line per rank.
- **Diagnostic artifact path:** sweep results should land in `output/circuit_analysis/n_latent_sweep/` (one subdirectory per rank) so the Pareto curve can be assembled programmatically.

</specifics>

<deferred>
## Deferred Ideas

These are flagged as candidates only if Wave A's n_latent sweep doesn't produce a usable Q (corr stays poor and NMSE_y also stays poor across all ranks — pointing to T=75 padding rather than over-parameterisation):

- **Masked-loss fitting:** modify `LatentNet.fit()` to compute NMSE_y only over per-trial task-active timesteps (extracted from NeuroGym `trial_info`). Tests T=75-padding hypothesis directly.
- **Shorter T regen:** circuit_data.npz at T~25–40 with `delay=0` and tighter `max_steps`. Tests padding hypothesis without code changes to LatentNet.
- **Condition-sliced fitting:** fit LatentNet separately on task-active vs blank periods to quantify each region's contribution to w_rec.

Multi-task / cross-task v2 work (already in roadmap as v2 requirements):
- Multi-task latent circuit comparison (PIE vs context-DM circuits) — v2 ADV-02.
- dPCA / demixed PCA analysis — v2 ADV-01.
- Input-conditioned fixed points from latent circuit — v2 FP-02.
- Subset-Q fitting around change-point/oddball events (already a pending todo: `2026-03-28-subset-q-fitting-changepoint-oddball.md`).

</deferred>

---

*Phase: 03-latent-circuit-inference*
*Context originally gathered: 2026-03-19; updated 2026-04-25 with 03-03 decisions post-execution of 03-01 and 03-02.*
