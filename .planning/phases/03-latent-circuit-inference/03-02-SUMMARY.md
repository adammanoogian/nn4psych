---
phase: 03-latent-circuit-inference
plan: "02"
subsystem: analysis
tags: [latent-circuit, latent-net, circuit-inference, ensemble-fitting, context-dm, gpu, slurm, pytorch]

# Dependency graph
requires:
  - phase: 03-latent-circuit-inference
    provides: "Plan 01 — LatentNet vendored, collect_circuit_data(), dual-modality ContextDecisionMaking model"
provides:
  - "fit_latent_circuit_ensemble() — sequential n-init LatentNet fitting with best-by-NMSE_y selection"
  - "validate_latent_circuit() — invariant subspace, per-trial R², trial-averaged R² grouped by condition"
  - "Cluster GPU pipeline: cluster/run_circuit_ensemble.sh + cluster/setup_env.sh for 100-init LatentNet SLURM runs"
  - "output/circuit_analysis/validation_results.json + ensemble_diagnostics.json + best_latent_circuit.pt"
  - "ContinuousActorCritic (ReLU) RNN retrained for LatentNet compatibility (tanh ActorCritic → ReLU ContinuousActorCritic)"
  - "z switched from raw logits to softmax policy beliefs for paper-consistent loss scales"
  - "Regenerated circuit_data.npz: n_trials=1000 (500/context), T=75 — resolves 03-01 concerns (batch size < n_trials, excessive blank timesteps)"
affects:
  - "03-03-perturbation-analysis (uses best_latent_circuit.pt and the fitted Q/w_rec for Q-mapped rank-one perturbations)"
  - "Phase 4/5 cluster-adjacent work (proves GPU SLURM workflow for this project)"

# Tech tracking
tech-stack:
  added:
    - "Monash M3 SLURM GPU pipeline for LatentNet ensemble fitting"
    - "bash/SLURM cluster scripts with ENV var param sweep support (N_LATENT, L_Y, INCLUDE_OUTPUT_LOSS, FORCE_RECOLLECT)"
  patterns:
    - "Cluster-first compute for ensemble fitting: local CPU ~2h, single GPU ~3.5h wall for 100×16×500 (see timing caveat below)"
    - "Re-generate data/model binaries on cluster rather than git-track (see commit 10a2135)"
    - "Soft-fail validation model: report metric vs threshold, don't block downstream (plan 03-02 soft-fail → 03-03 inherits)"
    - "Ensemble diagnostics separate from validation report: all_nmse_y distribution saved so later analysis can detect multimodal landscape"
    - ".regen.lock sentinel file to serialize parallel data regeneration across concurrent SLURM jobs"

key-files:
  created:
    - cluster/run_circuit_ensemble.sh
    - cluster/setup_env.sh
    - cluster/environment_cluster.yml
    - output/circuit_analysis/validation_results.json
    - output/circuit_analysis/ensemble_diagnostics.json
    - output/circuit_analysis/best_latent_circuit.pt
    - data/processed/rnn_behav/.regen.lock
  modified:
    - src/nn4psych/analysis/circuit_inference.py (added fit_latent_circuit_ensemble, validate_latent_circuit; softmax z)
    - src/nn4psych/analysis/latent_net.py (verbose param, device handling, forward-pass pre-allocation, GPU bottleneck fixes, include_output_loss toggle)
    - src/nn4psych/models/ (ContinuousActorCritic wiring for LatentNet-compatible RNN)
    - scripts/training/train_context_dm.py (ContinuousActorCritic path)
    - data/processed/rnn_behav/circuit_data.npz (regenerated: T 500→75, n_trials 40→1000)

key-decisions:
  - "n_latent=16 for benchmark run (plan spec: 8; cluster default: 8) — chosen to push reconstruction quality; flagged as parameter to revisit in 03-03"
  - "Switch RNN from tanh ActorCritic to ReLU ContinuousActorCritic (commit 77e6b1c) — required for LatentNet (ReLU-based) invariant-subspace alignment"
  - "z = softmax(logits) instead of raw logits (commit 35dbb02) — drops mse_z from ~30 to ~0.3, balanced with nmse_y at l_y=1.0 per paper"
  - "Data regen: n_trials=40→1000 (500/context), T=500→75 — addresses 03-01 concerns (batch_size 128 > n_trials; T=500 had only ~5% task-relevant steps)"
  - "Cluster-first: 100-init ensemble run on Monash M3 GPU, not local CPU — 16GB RAM local would have taken 1-2 hours per init with 1000 trials"
  - "Soft-fail on invariant subspace (corr=0.703 < 0.85) is NON-BLOCKING per plan — proceed to 03-03 with caveat"
  - "Ensemble diagnostics saved separately (all 100 nmse_y values + convergence stats) so we can detect multimodality in the loss landscape"

patterns-established:
  - "Cluster SLURM for ensemble compute: env-var parameter sweeps via run_circuit_ensemble.sh, .regen.lock for data-regen serialization"
  - "ContinuousActorCritic = ReLU alternative when downstream analyses assume ReLU (LatentNet, Langdon & Engel 2025)"
  - "softmax z for policy-belief reconstruction (vs raw logits) — loss scales naturally balanced, easier paper-consistent interpretation"

# Metrics
duration: ~205min (100-init ensemble GPU wall); ~4 weeks elapsed including iteration
completed: 2026-04-24
---

# Phase 3 Plan 02: Latent Circuit Ensemble Fitting & Validation Summary

**100-init LatentNet ensemble fitted on Monash M3 GPU producing best NMSE_y=0.239 and trial-averaged R²=0.98; invariant subspace soft-fails (corr=0.703 < 0.85), meaning the fit reconstructs activity well but does not fully linearise the RNN's recurrent connectivity.**

## Performance

- **Ensemble run:** 204.5 min wall on CUDA (single GPU, 100 sequential inits × 16 latent × 500 epochs × 1000 trials × T=75)
- **Elapsed calendar time:** 2026-03-20 → 2026-04-24 (~5 weeks, including RNN retrain + data regen + GPU bottleneck fixes)
- **Tasks:** 2/2 (implementation + ensemble run)
- **Files modified:** 10+ across multiple commits

## Accomplishments

- **Implementation** (commit `251b667`): `fit_latent_circuit_ensemble()` and `validate_latent_circuit()` added to `src/nn4psych/analysis/circuit_inference.py` with 321 LOC; verbose=True flag added to `latent_net.py` fit()
- **RNN retrain** (commit `77e6b1c`): ContextDecisionMaking agent moved from tanh `ActorCritic` to ReLU `ContinuousActorCritic` for LatentNet (ReLU-based) compatibility
- **Data regen** (commits `10a2135`, `56603f8`): circuit_data.npz regenerated with n_trials=1000 (500/context) and T=75 — resolves both 03-01-flagged concerns
- **z reformulation** (commit `35dbb02`): softmax beliefs instead of raw logits; mse_z from ~30 → ~0.3, balanced with nmse_y at l_y=1.0
- **GPU perf** (commits `311d634`, `8adc90f`, `4aee60d`, `973d50f`): pre-allocated forward pass, 6 GPU bottleneck fixes, device placement fixes (q recomputed on correct device at start of fit/forward)
- **Cluster pipeline** (new files): `cluster/run_circuit_ensemble.sh` with env-var parameter sweep support; `cluster/setup_env.sh`; `.regen.lock` for data-regen serialization
- **Final benchmark** (commit `8ff2b8e`): 100 inits × 16 latent × 500 epochs completed; best_latent_circuit.pt + validation_results.json + ensemble_diagnostics.json produced

## Task Commits

1. **Impl**: `251b667` feat(03-02) — fit_latent_circuit_ensemble + validate_latent_circuit
2. **RNN switch**: `77e6b1c` — ContinuousActorCritic (ReLU) for LatentNet compat
3. **Data untrack**: `10a2135` — auto-regenerate data/model binaries on cluster
4. **Device fix**: `4aee60d` — proper env setup and GPU device handling
5. **Retrain**: `56603f8` — retrained RNN on cluster
6. **Perf (timing)**: `311d634` — pre-allocate LatentNet forward pass + timing
7. **Perf (GPU×6)**: `8adc90f` — 6 GPU bottleneck fixes
8. **Baseline**: `5e032c3` — circuit run without pytorch optimizations
9. **q device**: `973d50f` — recompute q on correct device at start of fit/forward
10. **z softmax**: `35dbb02` — switch z to softmax beliefs + include_output_loss toggle
11. **Benchmark**: `8ff2b8e` — 3 benchmark runs (final artifacts)

## Validation Results (from output/circuit_analysis/validation_results.json)

| Metric | Target | Result | Verdict |
|---|---|---|---|
| n_inits | 100 | 100 | ✓ |
| n_latent | 8 (plan) | **16** | ⚠ deviation |
| Epochs | 500 | 500 | ✓ |
| Device | any | CUDA | ✓ |
| Best NMSE_y | lower = better | 0.239 (init 95) | — |
| Best MSE_z | lower = better | 0.273 | — |
| NMSE_q | paper-low | 0.034 | ✓ |
| Activity R² (full, per-trial) | paper: 0.96 | **0.851** | — below paper, above chance |
| Activity R² (latent, per-trial) | — | 0.837 | — |
| **Trial-avg R² (full space)** | paper: 0.96 | **0.980** | ✓ **exceeds paper** |
| Trial-avg R² (latent space) | — | 0.983 | ✓ |
| Trial-avg R² (ctx=0 / ctx=1) | — | 0.979 / 0.980 | ✓ |
| **Invariant subspace corr** | ≥ 0.85 | **0.703** | ❌ **soft-fail** |

Ensemble landscape (from ensemble_diagnostics.json):
- mean NMSE_y = 0.270, std = 0.029, min = 0.239, max = 0.455
- Distribution is tight (most inits 0.24–0.30) with a handful of outlier inits at 0.35–0.46 — suggests a well-behaved landscape with a few bad local minima

## Decisions Made

1. **n_latent=16 for the benchmark run** despite plan spec n_latent=8. Not yet revisited at n_latent=8 under the regenerated (1000×75) data + softmax z + ReLU RNN configuration. Worth an A/B in 03-03 planning to determine if rank 16 is actually needed or if 8 suffices with the improved data.

2. **Full stack rewrite before benchmarking** (ReLU RNN, softmax z, regen data) rather than re-running 03-02 at the original (tanh, raw logits, 40×500) configuration. Justified by: LatentNet assumes ReLU dynamics; paper uses softmax-normalised z; 03-01 had flagged both 40-trial batch-size and 500-step padding as concerns.

3. **Benchmark run done on M3 GPU, not local CPU.** 16GB RAM + CPU would have been prohibitive at 1000 trials.

4. **Soft-fail accepted on invariant subspace.** Plan explicitly says non-blocking. But this is *the* key structural validity check (CIRC-04, SC-2) — the Q subspace only partially captures the RNN's recurrent mechanism. Carry into 03-03 as an open scientific concern.

## Deviations from Plan

### Category A — Improvements to plan-spec (addressed 03-01 carry-over concerns)

**1. Data regeneration: T=500 → T=75, n_trials=40 → 1000**
- **Found during:** cluster setup (post 03-01 Next Phase Readiness notes)
- **Issue:** 03-01 summary flagged (a) n_trials=40 < batch_size=128, so only partial batches per epoch; (b) T=500 had ~5% task-relevant steps, rest blank
- **Fix:** regenerate at n_trials=1000 (500/context), T=75 before benchmarking
- **Impact:** Fit quality likely improved; see "Open Concern" below for the remaining T=75 task-structure question.

**2. RNN architecture: ActorCritic (tanh) → ContinuousActorCritic (ReLU)**
- **Found during:** LatentNet integration
- **Issue:** LatentNet (Langdon & Engel 2025) assumes ReLU dynamics in both the full-N and latent-n spaces. Fitting a ReLU latent circuit to tanh RNN activity creates a fundamental architecture mismatch that would inflate NMSE_y regardless of rank.
- **Fix:** Retrain the context-DM agent with `ContinuousActorCritic` (ReLU). Verified via commit `77e6b1c` + `56603f8`.
- **Impact:** Invariant subspace is now a fair test of the paper's method (same nonlinearity both sides).

**3. z signal: raw logits → softmax(logits)**
- **Found during:** loss-scale analysis
- **Issue:** raw logits gave mse_z ~30 while nmse_y ~0.25, making l_y=1.0 (paper's default) ignore the task-output constraint. Paper uses softmax-bounded action beliefs.
- **Fix:** z = softmax(logits), bounded [0,1], sums to 1. mse_z → ~0.3, naturally balanced.
- **Commit:** `35dbb02`.

### Category B — Parameter choices

**4. n_latent=16 vs plan's 8**
- **Found during:** benchmark run setup
- **Issue:** Plan spec was 8; cluster default is 8; benchmark was run with `N_LATENT=16` via env var
- **Justification:** unclear from commit history — likely exploratory to push reconstruction quality
- **Impact:** The trial-avg R² = 0.98 (above paper's 0.96) may be partly a higher-rank artifact, not inherent fit quality. Revisit in 03-03 at n_latent ∈ {4, 8, 12, 16} to find the knee.

### Category C — Infrastructure

**5. Cluster SLURM pipeline not in original plan**
- **Rationale:** 1000-trial ensemble at 100×500 epochs is infeasible on 16GB local CPU
- **Added files:** `cluster/run_circuit_ensemble.sh`, `cluster/setup_env.sh`, `cluster/environment_cluster.yml`, `data/processed/rnn_behav/.regen.lock`
- **Pattern:** env-var parameterisation (N_LATENT, L_Y, INCLUDE_OUTPUT_LOSS, FORCE_RECOLLECT) for ensemble parameter sweeps without editing code
- **Companion:** slurm-autopush skill now installed in `~/.claude/skills/` for future cluster-results auto-pull workflows

---

**Total deviations:** 5 (3 improvements addressing 03-01 concerns, 1 parameter choice open for re-examination, 1 infrastructure addition)
**Impact on plan:** All deviations scientifically justified except n_latent=16 (which warrants a sweep in 03-03). Core plan deliverables (100-init ensemble, validation report, saved best model) all produced.

## Issues Encountered

- **GPU device placement bug** (commit `973d50f`): `q` was not recomputed on the correct device at start of fit() and forward() — caused CPU/GPU tensor mismatch. Fixed.
- **GPU perf ceiling hit** without optimizations (commit `5e032c3` baseline → `8adc90f` 6× fixes): initial ensemble was prohibitively slow; 6 targeted optimizations brought wall time down enough for 100 inits.
- **Calendar drift**: 5 weeks elapsed between 03-01 complete (2026-03-20) and final benchmark (2026-04-24). Work was real (11 commits, substantial architecture changes) but the GSD workflow didn't track it — SUMMARY written post-hoc from git history + artifacts.

## Open Concern for 03-03: Task structure vs. padding at T=75

**The user raised a critical question:** if T=75 but actual task dynamics span only fixation+stim+delay+decision (~15–30 steps at dt=100ms for ContextDecisionMaking-v0), is the remaining ~45–60 steps of padding adding noise to the latent circuit fit? Could this be contributing to the 0.703 invariant subspace soft-fail?

**Evidence relevant to the concern:**
- T went 500 → 75 already (big improvement: from ~5% task-relevant to ~20–40% task-relevant)
- Trial-avg R² = 0.98 (very high, averaging-out helps) but per-trial R² = 0.85 (lower — single-trial dynamics include noise from blank periods)
- Invariant subspace fails (0.70 < 0.85) — structural mismatch between inferred w_rec and Q^TW_recQ. One plausible cause: w_rec is fitted on dynamics including padding, where the true recurrent computation is not engaged, so w_rec over-fits to drift/decay rather than task-relevant dynamics.

**Proposed investigation for 03-03 planning:**
- Option A: **Mask the loss** — compute NMSE_y only on task-active timesteps (fixation + stimulus + delay + decision), ignore padding. Requires per-trial timing metadata from NeuroGym.
- Option B: **Shorten T** — regenerate circuit_data.npz with max_steps tuned to cover just the task window (e.g., T=30 with delay=0).
- Option C: **Condition-sliced fitting** — fit LatentNet separately to task-active vs blank regions to quantify how much each contributes to the fit.
- Option D: **Parameter sweep n_latent ∈ {4, 8, 12, 16}** at current data to find the rank knee, before any further structural changes.

The decision here shapes 03-03's perturbation analysis: if Q doesn't faithfully linearise the task-relevant subspace, Q-mapped rank-one perturbations won't predict behavioral changes cleanly.

## Next Phase Readiness

**Phase 03-03 (Perturbation Analysis) — ready-with-caveats:**
- ✓ `best_latent_circuit.pt` loads cleanly into LatentNet(n=16, N=64, input_size=7, output_size=3)
- ✓ Inferred Q, w_rec, w_in, w_out all accessible for perturbation
- ✓ ContinuousActorCritic checkpoint available for "map latent perturbation back to RNN" step
- ⚠ Invariant subspace soft-fail (0.703) means Q is an imperfect map of RNN connectivity. Perturbations in latent space may not translate cleanly to behavioral deltas. Plan 03-03 should either (a) improve Q first (via the T-structure investigation above) or (b) proceed and report perturbation sensitivity as an empirical observation.
- ⚠ n_latent=16 vs plan's 8 — potentially inflates reconstruction metrics; sweep recommended.

**New cluster asset:** `cluster/run_circuit_ensemble.sh` is parameterised and reusable. 03-03 perturbation sweeps could run on the same SLURM infrastructure.

---
*Phase: 03-latent-circuit-inference*
*Plan 02 completed: 2026-04-24 (benchmark run 2026-04-24, summary written post-hoc from git history + validation artifacts)*
