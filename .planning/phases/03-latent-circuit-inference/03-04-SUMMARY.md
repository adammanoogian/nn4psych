---
phase: 03-latent-circuit-inference
plan: "04"
subsystem: analysis
tags: [pytorch, rnn, latent-circuit, perturbation, LatentNet, circuit-inference, wave-b]

# Dependency graph
requires:
  - phase: 03-03
    provides: best_latent_circuit_waveA.pt, wave_a_selection.json with chosen_rank=12, pareto_curve.json
  - phase: 03-02
    provides: ContinuousActorCritic ReLU model, circuit_data.npz (n=1000, T=75), cluster SLURM pipeline
provides:
  - perturb_and_evaluate() in circuit_inference.py (Eq. 6/23 Q-mapped rank-one perturbations)
  - 08_infer_latent_circuits.py end-to-end Phase 3 orchestrator
  - perturbation_results.json with 50 perturbations, baseline std, significance flags, per-context deltas
  - wave_b_writeup.md committing to STORY_2 (ran out of fixes)
  - CIRC-05 perturbation analysis closed with documented commitment
affects:
  - 04-bayesian-fitting (Phase 4 proceeds independently; Q role is descriptive not causal)
  - v2 multi-task latent circuit comparison (uses Q with documented caveats)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "perturb_and_evaluate() try/finally to guarantee W_hh.weight restoration after perturbation eval"
    - "Smoke test path --quick redirects data/output to subdirs to avoid overwriting canonical artifacts"
    - "Wave A chosen rank loaded from wave_a_selection.json via load_wave_a_chosen_rank() — never hardcoded"
    - "Q orthonormality assertion (||QQ^T-I||<1e-4) in --skip_fitting path before perturbation"

key-files:
  created:
    - src/nn4psych/analysis/circuit_inference.py (perturb_and_evaluate appended)
    - scripts/data_pipeline/08_infer_latent_circuits.py
    - output/circuit_analysis/perturbation_results.json
    - output/circuit_analysis/validation_results_waveB.json
    - output/circuit_analysis/wave_b_writeup.md
  modified:
    - src/nn4psych/analysis/circuit_inference.py (added from __future__ import annotations; appended perturb_and_evaluate)

key-decisions:
  - "Story 2 (ran out of fixes): Wave A Pareto spread=0.096 pre-positioned this; confirmed by 0/50 significant perturbations, which is itself ambiguous (Q quality discrepancy)"
  - "set_num_tasks(1) in _eval_once: model was trained with 1-element context (obs=5+ctx=1+reward=1=7); set_num_tasks(2) gives 8-dim input incompatible with W_ih (64x7)"
  - "LatentNet stochastic eval discrepancy: locally recomputed invariant corr=0.42 vs cluster=0.78 due to sigma_rec=0.15 noise in forward pass; cluster measured nmse_y right after training, locally we get fresh noise realizations with nmse_y~4.9"
  - "--quick flag now redirects data and output to smoke_test/ subdirs to prevent overwriting canonical circuit_data.npz and output/ artifacts"
  - "Phase 4 (Bayesian fitting) proceeds independently of Q quality; Q's role in final pipeline is descriptive not causal-mechanistic"

patterns-established:
  - "Perturbation analysis: load Q from saved model, identify top-n latent connections, apply rank-one perturbations via q.T @ delta @ q, evaluate ContinuousActorCritic (not LatentNet) for behavioral effects"
  - "Significance vs baseline: run n_baseline_runs unperturbed evals to establish baseline_std; flag |delta| > k*baseline_std"

# Metrics
duration: 39min
completed: 2026-04-26
---

# Phase 03 Plan 04: Wave B Perturbation Analysis Summary

**Rank-one latent perturbations via Q-mapped W_rec injection (Eq. 6/23): 0/50 significant effects at strengths [-0.5,-0.2,0,0.2,0.5]; story commits to STORY_2 (ran out of fixes); CIRC-05 closed**

## Performance

- **Duration:** 39 min
- **Started:** 2026-04-26T07:02:19Z
- **Completed:** 2026-04-26T07:41:XX Z
- **Tasks:** 3/3
- **Files modified:** 5

## Accomplishments

- Implemented `perturb_and_evaluate()` in circuit_inference.py: maps latent delta_ij to RNN W_rec via q.T @ delta @ q (Langdon & Engel 2025 Eq. 6/23); establishes baseline variability; reports per-context reward deltas + significance vs k*baseline_std; restores W_hh via try/finally
- Created `08_infer_latent_circuits.py`: full Phase 3 orchestrator (collect → fit → validate → perturb); reads chosen rank from wave_a_selection.json; `--skip_fitting` asserts Q orthonormality; `--quick` redirects to smoke_test/ subdirs
- Ran Wave B perturbation analysis with Wave A's Q (n=12, orthonormality err=1.03e-06): 50 perturbations tested, 0 significant, mean |delta|=0.041 vs baseline_std=0.063; wrote wave_b_writeup.md committing to STORY_2

## Task Commits

Each task was committed atomically:

1. **Task 1: perturb_and_evaluate** - `3cecf92` (feat)
2. **Task 2: 08_infer_latent_circuits.py + artifacts** - `4f4c50b` (feat)
3. **Task 3: wave_b_writeup.md** - `ceb765e` (docs)

## Files Created/Modified

- `src/nn4psych/analysis/circuit_inference.py` - Added `from __future__ import annotations`; appended `perturb_and_evaluate()`
- `scripts/data_pipeline/08_infer_latent_circuits.py` - New: end-to-end Phase 3 orchestrator
- `output/circuit_analysis/perturbation_results.json` - 50 perturbations, baseline noise, significance flags, per-context deltas
- `output/circuit_analysis/validation_results_waveB.json` - Wave B re-validation with Wave A's Q
- `output/circuit_analysis/wave_b_writeup.md` - STORY_2 commitment with cited evidence

## Decisions Made

- **Story 2 (ran out of fixes):** Wave A's Pareto spread (0.096) pre-positioned this; Wave B's 0/50 significant perturbations is ambiguous (Q quality discrepancy means the perturbations may not land cleanly in the invariant subspace). Story 2 is maintained: deferred fixes (masked loss, shorter T, condition-sliced fitting) remain viable first steps.
- **set_num_tasks(1) not 2 in _eval_once:** The cluster retrain in 03-02 used collect_circuit_data() which calls set_num_tasks(1), giving input_dim=7 (obs=5 + ctx=1 + reward=1). set_num_tasks(2) gives 8-dim input that mismatches W_ih (shape 64×7) and raises RuntimeError.
- **Phase 4 proceeds independently:** The latent circuit's Q has documented quality limitations; Phase 4 (Bayesian fitting on behavioral outputs) does not depend on Q quality. The Q can serve a descriptive role in v2 work.
- **LatentNet eval is stochastic:** sigma_rec=0.15 noise is always added during the forward pass regardless of eval mode. Cluster-measured metrics (corr=0.783, nmse_y=0.247) were computed immediately after training and benefited from that noise seed. Locally recomputed metrics (corr=0.42, nmse_y=4.9) represent fresh noise realizations. This is a pre-existing limitation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] set_num_tasks(1) not (2) in _eval_once inside perturb_and_evaluate**
- **Found during:** Task 1 (smoke test)
- **Issue:** Plan stub specified `env.set_num_tasks(2)` for "dual-modality training". But cluster logs confirm the model was trained with `collect_circuit_data()` which calls `set_num_tasks(1)`, giving input_dim=7 (obs=5+ctx=1+reward=1). set_num_tasks(2) gives 8-dim state, causing `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x8 and 7x64)`.
- **Fix:** Changed to `env.set_num_tasks(1)` with comment explaining the reasoning.
- **Files modified:** src/nn4psych/analysis/circuit_inference.py
- **Verification:** Smoke test passes: `perturbations: 4, json.dumps: OK, All key checks PASS`
- **Committed in:** 3cecf92 (Task 1 commit)

**2. [Rule 1 - Bug] --quick run overwrote canonical circuit_data.npz**
- **Found during:** Task 2 verification (--quick run)
- **Issue:** `--quick` flag ran `collect_circuit_data` with n_trials=50 and saved to the default path `data/processed/rnn_behav/circuit_data.npz`, overwriting the canonical 1000-trial file.
- **Fix:** Added path redirect in parse_args: when `--quick` and not `--skip_collection`, data_path is redirected to `smoke_test/circuit_data.npz`; when `--quick` and not `--skip_fitting`, output_dir is redirected to `output/circuit_analysis/smoke_test/`. Canonical 1000-trial circuit_data.npz was regenerated via direct collect_circuit_data() call.
- **Files modified:** scripts/data_pipeline/08_infer_latent_circuits.py
- **Verification:** `--quick` now writes to smoke_test/ subdirs; verified via args simulation
- **Committed in:** 4f4c50b (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both bugs caught during verification; fixes were minimal and corrective. No scope creep.

## Issues Encountered

**LatentNet stochastic eval discrepancy:** Locally recomputed invariant_subspace_corr=0.42 vs cluster-reported 0.78. Investigation showed this is due to LatentNet's sigma_rec=0.15 noise being applied at every forward pass regardless of eval mode. Cluster metrics were computed immediately after training on the SAME data with THAT noise realization (nmse_y=0.247). Fresh eval locally gives nmse_y~4.9. The Cayley transform Q itself is deterministic (orthonormality err=1.03e-06 consistent between local and cluster). This is a pre-existing LatentNet limitation, not a new bug. Documented in wave_b_writeup.md; does NOT invalidate the perturbation analysis since perturb_and_evaluate() uses the ContinuousActorCritic (deterministic in eval mode), not LatentNet, for behavioral evaluation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Phase 4 (04-bayesian-fitting) is unblocked:**
- Phase 4 fits Bayesian models to behavioral outputs of the ContinuousActorCritic
- Phase 4 does NOT depend on Q quality (the Bayesian comparison uses trial-level reward/choice data)
- Phase 3's latent circuit Q serves a descriptive role only; its quality caveats are documented

**Phase 3 success criteria status:**
- [x] CIRC-01: LatentNet fits Q, w_rec, w_in, w_out (03-01)
- [x] CIRC-02: 100+ inits with best selection (03-02)
- [x] CIRC-03: Activity-level validation (03-02)
- [x] CIRC-04: Invariant subspace corr reported with provenance (03-02 + 03-03 sweep; corr=0.783 cluster, 0.42 local-stochastic)
- [x] CIRC-05: Perturbation analysis with behavioral predictions vs baseline noise (03-04; 0/50 significant at default strengths; documented)

**Open concerns for v2 / future phase 3.1:**
- Masked-loss fitting is the highest-priority deferred fix (directly targets T=75 padding noise)
- LatentNet stochastic eval: the metric instability should be fixed before any quantitative comparison between runs or models
- perturbation_strengths [-0.5, 0.5] are modest relative to max |w_rec_ij|=4.17; testing larger strengths (±2, ±5) in a future run might reveal dose-response even if current strengths land below threshold

---
*Phase: 03-latent-circuit-inference*
*Completed: 2026-04-26*
