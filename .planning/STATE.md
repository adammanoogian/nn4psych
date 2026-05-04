# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** RNN agent trainable on multiple cognitive tasks with analyzable hidden representations comparable to human data via Bayesian model fitting
**Current focus:** Phase 4 (Bayesian Model Fitting / Nassar 2021). User pivot 2026-05-04: 04-04a re-scoped from "K=20 SLURM array re-training" to "axis-all cohort manifest from existing 1,884 checkpoints" (Kumar et al. 2025 CCN per-axis design). 04-04a complete; 04-04b ready to start. 04-03 still blocked on Brain2021Code download. Phase 3.1 closure (03-07/08) still deferred per 2026-04-29 user pivot.

## Current Position

Phase: 4 of 5 (Bayesian Model Fitting / Nassar 2021) — In progress
Plan: 04-03a COMPLETE (MATLAB parity + human data cleaning). Next options: (a) 04-03 (Human Data Fits — UNBLOCKED, forward model now validated); (b) 04-04b (replay-and-fit RBO over the 1,884-cohort, also unblocked); (c) BAYES-06 re-queue (50-dataset param recovery with corrected model).
Status: Phase 4 waves 1, 2, 3a, 4 complete. Wave 3 (04-03) now unblocked: Brain2021Code data cleaned, compute_rbo_forward math validated against MATLAB. BAYES-06 still open (need re-queue).
Last activity: 2026-05-04 — Ran human data cleaning pipeline (134 subjects, 55,992 valid trials). Fixed 3 math bugs in compute_rbo_forward: U_val constant, tot_sig=sigma_N/sqrt(1-tau), MATLAB ss R-update. All 7 tests pass. 3 task commits (d1014ac, ec8d151, 648e54c).

Progress: [██████████] ~95% (04-01/02/03a/04a done; 04-03/04b runnable; 04-05 user-gated optional)

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: ~14 min
- Total execution time: ~2.0 hours code + ~2 hours batch compute

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure-and-organization | 3/3 COMPLETE | ~24 min | ~8 min |
| 02-rnn-training-verification | 3/3 COMPLETE | ~42 min | ~14 min |
| 03-latent-circuit-inference | 4/4 COMPLETE | ~320 min compute + 5 weeks iteration | ~80 min |
| 04-bayesian-model-fitting | 3/6 in progress | ~147 min (code+behavior extract) + 33 min smoke | ~49 min |

**Recent Trend:**
- Last 9 plans: 01-01 (9 min), 01-02 (unknown), 01-03 (7 min), 02-01 (12 min), 02-02 (15 min), 02-03 (15 min), 03-01 (75 min), 03-02 (205 min GPU ensemble + ~5 weeks iteration), 03-03 (~?), 03-04 (39 min)
- Trend: 03-04 executed cleanly in 39 min; 2 Rule 1 bugs auto-fixed (input_dim mismatch, --quick path overwriting data)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: NumPyro/JAX chosen over PyMC for Bayesian models (JAX faster, composable)
- [Init]: PyEM models to be archived, not maintained
- [Init]: Latent circuit inference (Langdon & Engel 2025) chosen over standard dim reduction
- [Init]: Context-DM task required — primary task for latent circuit analysis
- [01-01]: compare_contexts() now accepts pre-computed negll floats — caller computes negll from own model
- [01-01]: cross_validate_k_fold() removed — was PyEM-only; NumPyro CV deferred to Phase 4
- [01-01]: batch_fit_bayesian.py not rewritten; TODO comment added; Phase 4 work
- [01-01]: Original bayesian/ root NOT deleted — Plan 01-02 handles archiving
- [01-03]: jax.lax.cond with operand=None and lambda _: pattern for closed-over variable branches
- [01-03]: is_changepoint = jnp.bool_(context == 'changepoint') outside step_fn — JAX tracer-compatible
- [01-03]: NeurogymWrapper.reset_epoch() resets self.trial = 0 plus all history lists
- [01-03]: env_params defaults to {} inside batch_extract_behavior (not as default arg) to avoid mutable default anti-pattern
- [02-01]: evaluate() method also needs hasattr guard — 4 sites total not 3; auto-fixed as Rule 1 bug
- [02-01]: obs_dim for ContextDecisionMaking is 1 + 2*dim_ring (not 1 + dim_ring); fixation + 2 modalities x ring_units
- [02-01]: neurogym installed in actinf-py-scripts conda env (v2.2.0); is the project's working Python environment
- [02-02]: extract_behavior_with_hidden uses argmax (deterministic) not sampling — consistent with extract_behavior(); reproducible for analysis
- [02-02]: NaN padding for hidden states (not zeros) so downstream masking can identify real zero activations vs padded timesteps
- [02-02]: DawTwoStep NeurogymWrapper produces max_T~1000 (long trials at dt=100ms) — may need trial length filtering before latent circuit fitting in Phase 3
- [02-03]: ContextDecisionMaking trials never signal done=True — all run to max_steps_per_trial (1000); hidden array has no NaN padding when all trial lengths equal
- [02-03]: Detach critic buffer values via .item() before GAE advantage computation — prevents RuntimeError: backward through graph a second time
- [02-03]: Detach hx after optimizer.step() when rollout buffer is smaller than trial length — required for RNNs with mid-trial updates
- [02-03]: Cast numpy int64 shape values to int() before json.dump() — numpy integers are not JSON serializable
- [03-01]: Vendor latent_net.py rather than pip-install from local path (no pyproject.toml in engellab/latentcircuit)
- [03-01]: z = raw actor logits (not softmax) as LatentNet target output — output_size=3
- [03-01]: ContextDecisionMaking accuracy must be measured by cumulative trial reward, NOT last action (done never fires)
- [03-01]: T=500 in circuit_data.npz — trials run to max_steps with all 500 timesteps captured (fixation+stim+delay+decision within 500 steps)
- [03-01]: Dual-modality training requires 50+ epochs (not the 10x20 smoke constraint) to achieve >55% accuracy
- [03-01]: n_trials=40 (20 per context) in circuit_data.npz per CRITICAL MEMORY CONSTRAINTS — less than batch_size=128 means 1 partial batch per fitting epoch
- [03-02]: Data regenerated on cluster — n_trials=40→1000 (500/context), T=500→75; addresses 03-01 batch-size and padding concerns
- [03-02]: RNN switched from tanh ActorCritic to ReLU ContinuousActorCritic for LatentNet compatibility (LatentNet assumes ReLU dynamics both sides)
- [03-02]: z = softmax(logits) not raw logits — mse_z drops from ~30 to ~0.3, balances with nmse_y at l_y=1.0 per paper
- [03-02]: n_latent=16 used for benchmark run (plan spec & cluster default: 8) — flagged as parameter to sweep in 03-03
- [03-02]: Cluster SLURM pipeline established (cluster/run_circuit_ensemble.sh, setup_env.sh, .regen.lock) — reusable for 03-03 perturbation sweeps
- [03-02]: Invariant subspace corr=0.703 < 0.85 threshold → soft-fail accepted per plan; carried into 03-03 as open structural-validity concern
- [03-02]: GPU device bug fixed — q must be recomputed on correct device at start of fit()/forward() (commit 973d50f)
- [03-03]: Wave A picked n_latent=12 (corr=0.7833 cluster, Pareto spread=0.096); recommended_story="ran_out_of_fixes"
- [03-04]: STORY_2 committed (ran out of fixes): Wave A Pareto spread pre-positioned; 0/50 significant perturbations at strengths [-0.5,0.5] is ambiguous (Q quality discrepancy — local corr=0.42 vs cluster=0.78 due to LatentNet stochastic eval)
- [03-04]: set_num_tasks(1) in _eval_once — model trained with obs=5+ctx=1+reward=1=input_dim=7; set_num_tasks(2) gives 8-dim input, mismatches W_ih (64x7)
- [03-04]: LatentNet sigma_rec=0.15 noise always active — cluster metrics (corr=0.78, nmse_y=0.247) were single-seed measurements right after training; fresh eval locally gives corr=0.42, nmse_y=4.9; stochastic eval is a pre-existing LatentNet limitation
- [03-04]: --quick flag now redirects data/output to smoke_test/ subdirs (prevents overwriting canonical circuit_data.npz and output artifacts)
- [03-04]: Phase 4 proceeds independently; Q's role in final pipeline is descriptive not causal-mechanistic
- [03-05]: Fixation+delay excluded from task_active_mask (sensorimotor-only; discrimination signal only in stimulus+decision periods)
- [03-05]: task_active_mask=None in fit_latent_circuit_ensemble is bit-identical to pre-03-05 Wave A behavior (verified smoke test)
- [03-05]: circuit_data.npz committed to git (un-ignored) so cluster can git pull mask-augmented file; fallback regen on cluster would produce file without task_active_mask key
- [03-05]: cluster/run_circuit_ensemble.sh MASKED=1 assertion raises KeyError if task_active_mask missing; prevents silent full-T fitting on stale checkout
- [03-05]: nmse_y_full always reported alongside masked nmse_y for Wave A cross-comparability
- [03-05]: mini-batch masked loop (batch_size=128, connectivity_masks() after each batch) mirrors LatentNet.fit() to preserve training dynamics
- [03-06]: sigma_rec=0.15 default (no eval-mode override) for per-context fits — cluster_same_seed_as_train matches Wave A pooled baseline for direct corr comparison
- [03-06]: circuit_data.npz sliced READ-ONLY by modality_context — no writes, no race condition with 03-05's task_active_mask
- [03-06]: afterany (not afterok) for autopush — push fires even if one context fit fails
- [03-06 Task 2]: Conclusion=AMBIGUOUS (by exhaustion): both per-context corrs LOWER than pooled by 0.12-0.14 — structural-separation hypothesis ruled out; Q quality cap is method/data-bound; converges with 03-05 STORY_1 evidence
- [03-05 Task 3]: Masked-loss corr=0.5699 at n=12 WORSE than Wave A 0.7833 (delta=-0.2134) — padding hypothesis ruled out; story direction for 03-08 is STORY_1 (method/data limit)
- [03-05 Task 3]: Negative delta (masked < full) is decisive regardless of Pareto spread (0.14) — masking hurt corr, direction is the signal
- [03-05 Task 3]: 03-07 still runs (crossed_85=false) but negative delta is strong prior shorter T won't help either
- [04-01]: All five RBO priors are FALLBACK weakly-informative defaults pending Nassar 2021 supplement; prior verification gated on 04-03 Task 1 Brain2021Code download
- [04-01]: Predictive NOT re-exported from nn4psych.bayesian (m9); downstream callers import from numpyro.infer directly
- [04-01]: run_mcmc arg order is bag_positions first, bucket_positions second (semantic clarity; differs from old numpyro_models.py)
- [04-01]: compute_rbo_forward uses full predictive-variance-weighted tau update (numpyro_models.py lines 133-138), not simplified tau/UU from metrics.py; flagged for 04-02 validation
- [04-01]: numpyro_models.py deprecated (DeprecationWarning at import) but retained for git-history continuity
- [04-01]: arviz pinned >=0.17.0,<0.25.0 in pyproject.toml (RESEARCH.md Open Question 5)
- [04-01]: XLA_FLAGS set in __init__.py via os.environ.setdefault() before any jax import; 4 virtual CPU devices verified
- [04-02]: stat_focus='stats'/'diagnostics' is the correct ArviZ 0.23.4 kwarg in az.summary (not kind=); validated at runtime (M3 fix)
- [04-02]: NumPyro get_extra_fields() always returns {'diverging': ...} in this version regardless of extra_fields kwarg — divergence access is always available
- [04-02]: Per-condition recovery design (CP+OB fits averaged) matches actual human fitting pipeline — correct identifiability validation
- [04-02]: Smoke N=4 r values are not formal BAYES-06 evidence; full 50-dataset run (bcaxsbh0c) is the gate
- [04-02]: Smoke MCMC settings (200 warmup) cause all fits to fail convergence gates + retry; full run (2000/4000 warmup) required for proper convergence
- [04-02]: nn4psych.bayesian scripts need both src/ AND project root on sys.path — src/ for package discovery, project root for envs.PIE_CP_OB_v2 in nn4psych.__init__
- [04-03a]: U_val in compute_rbo_forward must be constant (1/BAG_RANGE)^LW, not uniform.pdf(delta). MATLAB changLike = 1/300 because bag positions are always in [0,300]; pred_errors (bag-bucket) can be negative where uniform.pdf returns 0.
- [04-03a]: tot_sig = sigma_N/sqrt(1-tau) (MATLAB totSig = sqrt(sigmaE^2+sigmaU^2)); previous code used sigma_N/tau which was off by a sqrt factor. At tau=0.5: correct=28.28, previous=40.0.
- [04-03a]: Tau update now uses MATLAB second-moment ss formula (cp and ob variants via lax.cond). CP first term sigmaE^2/1; OB first term sigmaE^2/R. Residual: CP = -(1-tau)*delta; OB = tau*delta.
- [04-03a]: Accepted deviation: truncated-normal pI normalization (~3.7% max CP omega, ~0.12% OB) cannot be reproduced without absolute bucket position. Parity test tolerance = 5e-2, not 1e-3.
- [04-03a]: Human data cleaning: 134 subjects (102 patients, 32 NC), 55,992 valid trials (2.8% excluded by first-3-per-block rule). data/processed/nassar2021/ committed to git (gitignore exception added).
- [04-04a 2026-05-04]: PIVOT — discarded original K=20 SLURM-array re-training. Re-training homogeneous seeds collapses the schizophrenia-spectrum comparison; the existing 1,884 trained checkpoints in trained_models/checkpoints/model_params_101000/ already span the Kumar et al. 2025 CCN per-axis design (γ, p_reset, τ, β_δ × 50 seeds), making them the right cohort for 04-04b's RBO fitting. No cluster compute used.
- [04-04a]: 5-hyperparameter design in trained_models — paper publishes 4 axes (γ, β_δ, p_reset, τ) but the data has a 5th `td_penalty` axis swept around 0.0. Empirically td_penalty has no detectable effect on ΔArea; included in cohort tagged `tdpenalty` (supplementary).
- [04-04a]: Canonical config (γ=0.95, p_reset=0, τ=100, β_δ=1, td_penalty=0) has 50 unique seeds (0–49) — this is the "control"-region for the schizophrenia-spectrum projection in 04-04b/05.
- [04-04a]: ΔArea-by-γ replicates Kumar Fig. 2A almost exactly (monotonic 0.1→0.9, peak γ∈[0.8, 0.9], dip at γ=0.99). p_reset panel deviates qualitatively (paper: monotonic decrease; ours: roughly flat with high noise) — flagged for follow-up if 04-04b's RBO-vs-p_reset projection looks weird; not blocking.
- [04-04a]: --reclassify_only flag rebuilds manifest+figure from cached parquet without re-extracting behavior; saves ~80 min on classifier-logic iterations.
- [04-04a]: cohort_manifest.json schema_version 1.0 is the consumer schema for 04-04b. Per-checkpoint POSIX paths; checkpoint_metrics.parquet stays gitignored (regenerable cache).
- [04-04a]: Stale K=20 SLURM commits (ec02730, 25b481f) dropped via `git stash → reset --hard HEAD~2 → stash pop`; 16 legitimate phase-4 commits pushed to origin/main on 2026-05-04 (origin had been 16 behind; cluster's git-up-to-date claim implied a pre-Phase 4 state).

### Pending Todos

5 pending:
- `explore-sac-continuous-actions` (v2/exploratory)
- `multitask-latent-circuit-comparison` (v2/analysis — cross-task Q comparison + fixed points in latent space)
- `subset-q-fitting-changepoint-oddball` (v2/analysis — fit Q to trial subsets around task events)
- `gpu-performance-bottlenecks-latentnet` (optimization — torch.compile, kernel fusion, Cayley frequency)
- `plan-cluster-gpu-adoption-for-latent-fitting` (planning — breakeven math + when to fold cluster GPU into a phase)

### Blockers/Concerns

- [RESOLVED 01-03]: JAX tracing bug in numpyro_models.py — FIXED with jax.lax.cond
- [RESOLVED 01-03]: extract_behavior private env API — FIXED with reset_epoch() public method
- [RESOLVED 03-01]: Latent circuit rank selection — start with n=8 (Tutorial default), validate with invariant subspace correlation check
- [RESOLVED 03-01]: Context-DM trial length — T=500 (runs to max_steps); LatentNet handles arbitrary T
- [RESOLVED 03-02]: n_trials=40 < batch_size=128 — FIXED by regen at n_trials=1000 (500/context)
- [RESOLVED 03-02]: T=500 with ~5% task-relevant — PARTIALLY FIXED by T=75 regen (now ~20-40% task-relevant, not full fix)
- [RESOLVED 03-04]: n_latent sweep completed (03-03); rank n=12 selected as best of tried; STORY_2 committed; CIRC-05 closed (with caveats per writeup)
- [RESOLVED 2026-04-29 — Gap 1, priority 1 — 03-05]: Masked-loss sweep COMPLETE. chosen_rank=12, corr=0.5699, crossed_85=false, delta_vs_wave_a=-0.2134. Padding hypothesis ruled out. wave_a_masked_selection.json written. Story tilts to STORY_1 (method/data limit).
- [RESOLVED 2026-04-29 — Gap 3, priority 3 / diagnostic — 03-06]: Per-context fitting COMPLETE. ctx-0 corr=0.6628, ctx-1 corr=0.6406, both BELOW pooled 0.7833 (deltas -0.1205 and -0.1427). Conclusion=AMBIGUOUS. Structural-separation hypothesis ruled out. per_context_results.json written.
- [Phase 3.1 DEFERRED — 03-07]: Shorter T regen (T≈30 with delay=0). Skipped on 2026-04-29 by user pivot — 03-05 negative delta (-0.21) is stronger refutation of padding hypothesis than 03-07's binary skip rule anticipated; 03-07 would replicate same answer at higher cluster cost. Plan committed (01d2fdf) preserved for v2 future work.
- [Phase 3.1 DEFERRED — 03-08]: Phase 3.1 closure writeup (STORY_1 method/data limit). Skipped on 2026-04-29 by user pivot to Phase 4. Will be picked up after Phase 4 ships — STORY_1 commitment is well-supported by Wave 5 evidence already on disk; the writeup is a documentation task, not blocking science.
- [Phase 3.1 OPEN — confound]: LatentNet stochastic eval — sigma_rec noise always active; cluster/local metric discrepancy is a pre-existing limitation; should fix or pin a single eval seed before quantitative comparisons between runs (Gap 4 candidate?).
- [Phase 3.1 OPEN — diagnostic]: Perturbation strengths [-0.5, 0.5] are modest relative to max |w_rec_ij|=4.17; if Q quality is fixed, re-running with stronger strengths may give cleaner SC-4 evidence.
- [RESOLVED 2026-05-04 — Phase 4 planning]: Nassar 2021 .mat file structure confirmed working; extract_nassar_trials.py successfully loaded 134 subjects' statusData with blockCompletedTrials, currentOutcome, currentPrediction, currentUpdate, currentDelta, currentMean, isChangeTrial fields.
- [OPEN — 04-02]: Full 50-dataset param recovery run from 2026-04-30 (background_task_id=bcaxsbh0c) appears to have failed silently or run elsewhere — no recovery_report.json on disk, .bg-shell/manifest.json empty, only smoke synth_000-003 fits present (~7 of 8). BAYES-06 NOT YET CLOSED. 04-03 Task 2 gates on recovery_report.json with all r >= 0.85; will need to re-queue (likely on cluster given 16GB RAM constraint here).
- [OPEN — 04-02 UPDATED]: Smoke convergence behavior — 200 warmup insufficient. ADDITIONALLY: tau update equation in compute_rbo_forward was wrong (RESEARCH.md Pitfall 1 confirmed). Now fixed in 04-03a. Re-queue with corrected model.
- [RESOLVED 2026-05-04 — 04-03 prereq]: Brain2021Code cleaned. 134 subjects loaded; data/processed/nassar2021/ committed. 04-03 now unblocked.
- [RESOLVED 2026-05-04 — 04-04a]: Cohort manifest built. 1,884 PIE_CP_OB_v2 checkpoints inventoried, ΔArea computed, manifest written. Kumar Fig. 2A replicated. cohort_manifest.json + figure committed (95fe5ca).
- [OPEN — 04-04a deviations]: p_reset panel doesn't match Kumar Fig. 2C monotonic decrease (we see flat noise). Possible env-param drift or RNG-seed difference; non-blocking but flag for 04-04b verification.
- [OPEN — 04-03a accepted deviation]: truncated-normal pI normalization not applied in compute_rbo_forward (needs absolute bucket position, not available in pred_error space). ~3.7% max CP omega offset. Consider refactoring to accept bag+bucket separately if BAYES-06 r < 0.85 after re-run.

## Session Continuity

Last session: 2026-05-04T17:33Z
Stopped at: Completed 04-03a-PLAN.md. Cleaned 134-subject Brain2021Code data (55,992 valid trials). Fixed 3 math bugs in compute_rbo_forward (U_val constant, tot_sig formula, MATLAB ss R-update). All 7 RBO tests pass. 3 task commits + 1 docs commit (d1014ac, ec8d151, 648e54c, + docs).
Resume options:
1. **04-03** (Human Data Fits) — UNBLOCKED. data/processed/nassar2021/ ready, forward model validated. Still gated on BAYES-06 for Task 2.
2. **BAYES-06 re-queue** — re-run 50-dataset param recovery with corrected compute_rbo_forward. Likely needs cluster (16GB RAM constraint).
3. **04-04b** (replay-and-fit RBO over the 1,884-cohort) — runnable now; same forward model fixes apply.
Resume file: None
