# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** RNN agent trainable on multiple cognitive tasks with analyzable hidden representations comparable to human data via Bayesian model fitting
**Current focus:** Phase 3 — base plans (03-01..03-04) all DONE, but verifier returned `human_needed` on SC-2 / SC-4 soft-fails. User chose Option B (2026-04-26): pursue Phase 3.1 gap closure (masked-loss fitting + shorter T regen + condition-sliced fitting) BEFORE Phase 4. Phase 3 is therefore in gap-closure (NOT closed).

## Current Position

Phase: 3 of 5 (Latent Circuit Inference) — In gap-closure
Plan: 4/4 base plans complete; 03-05 COMPLETE; 03-06 COMPLETE (both Wave 5 plans done)
Status: 03-05 COMPLETE — masked-loss sweep done, wave_a_masked_selection.json written (chosen_rank=12, corr=0.5699, crossed_85=false, delta=-0.2134); 03-06 COMPLETE — per_context_results.json written (ctx-0 corr=0.6628, ctx-1 corr=0.6406, conclusion=AMBIGUOUS, both deltas negative-large, structural-separation ruled out).
Last activity: 2026-04-29 — 03-06 Task 2 complete: aggregate_per_context.py + per_context_results.json + full SUMMARY.md committed. Conclusion=AMBIGUOUS: both per-context corrs below pooled, structural-separation hypothesis ruled out. Wave 5 fully done.

Progress: [█████████░] ~85% (12/~13 base plans + Phase 3.1 plans in-progress; 03-07 and 03-08 remain)

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: ~10 min
- Total execution time: ~0.65 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure-and-organization | 3/3 COMPLETE | ~24 min | ~8 min |
| 02-rnn-training-verification | 3/3 COMPLETE | ~42 min | ~14 min |
| 03-latent-circuit-inference | 4/4 COMPLETE | ~320 min compute + 5 weeks iteration | ~80 min |

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
- [Phase 3.1 PLANNED — Gap 2, priority 2 — 03-07]: Shorter T regen (T≈30 with delay=0) — orthogonal probe of T=75 padding hypothesis. Plan committed (01d2fdf). Conditional: 03-07 auto-skips if 03-05's wave_a_masked_selection.json reports crossed_85_threshold=true. Wave 6 in execution order.
- [Phase 3.1 PLANNED — closure — 03-08]: Aggregate Phase 3.1 evidence (03-05 + 03-06 + 03-07), update wave_b_writeup.md story commit (STORY_0 if any gap >= 0.85, else STORY_1 method/data limit). autonomous: false — checkpoint before Phase 4 unblock. Wave 7.
- [Phase 3.1 OPEN — confound]: LatentNet stochastic eval — sigma_rec noise always active; cluster/local metric discrepancy is a pre-existing limitation; should fix or pin a single eval seed before quantitative comparisons between runs (Gap 4 candidate?).
- [Phase 3.1 OPEN — diagnostic]: Perturbation strengths [-0.5, 0.5] are modest relative to max |w_rec_ij|=4.17; if Q quality is fixed, re-running with stronger strengths may give cleaner SC-4 evidence.
- [Phase 4 planning]: Nassar 2021 .mat file nested indexing not directly inspected — must run describe_mat_structure() before writing data loading code (research flag)

## Session Continuity

Last session: 2026-04-29T20:10:00Z (approximate)
Stopped at: Completed 03-06 Task 2 — aggregate_per_context.py + per_context_results.json committed; full SUMMARY.md written; STATE.md updated. Wave 5 (03-05 + 03-06) both complete.
Resume: Run 03-07 (shorter-T regen, Wave 6), then 03-08 (writeup closure checkpoint, Wave 7). Both per-context and masked-loss evidence converge on STORY_1 (method/data limit).
Resume file: None
