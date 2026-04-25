# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** RNN agent trainable on multiple cognitive tasks with analyzable hidden representations comparable to human data via Bayesian model fitting
**Current focus:** Phase 3 — Latent Circuit Inference (03-01 + 03-02 complete; soft-fail on invariant subspace carried into 03-03)

## Current Position

Phase: 3 of 5 (Latent Circuit Inference) — In progress
Plan: 2 of 3 in phase 03 complete (03-01 + 03-02 done)
Status: In progress — Phase 3, Plan 2 complete with documented soft-fail
Last activity: 2026-04-24 — Completed 03-02 (100-init GPU ensemble on cluster; invariant subspace corr=0.703 soft-fails; trial-avg R²=0.98)

Progress: [████████░░] ~62% (8/~13 total plans)

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
| 03-latent-circuit-inference | 2/3 | ~280 min compute + 5 weeks iteration | ~140 min |

**Recent Trend:**
- Last 8 plans: 01-01 (9 min), 01-02 (unknown), 01-03 (7 min), 02-01 (12 min), 02-02 (15 min), 02-03 (15 min), 03-01 (75 min), 03-02 (205 min GPU ensemble + ~5 weeks iteration)
- Trend: 03-02 was substantially larger than plan scope — 11 commits including RNN rearch (tanh→ReLU), data regen, z reformulation, GPU perf, cluster SLURM pipeline. Summary written post-hoc from artifacts.

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
- [Phase 03-03 open]: Invariant subspace corr=0.703 < 0.85 — Q only partially linearises RNN connectivity. May contaminate perturbation analysis. Candidate causes: n_latent=16 over-parameterised, remaining task-structure/padding noise at T=75, or method limit at current data quality.
- [Phase 03-03 open]: T=75 still has ~45-60 steps of blank/padding beyond task-active window (~15-30 steps at dt=100ms for ContextDecisionMaking). User flagged as concern — investigate whether this is noise affecting fit quality. Options: mask loss to task-active timesteps, shorten T, sliced fitting, or n_latent sweep.
- [Phase 03-03 open]: n_latent=16 benchmark vs plan's 8 — sweep n_latent ∈ {4, 8, 12, 16} to find knee before committing to 16 for perturbation analysis.
- [Phase 4 planning]: Nassar 2021 .mat file nested indexing not directly inspected — must run describe_mat_structure() before writing data loading code (research flag)

## Session Continuity

Last session: 2026-04-24
Stopped at: Completed 03-02-PLAN.md — Phase 3 Plan 2 done (100-init GPU ensemble + validation; soft-fail on invariant subspace documented). Summary written post-hoc from cluster artifacts. Ready to discuss/plan 03-03 with open task-structure concern.
Resume file: None
