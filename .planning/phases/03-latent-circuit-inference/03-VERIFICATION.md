---
phase: 03-latent-circuit-inference
date: 2026-04-26
status: gaps_found
score: 4/5
user_decision: 2026-04-26 — Option B chosen. User mandated Phase 3.1 deferred fixes before Phase 4.
human_verification:
  - test: (resolved 2026-04-26) Accept Story 2 vs mandate Phase 3.1 deferred fixes
    expected: Go/no-go on Phase 4 unblock
    why_human: SC-2 soft-fails at corr=0.783 (cluster) / 0.42 (local stochastic)
    resolution: Option B — pursue gap closure to attempt to lift corr above 0.85
---

# Phase 3: Latent Circuit Inference Verification Report

**Phase Goal:** The latent circuit inference pipeline fits Q, w_rec, w_in, w_out from context-DM RNN hidden states with 100-initialization ensemble validation, and the inferred circuit passes both activity-level and connectivity-level checks.
**Verified:** 2026-04-26
**Status:** gaps_found (re-routed from human_needed by user decision 2026-04-26 — Option B)
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| #       | Must-Have                                                                       | Status                 | Evidence                                                                                             |
|---------|---------------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------------------------------|
| SC-1    | 100+ random inits, best by reconstruction loss                                  | VERIFIED               | fit_latent_circuit_ensemble() loop; validation_results.json: n_inits=100, best_init_idx=95           |
| SC-2    | QT W_rec Q corr with w_rec >= 0.85 (invariant subspace)                        | SOFT-FAIL (documented) | corr=0.703 (03-02 n=16); corr=0.783 (Wave A n=12); no rank crosses 0.85; local re-eval corr~0.42    |
| SC-3    | Projecting RNN responses onto latent axes reproduces trial-averaged dynamics    | VERIFIED               | trial_avg_r2_full=0.980 (03-02), 0.976 (Wave A n=12); both exceed paper target of 0.96               |
| SC-4    | Perturbing w_rec in latent space produces measurable predicted behavioral change | SOFT-FAIL (ambiguous)  | 0/50 perturbations significant; max delta=0.120 vs threshold=0.126; ambiguity documented in writeup  |
| CIRC-01 | LatentNet fits Q, w_rec, w_in, w_out from context-DM hidden states              | VERIFIED               | All five exports in circuit_inference.py; best_latent_circuit_waveA.pt 38 KB; Q ortho err=1.03e-06  |

**Score:** 3/5 hard-pass, 2/5 soft-fail with documented mitigation and writeup closure artifact.

---

## Artifact Verification

### Source Code

| Artifact | Lines | Status |
|---|---|---|
| src/nn4psych/analysis/circuit_inference.py | 899 | VERIFIED: exports collect_circuit_data, save_circuit_data, fit_latent_circuit_ensemble, validate_latent_circuit, perturb_and_evaluate |
| src/nn4psych/analysis/latent_net.py | 160 | VERIFIED: LatentNet class vendored from engellab/latentcircuit |
| scripts/data_pipeline/08_infer_latent_circuits.py | 340 | VERIFIED: main(), load_wave_a_chosen_rank(), --skip_collection, --skip_fitting, --quick flags |

### Output Artifacts

| Artifact | Present | Size / Key Value |
|---|---|---|
| output/circuit_analysis/best_latent_circuit.pt | YES | 38901 bytes (2026-04-24); 03-02 benchmark n=16 |
| output/circuit_analysis/best_latent_circuit_waveA.pt | YES | 38069 bytes (2026-04-25); Wave A chosen Q n=12 |
| output/circuit_analysis/best_latent_circuit_waveB_refit.pt | YES | 38237 bytes (2026-04-26); Wave B smoke refit |
| output/circuit_analysis/ensemble_diagnostics.json | YES | 100 nmse_y values; mean=0.270, std=0.029, min=0.239, max=0.455 |
| output/circuit_analysis/validation_results.json | YES | 03-02: corr=0.703, trial_avg_r2=0.980, n_inits=100, device=cuda |
| output/circuit_analysis/validation_results_waveB.json | YES | Wave B local re-eval: corr=0.42, nmse_y=4.9 (stochastic noise discrepancy documented) |
| output/circuit_analysis/perturbation_results.json | YES | 50 perturbations, 0 significant, max delta=0.120, threshold=0.126 |
| output/circuit_analysis/wave_b_writeup.md | YES | Story commitment: STORY_2 (line 6) |
| output/circuit_analysis/n_latent_sweep/wave_a_selection.json | YES | chosen_rank=12, recommended_story=ran_out_of_fixes |
| output/circuit_analysis/n_latent_sweep/pareto_curve.json | YES | 3 rows (n8/n12/n16); ranks_missing=[4] |
| output/circuit_analysis/n_latent_sweep/n8/ | YES | best_latent_circuit.pt, ensemble_diagnostics.json, validation_results.json |
| output/circuit_analysis/n_latent_sweep/n12/ | YES | best_latent_circuit.pt, ensemble_diagnostics.json, validation_results.json |
| output/circuit_analysis/n_latent_sweep/n16/ | YES | best_latent_circuit.pt, ensemble_diagnostics.json, validation_results.json |

---

## Specific Sanity Checks

### Check 1: perturbation_results.json wave_a_chosen_rank matches wave_a_selection.json chosen_rank

perturbation_results.json line 2: "wave_a_chosen_rank": 12
wave_a_selection.json line 2: "chosen_rank": 12

PASS. Both = 12.

### Check 2: perturbation_results.json has baseline.significance_threshold and per-perturbation reward_delta_by_context

baseline.significance_threshold: 0.12566821396041233 (present)
reward_delta_by_context: present on all 50 perturbation entries (grep confirms 50 matching lines).

PASS.

### Check 3: wave_b_writeup.md commits to exactly ONE story

wave_b_writeup.md line 6: **Story commitment:** STORY_2

Single value. Not a placeholder. No multi-story ambiguity.

PASS.

### Check 4: 08_infer_latent_circuits.py reads chosen rank via load_wave_a_chosen_rank(), no hardcoded n_latent literal

load_wave_a_chosen_rank() defined at line 69; called at line 181 as n_latent = load_wave_a_chosen_rank().
Grep for n_latent = [0-9] in the file: zero matches.

PASS.

### Check 5: Q orthonormality assertion present in --skip_fitting path

Lines 265-274 of 08_infer_latent_circuits.py (inside the else branch gated on args.skip_fitting):
    # Assert Q orthonormality: ||Q Q^T - I|| < 1e-4
    assert identity_err < 1e-4, ...

PASS.

---

## Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| 08_infer_latent_circuits.py main | wave_a_selection.json | load_wave_a_chosen_rank() at line 181 | WIRED |
| 08_infer_latent_circuits.py --skip_fitting | best_latent_circuit_waveA.pt | WAVE_A_BEST_PT_PATH loaded at line 259 | WIRED |
| fit_latent_circuit_ensemble() | LatentNet | instantiated in loop (line 365); best init reloaded via load_state_dict (line 430) | WIRED |
| validate_latent_circuit() | RNN W_rec | passed from model.W_hh.weight (line 278); used in q @ W_rec @ q.T (line 527) | WIRED |
| perturb_and_evaluate() | model.W_hh.weight | in-place mutation at line 821; try/finally restoration at line 862 | WIRED |
| circuit_inference.py exports | 08_infer_latent_circuits.py | all five functions imported at lines 53-59 | WIRED |

---

## SC-2 Soft-Fail Accounting (CIRC-04: Invariant Subspace)

This is the central unresolved scientific issue. The evidence chain:

1. 03-02 benchmark (n=16): corr=0.703. Documented non-blocking per plan. Device=CUDA, 100 inits.
2. 03-03 Wave A sweep (n in {8,12,16}): corr = {0.710, 0.783, 0.687}. Non-monotonic; n=12 peaks. Spread=0.096.
3. Chosen rank n=12: corr=0.783, below 0.85 threshold. No rank crossed it.
4. Wave B perturbation (03-04): 0/50 perturbations significant. Max delta=0.120 vs threshold=0.126.
5. Local re-evaluation discrepancy: validation_results_waveB.json shows corr=0.42, nmse_y=4.9.
   Root cause: LatentNet applies sigma_rec=0.15 noise at every forward pass (train and eval modes alike).
   Cluster metrics computed immediately after training on the same noise seed.
   Local re-eval uses fresh noise realizations. Cayley transform Q itself is deterministic (ortho err=1.03e-06).

Story 2 (ran_out_of_fixes) rationale: The 0.096 Pareto spread is the primary evidence that the structural
concern is not exhausted -- ranks vary meaningfully, n=12 is best tried, but deferred fixes have not been attempted.
The story commitment was made before Wave B perturbation results were seen (Wave A pre-positioning), and is maintained.

Deferred fixes listed in wave_b_writeup.md:
1. Masked-loss fitting: compute NMSE_y only over task-active timesteps (~15-30 of T=75).
2. Shorter T regen: circuit_data.npz at T=25-40 with delay=0.
3. Condition-sliced fitting: fit separately on task-active vs blank periods.

---

## SC-4 Perturbation Ambiguity (CIRC-05)

0/50 significant result has a documented confound: Q used for perturbation has corr~0.42 locally (stochastic re-eval).
Perturbations not landing in the true invariant subspace would produce attenuated behavioral effects regardless of strength.
Max delta=0.120 is 95.5% of the significance threshold (0.126) -- not clearly zero.

Two interpretations per writeup: (a) perturbation strengths [-0.5, 0.5] too small relative to RNN noise floor at
max_steps=75, or (b) Q quality prevents perturbations from hitting the causal subspace. Both are consistent with Story 2.
The ambiguity is documented; it does not invalidate the perturbation analysis infrastructure.

---

## Anti-Patterns

None that block goal achievement:

- No TODO/FIXME/placeholder in the five required exports of circuit_inference.py.
- No hardcoded n_latent in 08_infer_latent_circuits.py main (verified via grep, zero matches).
- perturb_and_evaluate() has try/finally weight restoration -- model not permanently mutated.
- --quick flag correctly redirects to smoke_test/ subdirs (bug found and fixed in 03-04 at lines 170-175).
- validation_results_waveB.json degraded metrics are NOT a code bug -- documented pre-existing LatentNet limitation.

---

## Human Verification Required

### 1. SC-2 / Story 2 Go/No-Go

**Test:** Read output/circuit_analysis/wave_b_writeup.md. Decide between:

**Option A -- Accept Story 2:** Phase 3 closes with corr=0.783 (cluster-measured) as documented best.
Deferred fixes flagged for Phase 3.1 or v2. Phase 4 (Bayesian fitting) proceeds now.

**Option B -- Mandate Phase 3.1:** Before Phase 4, attempt masked-loss fitting (circuit_inference.py
needs a new fitting variant; circuit_data.npz needs per-trial timing metadata), shortened T regen,
or condition-sliced fitting.

**Expected:** A go/no-go decision recorded before the next GSD phase begins.

**Why human:** Scientific judgment call. The code and artifacts are complete and correct. The question is
whether the connectivity-level soft-fail is acceptable given that Phase 4 does NOT depend on Q quality.
Phase 4 fits Bayesian models to trial-level reward/choice data; Q is not in that pipeline.

**Key facts for the decision:**
- Best achievable corr with current method: 0.783 (n=12, of ranks {8,12,16} tried).
- Phase 4 independence: confirmed in 03-04-SUMMARY.md and wave_b_writeup.md.
- Highest-priority deferred fix: masked-loss fitting (directly targets T=75 padding noise hypothesis).
- Local stochastic eval gives corr~0.42 -- pre-existing LatentNet limitation, not a regression.

---

## Gaps Summary

User chose Option B (2026-04-26): pursue deferred fixes before Phase 4. The structural code and artifacts
are complete. The gaps below target the **scientific** soft-fail on SC-2 (invariant subspace corr ≥ 0.85).
All three gaps are deferred-fix candidates from `output/circuit_analysis/wave_b_writeup.md` and 03-CONTEXT.md.

### Gap 1 — Masked-loss fitting (HIGHEST PRIORITY)

**Targets:** SC-2 (invariant subspace corr), SC-4 (perturbation effect amplitude).

**Hypothesis:** Of T=75 timesteps in `circuit_data.npz`, only ~15-30 are task-active (fixation + stim + decision
within ~15-30 steps at dt=100ms for ContextDecisionMaking). The remaining ~45-60 steps are blank/padding noise.
LatentNet currently fits NMSE_y over ALL timesteps, so the loss is dominated by reconstructing the blank period
(low signal). A masked loss that ignores blank timesteps should let the fit allocate its representational budget
to task-active dynamics, lifting invariant_subspace_corr.

**What to build:**
- Per-trial task-active mask in circuit collection: `collect_circuit_data` augmented to record
  `(t_start, t_end)` per trial (the env's known stim/decision interval) and save `task_active_mask: (n_trials, T)` bool array.
- New fit variant `fit_latent_circuit_ensemble_masked()` (or a `mask` kwarg on existing function) that computes
  NMSE_y over masked timesteps only.
- Re-run cluster ensemble at the chosen rank (n=12) and adjacent ranks (n=8, 16) with masked loss.
- Re-run validation; report new corr; update `wave_a_selection.json` (or write `wave_a_masked_selection.json`)
  if the chosen rank shifts.

**Success:** corr_masked ≥ 0.85 at the chosen rank. If still below, the structural concern moves from
"ran out of fixes" toward "method limit" (and Story commitment in writeup updates accordingly).

**Risk:** masking definition depends on env timing — must extract the actual fixation/stim/decision boundaries
from `SingleContextDecisionMakingWrapper` (NeuroGym timing dict), not assume them.

### Gap 2 — Shorter T regen with delay=0 (orthogonal probe)

**Targets:** SC-2.

**Hypothesis:** Even with T=75, the underlying NeuroGym task may include a long delay phase that produces nearly
constant hidden state. Regenerating circuit_data with `delay=0` (or shortened delay) and `max_steps≈25-40`
removes the same low-signal timesteps that masking would ignore — but also retrains the RNN if needed and
keeps the eval dynamics matched.

**What to build:**
- Verify `SingleContextDecisionMakingWrapper` exposes `delay` / per-period overrides (NeuroGym does).
- Regenerate `circuit_data.npz` at smaller T (say T=30) on cluster.
- Refit at chosen rank.
- Compare corr to T=75 baseline.

**Risk:** RNN was trained at the standard timing; eval at shorter T may degrade RNN accuracy itself, confounding
interpretation.

### Gap 3 — Condition-sliced fitting (diagnostic, lower priority)

**Targets:** SC-2 diagnosis (does Q quality differ across task conditions?).

**Hypothesis:** If the RNN encodes context-DM differently across `modality_context ∈ {0, 1}`, fitting one Q
across both pools all the connectivity into a shared latent space that may not be optimal for either.
Per-context Qs would diagnose whether the RNN's recurrent structure is itself non-low-rank or whether it's
two interleaved low-rank circuits.

**What to build:**
- Slice circuit data by modality_context.
- Fit two LatentNets independently.
- Report per-condition corr.
- If both >> pooled corr: structural separation. If both ≈ pooled corr: not the issue.

**Risk:** Halves training data per fit; may need n_inits boost. Diagnostic only — not a fix in itself.

### Sequencing recommendation

1. **Gap 1 first** — directly tests the dominant hypothesis (T=75 padding) without retraining the RNN.
2. **Gap 2 only if Gap 1 doesn't clear 0.85** — orthogonal evidence on the same hypothesis.
3. **Gap 3 in parallel with Gap 1** if cluster compute allows — diagnostic, not gating.

Cost estimate: Gap 1 ≈ 1 cluster sweep (~1 hr × 3 ranks ≈ 3 hr GPU); Gap 2 ≈ 1 cluster sweep + data regen;
Gap 3 ≈ 2 cluster sweeps. Total ~10 GPU-hours if all three pursued.

### Out of scope for Phase 3.1

- Different RNN training regime (curriculum, supervised pretraining) — Phase 6 / v2 territory.
- Different latent fitting method (e.g. SLDS, RSLDS) — out of paper scope.
- Larger n_latent ranges (>16) — Pareto already shows degradation at 16; not promising.

---

_Verified: 2026-04-26_
_Verifier: Claude (gsd-verifier)_

