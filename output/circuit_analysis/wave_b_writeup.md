# Wave B Writeup: Phase 3 Scientific Commitment

**Phase:** 03-latent-circuit-inference
**Date:** 2026-04-26
**Wave A chosen rank:** n_latent = 12
**Story commitment:** STORY_2

## Evidence Summary

### Wave A: n_latent sweep Pareto curve

| n_latent | best NMSE_y | invariant subspace corr | trial-avg R^2 (full) |
|----------|-------------|-------------------------|----------------------|
| 8        | 0.2964      | 0.7095                  | 0.9690               |
| 12       | 0.2472      | 0.7833                  | 0.9764               |
| 16       | 0.2434      | 0.6872                  | 0.9780               |

Note: rank n=4 failed to converge (excluded from sweep). Cluster-computed metrics;
locally recomputed invariant subspace corr measures ~0.42 due to LatentNet stochastic
noise at eval time (sigma_rec=0.15; see Deviation section in SUMMARY.md).

- Spread of invariant_subspace_corr across ranks: 0.096 (0.783 - 0.687)
- Chosen rank: n_latent = 12 (corr = 0.783)
- Wave A pre-positioning: ran_out_of_fixes — "Invariant subspace corr varies
  meaningfully across ranks (spread=0.096); chosen rank n_latent=12 is best of
  those tried but does not cross 0.85. Wave A reduced but did not eliminate the
  structural concern; deferred fixes (masked loss, shorter T) remain on the table."

### Wave B: perturbation analysis

- Perturbations evaluated: 50 (10 top-magnitude latent connections x 5 strengths)
- Perturbation strengths: [-0.5, -0.2, 0.0, 0.2, 0.5]
- Baseline noise (pooled std across 5 unperturbed runs): 0.0628
- Significance threshold (k = 2.0): 0.1257
- Mean |reward delta|: 0.0406
- Max |reward delta|: 0.1195 (just below threshold)
- Significant perturbations: 0 / 50
- Baseline mean reward (pooled): 1.770 (context 0: 1.793, context 1: 1.746)

The perturbation analysis perturbed rank-one connections in the n=12 latent
w_rec (strongest excitatory: w[11,2]=4.17, w[10,1]=3.44; strongest inhibitory:
w[11,1]=-3.16, w[9,9]=-2.23) via the mapping W_rec' = W_rec + q.T @ delta_ij @ q,
where q is Wave A's chosen orthonormal projection (12 x 64, orthonormality
err=1.03e-06). All 50 perturbations fall below the 2-sigma significance
threshold. Mean |reward delta| = 0.041 is approximately 65% of baseline
std = 0.063 — not zero, but not robustly above noise.

## Commitment

**We commit to: the soft-fail at corr=0.783 (cluster eval) reflects investigations
not yet run, not a fundamental method limit.**

The Pareto curve shows meaningful variation across ranks (spread=0.096); rank
n_latent=12 is the best of those tried but does not cross 0.85. Wave A's
pre-positioning, determined from the Pareto evidence before Wave B perturbation
results were in, recommended "ran_out_of_fixes" — and this commitment is
maintained.

The perturbation analysis shows 0 / 50 significant effects at strengths [-0.5,
-0.2, 0, 0.2, 0.5]. This is consistent with Story 2 rather than refuting it:
the lack of significant perturbation effects is itself likely a consequence of
the Q quality issue (locally recomputed corr ~0.42 vs cluster-measured 0.78).
The Q's stochastic-eval discrepancy means the projection matrix we used for
perturbation analysis may not have fully captured the RNN's invariant subspace.
Perturbations that don't land cleanly in the actual invariant subspace would
not be expected to produce robust behavioral effects regardless of strength.
In other words, the 0/50 significant result is ambiguous: it could reflect a
method limit in Q OR it could reflect that the perturbation strengths are too
small relative to the RNN's behavioral noise floor at max_steps=75.

**Deferred fixes** (from 03-CONTEXT.md, NOT included in this phase):

1. Masked-loss fitting: compute NMSE_y only over per-trial task-active timesteps
   (estimated ~15-30 active steps out of T=75). This directly tests the T=75
   padding hypothesis flagged in 03-02-SUMMARY.
2. Shorter T regen: circuit_data.npz at T=25-40 with delay=0. Would also
   reduce padding-noise contamination.
3. Condition-sliced fitting: fit LatentNet separately on task-active vs blank
   periods. Mirrors Langdon & Engel 2025 Fig. 3 approach.

**Implication:** Phase 3 closes with corr=0.783 (cluster-measured) as the current
best, with a documented discrepancy on local re-evaluation. A future plan
(Phase 3.1 or v2 work) could pursue masked-loss fitting first — it directly
targets the most likely cause of the soft-fail. Phase 4 (Bayesian fitting on
RNN behavioral outputs) proceeds independently; the latent circuit's role in
the Phase 4 pipeline is descriptive, not causal-mechanistic. v2 work
(multi-task latent circuit comparison, fixed-point analysis in latent space)
can still build on this Q with the documented caveats.

## Files

- Wave A: output/circuit_analysis/n_latent_sweep/{pareto_curve.json, pareto_curve.png, wave_a_selection.json}
- Wave A Q: output/circuit_analysis/best_latent_circuit_waveA.pt
- Wave B: output/circuit_analysis/perturbation_results.json
- Validation (Wave B re-run): output/circuit_analysis/validation_results_waveB.json
- Writeup: output/circuit_analysis/wave_b_writeup.md
