---
phase: 03-latent-circuit-inference
plan: "03"
status: complete
date_completed: 2026-04-25
---

# 03-03 Summary: Wave A — n_latent Sweep & Q Selection

## Outcome

Wave A locked the input to Wave B (Plan 03-04): **chosen rank n_latent = 12**,
loaded from `output/circuit_analysis/best_latent_circuit_waveA.pt`. Wave A
reduced — but did not eliminate — the 03-02 invariant-subspace soft-fail
(0.703 → 0.783; threshold 0.85). Story prepositioning: **ran_out_of_fixes**.

## Wave A findings

### Pareto table (4 ranks attempted, 3 succeeded)

| n_latent | best NMSE_y | invariant subspace corr | trial-avg R² (full) | wall (min) |
|----------|-------------|--------------------------|---------------------|------------|
| 4        | — (assertion fail; n < max(input=7, output=3) = 7) | — | — | — |
| 8        | 0.2964      | 0.7095                   | 0.9690              | 187.3      |
| 12       | **0.2472**  | **0.7833**               | 0.9764              | 95.3       |
| 16       | 0.2434      | 0.6872                   | 0.9780              | 91.0       |

### Rank-vs-corr relationship

- **Spread of invariant_subspace_corr = 0.096** (min 0.687 at n=16, max 0.783 at n=12).
- **Non-monotonic.** corr peaks at n=12 and drops at n=16, the classic
  over-parameterised-latent signature: the n=16 fit can spread the same
  task computation across more dimensions, lowering NMSE_y marginally
  (0.247 → 0.243) at the cost of distributing connectivity across redundant
  latent axes that no longer align with the RNN's true low-rank subspace.
- **n=8 also loses corr (0.710)** — under-rank from the other side.
- **No rank crosses 0.85.** Wave A reduced the corr deficit (0.703 → 0.783)
  but did not close it.

### Chosen rank: n_latent = 12

- Selected by `argmax(invariant_subspace_corr)` with tie-breaker
  `lower-rank` (parsimony) — applied deterministically in
  `select_rank()` via key `(corr, -n_latent)`.
- Provenance: `output/circuit_analysis/n_latent_sweep/n12/{best_latent_circuit.pt, validation_results.json, ensemble_diagnostics.json}` (cluster job 54946904, autopushed in commit a550003).
- Wave A Q archived as `output/circuit_analysis/best_latent_circuit_waveA.pt`.
- Orthonormality check: `||Q Qᵀ − I|| = 1.03e-06` (well below 1e-4 threshold).

### Wave B writeup pre-positioning

`recommended_story = "ran_out_of_fixes"` (NOT method_limit), because the
0.096 spread across ranks exceeds the 0.05 flatness cutoff. The
prepositioning is a starting hypothesis that 03-04 will RE-EVALUATE with
perturbation evidence — but it tells Wave B's writeup which scientific
narrative the Pareto curve alone supports.

**Concrete signal that the structural concern is not yet exhausted:**
n=12 outperforms both neighbours on corr while sitting between them on
NMSE_y. That's not "method/data limit, all ranks plateau"; that's "we
found a rank-knee but the absolute level is still below threshold". The
deferred fixes from 03-CONTEXT.md (masked loss, shorter T regen,
condition-sliced fitting) remain viable directions.

### Per-rank failures and notable wall times

- **n=4:** Expected LatentNet `connectivity_masks()` assertion failure
  (requires n ≥ max(input=7, output=3) = 7). The job submitter
  (`run_n_latent_sweep.sh`) intentionally launches it anyway so the
  failure is recorded; the autopush includes no n4 directory because no
  artifacts were produced. Documented in `pareto_curve.json`'s
  `ranks_missing: [4]`.
- **n=8 wall ≈ 2× n=12/16** (187 min vs 95/91). All three ran on the
  same L40S node (`m3g107`); GPU-utilisation summaries don't show
  contention. Plausible cause: a slower-converging init or differing
  early-stop pattern. Worth a quick `gpu_stats_54946903.csv` glance next
  cluster session, but doesn't affect the Pareto comparison
  (selection is by final corr, not wall).

## Wave B (Plan 03-04) — what is now locked

| Decision | Value | Source |
|----------|-------|--------|
| n_latent | **12** | `wave_a_selection.json::chosen_rank` |
| Q (LatentNet state_dict) | `best_latent_circuit_waveA.pt` | copied from n12/ |
| Starting story hypothesis | `ran_out_of_fixes` | `wave_a_selection.json::wave_b_prepositioning.recommended_story` |
| Threshold not crossed | corr 0.783 < 0.85 | `wave_a_selection.json::metrics.invariant_subspace_corr` |

**03-04 entry point:** `python scripts/data_pipeline/08_infer_latent_circuits.py --skip_collection --skip_fitting`
will load this n=12 Q, recompute Q via `cayley_transform`, assert
orthonormality (< 1e-4 — already verified at 1e-6 here), and run
perturbation analysis on the top-10 magnitude latent connections of the
n=12 `recurrent_layer.weight`.

The perturbation evidence then either upgrades the writeup commitment to
**story 1** (perturbations don't beat baseline noise → Q doesn't drive
behavior either → method limit), keeps it at **story 2** (perturbations
DO land → Q captures causal mechanism → deferred fixes are warranted),
or — vanishingly unlikely given corr=0.783 — flips to **story 0**
(passed). The decision rule is in 03-04-PLAN Task 3.

## Artifacts produced

- `cluster/run_n_latent_sweep.sh` (Task 1; autopush wired in ebcf836)
- `scripts/analysis/aggregate_n_latent_sweep.py` (Task 3)
- `output/circuit_analysis/n_latent_sweep/{n8,n12,n16}/` per-rank cluster outputs
- `output/circuit_analysis/n_latent_sweep/pareto_curve.json` (3 successful rows + ranks_missing=[4])
- `output/circuit_analysis/n_latent_sweep/pareto_curve.png` (3-panel plot, threshold line at 0.85)
- `output/circuit_analysis/n_latent_sweep/wave_a_selection.json` (chosen_rank=12, prepositioning=ran_out_of_fixes)
- `output/circuit_analysis/best_latent_circuit_waveA.pt` (locked Q for Wave B)

The 03-02 benchmark `output/circuit_analysis/best_latent_circuit.pt` is
preserved (not overwritten) per the must-have.
