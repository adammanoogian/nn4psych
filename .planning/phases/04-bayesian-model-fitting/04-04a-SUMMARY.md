---
phase: 04-bayesian-model-fitting
plan: 04a
status: complete
started: 2026-05-04T08:46Z
completed: 2026-05-04T18:30Z
---

# 04-04a Summary — RNN Cohort Manifest (Axis-All)

## Objective recap

Build a per-checkpoint cohort manifest from the existing 1,884 trained PIE_CP_OB_v2 RNNs in `trained_models/checkpoints/model_params_101000/`, using the per-axis sweep design from Kumar et al. 2025 CCN ("Neurocomputational Underpinnings of Suboptimal Beliefs in Reinforcement Learning Agents"). Each checkpoint gets a recomputed ΔArea = Area_CP − Area_OB metric (Eq. 8 of the paper), and the manifest is the consumer-facing schema for Plan 04-04b's Reduced Bayesian Observer fitting.

This plan **superseded** the original K=20 SLURM-array re-training. That approach was wrong-direction work — re-training homogeneous seeds throws away the published behavioral diversity already encoded in the existing 1,884-checkpoint sweep around γ=0.95, p_reset=0, τ=100, β_δ=1.

## What was built

### Script (`scripts/data_pipeline/09b_build_rnn_cohort_manifest.py`, 419 → 480 LOC)
- Filename parser (gamma, preset_memory, rollout_size, td_scale, td_penalty, seed, performance_score, ...).
- 5-axis classifier: tags each checkpoint as `gamma`/`preset`/`rollout`/`tdscale`/`tdpenalty`/`canonical`/`off-axis` against the canonical (γ=0.95, p_reset=0, τ=100, β_δ=1, td_penalty=0).
- Resumable behavior extraction at 8 epochs (parquet checkpointing every 100 entries).
- Manifest writer (schema v1.0) and Kumar Fig. 2 replication plotter (2×3).
- `--reclassify_only` mode reloads the parquet, re-applies classification, and rewrites the manifest+figure in seconds (no behavior re-extraction).

### Outputs
| File | Size | Tracked? |
|------|------|----------|
| `data/processed/rnn_cohort/checkpoint_metrics.parquet` | ~70 KB | local cache only |
| `data/processed/rnn_cohort/cohort_manifest.json` | ~770 KB | **git** |
| `figures/rnn_cohort/delta_area_by_axis.png` + `.svg` | ~120 KB | **git** |

### Commits (5)
- `03ff4ec` feat — manifest builder script (initial)
- `68c4f15` docs — pivot PLAN.md (rewrite from K=20 SLURM-array → axis-all local)
- `2b10e0e` feat — td_penalty 5th axis + `--reclassify_only` flag (post-build fix)
- `f788bc9` chore — `.gitignore` unignore for manifest + figure
- `95fe5ca` feat — cohort manifest + figure outputs

## Cohort composition

```
n_total_checkpoints: 1884
n_in_cohort:         1884
n_off_axis_excluded: 0   (no checkpoints vary >1 axis off canonical)

axis_swept counts:
  canonical : 50    (γ=0.95, p_reset=0, τ=100, β_δ=1, td_penalty=0; 50 unique seeds)
  gamma     : 350   (7 non-canonical γ values × 50 seeds)
  preset    : 349   (7 non-canonical p_reset values × 50; one cell short)
  rollout   : 385   (9 non-canonical τ values, partial seed coverage)
  tdscale   : 400   (8 non-canonical β_δ values × 50)
  tdpenalty : 350   (7 non-canonical td_penalty values × 50; supplementary, not in Kumar 2025)
```

## ΔArea results

### By axis (mean ± std, range)
| axis_swept | n   | mean ΔArea | std   | min    | max    |
|-----------|----:|-----------:|------:|-------:|-------:|
| canonical | 50  | 21.04      | 17.74 | -23.48 |  61.64 |
| gamma     | 350 | 13.66      | 20.02 | -26.29 |  82.92 |
| preset    | 349 | 21.04      | 20.75 | -26.70 |  87.64 |
| rollout   | 385 | 17.25      | 22.79 | -44.36 | 104.59 |
| tdscale   | 400 | 19.82      | 19.89 | -26.29 | 124.37 |
| tdpenalty | 350 | 21.30      | 18.61 | -34.67 |  72.41 |

### γ panel (per Kumar Fig. 2A)
| γ    | n  | mean ΔArea | std   |
|-----:|---:|-----------:|------:|
| 0.10 | 50 |   2.87     | 13.65 |
| 0.25 | 50 |   9.24     | 16.89 |
| 0.50 | 50 |  10.75     | 19.32 |
| 0.70 | 50 |  16.99     | 19.59 |
| 0.80 | 50 |  24.91     | 22.97 |
| 0.90 | 50 |  25.09     | 20.82 |
| 0.99 | 50 |   5.75     | 13.40 |

Reproduces Kumar 2025 Fig. 2A: monotonic increase 0.1 → 0.9, peak at γ ∈ [0.8, 0.9], dip at γ=0.99 (credit-assignment / unbounded value-function regime).

## Decisions logged

- **epochs=8** for behavior averaging (script default; matches Stage 05's existing convention).
- **Off-axis exclusion** = checkpoints varying ≥2 axes off canonical → tagged `off-axis` and excluded from cohort. Empirically zero off-axis checkpoints in this sweep — the entire 1,884-set is pure axis-aligned.
- **5th axis (td_penalty) included** in the cohort despite not being in Kumar's published 4-axis design. Empirically td_penalty has no detectable effect on ΔArea (means within canonical's 1σ across all 7 non-zero values), so its inclusion just doubles the canonical-region sample size for 04-04b's RBO fitting without introducing a new behavioral degree of freedom.
- **POSIX checkpoint paths** in manifest (via `Path.relative_to(...).as_posix()`) so the JSON is cluster/Linux-portable.
- **Parquet stays local** — regenerable cache, ~70 KB only but binary; LFS would be the right path if we ever need it shared. Manifest JSON contains all the data anyway.

## Performance

- Wall clock: 81 minutes (8 epochs × 1,884 checkpoints × ~2.6s/ck on 16 GB / 8-core local box, `torch.set_num_threads(1)`).
- Resumable: parquet saved every 100 entries; re-runs skip done checkpoints.
- Reclassify pass (`--reclassify_only`) takes ~3 seconds.
- 0 extraction failures; status='ok' for all 1,884.

## Open follow-ups

- **p_reset panel deviates qualitatively** from Kumar 2025 Fig. 2C: paper shows monotonic decrease, my replication shows roughly flat with high noise. Could be (a) different RNG seeds, (b) subtle env parameter difference (max_displacement=10 vs 15 in v1?), or (c) different `get_lrs_v2` threshold. Not blocking 04-04b — the absolute ΔArea per checkpoint is consistent and that's what RBO fitting consumes. Worth a follow-up if 04-04b's RBO-vs-p_reset projection looks weird.
- **rollout count is 385** rather than 9 × 50 = 450. Some (rollout × seed) cells are missing checkpoints. Cells where this matters can be tracked from the manifest's per-cell counts; doesn't affect the 04-04b iteration.

## Required by next plan (04-04b)

Plan 04-04b will:
1. Read `data/processed/rnn_cohort/cohort_manifest.json` and iterate `seeds[*]`.
2. For each (checkpoint, condition ∈ {CP, OB}) pair, replay one of the Nassar 2021 human bag-position sequences through the loaded `ActorCritic` model and capture the bucket trajectory.
3. Fit Reduced Bayesian Observer (from `nn4psych.bayesian.reduced_bayesian`) per (checkpoint × condition).
4. Project fitted RBO parameters onto the schizophrenia-spectrum ΔArea axis from this manifest, alongside the per-subject human RBO fits from 04-03 (currently blocked on Brain2021Code download).

## Verification

```
$ /c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import json
with open('data/processed/rnn_cohort/cohort_manifest.json') as f:
    m = json.load(f)
assert m['schema_version'] == '1.0'
assert m['n_in_cohort'] + m['n_off_axis_excluded'] == m['n_total_checkpoints']
for axis in ['gamma', 'preset', 'rollout', 'tdscale']:
    assert m['axis_counts'].get(axis, 0) >= 50, f'{axis} cohort under 50: {m[\"axis_counts\"]}'
print(f'manifest OK: n_in_cohort={m[\"n_in_cohort\"]}, axis_counts={m[\"axis_counts\"]}')
"
manifest OK: n_in_cohort=1884, axis_counts={'canonical': 50, 'tdscale': 400, 'rollout': 385, 'tdpenalty': 350, 'gamma': 350, 'preset': 349}

$ ls -la figures/rnn_cohort/delta_area_by_axis.png
-rw-r--r-- ...  ~80 KB
```

All success criteria met.
