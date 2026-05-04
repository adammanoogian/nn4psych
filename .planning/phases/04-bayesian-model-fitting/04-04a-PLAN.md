---
phase: 04-bayesian-model-fitting
plan: 04a
type: execute
wave: 4
depends_on: ["04-01"]
gap_closure: false
autonomous: true
files_modified:
  - scripts/data_pipeline/09b_build_rnn_cohort_manifest.py
  - data/processed/rnn_cohort/checkpoint_metrics.parquet
  - data/processed/rnn_cohort/cohort_manifest.json
  - figures/rnn_cohort/delta_area_by_axis.png
  - .gitignore
must_haves:
  truths:
    - "Existing 1,884 trained checkpoints in trained_models/checkpoints/model_params_101000/ are inventoried and tagged with axis_swept ∈ {gamma, preset, rollout, tdscale, canonical, off-axis} against the Kumar et al. 2025 canonical (γ=0.95, p_reset=0.0, τ=100, β_δ=1.0)."
    - "Each checkpoint has a recomputed ΔArea (Area_CP − Area_OB) metric from re-running PIE_CP_OB_v2 behavior extraction at 8 epochs."
    - "ΔArea distribution by gamma reproduces Kumar 2025 Fig. 2A: monotonic increase from γ≈0.1 to γ≈0.95 with a non-monotonic dip at γ=0.99 (credit-assignment / unbounded-V regime)."
    - "cohort_manifest.json enumerates every axis-sweep checkpoint with its model_path, parsed hyperparameters, axis_swept tag, axis_value, and ΔArea — directly consumable by Plan 04-04b's RBO fit loop."
    - "Off-axis checkpoints (those varying ≥2 hyperparameters off canonical) are excluded from the cohort but counted in n_off_axis_excluded."
  artifacts:
    - path: "scripts/data_pipeline/09b_build_rnn_cohort_manifest.py"
      provides: "Resumable script that inventories checkpoints, classifies axis_swept, runs behavior extraction, computes ΔArea, writes manifest + sanity figure."
      contains: "build_inventory"
    - path: "data/processed/rnn_cohort/checkpoint_metrics.parquet"
      provides: "Per-checkpoint table: gamma, preset_memory, rollout_size, td_scale, seed, axis_swept, axis_value, area_cp, area_ob, delta_area, status."
      contains: "delta_area"
    - path: "data/processed/rnn_cohort/cohort_manifest.json"
      provides: "Schema-versioned manifest consumed by 04-04b: {schema_version, n_total_checkpoints, n_in_cohort, canonical, axis_counts, seeds: [...]}."
      contains: "schema_version"
    - path: "figures/rnn_cohort/delta_area_by_axis.png"
      provides: "2×2 panel of ΔArea vs each axis (Kumar 2025 Fig. 2 replication) — sanity check that the cohort matches published behavior."
      contains: "delta_area_by_axis"
  key_links:
    - from: "data/processed/rnn_cohort/cohort_manifest.json"
      to: "scripts/data_pipeline/10_fit_rnn_data.py (Plan 04-04b)"
      via: "04-04b reads manifest to iterate over the per-axis cohort and fit Reduced Bayesian Observer per checkpoint."
      pattern: "cohort_manifest"
---

<objective>
Build a per-checkpoint cohort manifest from the existing 1,884 trained RNNs in
`trained_models/checkpoints/model_params_101000/`. Use the per-axis sweep
design from Kumar et al. 2025 (CCN, "Neurocomputational Underpinnings of
Suboptimal Beliefs in Reinforcement Learning Agents") which varies four
hyperparameters — γ (discount), p_reset (working-memory disruption), τ
(rollout / episodic memory capacity), β_δ (advantage scaling) — one at a time
around the canonical (γ=0.95, p_reset=0.0, τ=100, β_δ=1.0). Each checkpoint
gets a re-extracted ΔArea metric (Area_CP − Area_OB; Eq. 8 of the paper) so
Plan 04-04b can fit Reduced Bayesian Observer per checkpoint and project RBO
parameters onto the schizophrenia-spectrum ΔArea axis.

This supersedes the original 04-04a (K=20 SLURM array of identical seeds),
which was discarded as wrong-direction work because re-training homogeneous
seeds throws away the existing ~50-seed × 4-axis behavioral diversity
already published in Kumar 2025. No cluster compute required.

Output:
- `scripts/data_pipeline/09b_build_rnn_cohort_manifest.py`
- `data/processed/rnn_cohort/checkpoint_metrics.parquet` (cache)
- `data/processed/rnn_cohort/cohort_manifest.json` (consumer schema for 04-04b)
- `figures/rnn_cohort/delta_area_by_axis.png` (Kumar Fig. 2 replication)
</objective>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/04-bayesian-model-fitting/04-CONTEXT.md
@.planning/phases/04-bayesian-model-fitting/04-RESEARCH.md
@docs/reference_papers/Kumar et al. 2025 CCN — Neurocomputational Underpinnings of Suboptimal Beliefs in RL Agents
@scripts/data_pipeline/03_analyze_hyperparameter_sweeps.py  (existing parse_model_filename)
@scripts/data_pipeline/04_visualize_behavioral_summary.py   (existing extract_model_behavior)
@scripts/data_pipeline/05_visualize_hyperparameter_effects.py  (existing calculate_area_metric)
@config.py  (GAMMA_VALUES, ROLLOUT_VALUES, PRESET_VALUES, SCALE_VALUES, paths)
@trained_models/checkpoints/model_params_101000/  (1,884 .pth files)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Inventory + axis classification</name>
  <files>scripts/data_pipeline/09b_build_rnn_cohort_manifest.py</files>
  <action>
Parse all `.pth` filenames in `trained_models/checkpoints/model_params_101000/`
using a parser equivalent to `03_analyze_hyperparameter_sweeps.parse_model_filename`,
extracting: gamma, preset_memory, rollout_size, td_scale, td_penalty, seed,
performance_score, hidden_dim, epochs_trained, max_displacement, reward_size.

Tag each checkpoint with `axis_swept`:
- 0 axes off-canonical → 'canonical' (γ=0.95 ∧ p_reset=0.0 ∧ τ=100 ∧ β_δ=1.0)
- 1 axis off → 'gamma' / 'preset' / 'rollout' / 'tdscale'
- ≥2 axes off → 'off-axis' (excluded from cohort but counted)

Set `axis_value` to the value of the swept axis (NaN for off-axis).
  </action>
  <verify>
Smoke at `--max_models 8`: report axis_swept counts, ensure all 8 are tagged
'gamma' (since the smoke set is dominated by gamma sweeps in alphabetical order).
  </verify>
  <done>
DataFrame with one row per checkpoint, columns include checkpoint_path
(POSIX), axis_swept, axis_value, and all parsed hyperparameters.
  </done>
</task>

<task type="auto">
  <name>Task 2: Re-extract behavior + compute ΔArea (resumable)</name>
  <files>scripts/data_pipeline/09b_build_rnn_cohort_manifest.py, data/processed/rnn_cohort/checkpoint_metrics.parquet</files>
  <action>
For each checkpoint, load via `ActorCritic` and run
`extract_model_behavior(model_path, epochs=8)` from
`scripts/data_pipeline/04_visualize_behavioral_summary.py` — yields states of
shape `(epochs, 2, 5, 200) = (epoch, condition, channels, trial)`.

Compute Area_CP and Area_OB separately by integrating learning rate vs
prediction error over `threshold=20` filtered trials, using `get_lrs_v2` from
`nn4psych.utils.metrics`. ΔArea = Area_CP − Area_OB.

Cache results to `data/processed/rnn_cohort/checkpoint_metrics.parquet` every
100 entries (resumable: subsequent runs skip checkpoints already in the
parquet's checkpoint_id set).

Use `torch.set_num_threads(1)` to avoid thrashing on the 16 GB / 8-core local
machine. ~50 minutes total for 1,884 checkpoints @ 8 epochs.
  </action>
  <verify>
- Resume works: rerun without `--no_resume` finds parquet rows already present and skips.
- Smoke 8 ck @ 4 epochs: ΔArea at γ=0.1 is strongly negative (e.g. −1.27, near Kumar 2025 Fig. 2A's leftmost point).
  </verify>
  <done>
Parquet exists with 1,884 rows. `status='ok'` for all (or `extraction_failed`
documented in SUMMARY for any failures).
  </done>
</task>

<task type="auto">
  <name>Task 3: Write cohort_manifest.json</name>
  <files>data/processed/rnn_cohort/cohort_manifest.json</files>
  <action>
Filter `checkpoint_metrics` to axis_swept ∈ {gamma, preset, rollout, tdscale,
canonical}. For each row emit a `seeds[*]` record:

```json
{
  "checkpoint_id": "<filename without .pth>",
  "model_path": "trained_models/checkpoints/model_params_101000/<file>.pth",
  "axis_swept": "gamma",
  "axis_value": 0.5,
  "gamma": 0.5,
  "preset_memory": 0.0,
  "rollout_size": 100,
  "td_scale": 1.0,
  "seed": 12,
  "delta_area": -0.42,
  "area_cp": ...,
  "area_ob": ...,
  "performance_score": ...,
  "status": "ok"
}
```

Top-level fields: schema_version="1.0", n_total_checkpoints, n_in_cohort,
n_off_axis_excluded, canonical, axis_counts, seeds.
  </action>
  <verify>
- JSON loads cleanly.
- `n_in_cohort + n_off_axis_excluded == n_total_checkpoints`.
- For each axis ∈ {gamma, preset, rollout, tdscale}: at least 50 entries (one per seed × per varied axis_value).
  </verify>
  <done>
File exists; downstream `04-04b` can parse and iterate without re-deriving
hyperparameters from filenames.
  </done>
</task>

<task type="auto">
  <name>Task 4: Sanity figure (Kumar Fig. 2 replication)</name>
  <files>figures/rnn_cohort/delta_area_by_axis.png</files>
  <action>
2×2 panel: ΔArea vs each axis (γ, p_reset, τ log-x, β_δ), error bars = 95% CI
across seeds within each cell. Add a horizontal zero line and the canonical
ΔArea as a reference dot.

Expected:
- γ panel: monotonic increase 0.1 → 0.95, dip at 0.99 (Kumar Fig. 2A).
- β_δ panel: monotonic increase (Fig. 2B).
- p_reset panel: monotonic decrease (Fig. 2C).
- τ panel: non-monotonic with peak around τ ∈ [20, 100] (Fig. 2D).

If any panel deviates qualitatively from the paper's published curve, document
in SUMMARY as an open question (could indicate a different RNG state, env
parameter, or learning-rate-area helper version).
  </action>
  <verify>
PNG + SVG saved. File size > 50 KB (i.e. not blank).
  </verify>
  <done>
Sanity figure exists and qualitatively matches Kumar 2025 Fig. 2.
  </done>
</task>

<task type="auto">
  <name>Task 5: Update .gitignore to allow tracking the manifest + figure</name>
  <files>.gitignore</files>
  <action>
Add unignore rules so collaborators (and future cluster pulls) get the
manifest and sanity figure without binary-cache (parquet) noise:

```
# 04-04a: cohort manifest is committed; parquet stays local
!/data/processed/rnn_cohort/cohort_manifest.json
!/figures/rnn_cohort/
!/figures/rnn_cohort/*.png
!/figures/rnn_cohort/*.svg
```

Do NOT unignore the parquet (it's a cache; regenerable from the script in
~50 min and large enough that LFS would be more appropriate if we ever need
it shared).
  </action>
  <verify>
`git check-ignore -v data/processed/rnn_cohort/cohort_manifest.json` reports
the unignore rule (not the parent ignore).
  </verify>
  <done>
.gitignore allows `cohort_manifest.json` and `figures/rnn_cohort/*` to be
staged; parquet still ignored.
  </done>
</task>

</tasks>

<verification>
After all tasks:
```
# Manifest exists and validates
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import json
with open('data/processed/rnn_cohort/cohort_manifest.json') as f:
    m = json.load(f)
assert m['schema_version'] == '1.0'
assert m['n_in_cohort'] + m['n_off_axis_excluded'] == m['n_total_checkpoints']
for axis in ['gamma', 'preset', 'rollout', 'tdscale']:
    assert m['axis_counts'].get(axis, 0) >= 50, f'{axis} cohort under 50: {m[\"axis_counts\"]}'
print(f'manifest OK: n_in_cohort={m[\"n_in_cohort\"]}, axis_counts={m[\"axis_counts\"]}')
"

# Figure exists, non-empty
ls -la figures/rnn_cohort/delta_area_by_axis.png
test $(stat -c %s figures/rnn_cohort/delta_area_by_axis.png 2>/dev/null || stat -f%z figures/rnn_cohort/delta_area_by_axis.png) -gt 50000
```
</verification>

<success_criteria>
- All 1,884 checkpoints inventoried; ≥99% have status='ok'.
- cohort_manifest.json contains ≥50 entries per axis (gamma/preset/rollout/tdscale).
- Sanity figure qualitatively matches Kumar 2025 Fig. 2 (monotonicity directions).
- 04-04b can iterate `cohort_manifest.json["seeds"]` and fit RBO without further filename parsing.
- No cluster compute consumed.
</success_criteria>

<output>
Create `.planning/phases/04-bayesian-model-fitting/04-04a-SUMMARY.md`:
- What was built: 09b script, parquet, manifest, sanity figure.
- Total run time and per-checkpoint median.
- ΔArea distribution by axis_swept (mean ± SD per axis).
- Any extraction_failed checkpoints + reasons.
- Decisions logged: epochs=8 default; off-axis exclusion criterion; canonical fix at γ=0.95/p_reset=0/τ=100/β_δ=1.
- Required-by-next-plan: 04-04b reads `cohort_manifest.json["seeds"]` and replays human Nassar 2021 sequences through each checkpoint, then fits RBO per (checkpoint, condition).
</output>
