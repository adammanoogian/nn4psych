---
phase: 04-bayesian-model-fitting
plan: 04b
type: execute
wave: 5
depends_on: ["04-01", "04-02", "04-03", "04-04a"]
gap_closure: false
autonomous: false
files_modified:
  - scripts/data_pipeline/10_fit_rnn_data.py
  - scripts/data_pipeline/replay_human_sequences.py
  - cluster/run_rnn_fits.sh
  - data/processed/bayesian/rnn_fits/
user_setup:
  - service: monash_m3_cluster
    why: "20 seeds x ~134 subjects x 2 conditions = 5360 MCMC fits; CPU array job on cluster (~6h wall)"
    dashboard_config:
      - task: "git push and SSH to M3 to submit cluster/run_rnn_fits.sh after 04-04a checkpoints are pulled"
        location: "Monash M3 cluster"
must_haves:
  truths:
    - "Each RNN seed (k=0..19) replays each human subject's exact (bag_position, condition) sequence to produce RNN-generated bucket trajectories"
    - "Reduced Bayesian observer is fit per (seed, subject, condition) cell with R-hat <= 1.01 AND ESS_bulk >= 400 (BAYES-05)"
    - "Fits are pooled across modality_context (PIE-RNN's within-task feature, NOT the CP/OB experimental condition) per CONTEXT.md decision"
    - "Per-fit JSON written to data/processed/bayesian/rnn_fits/per_fit/seed_{k:02d}_{subject_id}_{condition}.json"
    - "Aggregate summary CSV usable by Phase 5 cohort comparison"
    - "Replay logic preserves human stimulus sequences exactly (no re-randomization); RNN sees identical bag observations as the corresponding human subject"
  artifacts:
    - path: "scripts/data_pipeline/replay_human_sequences.py"
      provides: "replay_seed_on_subject(seed_idx, subject_trial_data) -> (bag_positions, bucket_positions); drives RNN forward on human sequence and extracts predicted bucket positions"
      contains: "def replay_seed_on_subject"
    - path: "scripts/data_pipeline/10_fit_rnn_data.py"
      provides: "End-to-end driver: load cohort_manifest.json + subject_trials.npy → for each (seed, subject, condition) call replay+fit_with_retry → write JSONs"
      contains: "if __name__ == '__main__'"
    - path: "cluster/run_rnn_fits.sh"
      provides: "SLURM array (--array=0-19) over seeds; each task fits ALL subjects for one seed sequentially (~6h wall per seed; total ~6h with 20 parallel)"
      contains: "SBATCH --array=0-19"
    - path: "data/processed/bayesian/rnn_fits/per_fit/"
      provides: "5360 per-fit JSONs (20 seeds x 134 subjects x 2 conditions)"
    - path: "data/processed/bayesian/rnn_fits/summary.csv"
      provides: "rnn_seed, subject_id, condition, status, H_mean, LW_mean, UU_mean, sigma_motor_mean, sigma_LR_mean, rhat_max, ess_min, n_divergences"
  key_links:
    - from: "scripts/data_pipeline/10_fit_rnn_data.py"
      to: "data/processed/rnn_cohort/cohort_manifest.json"
      via: "iterates over manifest['seeds']"
      pattern: "cohort_manifest"
    - from: "scripts/data_pipeline/10_fit_rnn_data.py"
      to: "scripts/data_pipeline/replay_human_sequences.py"
      via: "from replay_human_sequences import replay_seed_on_subject"
      pattern: "replay_seed_on_subject"
    - from: "scripts/data_pipeline/10_fit_rnn_data.py"
      to: "data/processed/nassar2021/subject_trials.npy"
      via: "loads human bag/condition sequences for replay"
      pattern: "subject_trials.npy"
---

<objective>
Replay each human subject's bag-position sequence through each of the K=20 RNN seeds (from 04-04a), then fit the Reduced Bayesian observer to the resulting RNN bucket trajectories. Pool across modality_context per CONTEXT.md decision. Run on cluster as a SLURM array over seeds (one task per seed, processes all subjects sequentially).

`autonomous: false` for cluster submission (same SSH/SLURM pattern as 04-04a).

Purpose: BAYES-05 + ROADMAP Phase 4 SC-4 evidence. Produces RNN posterior parameter distributions matched to humans (same stimulus sequences) for Phase 5 schizophrenia-vs-control-vs-RNN comparison.

Output:
- `replay_human_sequences.py` (RNN forward driver on human stimulus sequences)
- `10_fit_rnn_data.py` (per-seed-per-subject-per-condition fit driver)
- `cluster/run_rnn_fits.sh` (SLURM array over seeds)
- `data/processed/bayesian/rnn_fits/per_fit/*.json` + `summary.csv`
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/04-bayesian-model-fitting/04-CONTEXT.md
@.planning/phases/04-bayesian-model-fitting/04-RESEARCH.md
@.planning/phases/04-bayesian-model-fitting/04-01-SUMMARY.md
@.planning/phases/04-bayesian-model-fitting/04-02-SUMMARY.md
@.planning/phases/04-bayesian-model-fitting/04-03-SUMMARY.md
@.planning/phases/04-bayesian-model-fitting/04-04a-SUMMARY.md
@scripts/training/train_rnn_canonical.py
@scripts/data_pipeline/extract_nassar_trials.py
@src/nn4psych/bayesian/reduced_bayesian.py
@src/nn4psych/bayesian/diagnostics.py
@cluster/run_rnn_cohort.sh
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement replay_human_sequences.py — drive RNN forward on human stimuli</name>
  <files>scripts/data_pipeline/replay_human_sequences.py</files>
  <action>
Create `scripts/data_pipeline/replay_human_sequences.py`. Follow project conventions: `from __future__ import annotations`, `matplotlib.use('Agg')` only if pyplot is imported (likely not needed here), NumPy-style docstrings, absolute imports, line length 88, Python 3.10+ types.

Module setup:
```python
from __future__ import annotations
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from pathlib import Path
import numpy as np
import torch

from nn4psych.models.actor_critic import ActorCritic   # check exact import path; Phase 2 should have it
```

Public function:
```python
def replay_seed_on_subject(
    model_path: Path,
    bag_positions: np.ndarray,
    condition: str,
    *,
    pool_modality_context: bool = True,
    device: str = 'cpu',
) -> tuple[np.ndarray, np.ndarray]:
    """Drive a trained RNN forward on a human bag-position sequence.

    Parameters
    ----------
    model_path : Path
        Path to the trained checkpoint (model.pt from 04-04a).
    bag_positions : np.ndarray, shape (n_trials,)
        Human-observed bag positions (in bag-screen coordinates 0–300).
    condition : str
        'changepoint' or 'oddball'. Mapped to environment context flag for the RNN.
    pool_modality_context : bool, default True
        If True, run the RNN once with modality_context=0 and once with =1, then
        average bucket predictions per trial (CONTEXT.md decision: pool RNN fits
        across modality_context). If False, return only modality_context=0 trace
        (for debugging).
    device : str
        torch device.

    Returns
    -------
    bag_positions : np.ndarray, shape (n_trials,)
        Echoed input (for downstream fitting interface compatibility).
    bucket_positions : np.ndarray, shape (n_trials,)
        RNN-generated bucket positions, derived from RNN action choices each trial.

    Notes
    -----
    The PIE_CP_OB_v2 environment expects observations including modality_context,
    bag, and previous reward. This function constructs synthetic per-trial
    observations from the human bag sequence and the RNN's previous action
    (bucket position from prior trial) to maintain the closed-loop assumption.
    """
```

Implementation strategy:

1. Load checkpoint via `ckpt = torch.load(model_path, map_location=device, weights_only=False)`. Read `config` to get model dims (hidden_dim, input_dim, output_dim from training).

2. Construct `ActorCritic` with same config; load `model.load_state_dict(ckpt['model_state_dict'])`; `model.eval()`.

3. Per-trial loop (no PyTorch grad):
```python
hx = torch.zeros(1, hidden_dim, device=device)
prev_action = torch.zeros(1, 1, device=device)  # bucket position
prev_reward = torch.zeros(1, 1, device=device)
bucket_positions = []

for t in range(n_trials):
    bag_t = torch.tensor([[bag_positions[t]]], dtype=torch.float32, device=device)
    # Build observation: modality_context, bag_t, prev_action, prev_reward, condition_flag
    # Match exact obs schema used by train_rnn_canonical.py — read the script to confirm
    # the obs concatenation order before implementing.
    obs = build_obs(modality_context_value, bag_t, prev_action, prev_reward, condition_flag)
    with torch.no_grad():
        logits, value, hx = model(obs, hx)
    action = logits.argmax(dim=-1).item()  # deterministic; matches Phase 2 02-02 decision
    bucket_t = action_to_bucket(action)    # depends on env action space; document mapping
    bucket_positions.append(bucket_t)
    # update reward from env mechanics? Replay does not need true reward — use the simulated update:
    prev_reward = compute_replay_reward(bag_positions[t], bucket_t)  # documented placeholder
    prev_action = torch.tensor([[bucket_t]], dtype=torch.float32, device=device)
```

**CRITICAL CAVEAT for executor:** The exact obs schema for PIE_CP_OB_v2 must be inferred from `scripts/training/train_rnn_canonical.py` and `envs/pie_environment.py` (project structure shows these exist). Read those files BEFORE implementing `build_obs`. Document the observed schema in the script's docstring. If the schema changes between Phase 2 and Phase 4 (it should not — TRAIN-01 stable), this is a regression to flag.

If `pool_modality_context=True`: run the loop twice (modality_context=0 and =1), average the two bucket trajectories trial-by-trial, return the averaged sequence.

4. Return `(bag_positions, np.array(bucket_positions, dtype=np.float64))`.

Add a CLI for one-off smoke testing:
```python
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_trials', type=int, default=100)
    args = parser.parse_args()
    bag = np.random.normal(150, 20, size=args.n_trials)
    bag, bucket = replay_seed_on_subject(Path(args.model_path), bag, condition='changepoint')
    print(f'bag shape: {bag.shape}; bucket shape: {bucket.shape}')
    print(f'bucket range: [{bucket.min():.1f}, {bucket.max():.1f}]')
```
  </action>
  <verify>
```
# Smoke test on a single 04-04a checkpoint (smoke if cohort exists; otherwise on /tmp/rnn_smoke from 04-04a Task 1)
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/replay_human_sequences.py --model_path data/processed/rnn_cohort/seed_00/model.pt --n_trials 50
# Expected: prints bag/bucket shapes (50,) and bucket value range; no exception

# Or with smoke checkpoint:
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/replay_human_sequences.py --model_path /tmp/rnn_smoke/model.pt --n_trials 30
```
  </verify>
  <done>
- `replay_human_sequences.py` exists with `replay_seed_on_subject`.
- Smoke run succeeds against at least one checkpoint.
- Obs schema documented in module docstring.
- Pool-across-modality_context implemented (enabled by default).
  </done>
</task>

<task type="auto">
  <name>Task 2: Implement 10_fit_rnn_data.py — per-(seed, subject, condition) fit driver</name>
  <files>scripts/data_pipeline/10_fit_rnn_data.py</files>
  <action>
Create `scripts/data_pipeline/10_fit_rnn_data.py`. Mirror the structure of `09_fit_human_data.py` from 04-03 — same diagnostics + retry helpers, same JSON shape via `make_fit_summary`, same exception/skip handling.

Module setup (identical preamble to 09_fit_human_data.py):
```python
from __future__ import annotations
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='arviz')

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import jax

import nn4psych.bayesian as bayes
from nn4psych.bayesian import reduced_bayesian_model, fit_with_retry, make_fit_summary, to_jsonable

from scripts.data_pipeline.replay_human_sequences import replay_seed_on_subject
```

CLI:
- `--cohort_manifest` (default `data/processed/rnn_cohort/cohort_manifest.json`)
- `--human_trials` (default `data/processed/nassar2021/subject_trials.npy`)
- `--human_metadata` (default `data/processed/nassar2021/subject_metadata.csv`)
- `--output_dir` (default `data/processed/bayesian/rnn_fits/`)
- `--seed_filter` (optional comma-separated seed indices; default all 20)
- `--subject_filter` (optional)
- `--num_warmup`, `--num_samples`, `--num_chains` (defaults 2000/2000/4)
- `--smoke` (overrides: 1 seed, first 2 subjects, num_warmup=200, num_samples=200, num_chains=2)
- `--single_seed_idx` (for SLURM array — process only this seed; used by cluster/run_rnn_fits.sh)

Pipeline:

1. Load cohort manifest. Filter to OK seeds.
2. Load human subject trials and metadata.
3. **Outer loop over seeds, inner loop over (subject, condition):**
   ```
   for entry in manifest['seeds'] filtered:
       if entry['status'] != 'OK': continue
       seed_idx = entry['seed_idx']
       model_path = entry['model_path']
       for subject_id in subjects:
           for condition in ['changepoint', 'oddball']:
               # Skip if JSON exists (resume support)
               out_path = output_dir / 'per_fit' / f'seed_{seed_idx:02d}_{subject_id}_{condition}.json'
               if out_path.exists(): continue
               # Get human bag sequence for this (subject, condition)
               bag_human = subject_trial_data[subject_id][condition]['bag']  # numpy array
               if len(bag_human) < 50:
                   write_skipped(out_path, reason='LOW_TRIALS', n_trials=len(bag_human))
                   continue
               # Replay through RNN
               bag, bucket = replay_seed_on_subject(model_path, bag_human, condition=condition, pool_modality_context=True)
               # Fit
               try:
                   mcmc, status, attempts = fit_with_retry(
                       reduced_bayesian_model,
                       {'bag_positions': bag, 'bucket_positions': bucket, 'context': condition},
                       seed=hash((seed_idx, subject_id, condition)) % 2**31,
                       num_warmup_first=args.num_warmup,
                       num_samples=args.num_samples,
                       num_chains=args.num_chains,
                   )
                   summary = make_fit_summary(mcmc, status=status, attempts=attempts,
                       var_names=['H','LW','UU','sigma_motor','sigma_LR'],
                       rnn_seed=int(seed_idx), subject_id=subject_id, condition=condition,
                       n_trials=int(len(bag)))
               except Exception as exc:
                   summary = {'rnn_seed': int(seed_idx), 'subject_id': subject_id, 'condition': condition,
                              'status': 'ERROR', 'error': str(exc)}
               with open(out_path, 'w') as f:
                   json.dump(to_jsonable(summary), f, indent=2)
   ```

4. **Compute budget per RESEARCH.md Section 6:** 20 seeds x 134 subjects x 2 conditions = 5360 fits, ~2-10 min each = 18-90 hours sequential. With cluster array over seeds (Task 3), wall time = single-seed sequential (~6 hours) per CONTEXT.md.

5. Implement `aggregate_summary(output_dir)` analogous to 04-03 Task 4: glob per-fit JSONs and produce `summary.csv` with columns `rnn_seed, subject_id, condition, status, H_mean, ..., rhat_max, ess_min, n_divergences, n_attempts`. Add `--aggregate_only` flag.

6. Local smoke test:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/10_fit_rnn_data.py --smoke
# Expected: 1 seed * 2 subjects * 2 conditions = 4 JSONs in data/processed/bayesian/rnn_fits/per_fit/
```
  </action>
  <verify>
```
# Smoke run completes
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/10_fit_rnn_data.py --smoke
ls data/processed/bayesian/rnn_fits/per_fit/ | wc -l
# Expected: >= 4

# JSON shape includes rnn_seed
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import json, glob
files = glob.glob('data/processed/bayesian/rnn_fits/per_fit/*.json')
assert files, 'no per-fit JSONs'
with open(files[0]) as f:
    fit = json.load(f)
required = {'rnn_seed', 'subject_id', 'condition', 'status', 'params'}
assert required.issubset(fit.keys()), f'missing keys: {required - set(fit.keys())}'
print('shape OK; status=', fit['status'], 'rnn_seed=', fit['rnn_seed'])
"

# Aggregate (smoke)
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/10_fit_rnn_data.py --aggregate_only
ls data/processed/bayesian/rnn_fits/summary.csv
```
  </verify>
  <done>
- `10_fit_rnn_data.py` smoke run produces per-fit JSONs with `rnn_seed` column.
- `--single_seed_idx` flag works (for cluster array).
- `aggregate_summary` produces summary.csv.
- Exception handler verified.
  </done>
</task>

<task type="auto">
  <name>Task 3: Write cluster/run_rnn_fits.sh SLURM array (one task per seed)</name>
  <files>cluster/run_rnn_fits.sh</files>
  <action>
Create `cluster/run_rnn_fits.sh` modeled on `cluster/run_rnn_cohort.sh` (04-04a) but CPU-only (no GPU needed for MCMC; NumPyro NUTS runs on CPU per Phase 1 decision):

```bash
#!/bin/bash
# =============================================================================
# SLURM: K=20 RNN Cohort x Human Subjects Fit Array
# =============================================================================
# For each RNN seed, replay all human subjects through the RNN and fit
# Reduced Bayesian per (seed, subject, condition).
#
# Usage:
#   sbatch cluster/run_rnn_fits.sh
# Auto-push:
#   FIT_JID=$(sbatch --parsable cluster/run_rnn_fits.sh)
#   sbatch --parsable --dependency=afterany:${FIT_JID} \
#       --export=ALL,PARENT_JOBS="${FIT_JID}" \
#       cluster/99_push_results.slurm
#
# Performance estimate (per seed, CPU):
#   ~6 hr for 134 subjects x 2 conditions (with 4 chains via XLA_FLAGS)
#   Total wall ≈ 6 hr if all 20 array tasks parallel
# =============================================================================

#SBATCH --job-name=rnn_fits
#SBATCH --output=cluster/logs/rnn_fits_%A_%a.out
#SBATCH --error=cluster/logs/rnn_fits_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=10:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

NUM_WARMUP=${NUM_WARMUP:-2000}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
NUM_CHAINS=${NUM_CHAINS:-4}

module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/setup_env.sh

# 4 virtual JAX devices for 4-chain NUTS
export XLA_FLAGS='--xla_force_host_platform_device_count=4'
export JAX_PLATFORM_NAME=cpu

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "[rnn_fits] seed=${SEED_IDX} num_warmup=${NUM_WARMUP} num_samples=${NUM_SAMPLES} num_chains=${NUM_CHAINS}"

python scripts/data_pipeline/10_fit_rnn_data.py \
    --single_seed_idx ${SEED_IDX} \
    --num_warmup ${NUM_WARMUP} \
    --num_samples ${NUM_SAMPLES} \
    --num_chains ${NUM_CHAINS}

EXIT_CODE=$?
echo "[rnn_fits] seed=${SEED_IDX} exit_code=${EXIT_CODE}"
exit ${EXIT_CODE}
```

Notes:
- `--cpus-per-task=4` matches the 4 chains; SLURM allocates 4 CPU cores per array task.
- `--partition=comp` (CPU partition on M3 — confirm naming with cluster docs; if `comp` is wrong, document in SUMMARY for fix).
- `--time=10:00:00` is conservative; per-seed timing was 6h estimate per RESEARCH.md Section 6.
- Autopush staging needs `data/processed/bayesian/rnn_fits/` covered. Verify in `cluster/99_push_results.slurm`; add if missing.

Run CRLF strip:
```
sed -i 's/\r$//' cluster/run_rnn_fits.sh
chmod +x cluster/run_rnn_fits.sh
```
  </action>
  <verify>
```
ls -la cluster/run_rnn_fits.sh
grep -c "^#SBATCH" cluster/run_rnn_fits.sh
# Expected: at least 8

grep -E "^#SBATCH --array=0-19" cluster/run_rnn_fits.sh
grep -c "10_fit_rnn_data.py" cluster/run_rnn_fits.sh
grep -c "single_seed_idx" cluster/run_rnn_fits.sh

# Autopush coverage
grep -c "rnn_fits\|bayesian/rnn" cluster/99_push_results.slurm
# Expected: >= 1
```
  </verify>
  <done>
- `cluster/run_rnn_fits.sh` exists; CRLF stripped; executable.
- Array directive covers 0-19; CPU partition specified; XLA_FLAGS exported.
- Autopush includes `data/processed/bayesian/rnn_fits/`.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: User submits cluster array; pull aggregated fits and run summary aggregation</name>
  <what-built>
- `replay_human_sequences.py` (RNN forward driver)
- `10_fit_rnn_data.py` (fit driver with cluster + smoke modes)
- `cluster/run_rnn_fits.sh` (SLURM array over 20 seeds)
- Autopush integration
  </what-built>
  <how-to-verify>
**Step 1 (user):** Push and submit:
```
git push origin main
ssh adam@m3.massive.org.au
cd /path/to/nn4psych && git pull origin main
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh
FIT_JID=$(sbatch --parsable cluster/run_rnn_fits.sh)
sbatch --parsable --dependency=afterany:${FIT_JID} \
    --export=ALL,PARENT_JOBS="${FIT_JID}" \
    cluster/99_push_results.slurm
echo "submitted: $FIT_JID"
```

**Step 2 (user):** Wait ~6-12 hr. Monitor:
```
squeue -u $USER -j $FIT_JID
sacct -j $FIT_JID --format=JobID,State,ExitCode,Elapsed
```

**Step 3 (user):** After autopush, pull locally:
```
git pull origin main
ls data/processed/bayesian/rnn_fits/per_fit/ | wc -l
# Expected: ~5360 (20 seeds * 134 subjects * 2 conditions)
# Acceptable: >= 5000 (some seeds may have low-trial skips or errors)
```

**Step 4 (Claude after resume):** Run aggregation:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/10_fit_rnn_data.py --aggregate_only
ls data/processed/bayesian/rnn_fits/summary.csv
```

Report pass-rate breakdown:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import pandas as pd
df = pd.read_csv('data/processed/bayesian/rnn_fits/summary.csv')
print('total fits:', len(df))
print('status counts:', df['status'].value_counts().to_dict())
print('per-seed counts:', df.groupby('rnn_seed').size().to_dict())
print(f'pass rate: {(df[\"status\"]==\"PASS\").mean():.2%}')
print(f'median rhat_max: {df[df[\"status\"]==\"PASS\"][\"rhat_max\"].median():.4f}')
print(f'median ess_min:  {df[df[\"status\"]==\"PASS\"][\"ess_min\"].median():.0f}')
print(f'median n_divergences: {df[df[\"status\"]==\"PASS\"][\"n_divergences\"].median()}')
"
```
  </how-to-verify>
  <resume-signal>
Type "pulled" when local has at least 5000 per-fit JSONs and `summary.csv` is generated. Type "blocked: <reason>" if cluster job failed catastrophically (e.g. all 20 seeds error out — likely indicates replay_human_sequences.py has an obs-schema bug; debug locally before re-submitting).
  </resume-signal>
</task>

</tasks>

<verification>
End-to-end Phase 4 RNN-fit verification:
```
# Per-fit JSONs and summary
ls data/processed/bayesian/rnn_fits/summary.csv
ls data/processed/bayesian/rnn_fits/per_fit/ | wc -l    # >= 5000

# CSV shape
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import pandas as pd
df = pd.read_csv('data/processed/bayesian/rnn_fits/summary.csv')
required = {'rnn_seed', 'subject_id', 'condition', 'status', 'H_mean', 'LW_mean', 'UU_mean', 'sigma_motor_mean', 'sigma_LR_mean', 'rhat_max', 'ess_min', 'n_divergences'}
assert required.issubset(df.columns), f'missing: {required - set(df.columns)}'
pass_rate = (df['status'] == 'PASS').mean()
print(f'rows={len(df)} pass={pass_rate:.2%}')
assert pass_rate >= 0.80, f'pass rate too low: {pass_rate:.2%} (gate: 80%)'
print('PASS')
"
```
</verification>

<success_criteria>
- BAYES-05 met: 20 RNN seeds x 134 subjects x 2 conditions fit cells produced (allowing some skips/errors per CONTEXT.md FAILED policy).
- ROADMAP Phase 4 SC-4 evidence: median R-hat <= 1.01 and median ESS_bulk >= 400 across PASS fits; divergence distribution documented.
- Pool-across-modality_context implemented and verified in replay logic.
- summary.csv ready for Phase 5 cohort comparison.
- Pass rate >= 80% across all (seed, subject, condition) cells.
</success_criteria>

<output>
After completion, create `.planning/phases/04-bayesian-model-fitting/04-04b-SUMMARY.md`:
- What was built: replay driver, fit driver, cluster script
- Cluster job ID, timing, per-seed completion
- Pass-rate breakdown by seed and by condition
- Median R-hat, ESS_bulk, divergences
- Decisions logged: obs schema confirmed for replay; modality_context pooling method
- Tech-stack additions: none new
- Required-by-next-plan: Phase 5 ingests human summary.csv (04-03) + RNN summary.csv (04-04b) for group comparison
- Open follow-ups: any seeds with high error rates; whether to re-train problematic seeds before Phase 5
</output>
