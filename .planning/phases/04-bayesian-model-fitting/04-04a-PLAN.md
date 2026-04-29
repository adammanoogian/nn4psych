---
phase: 04-bayesian-model-fitting
plan: 04a
type: execute
wave: 4
depends_on: ["04-01"]
gap_closure: false
autonomous: false
files_modified:
  - cluster/run_rnn_cohort.sh
  - cluster/setup_env.sh
  - cluster/99_push_results.slurm
  - data/processed/rnn_cohort/
  - scripts/training/train_rnn_canonical.py
user_setup:
  - service: monash_m3_cluster
    why: "Cluster GPU allocation for SLURM array K=20 RNN training; SSH/SLURM submission requires user-side authentication"
    dashboard_config:
      - task: "Push branch to origin so cluster can git pull updated cluster/run_rnn_cohort.sh and train_rnn_canonical.py"
        location: "Local repo (git push origin main)"
      - task: "SSH to Monash M3, pull repo, sbatch cluster/run_rnn_cohort.sh, and run cluster/99_push_results.slurm afterany dependency"
        location: "Monash M3 cluster (m3.massive.org.au)"
must_haves:
  truths:
    - "K=20 independent seeds of the canonical PIE_CP_OB_v2 RNN are trained via SLURM array job"
    - "Each seed's trained model checkpoint and final reward curve are saved to data/processed/rnn_cohort/seed_{i:02d}/"
    - "Checkpoints are pulled from cluster to local via the established autopush mechanism (run_circuit_ensemble.sh pattern), and 99_push_results.slurm has the required stage_files calls for rnn_cohort artifacts (M6 fix)"
    - "All 20 checkpoints loadable via existing nn4psych.models.actor_critic.ActorCritic loader"
    - "Training reward curves indicate learning (final reward not flat or diverging) for at least 18/20 seeds; failed seeds documented but not blocking 04-04b"
  artifacts:
    - path: "cluster/run_rnn_cohort.sh"
      provides: "SLURM array job (--array=0-19) that trains 20 RNN seeds with --seed ${SLURM_ARRAY_TASK_ID}; SBATCH directives, env setup, output to data/processed/rnn_cohort/seed_$SLURM_ARRAY_TASK_ID/"
      contains: "SBATCH --array"
    - path: "data/processed/rnn_cohort/seed_00/model.pt"
      provides: "Trained ActorCritic checkpoint for seed 0 (analogous for seeds 01..19)"
    - path: "data/processed/rnn_cohort/seed_00/training_log.json"
      provides: "Per-epoch reward curve and final stats for QA"
      contains: "rewards"
    - path: "data/processed/rnn_cohort/cohort_manifest.json"
      provides: "List of seed indices with checkpoint paths and final reward; consumed by 04-04b for RNN cohort iteration"
      contains: "seeds"
    - path: "cluster/99_push_results.slurm"
      provides: "Autopush script extended with stage_files calls for rnn_cohort artifacts (M6 fix)"
      contains: "rnn_cohort"
  key_links:
    - from: "cluster/run_rnn_cohort.sh"
      to: "scripts/training/train_rnn_canonical.py"
      via: "python scripts/training/train_rnn_canonical.py --seed ${SLURM_ARRAY_TASK_ID} --output_dir data/processed/rnn_cohort/seed_${SLURM_ARRAY_TASK_ID}"
      pattern: "train_rnn_canonical.py"
    - from: "data/processed/rnn_cohort/cohort_manifest.json"
      to: "scripts/data_pipeline/10_fit_rnn_data.py (Plan 04-04b)"
      via: "04-04b reads manifest to iterate over K=20 seeds"
      pattern: "cohort_manifest"
---

<objective>
Re-train K=20 independent seeds of the canonical PIE_CP_OB_v2 RNN on the Monash M3 cluster via a SLURM array job, then pull all checkpoints to local. This is cluster-side compute that produces the RNN cohort consumed by 04-04b for replay-and-fit.

`autonomous: false` because cluster job submission requires SSH/SLURM authentication that Claude cannot perform headlessly — the user must `git push`, SSH to M3, and submit the job. After submission, cluster autopush returns checkpoints to local without further user action.

Purpose: BAYES-05 prerequisite. Phase 4 needs 20 RNN seeds to fit Reduced Bayesian against (RNN-as-cognitive-cohort). This plan reuses the established Phase 3 cluster infrastructure (cluster/setup_env.sh, autopush via 99_push_results.slurm).

Output:
- `cluster/run_rnn_cohort.sh` (new SLURM array script)
- `cluster/99_push_results.slurm` extended with `stage_files` calls for rnn_cohort (M6 fix)
- `data/processed/rnn_cohort/seed_{00..19}/` (20 directories with model.pt + training_log.json)
- `data/processed/rnn_cohort/cohort_manifest.json`
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
@cluster/run_circuit_ensemble.sh
@cluster/setup_env.sh
@cluster/99_push_results.slurm
@scripts/training/train_rnn_canonical.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add --seed and --output_dir CLI to train_rnn_canonical.py if not present; verify smoke locally</name>
  <files>scripts/training/train_rnn_canonical.py</files>
  <action>
Read the current `scripts/training/train_rnn_canonical.py` to confirm its CLI surface. Per project CLAUDE.md, this script accepts `--epochs --trials --maxt`. For 04-04a, we need:
- `--seed N` (integer; sets numpy / torch / random seeds; passes to env constructor for reproducible trial sequences)
- `--output_dir PATH` (directory where the final model checkpoint and training log are saved as `model.pt` and `training_log.json`)

If these flags ALREADY exist (likely from Phase 2), confirm and skip the edit. If they DO NOT exist, add them with minimal disruption:

1. Add to argparse:
```python
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--output_dir', type=str, default='data/processed/rnn_canonical/', help='Output directory for checkpoint and log')
```

2. Apply seed at script start:
```python
import random
import numpy as np
import torch

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
Call `set_seeds(args.seed)` before any model construction.

3. After training, save:
```python
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {...},
    'seed': args.seed,
}, output_dir / 'model.pt')
with open(output_dir / 'training_log.json', 'w') as f:
    json.dump({
        'rewards': [float(r) for r in episode_rewards],   # cast for JSON
        'final_reward': float(episode_rewards[-1]),
        'n_epochs': int(args.epochs),
        'seed': int(args.seed),
    }, f, indent=2)
```

Cast all numpy / torch scalars to Python builtins before json.dump (recurring lesson STATE.md decision 02-03).

4. Smoke test locally per project CLAUDE.md (small params, single process, 16GB RAM safe):
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/training/train_rnn_canonical.py --epochs 3 --trials 10 --maxt 30 --seed 99 --output_dir /tmp/rnn_smoke
```
Confirm `/tmp/rnn_smoke/model.pt` and `/tmp/rnn_smoke/training_log.json` exist.

If train_rnn_canonical.py is missing the canonical training loop (i.e. it's a placeholder), STOP and surface this to the user — do NOT scope-creep this plan into rewriting Phase 2 work. Reference: `scripts/training/train_rnn_canonical.py` should already be functional from Phase 2 (TRAIN-01 completed).
  </action>
  <verify>
```
# Smoke run
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/training/train_rnn_canonical.py --epochs 3 --trials 10 --maxt 30 --seed 99 --output_dir /tmp/rnn_smoke
ls /tmp/rnn_smoke/model.pt /tmp/rnn_smoke/training_log.json
# Both exist; training_log.json contains 'rewards' list
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import json
with open('/tmp/rnn_smoke/training_log.json') as f:
    log = json.load(f)
assert 'rewards' in log and 'seed' in log and log['seed'] == 99, f'log shape wrong: {log.keys()}'
print('smoke OK; final_reward=', log['final_reward'])
"
```
  </verify>
  <done>
- `train_rnn_canonical.py` accepts `--seed` and `--output_dir`.
- Smoke run produces `model.pt` and `training_log.json` with seed-tagged content.
- All numpy/torch scalars JSON-serializable.
  </done>
</task>

<task type="auto">
  <name>Task 2: Write cluster/run_rnn_cohort.sh SLURM array job (K=20 seeds) and add stage_files calls to 99_push_results.slurm</name>
  <files>cluster/run_rnn_cohort.sh,cluster/99_push_results.slurm</files>
  <action>
**A. Create `cluster/run_rnn_cohort.sh`** modeled on `cluster/run_circuit_ensemble.sh` (which is the reference for Phase-3-established cluster infrastructure: setup_env.sh, miniforge3 module, autopush pattern, CRLF strip).

Structure:

```bash
#!/bin/bash
# =============================================================================
# SLURM: K=20 RNN Cohort Re-Training (PIE_CP_OB_v2)
# =============================================================================
# Trains K=20 independent seeds of the canonical PIE RNN as a SLURM array job.
# Each array task trains one seed and writes to data/processed/rnn_cohort/seed_${SLURM_ARRAY_TASK_ID}/.
#
# Usage:
#   sbatch cluster/run_rnn_cohort.sh
#   sbatch --export=K=20,EPOCHS=150,TRIALS=200 cluster/run_rnn_cohort.sh
#
# Auto-push results after all array tasks finish (Phase 3 pattern):
#   sed -i 's/\r$//' cluster/*.slurm cluster/*.sh
#   FIT_JID=$(sbatch --parsable cluster/run_rnn_cohort.sh)
#   sbatch --parsable --dependency=afterany:${FIT_JID} \
#       --export=ALL,PARENT_JOBS="${FIT_JID}" \
#       cluster/99_push_results.slurm
#
# Performance estimate (per seed, GPU):
#   ~30-60 min per seed; total wall ≈ 60 min if K parallel array tasks
#
# Reference: cluster/run_circuit_ensemble.sh (Phase 3) for the env-setup pattern
# =============================================================================

#SBATCH --job-name=rnn_cohort
#SBATCH --output=cluster/logs/rnn_cohort_%A_%a.out
#SBATCH --error=cluster/logs/rnn_cohort_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2

# Configuration overrides via --export
EPOCHS=${EPOCHS:-150}
TRIALS=${TRIALS:-200}
MAXT=${MAXT:-100}

# Environment setup (Phase 3 pattern)
module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Activate env via setup_env.sh (creates if missing)
source cluster/setup_env.sh

# Output directory: data/processed/rnn_cohort/seed_${SLURM_ARRAY_TASK_ID}/
SEED_IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
OUT_DIR="data/processed/rnn_cohort/seed_${SEED_IDX}"
mkdir -p "${OUT_DIR}"

echo "[rnn_cohort] seed=${SLURM_ARRAY_TASK_ID} epochs=${EPOCHS} trials=${TRIALS} maxt=${MAXT} out=${OUT_DIR}"
echo "[rnn_cohort] hostname=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null || echo 'no nvidia-smi')"

python scripts/training/train_rnn_canonical.py \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --epochs ${EPOCHS} \
    --trials ${TRIALS} \
    --maxt ${MAXT} \
    --output_dir "${OUT_DIR}"

EXIT_CODE=$?
echo "[rnn_cohort] seed=${SLURM_ARRAY_TASK_ID} exit_code=${EXIT_CODE}"
exit ${EXIT_CODE}
```

Notes:
- `--array=0-19` means 20 array tasks indexed 0-19, mapping to seed indices 0-19.
- `EXIT_CODE` propagated so SLURM marks failed seeds as failed (autopush still fires via afterany).
- GPU partition matches Phase 3 setup; if M3 partition naming differs, document in SUMMARY (planner has limited cluster-side visibility).
- `setup_env.sh` already creates the cluster conda env per Phase 3 — reuse, do not duplicate.

Run CRLF strip locally:
```
sed -i 's/\r$//' cluster/run_rnn_cohort.sh
chmod +x cluster/run_rnn_cohort.sh
```

**B. M6 fix — Extend `cluster/99_push_results.slurm` with explicit `stage_files` calls for rnn_cohort artifacts.**

The existing `cluster/99_push_results.slurm` defines a `stage_files <pattern> <description>` function (around line 205) that all autopush rules use. **Bare `git add` lines bypass dry-run logic and break parity with the rest of the autopush workflow.** Per checker iteration 1, the previous draft suggested a "defensive `git add`" — that is INCORRECT for this script. Use `stage_files` only.

Locate the `stage_files` block in `99_push_results.slurm` (after the existing rules for `output/circuit_analysis/...` and `data/processed/rnn_behav/...`). Append these EXACT lines (must match RNN-cohort artifact paths exactly):

```bash
# Phase 4 RNN cohort (04-04a)
stage_files "data/processed/rnn_cohort/seed_*/model.pt" "RNN cohort checkpoints"
stage_files "data/processed/rnn_cohort/seed_*/training_log.json" "RNN cohort training logs"
stage_files "data/processed/rnn_cohort/cohort_manifest.json" "RNN cohort manifest"
```

Do NOT use `git add` for these paths. Do NOT add wildcards above the seed level. Do NOT touch the existing rules.

Verify the patterns are present and use the `stage_files` function (not bare `git add`):
```
grep -c "stage_files \"data/processed/rnn_cohort/" cluster/99_push_results.slurm
# Expected: 3 (one for model.pt, one for training_log.json, one for cohort_manifest.json)

# Defensive: confirm no bare git-add lines for rnn_cohort were introduced
grep -E "^\s*git add.*rnn_cohort" cluster/99_push_results.slurm | wc -l
# Expected: 0
```

Run CRLF strip on 99_push_results.slurm:
```
sed -i 's/\r$//' cluster/99_push_results.slurm
```
  </action>
  <verify>
```
# File exists and is executable
ls -la cluster/run_rnn_cohort.sh
# SBATCH directives parseable (lint via grep)
grep -c "^#SBATCH" cluster/run_rnn_cohort.sh
# Expected: at least 8 (job-name, output, error, array, time, mem, gres, partition, cpus-per-task)

# Array directive correct
grep -E "^#SBATCH --array=0-19" cluster/run_rnn_cohort.sh

# train_rnn_canonical.py invocation present with all required args
grep -c "scripts/training/train_rnn_canonical.py" cluster/run_rnn_cohort.sh
grep -c -- "--seed \${SLURM_ARRAY_TASK_ID}" cluster/run_rnn_cohort.sh

# M6 fix: autopush staging uses stage_files (NOT bare git add) for rnn_cohort
grep -c "stage_files \"data/processed/rnn_cohort/" cluster/99_push_results.slurm
# Expected: 3

# M6 fix: no bare git-add for rnn_cohort
test "$(grep -E "^\s*git add.*rnn_cohort" cluster/99_push_results.slurm | wc -l)" -eq 0 && echo "no bare git-add OK"
```
  </verify>
  <done>
- `cluster/run_rnn_cohort.sh` exists with #SBATCH --array=0-19, GPU directive, and full training command.
- CRLF stripped, executable.
- M6 fix: `cluster/99_push_results.slurm` has exactly 3 new `stage_files` calls covering `seed_*/model.pt`, `seed_*/training_log.json`, and `cohort_manifest.json`. No bare `git add` lines were introduced.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: User submits SLURM array job and reports cohort manifest after autopush returns</name>
  <what-built>
- `cluster/run_rnn_cohort.sh` SLURM array (K=20)
- `train_rnn_canonical.py` updated with --seed and --output_dir
- `cluster/99_push_results.slurm` updated with 3 `stage_files` calls for rnn_cohort artifacts (M6)
  </what-built>
  <how-to-verify>
**Step 1 (user):** Push branch and SSH to M3.
```
git push origin main   # or current branch; cluster pulls from origin
ssh adam@m3.massive.org.au   # or the configured M3 SSH alias
cd /path/to/nn4psych  # cluster checkout
git pull origin main
```

**Step 2 (user):** Submit array job + autopush dependency:
```
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh
FIT_JID=$(sbatch --parsable cluster/run_rnn_cohort.sh)
echo "submitted array job: $FIT_JID"
sbatch --parsable --dependency=afterany:${FIT_JID} \
    --export=ALL,PARENT_JOBS="${FIT_JID}" \
    cluster/99_push_results.slurm
```

**Step 3 (user):** Wait ~60-120 min for the array to complete. Monitor:
```
squeue -u $USER -j $FIT_JID
sacct -j $FIT_JID --format=JobID,State,ExitCode,Elapsed
ls cluster/logs/rnn_cohort_${FIT_JID}_*.out | head -3
```

**Step 4 (user):** After autopush completes, the local repo will have new commits with `data/processed/rnn_cohort/seed_*/` directories. Pull locally:
```
git pull origin main
ls data/processed/rnn_cohort/
```

Expected: 20 directories `seed_00/` through `seed_19/`, each with `model.pt` and `training_log.json`.

**Step 5 (Claude, after resume-signal):** Build `data/processed/rnn_cohort/cohort_manifest.json`:
```python
import json, glob
from pathlib import Path
manifest = {'seeds': []}
for d in sorted(glob.glob('data/processed/rnn_cohort/seed_*')):
    seed_idx = int(Path(d).name.split('_')[1])
    log_path = Path(d) / 'training_log.json'
    model_path = Path(d) / 'model.pt'
    if log_path.exists() and model_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        manifest['seeds'].append({
            'seed_idx': seed_idx,
            'model_path': str(model_path),
            'training_log_path': str(log_path),
            'final_reward': log.get('final_reward'),
            'n_epochs': log.get('n_epochs'),
            'status': 'OK',
        })
    else:
        manifest['seeds'].append({'seed_idx': seed_idx, 'status': 'MISSING_FILES'})
manifest['n_seeds_ok'] = sum(1 for s in manifest['seeds'] if s.get('status') == 'OK')
with open('data/processed/rnn_cohort/cohort_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'manifest written; {manifest["n_seeds_ok"]}/20 seeds OK')
```
  </how-to-verify>
  <resume-signal>
Type "pulled" once `git pull` brought down at least 18/20 seed directories (per must_haves "at least 18/20 seeds"). Then Claude builds cohort_manifest.json and validates pass-rate. Type "blocked: <reason>" if cluster job failed entirely (e.g. partition unavailable, env-build error) and the planner will route to a cluster diagnosis follow-up.
  </resume-signal>
</task>

</tasks>

<verification>
After Task 3 resume:
```
# Manifest exists
ls data/processed/rnn_cohort/cohort_manifest.json

# At least 18/20 seeds OK
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import json
with open('data/processed/rnn_cohort/cohort_manifest.json') as f:
    manifest = json.load(f)
n_ok = manifest['n_seeds_ok']
assert n_ok >= 18, f'expected at least 18 seeds OK, got {n_ok}'
final_rewards = [s['final_reward'] for s in manifest['seeds'] if s.get('final_reward') is not None]
print(f'n_ok={n_ok}/20; final_reward median={sorted(final_rewards)[len(final_rewards)//2]:.3f}')
print('PASS')
"

# Each OK seed has loadable model.pt
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import torch
import json
with open('data/processed/rnn_cohort/cohort_manifest.json') as f:
    manifest = json.load(f)
for entry in manifest['seeds'][:3]:
    if entry.get('status') == 'OK':
        ckpt = torch.load(entry['model_path'], map_location='cpu', weights_only=False)
        assert 'model_state_dict' in ckpt, f'malformed checkpoint: {ckpt.keys()}'
        print(f'seed {entry[\"seed_idx\"]} loadable')
print('checkpoints OK')
"
```
</verification>

<success_criteria>
- `cluster/run_rnn_cohort.sh` SLURM array submitted and completed.
- `cluster/99_push_results.slurm` has 3 `stage_files` calls for rnn_cohort artifacts (M6 fix; no bare `git add` lines).
- 18+ of 20 seeds have valid `model.pt` and `training_log.json`.
- `cohort_manifest.json` enumerates all OK seeds.
- Failed seeds (if any < 2) documented in SUMMARY but do not block 04-04b.
- Final reward distribution shows learning curves (not flat) per Phase 2 TRAIN-01 acceptance.
</success_criteria>

<output>
After completion, create `.planning/phases/04-bayesian-model-fitting/04-04a-SUMMARY.md`:
- What was built: SLURM array, train script CLI updates, manifest, autopush stage_files extension (M6)
- Cluster job ID and timing
- Per-seed final reward distribution
- Decisions logged: any partition / wall-time tweaks; failed seeds and why
- Required-by-next-plan: 04-04b iterates over `cohort_manifest.json` seeds[*]['model_path']
- Open follow-ups: any seeds that need re-training before Phase 5 cohort comparison
</output>
