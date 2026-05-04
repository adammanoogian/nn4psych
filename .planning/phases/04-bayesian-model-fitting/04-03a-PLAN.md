---
phase: 04-bayesian-model-fitting
plan: 03a
type: execute
wave: 2.5
depends_on: ["04-01", "04-02"]
gap_closure: false
autonomous: true
files_modified:
  - scripts/data_pipeline/extract_nassar_trials.py
  - data/processed/nassar2021/subject_trials.npy
  - data/processed/nassar2021/subject_metadata.csv
  - scripts/validation/validate_rbo_vs_matlab.py
  - src/nn4psych/bayesian/_frugfun_reference.py
  - src/nn4psych/bayesian/reduced_bayesian.py  # only if math divergences are real
  - tests/test_reduced_bayesian.py             # add MATLAB-parity test
  - .planning/phases/04-bayesian-model-fitting/04-03a-SUMMARY.md
must_haves:
  truths:
    - "extract_nassar_trials.py NASSAR_DIR points at data/raw/nassar2021/Brain2021Code/ (not the legacy Github/Nassar_et_al_2021 path)"
    - "Cleaning runs successfully on all 134 subjects (32 NC + 46 Patients + 56 Patients2); per-subject excluded-trial counts written to subject_metadata.csv"
    - "NumPyro compute_rbo_forward and a Python port of frugFun5.m produce numerically equivalent learning_rate, omega, and belief trajectories on a shared synthetic dataset (max abs diff < 1e-3 for B; < 1e-4 for omega/alpha at LW=1.0)"
    - "If divergences exist: documented (truncated normal, log-space LW, drift, R-vs-tau-update form) and EITHER fixed in compute_rbo_forward OR explicitly justified as 'paper-deviation accepted' in 04-03a-SUMMARY.md"
    - "tests/test_reduced_bayesian.py gains a test_matlab_parity test that fails-fast if math drifts"
    - "All existing 6 tests continue to pass after any compute_rbo_forward changes"
  artifacts:
    - path: "scripts/validation/validate_rbo_vs_matlab.py"
      provides: "Side-by-side comparator: runs both NumPyro forward and Python-port frugFun5 reference on synthetic deltas; prints per-trajectory diffs and writes a CSV"
      contains: "if __name__"
    - path: "src/nn4psych/bayesian/_frugfun_reference.py"
      provides: "Faithful Python/NumPy port of frugFun5.m (CPU-only reference; not used in production fits, only validation). Underscore prefix marks internal."
      contains: "def frugfun5_reference"
    - path: "data/processed/nassar2021/subject_trials.npy"
      provides: "Cleaned per-subject behavioral data from extract_nassar_trials.py (134 subjects × 4 conditions × ~100 trials each, after AASP cleaning)"
    - path: "data/processed/nassar2021/subject_metadata.csv"
      provides: "Per-subject metadata: subject_id, is_patient, n_trials, n_excluded"
      contains: "n_trials"
    - path: "tests/test_reduced_bayesian.py"
      provides: "Now includes test_matlab_parity guard"
      contains: "test_matlab_parity"
  key_links:
    - from: "scripts/validation/validate_rbo_vs_matlab.py"
      to: "src/nn4psych/bayesian/_frugfun_reference.py"
      via: "imports frugfun5_reference for the reference trajectory"
      pattern: "_frugfun_reference"
    - from: "tests/test_reduced_bayesian.py::test_matlab_parity"
      to: "src/nn4psych/bayesian/_frugfun_reference.py"
      via: "imports frugfun5_reference; ensures parity holds at HEAD"
      pattern: "frugfun5_reference"
---

<objective>
Two prereq tasks gating 04-03 (Human Data Fits): (1) actually run the human-data cleaning pipeline against the now-extracted Brain2021Code so we can see how much dirty data exists per subject, and (2) cross-validate the NumPyro Reduced Bayesian Observer (`compute_rbo_forward`) against the canonical MATLAB implementation `frugFun5.m` from the Nassar lab.

Why this is prereq, not part of 04-03: re-running the 50-dataset param recovery (BAYES-06 gate for 04-03 Task 2) and running 268 per-subject MCMC fits both consume the SAME forward model. If the NumPyro math diverges from MATLAB in a non-trivial way (truncated-normal omega, log-space likeWeight, drift parameter, R-vs-tau-update form), every downstream fit is paper-incomparable. Catching that *before* burning ~17 h on recovery + ~5 h on human fits is the cheap win.

`autonomous: true` — local-only, no SSH/SLURM, no external user action. Runs in ~30 min wall.

Output:
- Cleaned human data on disk + per-subject stats
- Side-by-side validation that NumPyro and MATLAB math agree (or a documented discrepancy with corresponding fix)
- One new pytest `test_matlab_parity` that fails-fast on future drift
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
@scripts/data_pipeline/extract_nassar_trials.py            (existing, has wrong path)
@src/nn4psych/bayesian/reduced_bayesian.py                 (compute_rbo_forward, the JAX impl)
@data/raw/nassar2021/Brain2021Code/functionCodes/frugFun5.m       (MATLAB reference)
@data/raw/nassar2021/Brain2021Code/functionCodes/frugFun5_uniformOddballs.m  (OB variant if applicable)
@data/raw/nassar2021/Brain2021Code/functionCodes/computeLR.m      (LR helpers)
@data/raw/nassar2021/Brain2021Code/functionCodes/computePrecFromRelWeights.m  (tau↔R conversions)
@data/raw/nassar2021/Brain2021Code/functionCodes/fitFrugFunSchiz.m  (fitting driver showing fmincon bounds)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix extract_nassar_trials.py path and run cleaning</name>
  <files>scripts/data_pipeline/extract_nassar_trials.py,data/processed/nassar2021/subject_trials.npy,data/processed/nassar2021/subject_metadata.csv</files>
  <action>
The existing `extract_nassar_trials.py` has `NASSAR_DIR = Path('C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code')` — points to a different repo that doesn't exist on this machine. Fix the path to use the repo's `RAW_DATA_DIR`:

```python
from config import RAW_DATA_DIR  # already imported as OUTPUT_DIR; add RAW_DATA_DIR
NASSAR_DIR = RAW_DATA_DIR / 'nassar2021' / 'Brain2021Code'
```

(The actual extracted data was just placed at `data/raw/nassar2021/Brain2021Code/` by Plan 04-03 Task 1 prework on 2026-05-04.)

Then run the script:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/extract_nassar_trials.py
```

Expected output:
- `data/processed/nassar2021/subject_trials.npy` — list of 134 dicts (per-subject)
- `data/processed/nassar2021/subject_metadata.csv` — 134 rows
- stdout: per-subject data-cleaning statistics (total trials, dropped, non-finite, valid)

If the script raises errors:
- Wrong field name in `.mat` (e.g. `blockCompletedTrials` doesn't exist in this version) → debug minimally with `scipy.io.loadmat` inspection. Document the actual field names in the script docstring. Do NOT rewrite the cleaning logic.
- Wrong directory layout (e.g. subjects directly under `realSubjects/` not under `Patients/Patients2/Normal Controls/` subdirs) → adjust glob patterns. The actual layout: `realSubjects/{Patients,Patients2,Normal Controls}/SP_*/<files>.mat`.
- Subject directories use `SP_*` prefix; verify with `ls`.

After successful run, print to stdout (the executor reports this in SUMMARY):
- Total subjects loaded
- Total trials before cleaning, total after
- % trials excluded
- Per-cohort breakdown: NC mean trials/subject, Patients mean, Patients2 mean
  </action>
  <verify>
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/data_pipeline/extract_nassar_trials.py 2>&1 | tail -30

ls -la data/processed/nassar2021/subject_trials.npy data/processed/nassar2021/subject_metadata.csv

/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "
import numpy as np, pandas as pd
trials = np.load('data/processed/nassar2021/subject_trials.npy', allow_pickle=True)
meta = pd.read_csv('data/processed/nassar2021/subject_metadata.csv')
print(f'n_subjects = {len(trials)}')
print(f'n_subjects in metadata = {len(meta)}')
print(f'patients = {meta[\"is_patient\"].sum()}, controls = {(~meta[\"is_patient\"]).sum()}')
print(f'mean valid trials/subject = {meta[\"n_trials\"].mean():.1f}')
print(f'median valid trials/subject = {meta[\"n_trials\"].median():.1f}')
print(f'mean excluded/subject = {meta[\"n_excluded\"].mean():.1f}')
assert len(trials) >= 130, f'expected ~134 subjects, got {len(trials)}'
assert meta['n_trials'].min() > 100, f'min n_trials too low: {meta[\"n_trials\"].min()}'
print('cleanup OK')
"
```
  </verify>
  <done>
- subject_trials.npy and subject_metadata.csv exist.
- 130+ subjects loaded successfully.
- Per-subject excluded-trial counts in metadata.
- SUMMARY records cleaning statistics.
  </done>
</task>

<task type="auto">
  <name>Task 2: Port frugFun5.m to NumPy reference and write side-by-side validator</name>
  <files>src/nn4psych/bayesian/_frugfun_reference.py,scripts/validation/validate_rbo_vs_matlab.py</files>
  <action>
**A. Create `src/nn4psych/bayesian/_frugfun_reference.py`** — a faithful, line-by-line port of `frugFun5.m` from `data/raw/nassar2021/Brain2021Code/functionCodes/frugFun5.m`. Pure NumPy, no JAX. Underscore prefix marks it as internal validation code, NOT for production fits.

Function signature (matching MATLAB exactly):
```python
def frugfun5_reference(
    data: np.ndarray,           # bag positions, shape (n,)
    Hazard: float,              # H ∈ [0, 1]
    noise: float,               # sigmaE (= SIGMA_N = 20.0)
    drift: float = 0.0,         # OB drift; 0 for default CP/OB
    likeWeight: float = 1.0,    # LW ∈ [0, 1]
    trueRun: int = 0,           # 0 = second-moment, 1 = run-length mean
    initGuess: float = 150.0,   # B(1) initial belief
    inRun: float = 1.0,         # R(1) initial run length
) -> dict:
    """Faithful NumPy port of frugFun5.m for cross-validation only.

    Returns
    -------
    {'B': beliefs (n+1,), 'totSig': (n,), 'R': (n+1,), 'pCha': (n,),
     'sigmaU': (n,), 'alpha': (n,)}
    """
```

Translation rules (match MATLAB IEEE arithmetic, keep tiny epsilons OUT of the reference — paper math has none):

1. **Truncated-normal `pI`** (lines 86-90):
   ```python
   pI = norm.pdf(data[i], B[i], totSig[i])
   normalize = norm.cdf(300, B[i], totSig[i]) - norm.cdf(0, B[i], totSig[i])
   pI = pI / normalize
   ```

2. **Uniform component** (lines 73, 93-97): `d = ones(300)/300`. So `changLike = 1/300` for `data ∈ [1, 300]`. Outside that range, paper code uses `d(1) = 1/300` (line 94) — i.e. fixed at 1/300. This means the U component is just the constant `1/300` regardless of value (because `d` is uniform).

3. **Log-space changeRatio** (line 99):
   ```python
   change_ratio = np.exp(likeWeight * np.log(changLike / pI) + np.log(Hazard / (1 - Hazard)))
   pCha[i] = change_ratio / (change_ratio + 1) if np.isfinite(change_ratio) else 1.0
   ```

4. **Belief update** (lines 110-115):
   ```python
   yInt = 1.0 / (R[i] + 1)
   slope = 1 - yInt
   alpha = yInt + pCha[i] * slope
   delta_t = data[i] - B[i]
   B[i+1] = B[i] + alpha * delta_t
   ```

5. **R-update** (lines 117-127), default `trueRun=0`:
   ```python
   ss = (
       pCha[i] * (sigmaE**2 / 1)
       + (1 - pCha[i]) * (sigmaE**2 / (R[i] + 1))
       + pCha[i] * (1 - pCha[i]) * ((B[i] + yInt * delta_t) - data[i]) ** 2
   )
   R[i+1] = sigmaE**2 / ss   # second-moment match (run-length-equivalent)
   ```

   For `trueRun=1`: `R[i+1] = (R[i] + 1) * (1 - pCha[i]) + pCha[i]`.

6. **sigmaU** (line 81): `sigmaU = sqrt((sigmaE / sqrt(R))**2 + drift**2)`.

7. **R recompute** (line 82): `R[i] = noise**2 / sigmaU[i]**2` (overwrites with drift-aware value before pCha computation).

8. **totSig** (line 84): `totSig = sqrt(sigmaE**2 + sigmaU**2)`.

Add a docstring noting this is the reference, not for production. Imports: only numpy and scipy.stats.norm.

**B. Create `scripts/validation/validate_rbo_vs_matlab.py`** — runs both implementations on the same synthetic data and prints diffs.

```python
"""Cross-validate NumPyro compute_rbo_forward against MATLAB frugFun5.m port.

Generates synthetic helicopter task trials (CP and OB conditions), runs both
reference implementations with matched parameters, and compares the resulting
trajectories. Fails (exit 1) if any per-trial divergence exceeds tolerance.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
import jax.numpy as jnp

from nn4psych.bayesian.reduced_bayesian import compute_rbo_forward, SIGMA_N
from nn4psych.bayesian._frugfun_reference import frugfun5_reference


def generate_synthetic_trials(
    n_trials: int = 200,
    hazard_rate: float = 0.125,  # paper default
    sigma_N: float = 20.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (helicopter_pos, bag_pos) trial arrays."""
    rng = np.random.default_rng(rng_seed)
    helicopter_pos = np.zeros(n_trials)
    bag_pos = np.zeros(n_trials)
    helicopter_pos[0] = rng.uniform(80, 220)
    for i in range(n_trials):
        if i > 0 and rng.random() < hazard_rate:
            helicopter_pos[i] = rng.uniform(0, 300)
        elif i > 0:
            helicopter_pos[i] = helicopter_pos[i - 1]
        bag_pos[i] = np.clip(rng.normal(helicopter_pos[i], sigma_N), 0, 300)
    return helicopter_pos, bag_pos


def run_numpyro(
    bag_positions: np.ndarray,
    bucket_positions: np.ndarray,
    H: float, LW: float, UU: float,
    context: str,
) -> dict:
    """compute_rbo_forward consumes pred_errors = bag - bucket."""
    pred_errors = bag_positions - bucket_positions
    params = {"H": jnp.asarray(H), "LW": jnp.asarray(LW), "UU": jnp.asarray(UU)}
    lr, upd, omega, tau = compute_rbo_forward(params, jnp.asarray(pred_errors), context)
    return {"alpha": np.asarray(lr), "omega": np.asarray(omega), "tau": np.asarray(tau)}


def run_matlab(bag_positions: np.ndarray, H: float, LW: float, **kwargs) -> dict:
    return frugfun5_reference(bag_positions, Hazard=H, noise=SIGMA_N, likeWeight=LW, **kwargs)


def compare(numpyro_out: dict, matlab_out: dict, label: str) -> dict:
    """Compute max-abs and median-abs diffs across trajectories."""
    # NumPyro: alpha (n,), omega (n,), tau (n+1,)
    # MATLAB: alpha (n,), pCha (n,), R (n+1,) — note R != tau
    diffs = {
        "alpha_max_abs": float(np.max(np.abs(numpyro_out["alpha"] - matlab_out["alpha"]))),
        "alpha_median_abs": float(np.median(np.abs(numpyro_out["alpha"] - matlab_out["alpha"]))),
        "omega_max_abs": float(np.max(np.abs(numpyro_out["omega"] - matlab_out["pCha"]))),
        "omega_median_abs": float(np.median(np.abs(numpyro_out["omega"] - matlab_out["pCha"]))),
    }
    print(f"\n=== {label} ===")
    for k, v in diffs.items():
        print(f"  {k}: {v:.6e}")
    return diffs


def main():
    parser = argparse.ArgumentParser(description="NumPyro vs MATLAB frugFun5 parity check")
    parser.add_argument("--alpha_tol", type=float, default=1e-3)
    parser.add_argument("--omega_tol", type=float, default=1e-3)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/processed/bayesian/matlab_parity_diffs.csv"))
    args = parser.parse_args()

    helicopter, bag = generate_synthetic_trials(n_trials=args.n_trials)
    # The MATLAB code is "self-bucketing": it computes the bucket trajectory it
    # would itself predict. To make a fair comparison, run MATLAB ONCE to get
    # the bucket trajectory, then feed pred_errors = bag - bucket back into
    # compute_rbo_forward.

    rows = []
    for context_label, H, LW, UU in [
        ("changepoint_default", 0.125, 1.0, 1.0),
        ("changepoint_LW0.5",   0.125, 0.5, 1.0),
        ("oddball_default",     0.125, 1.0, 1.0),
    ]:
        # Pick canonical context-mapping. NumPyro's compute_rbo_forward
        # internally branches on context for the alpha rule; MATLAB's
        # frugFun5_reference is the CP version (use frugFun5_uniformOddballs
        # if a separate OB function exists in the lab code).
        is_cp = "changepoint" in context_label
        m = run_matlab(bag, H, LW)
        bucket = m["B"][:-1]  # last belief is post-final, drop
        n = run_numpyro(bag, bucket, H, LW, UU, "changepoint" if is_cp else "oddball")
        diffs = compare(n, m, context_label)
        diffs["scenario"] = context_label
        diffs["H"] = H; diffs["LW"] = LW; diffs["UU"] = UU
        rows.append(diffs)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nwrote: {args.output}")

    # Gate decision
    fails = []
    for _, row in df.iterrows():
        if row["alpha_max_abs"] > args.alpha_tol:
            fails.append(f"{row['scenario']}: alpha diff {row['alpha_max_abs']:.4e} > {args.alpha_tol}")
        if row["omega_max_abs"] > args.omega_tol:
            fails.append(f"{row['scenario']}: omega diff {row['omega_max_abs']:.4e} > {args.omega_tol}")

    if fails:
        print("\nFAIL: parity violated")
        for f in fails:
            print(f"  ✗ {f}")
        return 1
    print("\nPASS: NumPyro and MATLAB agree within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run it:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/validation/validate_rbo_vs_matlab.py
```

Three scenarios:
1. CP, default (H=0.125, LW=1.0, UU=1.0)
2. CP, LW=0.5 (tests log-space LW behaviour at non-trivial likeWeight)
3. OB, default

For OB the MATLAB function may need to be `frugFun5_uniformOddballs.m` if it exists separately in `functionCodes/`. Check first; if it exists, port that variant too as `frugfun5_oddball_reference`. If not, our `compute_rbo_forward` and `frugfun5_reference` should both branch on context internally.

Document the comparison results in 04-03a-SUMMARY.md as a table:

| Scenario | alpha max-abs | omega max-abs | Status |
|----------|--------------:|--------------:|:------:|
| CP default | ... | ... | ✓/✗ |
| CP LW=0.5 | ... | ... | ✓/✗ |
| OB default | ... | ... | ✓/✗ |
  </action>
  <verify>
```
ls src/nn4psych/bayesian/_frugfun_reference.py scripts/validation/validate_rbo_vs_matlab.py

# Reference is importable
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -c "from nn4psych.bayesian._frugfun_reference import frugfun5_reference; print('import OK')"

# Validator runs (may FAIL the parity gate; that's expected info, not a script crash)
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/validation/validate_rbo_vs_matlab.py
echo "exit=$?  # 0 = parity OK; 1 = divergence detected (Task 3 will fix)"
ls data/processed/bayesian/matlab_parity_diffs.csv
```
  </verify>
  <done>
- `_frugfun_reference.py` is a line-by-line port of frugFun5.m (truncated normal, log-space LW, drift, second-moment R-update).
- Validator script runs and writes diffs CSV.
- Three scenarios compared (CP default, CP LW=0.5, OB default).
- Diff table recorded in SUMMARY for 04-03a.
  </done>
</task>

<task type="auto">
  <name>Task 3: Fix divergences in compute_rbo_forward (if any) and add MATLAB-parity test</name>
  <files>src/nn4psych/bayesian/reduced_bayesian.py,tests/test_reduced_bayesian.py</files>
  <action>
Branching on Task 2's parity-check result:

**A. If parity holds (all diffs < tolerance):**
1. Document in 04-03a-SUMMARY.md: "NumPyro and MATLAB agree to <tolerance>; no production-code change needed."
2. Skip directly to test.

**B. If divergences are real:**
1. Identify which discrepancy drives each scenario's diff. Likely candidates from inspection:
   - **Truncated normal**: the existing `compute_rbo_forward` uses `jax_norm.pdf(delta, loc=0.0, scale=sigma_t) ** LW` without the `normcdf(300) - normcdf(0)` normalization. To fix:
     ```python
     N_pdf = jax_norm.pdf(delta, loc=0.0, scale=sigma_t)
     # Truncate to bag range (paper-faithful) — note the bag is in [0, 300]
     # but pred_errors = bag - bucket can range over [-300, +300]; the
     # MATLAB code computes truncation w.r.t. data ∈ [0, 300] using the
     # bucket B(i) as the location, NOT relative to delta.
     # Re-read frugFun5.m line 89: normalize is computed at the BAG-position
     # level, not the delta level. We need to refactor compute_rbo_forward
     # to use bag_positions and bucket_positions explicitly, not pred_errors.
     ```
     Note: this may be a deeper refactor — `compute_rbo_forward` currently consumes `pred_errors`, but the truncated-normal correction needs `bag` and `bucket` separately. Decision: refactor signature to accept both, deprecate the pred_errors-only path. Wire reduced_bayesian_model accordingly.

   - **Log-space LW**: replace
     ```python
     omega = (H * U^LW) / (H * U^LW + (1-H) * N^LW)
     ```
     with the MATLAB-equivalent log-space form:
     ```python
     log_change_ratio = LW * (jnp.log(U_val + 1e-30) - jnp.log(N_val + 1e-30)) + jnp.log(H / (1 - H + 1e-30))
     omega = jax.nn.sigmoid(log_change_ratio)
     ```
     This is numerically more stable AND matches MATLAB's exp/log structure exactly.

   - **R vs τ update**: confirm the MATLAB second-moment formula matches our predictive-variance-weighted form. They MAY already be algebraically equivalent (just different variable choice: τ = 1/(R+1) is the standard reparameterization). If equivalent, no fix; document the equivalence in code comments.

   - **drift parameter**: verify Nassar 2021 task uses drift=0 (no helicopter drift in CP/OB; helicopter is constant within a run). If yes, current implementation is correct (no drift). Document as "Nassar 2021 task uses drift=0; no production code change needed". If no, expose drift as an optional parameter (default 0.0) and pass through.

2. Apply the fix. Keep diffs minimal; don't refactor unrelated code.
3. Re-run `tests/test_reduced_bayesian.py` (existing 6 tests). All must still pass.
4. Re-run validator. All scenarios should now pass tolerance gates.

**C. Add the parity test:**

Append to `tests/test_reduced_bayesian.py`:

```python
def test_matlab_parity():
    """Guard against silent drift in the JAX forward model.

    Runs compute_rbo_forward and the NumPy port of frugFun5.m on the same
    synthetic deltas with a small but representative parameter set. Fails
    if max-abs divergence exceeds 1e-3 on alpha or omega.
    """
    import numpy as np
    import jax.numpy as jnp
    from nn4psych.bayesian.reduced_bayesian import compute_rbo_forward, SIGMA_N
    from nn4psych.bayesian._frugfun_reference import frugfun5_reference

    rng = np.random.default_rng(42)
    bag = np.clip(rng.normal(150.0, SIGMA_N, size=64), 0, 300)
    # MATLAB self-bucketing: derive bucket from MATLAB belief trajectory
    m = frugfun5_reference(bag, Hazard=0.125, noise=SIGMA_N, likeWeight=1.0)
    bucket = m["B"][:-1]
    pred_errors = bag - bucket

    params = {"H": jnp.asarray(0.125), "LW": jnp.asarray(1.0), "UU": jnp.asarray(1.0)}
    lr, _upd, omega, _tau = compute_rbo_forward(params, jnp.asarray(pred_errors), "changepoint")

    assert np.max(np.abs(np.asarray(lr) - m["alpha"])) < 1e-3
    assert np.max(np.abs(np.asarray(omega) - m["pCha"])) < 1e-3
```

Run the full test file:
```
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -m pytest tests/test_reduced_bayesian.py -v
```

Expected: 7 tests pass (6 existing + 1 new parity test).
  </action>
  <verify>
```
# All RBO tests pass including new parity test
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -m pytest tests/test_reduced_bayesian.py -v
# Expected: 7 passed

# Validator now passes
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/validation/validate_rbo_vs_matlab.py
echo "exit=$?  # must be 0"
```
  </verify>
  <done>
- All identified divergences are either fixed in compute_rbo_forward OR documented as accepted (with rationale).
- 7 tests pass in test_reduced_bayesian.py (6 existing + test_matlab_parity).
- Validator script exits 0.
- 04-03a-SUMMARY.md records the diff table BEFORE fix and (if applicable) AFTER fix.
  </done>
</task>

</tasks>

<verification>
End-to-end:
```
# Cleanup pipeline ran
ls data/processed/nassar2021/subject_trials.npy data/processed/nassar2021/subject_metadata.csv

# Reference + validator exist
ls src/nn4psych/bayesian/_frugfun_reference.py scripts/validation/validate_rbo_vs_matlab.py

# Tests pass
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe -m pytest tests/test_reduced_bayesian.py -v

# Parity check passes
/c/Users/aman0087/AppData/Local/miniforge3/envs/actinf-py-scripts/python.exe scripts/validation/validate_rbo_vs_matlab.py
```
</verification>

<success_criteria>
- Task 1: 134 subjects cleaned, per-subject excluded counts in metadata CSV.
- Task 2: Side-by-side validator runs; diffs table documented for ≥3 parameter scenarios.
- Task 3: Either no fix needed (parity holds) OR fix applied + tests pass; new test_matlab_parity guards future drift.
- Re-running BAYES-06 param recovery and 04-03 human fits is now safe — model is paper-faithful or accepted-deviations are documented.
</success_criteria>

<output>
Create `.planning/phases/04-bayesian-model-fitting/04-03a-SUMMARY.md`:
- What was built: cleanup run + frugfun5 reference port + validator + parity test
- Cleanup statistics: subjects loaded, total trials before/after, per-cohort breakdowns
- Diff table: NumPyro vs MATLAB across CP/OB scenarios (BEFORE any fix)
- Decisions logged: did we fix discrepancies or accept them? Why?
- Diff table AFTER fix (if applicable)
- Required-by-next-plan: 04-03 Task 2 BAYES-06 gate is now meaningful — re-run param recovery uses paper-faithful model
- Open follow-ups: any accepted-deviations that future work should revisit
</output>
