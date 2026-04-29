"""One-shot aggregator for per-context LatentNet fitting results.

Reads per-context validation_results.json files produced by cluster fits,
computes deltas vs pooled Wave A baseline, classifies conclusion, and writes
output/circuit_analysis/per_context/per_context_results.json.

Conclusion enum:
  STRUCTURAL_SEPARATION -- both delta_ctx >= 0.05 (each per-context fit beats pooled)
  NO_SEPARATION         -- both |delta_ctx| < 0.05 (per-context same as pooled)
  AMBIGUOUS             -- anything else (catch-all)

Usage
-----
python scripts/analysis/aggregate_per_context.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PER_CTX_DIR = REPO_ROOT / "output" / "circuit_analysis" / "per_context"
_SWEEP_DIR = REPO_ROOT / "output" / "circuit_analysis" / "n_latent_sweep"
WAVE_A_SEL = _SWEEP_DIR / "wave_a_selection.json"
OUT_PATH = PER_CTX_DIR / "per_context_results.json"

SEPARATION_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


ctx0 = _load_json(PER_CTX_DIR / "context_0" / "validation_results.json")
ctx1 = _load_json(PER_CTX_DIR / "context_1" / "validation_results.json")
wave_a = _load_json(WAVE_A_SEL)

pooled_corr: float = float(wave_a["metrics"]["invariant_subspace_corr"])

corr_ctx0: float = float(ctx0["invariant_subspace_corr"])
corr_ctx1: float = float(ctx1["invariant_subspace_corr"])
nmse_y_ctx0: float = float(ctx0["nmse_y"])
nmse_y_ctx1: float = float(ctx1["nmse_y"])
n_trials_ctx0: int = int(ctx0["n_trials_used"])
n_trials_ctx1: int = int(ctx1["n_trials_used"])

# ---------------------------------------------------------------------------
# Deltas and classification
# ---------------------------------------------------------------------------
delta_ctx0: float = round(corr_ctx0 - pooled_corr, 6)
delta_ctx1: float = round(corr_ctx1 - pooled_corr, 6)

both_above_threshold = (
    delta_ctx0 >= SEPARATION_THRESHOLD and delta_ctx1 >= SEPARATION_THRESHOLD
)
both_within_noise = (
    abs(delta_ctx0) < SEPARATION_THRESHOLD and abs(delta_ctx1) < SEPARATION_THRESHOLD
)

if both_above_threshold:
    conclusion = "STRUCTURAL_SEPARATION"
elif both_within_noise:
    conclusion = "NO_SEPARATION"
else:
    conclusion = "AMBIGUOUS"

either_above_85: bool = max(corr_ctx0, corr_ctx1) >= 0.85

# ---------------------------------------------------------------------------
# Print summary (visible in bash log)
# ---------------------------------------------------------------------------
print(f"Pooled corr (Wave A, n=12): {pooled_corr:.4f}")
print(
    f"Context 0:  corr={corr_ctx0:.4f}  delta={delta_ctx0:+.4f}"
    f"  nmse_y={nmse_y_ctx0:.4f}  n={n_trials_ctx0}"
)
print(
    f"Context 1:  corr={corr_ctx1:.4f}  delta={delta_ctx1:+.4f}"
    f"  nmse_y={nmse_y_ctx1:.4f}  n={n_trials_ctx1}"
)
print(f"Separation threshold: {SEPARATION_THRESHOLD}")
print(f"Conclusion: {conclusion}")
print(f"either_above_85: {either_above_85}")

# ---------------------------------------------------------------------------
# Write output JSON (all Python native types — no numpy scalars)
# ---------------------------------------------------------------------------
result: dict = {
    "n_latent": int(wave_a["chosen_rank"]),
    "context_0": {
        "corr": corr_ctx0,
        "nmse_y": nmse_y_ctx0,
        "n_trials": n_trials_ctx0,
    },
    "context_1": {
        "corr": corr_ctx1,
        "nmse_y": nmse_y_ctx1,
        "n_trials": n_trials_ctx1,
    },
    "pooled_corr_reference": pooled_corr,
    "delta_ctx0": delta_ctx0,
    "delta_ctx1": delta_ctx1,
    "separation_threshold": SEPARATION_THRESHOLD,
    "conclusion": conclusion,
    "eval_procedure": "cluster_same_seed_as_train",
    "either_above_85": either_above_85,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w") as fh:
    json.dump(result, fh, indent=2)

print(f"\nWrote: {OUT_PATH}")
