#!/usr/bin/env python3
"""Aggregate Wave A masked-loss n_latent sweep results into Pareto curve and selection.

Reads per-rank cluster artifacts from
output/circuit_analysis/n_latent_sweep_masked/n{N}/ (N in {8, 12, 16}),
assembles diagnostic metrics into a Pareto curve (JSON + PNG), and writes
wave_a_masked_selection.json for downstream 03-07 / 03-08 consumption.

Decision rule (mirrors Wave A / 03-03): chosen rank = argmax(invariant_subspace_corr)
across available ranks, with ties broken by preferring the LOWER rank (parsimony).

The masked sweep uses the same evaluation procedure as Wave A
(cluster_same_seed_as_train) for valid cross-comparison.

Usage
-----
    python scripts/analysis/aggregate_n_latent_sweep_masked.py
    python scripts/analysis/aggregate_n_latent_sweep_masked.py --ranks 8,12,16

Outputs
-------
output/circuit_analysis/n_latent_sweep_masked/
    pareto_curve_masked.json  — 3 rows of diagnostic metrics (+ masked/cross-ref fields)
    pareto_curve_masked.png   — two-curve overlay: full vs masked corr-vs-rank
    wave_a_masked_selection.json — chosen rank, metrics, crossed_85_threshold,
        delta_vs_wave_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # MUST come before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402

MASKED_SWEEP_DIR = Path("output/circuit_analysis/n_latent_sweep_masked")
FULL_SWEEP_DIR = Path("output/circuit_analysis/n_latent_sweep")
WAVE_A_FULL_SELECTION = FULL_SWEEP_DIR / "wave_a_selection.json"
WAVE_A_FULL_PARETO = FULL_SWEEP_DIR / "pareto_curve.json"

DEFAULT_RANKS: tuple[int, ...] = (8, 12, 16)

# Wave A pooled baseline n=12 corr — from wave_a_selection.json (cluster run 03-03)
WAVE_A_BASELINE_CORR = 0.7832673556806835


def load_rank_metrics(rank: int, sweep_dir: Path = MASKED_SWEEP_DIR) -> dict | None:
    """Load validation_results.json + ensemble_diagnostics.json for one rank.

    Parameters
    ----------
    rank : int
        n_latent value (subdirectory is n{rank}).
    sweep_dir : Path, optional
        Root sweep directory. Default is n_latent_sweep_masked.

    Returns
    -------
    dict or None
        Per-rank metrics dict with keys n_latent, best_nmse_y (masked),
        best_nmse_y_full, best_mse_z, invariant_subspace_corr,
        trial_avg_r2_full_space, per_trial_r2_full_space,
        ensemble_std_nmse_y, elapsed_minutes, status.
        Returns None if the rank's directory is missing or incomplete.
    """
    rank_dir = sweep_dir / f"n{rank}"
    val_path = rank_dir / "validation_results.json"
    diag_path = rank_dir / "ensemble_diagnostics.json"

    if not val_path.exists() or not diag_path.exists():
        return None

    val = json.loads(val_path.read_text())
    diag = json.loads(diag_path.read_text())

    return {
        "n_latent": int(rank),
        "best_nmse_y": float(val["best_nmse_y"]),  # masked NMSE_y
        "best_nmse_y_full": float(val["best_nmse_y_full"]),
        "best_mse_z": float(val["best_mse_z"]),
        "invariant_subspace_corr": float(val["invariant_subspace_corr"]),
        "trial_avg_r2_full_space": (
            float(val["trial_avg_r2_full_space"])
            if val.get("trial_avg_r2_full_space") is not None
            else None
        ),
        "per_trial_r2_full_space": float(val["activity_r2_full_space"]),
        "ensemble_std_nmse_y": float(
            diag["convergence_stats"]["std_nmse_y"]
        ),
        "elapsed_minutes": float(val.get("elapsed_minutes", 0.0)),
        "status": str(val["status"]),
        "masked": True,
    }


def build_pareto_curve(
    rank_metrics: list[dict],
    ranks_attempted: list[int],
    ranks_missing: list[int],
) -> dict:
    """Assemble per-rank metrics into a Pareto curve dict.

    Parameters
    ----------
    rank_metrics : list of dict
        One entry per successful rank from load_rank_metrics.
    ranks_attempted : list of int
        Ranks that were requested.
    ranks_missing : list of int
        Ranks that were requested but produced no artifacts.

    Returns
    -------
    dict
        Same schema as output/circuit_analysis/n_latent_sweep/pareto_curve.json
        plus top-level ``masked: true`` and a ``comparable_full_pareto``
        cross-reference path.
    """
    return {
        "masked": True,
        "comparable_full_pareto": str(WAVE_A_FULL_PARETO),
        "ranks_attempted": [int(r) for r in ranks_attempted],
        "ranks_succeeded": [int(m["n_latent"]) for m in rank_metrics],
        "ranks_missing": [int(r) for r in ranks_missing],
        "rows": rank_metrics,
    }


def select_rank(rank_metrics: list[dict]) -> dict:
    """Choose the rank with maximum invariant_subspace_corr (ties → lower rank).

    Mirrors the Wave A selection rule from aggregate_n_latent_sweep.py for
    valid cross-sweep comparison.

    Parameters
    ----------
    rank_metrics : list of dict
        Per-rank metrics. Must be non-empty.

    Returns
    -------
    dict
        {"chosen_rank": int, "metrics": dict, "rationale": str}
    """
    if not rank_metrics:
        raise ValueError(
            "Cannot select rank: no successful per-rank metrics. "
            f"Expected at least one entry from {DEFAULT_RANKS}."
        )
    chosen = max(
        rank_metrics,
        key=lambda r: (r["invariant_subspace_corr"], -r["n_latent"]),
    )
    rationale = (
        f"Selected n_latent={chosen['n_latent']} as it maximises "
        f"invariant_subspace_corr ({chosen['invariant_subspace_corr']:.4f}) "
        f"across the {len(rank_metrics)}-rank masked sweep "
        f"(ties broken by preferring lower rank for parsimony). "
        f"Mirrors Wave A selection rule for valid cross-comparison."
    )
    return {
        "chosen_rank": int(chosen["n_latent"]),
        "metrics": chosen,
        "rationale": rationale,
    }


def write_pareto_plot(
    masked_pareto: dict,
    full_pareto_path: Path,
    out_path: Path,
) -> None:
    """Save a two-curve overlay: full vs masked corr-vs-rank.

    Two panels (shared x-axis = n_latent):
      1. NMSE_y (full) vs rank — Wave A comparison baseline
      2. Invariant subspace corr: full (Wave A) overlay vs masked (this sweep)

    Parameters
    ----------
    masked_pareto : dict
        Output of build_pareto_curve for the masked sweep.
    full_pareto_path : Path
        Path to Wave A's pareto_curve.json for overlay.
    out_path : Path
        PNG path to save (matplotlib.use('Agg'), no plt.show()).
    """
    masked_rows = sorted(masked_pareto["rows"], key=lambda r: r["n_latent"])
    ns_masked = [r["n_latent"] for r in masked_rows]
    corr_masked = [r["invariant_subspace_corr"] for r in masked_rows]
    nmse_full_masked = [r["best_nmse_y_full"] for r in masked_rows]
    nmse_masked = [r["best_nmse_y"] for r in masked_rows]

    full_rows: list[dict] = []
    if full_pareto_path.exists():
        full_pareto = json.loads(full_pareto_path.read_text())
        full_rows = sorted(full_pareto["rows"], key=lambda r: r["n_latent"])

    ns_full = [r["n_latent"] for r in full_rows]
    corr_full = [r["invariant_subspace_corr"] for r in full_rows]
    nmse_full = [r["best_nmse_y"] for r in full_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: NMSE_y (full T) comparison
    ax = axes[0]
    if full_rows:
        ax.plot(ns_full, nmse_full, "o-", color="C0", label="Wave A (full)")
    ax.plot(ns_masked, nmse_full_masked, "s--", color="C0", alpha=0.6,
            label="Masked sweep (full T eval)")
    ax.plot(
        ns_masked, nmse_masked, "^:", color="C3", label="Masked sweep (masked eval)"
    )
    ax.set_xlabel("n_latent")
    ax.set_ylabel("best NMSE_y (lower = better)")
    ax.set_title("Reconstruction quality")
    ax.legend(fontsize=7)

    # Panel 2: Invariant subspace corr — decision metric overlay
    ax = axes[1]
    if full_rows:
        ax.plot(ns_full, corr_full, "o-", color="C1", label="Wave A (full)")
    ax.plot(ns_masked, corr_masked, "s--", color="C2", label="Masked sweep")
    ax.axhline(0.85, ls="--", color="grey", label="paper threshold (0.85)")
    ax.set_xlabel("n_latent")
    ax.set_ylabel("invariant subspace corr")
    ax.set_title("Connectivity alignment (decision metric)")
    ax.legend(loc="lower right", fontsize=7)

    # Panel 3: Delta corr (masked - full) per rank
    ax = axes[2]
    if full_rows:
        full_corr_by_n = {
            r["n_latent"]: r["invariant_subspace_corr"] for r in full_rows
        }
        deltas = [
            c - full_corr_by_n[n]
            for n, c in zip(ns_masked, corr_masked, strict=False)
            if n in full_corr_by_n
        ]
        ns_delta = [
            n for n in ns_masked if n in full_corr_by_n
        ]
        ax.bar(ns_delta, deltas, color=["C2" if d > 0 else "C3" for d in deltas])
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xlabel("n_latent")
        ax.set_ylabel("corr delta (masked − full)")
        ax.set_title("Change in corr: masked vs Wave A")
    else:
        ax.set_visible(False)

    fig.suptitle("Wave A masked-loss sweep vs Wave A full-loss (Gap 1 diagnostic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    """Aggregate masked sweep, write pareto_curve_masked.json and selection JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RANKS),
        help="Comma-separated ranks to aggregate (default: 8,12,16).",
    )
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]

    MASKED_SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    rank_metrics: list[dict] = []
    missing: list[int] = []
    for r in ranks:
        m = load_rank_metrics(r)
        if m is None:
            missing.append(r)
            print(f"WARNING: n_latent={r} missing artifacts; skipping.")
        else:
            rank_metrics.append(m)
            print(
                f"  n_latent={r}: masked_nmse_y={m['best_nmse_y']:.4f}, "
                f"nmse_y_full={m['best_nmse_y_full']:.4f}, "
                f"corr={m['invariant_subspace_corr']:.4f}, "
                f"trial_avg_R2={m['trial_avg_r2_full_space']:.4f}, "
                f"wall={m['elapsed_minutes']:.1f}min"
            )

    if not rank_metrics:
        print("ERROR: no successful ranks found in masked sweep directory.")
        return 1

    pareto = build_pareto_curve(rank_metrics, ranks, missing)
    pareto_json_path = MASKED_SWEEP_DIR / "pareto_curve_masked.json"
    pareto_json_path.write_text(json.dumps(pareto, indent=2))
    print(f"Wrote {pareto_json_path}")

    pareto_png_path = MASKED_SWEEP_DIR / "pareto_curve_masked.png"
    write_pareto_plot(pareto, WAVE_A_FULL_PARETO, pareto_png_path)
    print(f"Wrote {pareto_png_path}")

    selection = select_rank(rank_metrics)
    chosen_corr = float(selection["metrics"]["invariant_subspace_corr"])
    crossed_85 = bool(chosen_corr >= 0.85)
    delta_vs_wave_a = float(chosen_corr - WAVE_A_BASELINE_CORR)

    all_ranks_summary = [
        {"n": int(m["n_latent"]), "corr": float(m["invariant_subspace_corr"])}
        for m in rank_metrics
    ]

    # Compute Pareto spread to inform story direction
    corrs = [m["invariant_subspace_corr"] for m in rank_metrics]
    spread = float(max(corrs) - min(corrs))

    if crossed_85:
        story_note = (
            "SC-2 CLEARED — chosen rank crosses 0.85. 03-08 commits Story 0. "
            "03-07 (shorter-T regen) is NO-OP."
        )
    elif delta_vs_wave_a < 0.0:
        # Masked corr is WORSE than full-T Wave A baseline — padding hypothesis
        # is ruled out (masking hurt, not helped). The negative delta is the
        # decisive signal regardless of Pareto spread.
        story_note = (
            f"Masked corr WORSE than Wave A baseline (delta={delta_vs_wave_a:+.4f}). "
            "Padding hypothesis ruled out — focusing loss on task-active timesteps "
            "did not improve connectivity alignment. Tilt Phase 3.1 story toward "
            "STORY_1 (method/data limit). 03-07 (shorter T regen) will still RUN "
            "per its conditional skip rule (crossed_85=false), but the negative "
            "delta is a strong prior that shorter T also won't help."
        )
    elif spread < 0.05:
        story_note = (
            f"Pareto flat (spread={spread:.3f} < 0.05) — adding latent capacity does "
            "not help; padding hypothesis ruled out. Tilt Phase 3.1 story toward "
            "STORY_1 (method/data limit). 03-07 (shorter T) will still run "
            "(crossed_85=false) but result expected to be flat."
        )
    else:
        story_note = (
            f"Corr varies across ranks (spread={spread:.3f}) but chosen rank still "
            "below 0.85. Deferred fixes (shorter T via 03-07) remain viable."
        )

    selection_payload = {
        "chosen_rank": int(selection["chosen_rank"]),
        "metrics": {
            "invariant_subspace_corr": float(
                selection["metrics"]["invariant_subspace_corr"]
            ),
            "masked_nmse_y": float(selection["metrics"]["best_nmse_y"]),
            "nmse_y_full": float(selection["metrics"]["best_nmse_y_full"]),
            "trial_avg_r2_full": float(
                selection["metrics"]["trial_avg_r2_full_space"]
            ),
        },
        "all_ranks": all_ranks_summary,
        "crossed_85_threshold": crossed_85,
        "delta_vs_wave_a": round(delta_vs_wave_a, 6),
        "wave_a_baseline_corr": float(WAVE_A_BASELINE_CORR),
        "eval_procedure": "cluster_same_seed_as_train",
        "masked": True,
        "comparable_full_selection": str(WAVE_A_FULL_SELECTION),
        "pareto_spread": round(spread, 6),
        "story_note": story_note,
    }

    selection_path = MASKED_SWEEP_DIR / "wave_a_masked_selection.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2))
    print(f"Wrote {selection_path}")
    print(f"  Chosen rank: n_latent={selection['chosen_rank']}")
    print(f"  Masked corr: {chosen_corr:.4f}")
    print(f"  Wave A baseline corr: {WAVE_A_BASELINE_CORR:.4f}")
    print(f"  Delta vs Wave A: {delta_vs_wave_a:+.4f}")
    print(f"  Crossed 0.85 threshold: {crossed_85}")
    print(f"  Pareto spread: {spread:.4f}")
    print(f"  Story note: {story_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
