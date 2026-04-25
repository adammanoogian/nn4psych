#!/usr/bin/env python3
"""Aggregate Wave A n_latent sweep results into a Pareto curve and Q selection.

Reads per-rank cluster artifacts from output/circuit_analysis/n_latent_sweep/n{N}/
(N in {4, 8, 12, 16}), assembles diagnostic metrics into a Pareto curve
(JSON + PNG), and copies the chosen rank's best_latent_circuit.pt to a
top-level best_latent_circuit_waveA.pt for Wave B (Plan 03-04) consumption.

Decision rule (from 03-CONTEXT.md): chosen rank = argmax(invariant_subspace_corr)
across the available ranks, with ties broken by preferring the LOWER rank
(parsimony — fewer latent dimensions are easier to interpret and perturb).
Reported alongside as the rank-knee diagnostic: the rank-vs-NMSE_y curve.

Usage
-----
    python scripts/analysis/aggregate_n_latent_sweep.py
    python scripts/analysis/aggregate_n_latent_sweep.py --ranks 8,12,16

Outputs
-------
output/circuit_analysis/n_latent_sweep/
    pareto_curve.json       — 4 (or fewer) rows of diagnostic metrics
    pareto_curve.png        — multi-panel Pareto plot
    wave_a_selection.json   — chosen rank, metrics, rationale
output/circuit_analysis/
    best_latent_circuit_waveA.pt  — copy of chosen rank's best_latent_circuit.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # MUST come before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402

SWEEP_DIR = Path("output/circuit_analysis/n_latent_sweep")
TOP_LEVEL_DIR = Path("output/circuit_analysis")
DEFAULT_RANKS: tuple[int, ...] = (4, 8, 12, 16)


def load_rank_metrics(rank: int, sweep_dir: Path = SWEEP_DIR) -> dict | None:
    """Load validation_results.json + ensemble_diagnostics.json for one rank.

    Parameters
    ----------
    rank : int
        n_latent value (subdirectory is n{rank}).
    sweep_dir : Path, optional
        Root sweep directory. Default is output/circuit_analysis/n_latent_sweep.

    Returns
    -------
    dict or None
        Per-rank metrics dict with keys n_latent, best_nmse_y, best_mse_z,
        invariant_subspace_corr, trial_avg_r2_full_space, per_trial_r2_full_space,
        ensemble_std_nmse_y, status. Returns None if the rank's directory is
        missing or incomplete (e.g., n=4 failed the LatentNet n>=7 assertion).
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
        "best_nmse_y": float(val["best_nmse_y"]),
        "best_mse_z": float(val["best_mse_z"]),
        "invariant_subspace_corr": float(val["invariant_subspace_corr"]),
        "trial_avg_r2_full_space": (
            float(val["trial_avg_r2_full_space"])
            if val.get("trial_avg_r2_full_space") is not None
            else None
        ),
        "per_trial_r2_full_space": float(val["activity_r2_full_space"]),
        "ensemble_std_nmse_y": float(diag["convergence_stats"]["std_nmse_y"]),
        "elapsed_minutes": float(val.get("elapsed_minutes", 0.0)),
        "status": str(val["status"]),
    }


def build_pareto_curve(
    rank_metrics: list[dict], ranks_attempted: list[int], ranks_missing: list[int]
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
        {"rows": list of per-rank dicts, "ranks_attempted": list of int,
         "ranks_succeeded": list of int, "ranks_missing": list of int}
    """
    return {
        "ranks_attempted": [int(r) for r in ranks_attempted],
        "ranks_succeeded": [int(m["n_latent"]) for m in rank_metrics],
        "ranks_missing": [int(r) for r in ranks_missing],
        "rows": rank_metrics,
    }


def select_rank(rank_metrics: list[dict]) -> dict:
    """Choose the rank with maximum invariant_subspace_corr (ties → lower rank).

    Tie-breaker rationale: when two ranks score within numerical equality on
    invariant_subspace_corr (a plausible scenario, e.g., n=8 and n=12 both at
    ~0.71), prefer the LOWER rank for parsimony — fewer latent dimensions
    are easier to interpret in the perturbation analysis (Wave B), and the
    lower-rank Q is a strictly more constrained representation. Without an
    explicit tie-breaker, max() returns the first-seen entry, which is
    list-order dependent (silently nondeterministic).

    Parameters
    ----------
    rank_metrics : list of dict
        Per-rank metrics. Must be non-empty.

    Returns
    -------
    dict
        {"chosen_rank": int, "metrics": dict (the chosen rank's full row),
         "rationale": str (one-line decision sentence)}
    """
    if not rank_metrics:
        raise ValueError(
            "Cannot select rank: no successful per-rank metrics. "
            "Expected at least one entry from {4, 8, 12, 16}."
        )
    chosen = max(
        rank_metrics,
        key=lambda r: (r["invariant_subspace_corr"], -r["n_latent"]),
    )
    rationale = (
        f"Selected n_latent={chosen['n_latent']} as it maximises "
        f"invariant_subspace_corr ({chosen['invariant_subspace_corr']:.4f}) "
        f"across the {len(rank_metrics)}-rank sweep "
        f"(ties broken by preferring lower rank for parsimony)."
    )
    return {
        "chosen_rank": int(chosen["n_latent"]),
        "metrics": chosen,
        "rationale": rationale,
    }


def write_pareto_plot(pareto: dict, out_path: Path) -> None:
    """Save a multi-panel Pareto plot.

    Three panels (shared x-axis = n_latent):
      1. NMSE_y vs rank          (lower = better reconstruction)
      2. Invariant subspace corr (higher = better connectivity match)
      3. Trial-avg R^2 full      (higher = better averaged dynamics)

    Parameters
    ----------
    pareto : dict
        Output of build_pareto_curve.
    out_path : Path
        PNG path to save (matplotlib.use('Agg'), no plt.show()).
    """
    rows = sorted(pareto["rows"], key=lambda r: r["n_latent"])
    ns = [r["n_latent"] for r in rows]
    nmse = [r["best_nmse_y"] for r in rows]
    corr = [r["invariant_subspace_corr"] for r in rows]
    r2 = [r["trial_avg_r2_full_space"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(ns, nmse, "o-", color="C0")
    axes[0].set_xlabel("n_latent")
    axes[0].set_ylabel("best NMSE_y (lower = better)")
    axes[0].set_title("Reconstruction quality")

    axes[1].plot(ns, corr, "o-", color="C1")
    axes[1].axhline(0.85, ls="--", color="grey", label="paper threshold (0.85)")
    axes[1].set_xlabel("n_latent")
    axes[1].set_ylabel("invariant subspace corr")
    axes[1].set_title("Connectivity alignment (decision metric)")
    axes[1].legend(loc="lower right", fontsize=8)

    axes[2].plot(ns, r2, "o-", color="C2")
    axes[2].set_xlabel("n_latent")
    axes[2].set_ylabel("trial-avg R^2 (full space)")
    axes[2].set_title("Trial-averaged dynamics")

    fig.suptitle("Wave A: n_latent Pareto curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_wave_b_writeup_prepositioning(selection: dict, pareto: dict) -> dict:
    """Pre-position Wave B's story-1-vs-story-2 writeup choice.

    Returns a dict that the wave_a_selection.json embeds, telling Wave B
    which scientific narrative the Pareto curve supports.

    Heuristic
    ---------
    - Story 1 ("method/data limit"): invariant_subspace_corr is roughly flat
      across ranks AND no rank crosses 0.85 → chosen rank is the best Q
      available, and Wave B's writeup should commit to "we couldn't reach 0.85
      because of method/data limits".
    - Story 2 ("ran out of fixes to try"): if invariant corr varies
      meaningfully across ranks (over-/under-parameterisation visible), Wave A
      reduced but did not eliminate the structural concern; deferred fixes
      remain viable.
    - "passed": chosen rank crosses 0.85.

    Returns
    -------
    dict
        {"recommended_story": "method_limit" | "ran_out_of_fixes" | "passed",
         "evidence": str, "passes_paper_threshold": bool}
    """
    rows = sorted(pareto["rows"], key=lambda r: r["n_latent"])
    corrs = [r["invariant_subspace_corr"] for r in rows]
    chosen_corr = selection["metrics"]["invariant_subspace_corr"]
    passes = chosen_corr >= 0.85

    if not corrs:
        return {
            "recommended_story": "method_limit",
            "evidence": "No successful ranks; method limit by default.",
            "passes_paper_threshold": False,
        }

    spread = max(corrs) - min(corrs)
    if passes:
        story = "passed"
        evidence = (
            f"Chosen rank n_latent={selection['chosen_rank']} reaches "
            f"corr={chosen_corr:.3f} >= 0.85. Wave A closed the soft-fail."
        )
    elif spread < 0.05:
        story = "method_limit"
        evidence = (
            f"Invariant subspace corr is flat across ranks "
            f"(min={min(corrs):.3f}, max={max(corrs):.3f}, spread={spread:.3f}). "
            "Adding latent capacity does not improve connectivity alignment, "
            "supporting the method/data-limit interpretation."
        )
    else:
        story = "ran_out_of_fixes"
        evidence = (
            f"Invariant subspace corr varies meaningfully across ranks "
            f"(spread={spread:.3f}); chosen rank n_latent={selection['chosen_rank']} "
            f"is best of those tried but does not cross 0.85. Wave A reduced "
            "but did not eliminate the structural concern; deferred fixes "
            "(masked loss, shorter T) remain on the table."
        )
    return {
        "recommended_story": story,
        "evidence": evidence,
        "passes_paper_threshold": bool(passes),
    }


def main() -> int:
    """Aggregate sweep, write Pareto + selection, copy chosen Q to top level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RANKS),
        help="Comma-separated ranks to aggregate (default: 4,8,12,16).",
    )
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

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
                f"  n_latent={r}: nmse_y={m['best_nmse_y']:.4f}, "
                f"corr={m['invariant_subspace_corr']:.4f}, "
                f"trial_avg_R2={m['trial_avg_r2_full_space']:.4f}, "
                f"wall={m['elapsed_minutes']:.1f}min"
            )

    if not rank_metrics:
        print("ERROR: no successful ranks found in sweep directory.")
        return 1

    pareto = build_pareto_curve(rank_metrics, ranks, missing)
    pareto_json_path = SWEEP_DIR / "pareto_curve.json"
    pareto_json_path.write_text(json.dumps(pareto, indent=2))
    print(f"Wrote {pareto_json_path}")

    pareto_png_path = SWEEP_DIR / "pareto_curve.png"
    write_pareto_plot(pareto, pareto_png_path)
    print(f"Wrote {pareto_png_path}")

    selection = select_rank(rank_metrics)
    wave_b_prepos = write_wave_b_writeup_prepositioning(selection, pareto)
    selection_payload = {
        **selection,
        "pareto_summary": {
            "ranks_attempted": pareto["ranks_attempted"],
            "ranks_succeeded": pareto["ranks_succeeded"],
            "ranks_missing": pareto["ranks_missing"],
        },
        "wave_b_prepositioning": wave_b_prepos,
    }
    selection_path = SWEEP_DIR / "wave_a_selection.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2))
    print(f"Wrote {selection_path}")
    print(f"  Chosen rank: n_latent={selection['chosen_rank']}")
    print(f"  Wave B story prepositioning: {wave_b_prepos['recommended_story']}")

    chosen_pt = SWEEP_DIR / f"n{selection['chosen_rank']}" / "best_latent_circuit.pt"
    target_pt = TOP_LEVEL_DIR / "best_latent_circuit_waveA.pt"
    if not chosen_pt.exists():
        print(f"ERROR: chosen rank's best_latent_circuit.pt missing: {chosen_pt}")
        return 2
    shutil.copy(chosen_pt, target_pt)
    print(f"Copied {chosen_pt} -> {target_pt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
