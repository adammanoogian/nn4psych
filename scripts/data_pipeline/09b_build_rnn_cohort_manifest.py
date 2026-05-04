#!/usr/bin/env python3
"""Stage 09b: Build RNN cohort manifest for Bayesian model fitting.

For each trained checkpoint in ``trained_models/checkpoints/model_params_101000/``,
parse hyperparameters from the filename, classify which axis was swept (per-axis
design from Kumar et al. 2025 CCN, Fig. 2), re-run the canonical PIE_CP_OB_v2
behavior extraction, compute the ΔArea metric (Area_CP - Area_OB; Eq. 8 of the
paper), and write a per-checkpoint cohort manifest consumed by Plan 04-04b for
Reduced Bayesian Observer fitting.

Inputs
------
- ``trained_models/checkpoints/model_params_101000/*.pth`` (1,884 checkpoints)
- ``envs.PIE_CP_OB_v2`` (helicopter task)
- ``nn4psych.models.actor_critic.ActorCritic``

Outputs
-------
- ``data/processed/rnn_cohort/checkpoint_metrics.parquet`` (per-checkpoint table)
- ``data/processed/rnn_cohort/cohort_manifest.json`` (manifest for 04-04b)
- ``figures/rnn_cohort/delta_area_by_axis.png`` (Kumar Fig. 2 replication)

Notes
-----
The Kumar canonical configuration is γ=0.95, p_reset=0.0, τ=100, β_δ=1.0.
Each axis sweep varies one hyperparameter while holding the other three at the
canonical values; ``axis_swept`` is set accordingly. Checkpoints that don't
match any pure-axis cell are tagged ``off-axis`` and excluded from the default
cohort.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINTS_DIR, FIGURE_DIR, PROCESSED_DATA_DIR


# Kumar et al. 2025 canonical (per Fig. 1B top, Fig. 2 axes)
CANONICAL = {"gamma": 0.95, "preset_memory": 0.0, "rollout_size": 100, "td_scale": 1.0}
AXIS_LABEL = {
    "gamma": r"$\gamma$ (discount)",
    "preset_memory": r"$p_{reset}$",
    "rollout_size": r"$\tau$ (rollout)",
    "td_scale": r"$\beta_\delta$ (TD scale)",
}


def _load_module(name: str, path: Path):
    """Load a numbered pipeline script as an importable module."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_filename(filename: str) -> dict | None:
    """Parse hyperparameters from a checkpoint filename.

    Filename pattern: ``{perf}_V3_{gamma}g_{preset}rm_{rollout}bz_{td}td_{tdscale}tds_..._{seed}s.pth``.
    """
    parts = Path(filename).stem.split("_")
    try:
        params: dict = {"performance_score": float(parts[0]), "version": parts[1]}
        for part in parts[2:]:
            if part.endswith("tds"):
                params["td_scale"] = float(part[:-3])
            elif part.endswith("td"):
                params["td_penalty"] = float(part[:-2])
            elif part.endswith("g"):
                params["gamma"] = float(part[:-1])
            elif part.endswith("rm"):
                params["preset_memory"] = float(part[:-2])
            elif part.endswith("bz"):
                params["rollout_size"] = int(part[:-2])
            elif part.endswith("md"):
                params["max_displacement"] = int(part[:-2])
            elif part.endswith("rz"):
                params["reward_size"] = float(part[:-2])
            elif part.endswith("n"):
                params["hidden_dim"] = int(part[:-1])
            elif part.endswith("e"):
                params["epochs_trained"] = int(part[:-1])
            elif part.endswith("s"):
                params["seed"] = int(part[:-1])
        required = {"gamma", "preset_memory", "rollout_size", "td_scale", "seed"}
        if not required.issubset(params):
            return None
        return params
    except (ValueError, IndexError):
        return None


def classify_axis(row: pd.Series) -> str:
    """Tag which axis was swept for this checkpoint.

    Returns 'gamma', 'preset', 'rollout', 'tdscale', 'canonical', or 'off-axis'.
    """
    g_off = not np.isclose(row["gamma"], CANONICAL["gamma"])
    p_off = not np.isclose(row["preset_memory"], CANONICAL["preset_memory"])
    r_off = row["rollout_size"] != CANONICAL["rollout_size"]
    s_off = not np.isclose(row["td_scale"], CANONICAL["td_scale"])

    n_off = sum([g_off, p_off, r_off, s_off])
    if n_off == 0:
        return "canonical"
    if n_off > 1:
        return "off-axis"
    if g_off:
        return "gamma"
    if p_off:
        return "preset"
    if r_off:
        return "rollout"
    return "tdscale"


def axis_value(row: pd.Series) -> float:
    """Extract the swept-axis value for plotting (NaN for off-axis)."""
    mapping = {
        "gamma": "gamma",
        "preset": "preset_memory",
        "rollout": "rollout_size",
        "tdscale": "td_scale",
        "canonical": "gamma",
    }
    col = mapping.get(row["axis_swept"])
    return float(row[col]) if col else float("nan")


def build_inventory(checkpoint_dir: Path) -> pd.DataFrame:
    """Parse all checkpoint filenames and classify axis_swept."""
    files = sorted(checkpoint_dir.glob("*.pth"))
    rows = []
    for f in files:
        p = parse_filename(f.name)
        if p is None:
            continue
        p["checkpoint_path"] = f.relative_to(PROJECT_ROOT).as_posix()
        p["filename"] = f.name
        p["checkpoint_id"] = f.stem
        rows.append(p)
    df = pd.DataFrame(rows)
    df["axis_swept"] = df.apply(classify_axis, axis=1)
    df["axis_value"] = df.apply(axis_value, axis=1)
    return df


def extract_delta_area(
    checkpoint_path: Path,
    extract_fn,
    area_fn,
    epochs: int,
) -> tuple[float, float, float] | None:
    """Run behavior extraction and compute Area_CP, Area_OB, ΔArea.

    Returns
    -------
    (area_cp, area_ob, delta_area) or None on failure.
    """
    try:
        states = extract_fn(str(checkpoint_path), epochs=epochs)
    except Exception as exc:
        print(f"  [WARN] extract failed for {checkpoint_path.name}: {exc}")
        return None

    # Recompute area_cp / area_ob separately so we can save both.
    # (calculate_area_metric in 05_*.py returns only the difference.)
    threshold = 20
    helper = importlib.util.spec_from_file_location(
        "_h", str(PROJECT_ROOT / "scripts" / "data_pipeline" / "05_visualize_hyperparameter_effects.py")
    )
    if helper is None or helper.loader is None:
        delta = float(area_fn(states))
        return float("nan"), float("nan"), delta
    hmod = importlib.util.module_from_spec(helper)
    helper.loader.exec_module(hmod)
    from nn4psych.utils.metrics import get_lrs_v2

    areas: dict[str, float] = {}
    for c, cond in enumerate(["cp", "ob"]):
        pes_all, lrs_all = [], []
        for e in range(states.shape[0]):
            pe, lr = get_lrs_v2(states[e, c], threshold=threshold)
            pes_all.extend(pe)
            lrs_all.extend(lr)
        if pes_all:
            pes = np.asarray(pes_all)
            lrs = np.asarray(lrs_all)
            order = np.argsort(pes)
            areas[cond] = float(np.trapezoid(lrs[order], pes[order]))
        else:
            areas[cond] = 0.0
    return areas["cp"], areas["ob"], areas["cp"] - areas["ob"]


def compute_metrics(
    inventory: pd.DataFrame,
    epochs: int,
    parquet_out: Path,
    resume: bool,
) -> pd.DataFrame:
    """Run behavior extraction for each checkpoint, append ΔArea columns.

    Resumable: if ``parquet_out`` already exists, skip checkpoints already in it.
    """
    parquet_out.parent.mkdir(parents=True, exist_ok=True)

    extract_mod = _load_module(
        "viz", PROJECT_ROOT / "scripts" / "data_pipeline" / "04_visualize_behavioral_summary.py"
    )
    area_mod = _load_module(
        "hp", PROJECT_ROOT / "scripts" / "data_pipeline" / "05_visualize_hyperparameter_effects.py"
    )
    extract_fn = extract_mod.extract_model_behavior
    area_fn = area_mod.calculate_area_metric

    if resume and parquet_out.exists():
        cached = pd.read_parquet(parquet_out)
        done = set(cached["checkpoint_id"].tolist())
        print(f"resume: {len(done)} / {len(inventory)} already computed in {parquet_out}")
        todo = inventory[~inventory["checkpoint_id"].isin(done)].copy()
    else:
        cached = pd.DataFrame()
        todo = inventory.copy()

    n_total = len(todo)
    if n_total == 0:
        print("nothing to do (parquet up to date)")
        return cached

    print(f"computing ΔArea for {n_total} checkpoints @ {epochs} epochs each")
    results = []
    t_start = time.time()
    save_every = 100
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        ck_path = PROJECT_ROOT / row.checkpoint_path
        result = extract_delta_area(ck_path, extract_fn, area_fn, epochs)
        if result is None:
            area_cp, area_ob, delta = (float("nan"),) * 3
            status = "extraction_failed"
        else:
            area_cp, area_ob, delta = result
            status = "ok"
        results.append(
            {"checkpoint_id": row.checkpoint_id, "area_cp": area_cp, "area_ob": area_ob, "delta_area": delta, "status": status}
        )
        if i % 25 == 0 or i == n_total:
            elapsed = time.time() - t_start
            eta = elapsed / i * (n_total - i)
            print(f"  [{i}/{n_total}] elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min latest={row.filename[:60]}... ΔArea={delta:.3f}")
        if i % save_every == 0 or i == n_total:
            partial = pd.DataFrame(results)
            partial = inventory.merge(partial, on="checkpoint_id", how="inner")
            if not cached.empty:
                partial = pd.concat([cached, partial], ignore_index=True).drop_duplicates("checkpoint_id", keep="last")
            partial.to_parquet(parquet_out, index=False)
            print(f"  ↳ checkpoint saved: {parquet_out} ({len(partial)} rows)")

    final = pd.read_parquet(parquet_out)
    return final


def write_manifest(metrics: pd.DataFrame, manifest_out: Path) -> None:
    """Write cohort_manifest.json consumed by 04-04b."""
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    cohort = metrics[metrics["axis_swept"].isin(["gamma", "preset", "rollout", "tdscale", "canonical"])].copy()
    seeds_records = []
    for r in cohort.itertuples(index=False):
        seeds_records.append(
            {
                "checkpoint_id": r.checkpoint_id,
                "model_path": r.checkpoint_path,
                "axis_swept": r.axis_swept,
                "axis_value": float(r.axis_value),
                "gamma": float(r.gamma),
                "preset_memory": float(r.preset_memory),
                "rollout_size": int(r.rollout_size),
                "td_scale": float(r.td_scale),
                "seed": int(r.seed),
                "delta_area": float(r.delta_area),
                "area_cp": float(r.area_cp),
                "area_ob": float(r.area_ob),
                "performance_score": float(r.performance_score),
                "status": r.status,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "n_total_checkpoints": int(len(metrics)),
        "n_in_cohort": int(len(cohort)),
        "n_off_axis_excluded": int((metrics["axis_swept"] == "off-axis").sum()),
        "canonical": CANONICAL,
        "axis_counts": cohort["axis_swept"].value_counts().to_dict(),
        "seeds": seeds_records,
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written: {manifest_out} ({len(seeds_records)} cohort entries)")


def plot_axis_sanity(metrics: pd.DataFrame, fig_out: Path) -> None:
    """Replicate Kumar Fig. 2: ΔArea vs each axis (95% CI across seeds)."""
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    axes_to_plot = [
        ("gamma", "gamma"),
        ("preset", "preset_memory"),
        ("rollout", "rollout_size"),
        ("tdscale", "td_scale"),
    ]
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    for (axis_swept, col), ax in zip(axes_to_plot, axarr.ravel()):
        sub = metrics[(metrics["axis_swept"] == axis_swept) | (metrics["axis_swept"] == "canonical")]
        if len(sub) == 0:
            ax.set_title(f"(no data: {axis_swept})")
            continue
        grouped = sub.groupby(col)["delta_area"].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped.sort_values(col)
        sem = grouped["std"] / np.sqrt(grouped["count"]).replace(0, 1)
        ci95 = 1.96 * sem
        ax.errorbar(grouped[col], grouped["mean"], yerr=ci95, marker="o", capsize=3, linewidth=2, color="black")
        ax.fill_between(grouped[col], grouped["mean"] - ci95, grouped["mean"] + ci95, alpha=0.2, color="gray")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel(AXIS_LABEL[col])
        ax.set_ylabel(r"$\Delta$Area (CP - OB)")
        ax.set_title(f"{axis_swept}  (n_seeds≈{int(grouped['count'].median())} per cell)")
        if axis_swept == "rollout":
            ax.set_xscale("log")
    fig.suptitle("RNN cohort: ΔArea by hyperparameter axis (Kumar et al. 2025 Fig. 2 replication)")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=150)
    plt.savefig(fig_out.with_suffix(".svg"))
    plt.close()
    print(f"sanity figure written: {fig_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RNN cohort manifest for Plan 04-04b")
    parser.add_argument("--epochs", type=int, default=8, help="Behavior epochs per checkpoint (default 8)")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=CHECKPOINTS_DIR / "model_params_101000",
        help="Directory of trained .pth checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "rnn_cohort",
        help="Output directory for parquet + manifest",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=FIGURE_DIR / "rnn_cohort",
        help="Output directory for sanity figure",
    )
    parser.add_argument("--max_models", type=int, default=None, help="Cap (for smoke); default None = all")
    parser.add_argument("--no_resume", action="store_true", help="Force full recomputation")
    args = parser.parse_args()

    torch.set_num_threads(1)  # 16GB RAM machine; avoid thrashing

    print("=" * 60)
    print("STAGE 09b: BUILD RNN COHORT MANIFEST")
    print("=" * 60)

    print("\n1. Inventory checkpoints...")
    inventory = build_inventory(args.checkpoint_dir)
    if args.max_models:
        inventory = inventory.head(args.max_models)
    print(f"   total checkpoints: {len(inventory)}")
    print("   axis_swept counts:")
    for axis, n in inventory["axis_swept"].value_counts().items():
        print(f"     {axis}: {n}")

    print("\n2. Compute ΔArea per checkpoint (resumable)...")
    parquet_out = args.output_dir / "checkpoint_metrics.parquet"
    metrics = compute_metrics(inventory, args.epochs, parquet_out, resume=not args.no_resume)
    print(f"   metrics rows: {len(metrics)}")

    print("\n3. Write cohort manifest...")
    manifest_out = args.output_dir / "cohort_manifest.json"
    write_manifest(metrics, manifest_out)

    print("\n4. Plot sanity figure (Kumar Fig. 2 replication)...")
    fig_out = args.figure_dir / "delta_area_by_axis.png"
    plot_axis_sanity(metrics, fig_out)

    print("\n" + "=" * 60)
    print(f"DONE. Inputs to 04-04b:")
    print(f"  - {parquet_out}")
    print(f"  - {manifest_out}")
    print(f"  - {fig_out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
