"""Aggregate per-subject Reduced Bayesian Observer fits into summary tables.

Stage 09c: Reads all 268 per-fit JSONs produced by ``09b_fit_human_subject.py``,
builds a flat ``summary.csv`` with one row per (subject × condition), writes
``summary.md`` with cohort × condition median tables, and saves diagnostic
distribution plots.

Run after the SLURM array job (``cluster/09b_fit_human_subjects.slurm``)
completes:

    python scripts/data_pipeline/09c_aggregate_human_fits.py

Outputs:
    {output_dir}/summary.csv                                  — flat row-per-fit
    {output_dir}/summary.md                                   — cohort × condition tables
    {figures_dir}/r_distribution_by_cohort.png                — fit quality distributions
    {figures_dir}/posterior_param_distributions.png           — H/LW/log_UU posterior means
    {figures_dir}/diagnostics_rhat_ess.png                    — convergence diagnostics

Notes
-----
This script is read-only with respect to the per-fit outputs; running it
multiple times is idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ["changepoint", "oddball"]
PARAM_NAMES = ["H", "LW", "log_UU", "sigma_motor", "sigma_LR"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate per-subject RBO fits into summary tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/bayesian/human_validation",
        help="Directory containing per_fit/{subject}_{condition}.json",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures/bayesian/human_validation",
        help="Directory for output figures",
    )
    return parser.parse_args()


def load_fit_rows(input_dir: Path) -> list[dict]:
    """Load all per-fit JSONs and flatten into one row each.

    Parameters
    ----------
    input_dir : Path
        Directory containing ``per_fit/`` subdirectory.

    Returns
    -------
    list[dict]
        One dict per (subject, condition), with flat fields suitable for
        DataFrame construction.
    """
    per_fit_dir = input_dir / "per_fit"
    if not per_fit_dir.exists():
        raise FileNotFoundError(
            f"Expected per_fit/ subdirectory at {per_fit_dir}; not found"
        )

    rows: list[dict] = []
    for json_path in sorted(per_fit_dir.glob("*.json")):
        with open(json_path) as f:
            d = json.load(f)
        row = {
            "subject_id": d["subject_id"],
            "subject_idx": d["subject_idx"],
            "is_patient": d["is_patient"],
            "cohort": "patient" if d["is_patient"] else "control",
            "condition": d["condition"],
            "n_trials": d["n_trials"],
            "n_blocks": d["n_blocks"],
            "fit_status": d["fit"]["status"],
            "rhat_max": d["fit"].get("attempts", [{}])[-1].get("rhat_max", float("nan")),
            "ess_min": d["fit"].get("attempts", [{}])[-1].get("ess_min", float("nan")),
            "n_divergences": d["fit"].get("n_divergences", -1),
            "r_obs_vs_pred": d["behavior"]["r_obs_vs_pred"],
            "rmse": d["behavior"]["rmse"],
            "mae": d["behavior"]["mae"],
            "elapsed_seconds": d["elapsed_seconds"],
        }
        # Posterior summaries for the 5 fitted params
        params_block = d["fit"].get("params", {})
        for p in PARAM_NAMES:
            row[f"{p}_mean"] = params_block.get(p, {}).get("mean", float("nan"))
            row[f"{p}_sd"] = params_block.get(p, {}).get("sd", float("nan"))
            row[f"{p}_rhat"] = params_block.get(p, {}).get("rhat", float("nan"))
            row[f"{p}_ess_bulk"] = params_block.get(p, {}).get(
                "ess_bulk", float("nan")
            )
        rows.append(row)
    return rows


def write_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """Write a markdown summary report with cohort × condition median tables.

    Parameters
    ----------
    df : pd.DataFrame
        Flat row-per-fit dataframe.
    out_path : Path
        Markdown file to write.
    """
    n_total = len(df)
    n_pass = int((df["fit_status"] == "PASS").sum())
    n_failed = int((df["fit_status"] == "FAILED").sum())

    lines: list[str] = []
    lines.append("# Nassar 2021 RBO fits — aggregate summary\n")
    lines.append(f"- Total fits: **{n_total}**")
    lines.append(f"- Status PASS: **{n_pass}** / FAILED: **{n_failed}**")
    lines.append(f"- Subjects (unique): {df['subject_id'].nunique()}")
    lines.append(
        f"- Cohort split: patients={int(df[df['cohort']=='patient']['subject_id'].nunique())}, "
        f"controls={int(df[df['cohort']=='control']['subject_id'].nunique())}\n"
    )

    # Median fit quality table by cohort × condition
    lines.append("## Behavioral fit quality (median)\n")
    lines.append("| cohort | condition | r_obs_vs_pred | RMSE | n_divergences | rhat_max |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cohort in ["control", "patient"]:
        for cond in CONDITIONS:
            sub = df[(df["cohort"] == cohort) & (df["condition"] == cond)]
            if len(sub) == 0:
                continue
            lines.append(
                f"| {cohort} | {cond} | "
                f"{sub['r_obs_vs_pred'].median():.3f} | "
                f"{sub['rmse'].median():.2f} | "
                f"{sub['n_divergences'].median():.0f} | "
                f"{sub['rhat_max'].median():.3f} |"
            )

    # Median posterior means by cohort × condition
    lines.append("\n## Posterior means (median across subjects)\n")
    header = "| cohort | condition | " + " | ".join(PARAM_NAMES) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(PARAM_NAMES)) + "|"
    lines.append(header)
    lines.append(sep)
    for cohort in ["control", "patient"]:
        for cond in CONDITIONS:
            sub = df[(df["cohort"] == cohort) & (df["condition"] == cond)]
            if len(sub) == 0:
                continue
            cells = [f"{sub[f'{p}_mean'].median():.3f}" for p in PARAM_NAMES]
            lines.append(f"| {cohort} | {cond} | " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_distributions(
    df: pd.DataFrame, figures_dir: Path
) -> None:
    """Plot fit-quality and posterior distributions, one figure per family.

    Saves three PNGs:
    - ``r_distribution_by_cohort.png``: r_obs_vs_pred + RMSE histograms
    - ``posterior_param_distributions.png``: H, LW, log_UU posterior means
    - ``diagnostics_rhat_ess.png``: R-hat and ESS distributions

    Parameters
    ----------
    df : pd.DataFrame
        Flat row-per-fit dataframe.
    figures_dir : Path
        Output directory for figures.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    cohort_colors = {"control": "#1f77b4", "patient": "#d62728"}

    # ----- Fig 1: behavioral fit quality -----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col_idx, cond in enumerate(CONDITIONS):
        for cohort, color in cohort_colors.items():
            sub = df[(df["cohort"] == cohort) & (df["condition"] == cond)]
            axes[0, col_idx].hist(
                sub["r_obs_vs_pred"].dropna(),
                bins=20,
                alpha=0.55,
                label=cohort,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            axes[1, col_idx].hist(
                sub["rmse"].dropna(),
                bins=20,
                alpha=0.55,
                label=cohort,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
        axes[0, col_idx].set_title(f"{cond}: observed vs predicted r")
        axes[0, col_idx].set_xlabel("Pearson r")
        axes[0, col_idx].set_ylabel("subjects")
        axes[0, col_idx].legend()
        axes[1, col_idx].set_title(f"{cond}: RMSE")
        axes[1, col_idx].set_xlabel("RMSE (screen units)")
        axes[1, col_idx].set_ylabel("subjects")
        axes[1, col_idx].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "r_distribution_by_cohort.png", dpi=140)
    plt.close(fig)

    # ----- Fig 2: posterior parameter distributions -----
    cognitive_params = ["H", "LW", "log_UU"]
    fig, axes = plt.subplots(len(cognitive_params), 2, figsize=(11, 9))
    for row_idx, p in enumerate(cognitive_params):
        for col_idx, cond in enumerate(CONDITIONS):
            for cohort, color in cohort_colors.items():
                sub = df[(df["cohort"] == cohort) & (df["condition"] == cond)]
                axes[row_idx, col_idx].hist(
                    sub[f"{p}_mean"].dropna(),
                    bins=20,
                    alpha=0.55,
                    label=cohort,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
            axes[row_idx, col_idx].set_title(f"{cond}: posterior mean of {p}")
            axes[row_idx, col_idx].set_xlabel(p)
            axes[row_idx, col_idx].set_ylabel("subjects")
            if row_idx == 0:
                axes[row_idx, col_idx].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "posterior_param_distributions.png", dpi=140)
    plt.close(fig)

    # ----- Fig 3: convergence diagnostics -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cohort, color in cohort_colors.items():
        sub = df[df["cohort"] == cohort]
        axes[0].hist(
            sub["rhat_max"].dropna(),
            bins=30,
            alpha=0.55,
            label=cohort,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].hist(
            sub["ess_min"].dropna(),
            bins=30,
            alpha=0.55,
            label=cohort,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    axes[0].axvline(1.01, ls="--", color="black", label="gate (1.01)")
    axes[0].set_title("Max R-hat per fit")
    axes[0].set_xlabel("R-hat")
    axes[0].set_ylabel("fits")
    axes[0].legend()
    axes[1].axvline(400, ls="--", color="black", label="gate (400)")
    axes[1].set_title("Min ESS_bulk per fit")
    axes[1].set_xlabel("ESS")
    axes[1].set_ylabel("fits")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "diagnostics_rhat_ess.png", dpi=140)
    plt.close(fig)


def main() -> None:
    """Aggregate, write summary, and plot."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    figures_dir = Path(args.figures_dir)

    rows = load_fit_rows(input_dir)
    if not rows:
        raise RuntimeError(
            f"No per-fit JSON files found in {input_dir / 'per_fit'}; "
            f"expected output of 09b_fit_human_subject.py runs"
        )

    df = pd.DataFrame(rows)

    csv_path = input_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({len(df)} rows)")

    md_path = input_dir / "summary.md"
    write_summary_md(df, md_path)
    print(f"wrote {md_path}")

    plot_distributions(df, figures_dir)
    print(f"wrote figures to {figures_dir}/")

    # Console summary
    print("\n=== Aggregate summary ===")
    print(f"Total fits:       {len(df)}")
    print(f"Status PASS:      {(df['fit_status']=='PASS').sum()}")
    print(f"Status FAILED:    {(df['fit_status']=='FAILED').sum()}")
    print(f"Median r:         {df['r_obs_vs_pred'].median():.3f}")
    print(f"Median RMSE:      {df['rmse'].median():.2f}")
    print(f"Median rhat_max:  {df['rhat_max'].median():.4f}")
    print(f"Median ess_min:   {df['ess_min'].median():.1f}")


if __name__ == "__main__":
    main()
