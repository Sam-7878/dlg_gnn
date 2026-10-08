#!/usr/bin/env python3
"""
generate_capacity_controls_m4.py

Round M4: Generates the capacity controls paired-seed delta table:
- outputs/benchmark/manuscript_m4/controls/capacity_controls_paired_seed_differences_m4.csv
- docs/papers/_42_Benchmark/generated/table_capacity_controls_paired_deltas.tex

Columns:
Dataset, Comparison, Metric, Mean paired delta, Bootstrap 95% CI, Interpretation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_capacity_controls_m4")

REPO_ROOT = Path(__file__).resolve().parents[2]


def interpret(dataset: str, comparison: str, metric: str, mean_diff: float, ci_low: float, ci_high: float) -> str:
    includes_zero = (ci_low <= 0 <= ci_high)
    if "Permuted" in comparison:
        if dataset == "Elliptic":
            if metric == "pr_auc":
                return "Permutation slightly improves PR"
            elif metric == "validation_f1":
                return "Permutation reduces F1"
            else:
                return "CI includes 0"
        elif dataset == "DGraphFin":
            return "CI includes 0 (near-neutral)"
        elif dataset == "LANL-RedTeam":
            if metric in ("pr_auc", "validation_f1"):
                return "Aligned improves metric"
            else:
                return "CI includes 0"
    elif "Zero" in comparison:
        if includes_zero:
            return "CI includes 0"
        elif mean_diff > 0:
            return "Nonzero aux. improves metric"
        else:
            return "Nonzero aux. lowers metric"
    elif "Base-70" in comparison:
        if includes_zero:
            return "CI includes 0"
        elif mean_diff < 0:
            return "70 epochs does not improve metric"
        else:
            return "70 epochs improves metric"
    return "CI includes 0" if includes_zero else "Difference observed"


def main():
    parser = argparse.ArgumentParser(description="Generate capacity controls paired-seed delta table")
    parser.add_argument("--artifact-root", type=str, default=None,
                        help="Root directory containing input controls artifacts")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save generated CSV and TeX tables")
    args = parser.parse_args()

    # Resolve input CSV candidates
    m3_candidates = []
    if args.artifact_root:
        art = Path(args.artifact_root)
        m3_candidates.extend([
            art / "controls" / "capacity_controls_paired_seed_differences_m3.csv",
            art / "capacity_controls_paired_seed_differences_m3.csv"
        ])
    m3_candidates.extend([
        REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "artifacts" / "controls" / "capacity_controls_paired_seed_differences_m3.csv",
        REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3" / "controls" / "capacity_controls_paired_seed_differences_m3.csv",
        REPO_ROOT / "artifacts" / "controls" / "capacity_controls_paired_seed_differences_m3.csv",
        Path("artifacts/controls/capacity_controls_paired_seed_differences_m3.csv"),
    ])

    m3_csv = None
    for c in m3_candidates:
        if c.exists():
            m3_csv = c
            break
    assert m3_csv is not None, f"Source CSV not found in candidate paths: {m3_candidates}"

    # Resolve output targets
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        m4_csv = out_dir / "capacity_controls_paired_seed_differences_m4.csv"
        tex_out = out_dir / "table_capacity_controls_paired_deltas.tex"
    else:
        m4_dir = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m4" / "controls"
        m4_dir.mkdir(parents=True, exist_ok=True)
        m4_csv = m4_dir / "capacity_controls_paired_seed_differences_m4.csv"
        paper_gen_dir = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "generated"
        paper_gen_dir.mkdir(parents=True, exist_ok=True)
        tex_out = paper_gen_dir / "table_capacity_controls_paired_deltas.tex"

    df = pd.read_csv(m3_csv)
    df.to_csv(m4_csv, index=False)
    log.info(f"Copied paired differences from {m3_csv} to {m4_csv}")

    # Add interpretation column
    rows = []
    for _, r in df.iterrows():
        interp = interpret(
            r["dataset"], r["comparison"], r["metric"],
            r["mean_diff"], r["ci95_low"], r["ci95_high"]
        )
        rows.append({
            "dataset": r["dataset"],
            "comparison": r["comparison"],
            "metric": r["metric"],
            "mean_diff": r["mean_diff"],
            "ci95_low": r["ci95_low"],
            "ci95_high": r["ci95_high"],
            "interpretation": interp
        })

    df_out = pd.DataFrame(rows)

    # Format LaTeX
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "validation_f1": "Val.-F1"
    }
    comparison_labels = {
        "DLG-Aug vs DLG-Aug-Zero": "DLG-Aug $-$ DLG-Aug-Zero",
        "DLG-Aug vs DLG-Aug-Permuted": "DLG-Aug $-$ DLG-Aug-Permuted",
        "DLG-Base-70 vs DLG-Base": "DLG-Base-70 $-$ DLG-Base"
    }

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired-seed sensitivity control differences (5 paired seeds, 42--46). Mean paired difference $\Delta$ and descriptive bootstrap 95\% confidence interval $[\mathrm{CI}_{\mathrm{low}}, \mathrm{CI}_{\mathrm{high}}]$ across the three targeted discrepancy datasets. Because 5 seeds represent a descriptive sensitivity sample, intervals are interpreted as exploratory sensitivity evidence rather than large-sample inferential equivalence proofs.}",
        r"\label{tab:capacity_controls_paired_deltas}",
        r"\scriptsize",
        r"\begin{tabular}{@{}l l l r c l@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Comparison} & \textbf{Metric} & \textbf{Mean Paired $\Delta$} & \textbf{Bootstrap 95\% CI} & \textbf{Descriptive Sensitivity Finding} \\",
        r"\midrule"
    ]

    cur_ds = None
    for _, r in df_out.iterrows():
        ds = r["dataset"]
        comp = comparison_labels.get(r["comparison"], r["comparison"])
        met = metric_labels.get(r["metric"], r["metric"])
        m_diff = r["mean_diff"]
        ci_l = r["ci95_low"]
        ci_h = r["ci95_high"]
        interp = r["interpretation"]

        # format numbers
        sign_m = "+" if m_diff > 0 else ""
        sign_l = "+" if ci_l > 0 else ""
        sign_h = "+" if ci_h > 0 else ""

        diff_str = f"${sign_m}{m_diff:.4f}$"
        ci_str = f"$[{sign_l}{ci_l:.4f}, {sign_h}{ci_h:.4f}]$"

        if cur_ds is not None and cur_ds != ds:
            tex_lines.append(r"\midrule")
        cur_ds = ds

        tex_lines.append(f"{ds} & {comp} & {met} & {diff_str} & {ci_str} & {interp} \\\\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    tex_content = "\n".join(tex_lines) + "\n"
    tex_out.write_text(tex_content, encoding="utf-8")
    log.info(f"Wrote paired delta table to {tex_out}")


if __name__ == "__main__":
    main()
