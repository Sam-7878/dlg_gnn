#!/usr/bin/env python3
"""
generate_capacity_controls_m3.py

Phase A / C Reconciled Capacity Controls Generator for Round M3.
Integrates:
1. Authoritative frozen primary references (from audit/control_reference_manifest.json)
   - Elliptic DLG-Base (0.0687) / DLG-Aug (0.1037)
   - DGraphFin DLG-Base (0.0101) / DLG-Aug (0.0134)
   - LANL DLG-Base (0.1367) / DLG-Aug (0.1114)
2. Enriched sensitivity control runs (from controls/raw/*.json)
   - DLG-Aug-Zero
   - DLG-Aug-Permuted
   - DLG-Base-70
3. Paired-seed difference analysis (seeds 42..46)

Outputs:
- outputs/benchmark/manuscript_m3/controls/capacity_controls_with_frozen_references_m3.csv
- outputs/benchmark/manuscript_m3/controls/capacity_controls_paired_seed_differences_m3.csv
- outputs/benchmark/manuscript_m3/controls/table_capacity_controls_m3.tex
- dlg_gnn/docs/papers/_42_Benchmark/generated/table_capacity_controls.tex
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("capacity_controls_m3")

REPO_ROOT = Path(__file__).resolve().parents[2]
M3_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3"
CONTROLS_DIR = M3_DIR / "controls"
RAW_DIR = CONTROLS_DIR / "raw"
AUDIT_DIR = M3_DIR / "audit"
PAPER_GEN_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "generated"


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_GEN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Authoritative Frozen Primary References
    ref_manifest_path = AUDIT_DIR / "control_reference_manifest.json"
    assert ref_manifest_path.exists(), f"Missing control reference manifest: {ref_manifest_path}"
    ref_manifest = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
    ref_records = pd.DataFrame(ref_manifest["records"])

    # 2. Load Sensitivity Control Runs
    control_records = []
    for f in sorted(RAW_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("status") == "success":
                control_records.append(d)
        except Exception as e:
            log.warning(f"Error reading {f}: {e}")

    df_ctrl = pd.DataFrame(control_records)
    log.info(f"Loaded {len(df_ctrl)} sensitivity control runs from {RAW_DIR}")
    assert len(df_ctrl) >= 45, f"Expected at least 45 control runs, got {len(df_ctrl)}"

    # Combine for table generation
    all_summary_rows = []

    # Datasets and models
    datasets = ["Elliptic", "DGraphFin", "LANL-RedTeam"]
    
    # Check frozen reference sanity
    for ds in datasets:
        for m in ["DLG-Base", "DLG-Aug"]:
            sub_ref = ref_records[(ref_records["dataset"] == ds) & (ref_records["model"] == m)]
            assert len(sub_ref) == 5, f"Expected 5 seeds for {ds}/{m}, got {len(sub_ref)}"
            roc_m, roc_s = sub_ref["roc_auc"].mean(), sub_ref["roc_auc"].std()
            pr_m, pr_s = sub_ref["pr_auc"].mean(), sub_ref["pr_auc"].std()
            f1_m, f1_s = sub_ref["validation_f1"].mean(), sub_ref["validation_f1"].std()

            # Hard sanity checks matching Work Order Section 8
            if ds == "Elliptic":
                if m == "DLG-Base":
                    assert abs(pr_m - 0.0687) < 0.001, f"Elliptic DLG-Base PR mismatch: {pr_m}"
                    assert not (0.29 < pr_m < 0.30), "CRITICAL: BOGUS 0.2917 detected!"
                elif m == "DLG-Aug":
                    assert abs(pr_m - 0.1037) < 0.001, f"Elliptic DLG-Aug PR mismatch: {pr_m}"
                    assert not (0.27 < pr_m < 0.29), "CRITICAL: BOGUS 0.2798 detected!"
            elif ds == "DGraphFin":
                if m == "DLG-Base":
                    assert abs(pr_m - 0.0101) < 0.0005, f"DGraphFin DLG-Base PR mismatch: {pr_m}"
                    assert not (0.035 < pr_m < 0.040), "CRITICAL: BOGUS 0.0381 detected!"
                elif m == "DLG-Aug":
                    assert abs(pr_m - 0.0134) < 0.0005, f"DGraphFin DLG-Aug PR mismatch: {pr_m}"
                    assert not (0.025 < pr_m < 0.030), "CRITICAL: BOGUS 0.0267 detected!"
            elif ds == "LANL-RedTeam":
                if m == "DLG-Base":
                    assert abs(pr_m - 0.1367) < 0.001, f"LANL DLG-Base PR mismatch: {pr_m}"
                elif m == "DLG-Aug":
                    assert abs(pr_m - 0.1114) < 0.001, f"LANL DLG-Aug PR mismatch: {pr_m}"

            epochs = "50" if m == "DLG-Base" else "70"
            all_summary_rows.append({
                "dataset": ds,
                "model": f"{m} (Authoritative)",
                "category": "Frozen primary reference",
                "epochs": epochs,
                "n_seeds": 5,
                "roc_mean": roc_m, "roc_std": roc_s,
                "pr_mean": pr_m, "pr_std": pr_s,
                "f1_mean": f1_m, "f1_std": f1_s,
            })

        for c_model in ["DLG-Aug-Zero", "DLG-Aug-Permuted", "DLG-Base-70"]:
            sub_ctrl = df_ctrl[(df_ctrl["dataset"] == ds) & (df_ctrl["model"] == c_model)]
            assert len(sub_ctrl) == 5, f"Expected 5 seeds for {ds}/{c_model}, got {len(sub_ctrl)}"
            roc_m, roc_s = sub_ctrl["roc_auc"].mean(), sub_ctrl["roc_auc"].std()
            pr_m, pr_s = sub_ctrl["pr_auc"].mean(), sub_ctrl["pr_auc"].std()
            f1_m, f1_s = sub_ctrl["validation_f1"].mean(), sub_ctrl["validation_f1"].std()
            ep = "70" if c_model == "DLG-Base-70" else "50" if c_model == "DLG-Aug-Zero" else "70"

            all_summary_rows.append({
                "dataset": ds,
                "model": c_model,
                "category": "M3 sensitivity control",
                "epochs": ep,
                "n_seeds": 5,
                "roc_mean": roc_m, "roc_std": roc_s,
                "pr_mean": pr_m, "pr_std": pr_s,
                "f1_mean": f1_m, "f1_std": f1_s,
            })

    df_summary = pd.DataFrame(all_summary_rows)
    summary_csv_path = CONTROLS_DIR / "capacity_controls_with_frozen_references_m3.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    log.info(f"Saved capacity controls summary to {summary_csv_path}")

    # 3. Paired-Seed Difference Analysis (seeds 42..46)
    paired_rows = []
    for ds in datasets:
        # Get per-seed series
        sub_ref_base = ref_records[(ref_records["dataset"] == ds) & (ref_records["model"] == "DLG-Base")].sort_values("seed")
        sub_ref_aug = ref_records[(ref_records["dataset"] == ds) & (ref_records["model"] == "DLG-Aug")].sort_values("seed")
        sub_zero = df_ctrl[(df_ctrl["dataset"] == ds) & (df_ctrl["model"] == "DLG-Aug-Zero")].sort_values("seed")
        sub_perm = df_ctrl[(df_ctrl["dataset"] == ds) & (df_ctrl["model"] == "DLG-Aug-Permuted")].sort_values("seed")
        sub_b70 = df_ctrl[(df_ctrl["dataset"] == ds) & (df_ctrl["model"] == "DLG-Base-70")].sort_values("seed")

        comparisons = [
            ("DLG-Aug vs DLG-Aug-Zero", sub_ref_aug, sub_zero),
            ("DLG-Aug vs DLG-Aug-Permuted", sub_ref_aug, sub_perm),
            ("DLG-Base-70 vs DLG-Base", sub_b70, sub_ref_base),
        ]

        for comp_name, group_a, group_b in comparisons:
            for metric in ["roc_auc", "pr_auc", "validation_f1"]:
                diffs = group_a[metric].values - group_b[metric].values
                ci_low, ci_high = bootstrap_ci(diffs)
                paired_rows.append({
                    "dataset": ds,
                    "comparison": comp_name,
                    "metric": metric,
                    "mean_diff": float(np.mean(diffs)),
                    "std_diff": float(np.std(diffs, ddof=1)),
                    "min_diff": float(np.min(diffs)),
                    "max_diff": float(np.max(diffs)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "seeds": "42..46",
                })

    df_paired = pd.DataFrame(paired_rows)
    paired_csv_path = CONTROLS_DIR / "capacity_controls_paired_seed_differences_m3.csv"
    df_paired.to_csv(paired_csv_path, index=False)
    log.info(f"Saved paired-seed differences to {paired_csv_path}")

    # 4. Generate LaTeX Table
    tex_lines = [
        r"% Auto-generated M3 Capacity and Training-Budget Sensitivity Controls Table",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation and capacity controls on discrepancy datasets (Elliptic, DGraphFin, and LANL-RedTeam). Authoritative rows report the frozen primary benchmark references, while sensitivity rows isolate input capacity (DLG-Aug-Zero), structural node alignment (DLG-Aug-Permuted), and training budget (DLG-Base-70). Results are mean $\pm$ std across 5 seeds.}",
        r"\label{tab:capacity_controls}",
        r"\footnotesize",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Variant / Control} & \textbf{Epochs} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{Val F1} \\",
        r"\midrule",
    ]

    for ds in datasets:
        tex_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{ds}}}}} \\\\")
        sub_ds = df_summary[df_summary["dataset"] == ds]
        for _, r in sub_ds.iterrows():
            m_name = r["model"]
            ep = r["epochs"]
            roc_str = f"${r['roc_mean']:.4f} \\pm {r['roc_std']:.4f}$"
            pr_str = f"${r['pr_mean']:.4f} \\pm {r['pr_std']:.4f}$"
            f1_str = f"${r['f1_mean']:.4f} \\pm {r['f1_std']:.4f}$"
            tex_lines.append(f" & {m_name} & {ep} & {roc_str} & {pr_str} & {f1_str} \\\\")
        if ds != datasets[-1]:
            tex_lines.append(r"\midrule")
        else:
            tex_lines.append(r"\bottomrule")

    tex_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    tex_content = "\n".join(tex_lines)
    m3_tex_path = CONTROLS_DIR / "table_capacity_controls_m3.tex"
    with open(m3_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    log.info(f"Saved M3 LaTeX table to {m3_tex_path}")

    paper_tex_path = PAPER_GEN_DIR / "table_capacity_controls.tex"
    with open(paper_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    log.info(f"Updated paper LaTeX table at {paper_tex_path}")


if __name__ == "__main__":
    main()
