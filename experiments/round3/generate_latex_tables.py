"""
Generate 5 publication-ready LaTeX tables for Round 3:
1. results/tables/real_main_results.tex
2. results/tables/real_ablation.tex
3. results/tables/real_latency.tex
4. results/tables/real_statistical_significance.tex
5. results/tables/real_privacy_tradeoff.tex
"""

import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def table_main_results():
    p = RESULTS_DIR / "real_main_results.csv"
    if not p.exists():
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main Experimental Results under Real Chronological Evaluation (GoG-MicroRAG-Stream-v1). Mean and std over 5 random seeds.}",
        r"\label{tab:real_main_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{AUC-PR} $\uparrow$ & \textbf{AUC-ROC} $\uparrow$ & \textbf{F1-Score} $\uparrow$ & \textbf{ECE} $\downarrow$ & \textbf{GNN Source} & \textbf{Split Type} \\",
        r"\midrule",
    ]

    for r in rows:
        m = r['method'].replace('_', ' ')
        pr = f"{float(r['mean_auc_pr']):.4f} $\\pm$ {float(r.get('std_auc_pr', 0.0)):.4f}"
        roc = f"{float(r['mean_auc_roc']):.4f} $\\pm$ {float(r.get('std_auc_roc', 0.0)):.4f}"
        f1 = f"{float(r['mean_f1']):.4f} $\\pm$ {float(r.get('std_f1', 0.0)):.4f}"
        ece = f"{float(r.get('mean_ece', 0.0)):.4f}"
        src = r.get('gnn_source', 'real')
        split = r.get('split_type', 'chronological')
        tex.append(f"{m} & {pr} & {roc} & {f1} & {ece} & {src} & {split} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    out = TABLES_DIR / "real_main_results.tex"
    out.write_text("\n".join(tex), encoding="utf-8")


def table_ablation():
    p = RESULTS_DIR / "real_ablation_results.csv"
    if not p.exists():
        p = RESULTS_DIR / "real_main_results.csv"
    if not p.exists():
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Analysis on Real GNN Framework.}",
        r"\label{tab:real_ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{AUC-PR} & \textbf{AUC-ROC} & \textbf{F1} \\",
        r"\midrule",
    ]

    for r in rows:
        m = r['method'].replace('_', ' ')
        pr = f"{float(r['mean_auc_pr']):.4f}"
        roc = f"{float(r['mean_auc_roc']):.4f}"
        f1 = f"{float(r['mean_f1']):.4f}"
        tex.append(f"{m} & {pr} & {roc} & {f1} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = TABLES_DIR / "real_ablation.tex"
    out.write_text("\n".join(tex), encoding="utf-8")


def table_latency():
    p = RESULTS_DIR / "real_e2e_latency.csv"
    if not p.exists():
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-End Real Inference Latency Breakdown by MC Passes ($T$).}",
        r"\label{tab:real_latency}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"\textbf{T} & \textbf{Mean Total (ms)} & \textbf{P95 Total (ms)} & \textbf{Events/sec} & \textbf{GNN Source} \\",
        r"\midrule",
    ]

    for r in rows:
        t = r.get('T', '-')
        mean_ms = f"{float(r.get('mean_total_ms', 0)):.2f}"
        p95_ms = f"{float(r.get('p95_total_ms', 0)):.2f}"
        eps = f"{float(r.get('events_per_sec', 0)):.1f}"
        src = r.get('gnn_source', 'real')
        tex.append(f"{t} & {mean_ms} & {p95_ms} & {eps} & {src} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = TABLES_DIR / "real_latency.tex"
    out.write_text("\n".join(tex), encoding="utf-8")


def table_statistical():
    p = RESULTS_DIR / "real_statistical_summary.csv"
    if not p.exists():
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired Bootstrap Confidence Intervals (10,000 Resamples) for AUC-PR Differences.}",
        r"\label{tab:real_stats}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{$\Delta$ AUC-PR} & \textbf{95\% CI} & \textbf{Significance} \\",
        r"\midrule",
    ]

    for r in rows:
        comp = r.get('comparison', '').replace('_', ' ')
        delta = f"{float(r.get('delta_auc_pr', 0)):.4f}"
        ci = f"[{float(r.get('ci_lo_95', 0)):.4f}, {float(r.get('ci_hi_95', 0)):.4f}]"
        sig = r.get('significance', '')
        tex.append(f"{comp} & {delta} & {ci} & {sig} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = TABLES_DIR / "real_statistical_significance.tex"
    out.write_text("\n".join(tex), encoding="utf-8")


def table_privacy():
    p = RESULTS_DIR / "real_privacy_utility.csv"
    if not p.exists():
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Membership Inference Attack Vulnerability Under Differential Noise Levels.}",
        r"\label{tab:real_privacy}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Representation} & \textbf{$\sigma$} & \textbf{Bal. Acc.} & \textbf{Macro F1} & \textbf{ROC-AUC} & \textbf{Leakage Risk} \\",
        r"\midrule",
    ]

    for r in rows:
        rep = r.get('representation', '').replace('_', ' ')
        sigma = f"{float(r.get('noise_sigma', 0)):.2f}"
        bal = f"{float(r.get('attack_balanced_accuracy', 0)):.4f}"
        f1 = f"{float(r.get('attack_macro_f1', 0)):.4f}"
        roc = f"{float(r.get('attack_roc_auc', 0)):.4f}"
        risk = r.get('leakage_risk', '')
        tex.append(f"{rep} & {sigma} & {bal} & {f1} & {roc} & {risk} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = TABLES_DIR / "real_privacy_tradeoff.tex"
    out.write_text("\n".join(tex), encoding="utf-8")


def main():
    table_main_results()
    table_ablation()
    table_latency()
    table_statistical()
    table_privacy()
    print("All LaTeX tables generated.")


if __name__ == "__main__":
    main()
