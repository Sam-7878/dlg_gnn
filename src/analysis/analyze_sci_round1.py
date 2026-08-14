"""Generate evidence tables, statistics, and figures from round-1 raw results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import studentized_range

from gog_fraud.evaluation.statistics import friedman_dataset_test, paired_model_tests, spearman_with_bootstrap

METRICS = ("roc_auc", "pr_auc", "validation_f1")
TOPOLOGY = ("edge_homophily", "fraud_homophily", "adjusted_homophily", "label_assortativity", "positive_ratio", "avg_degree")


def _latex(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}"), encoding="utf-8")


def _performance_table(raw: pd.DataFrame, *, fraud_only: bool) -> pd.DataFrame:
    frame = raw.loc[raw.status.eq("success")].copy()
    if fraud_only:
        frame = frame.loc[frame.domain_group.eq("fraud_oriented")]
    seed_mean = frame.groupby(["dataset", "model"], as_index=False)[list(METRICS)].mean()
    overall = seed_mean.groupby("model")[list(METRICS)].agg(["mean", "std"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    ranks = seed_mean.copy()
    for metric in METRICS:
        ranks[f"{metric}_rank"] = ranks.groupby("dataset")[metric].rank(ascending=False, method="average")
    rank_columns = [f"{metric}_rank" for metric in METRICS]
    overall["composite_rank"] = ranks.groupby("model")[rank_columns].mean().mean(axis=1)
    return overall.reset_index().sort_values("composite_rank")


def _rank_heatmap(raw: pd.DataFrame, output: Path, *, fraud_only: bool, title: str) -> None:
    frame = raw.loc[raw.status.eq("success")]
    if fraud_only: frame = frame.loc[frame.domain_group.eq("fraud_oriented")]
    matrix = frame.groupby(["dataset", "model"], as_index=False).pr_auc.mean().pivot(index="model", columns="dataset", values="pr_auc")
    if matrix.empty:
        plt.figure(figsize=(7, 3)); plt.text(.5, .5, "No eligible data", ha="center", va="center")
        plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(output, dpi=220); plt.close(); return
    ranks = matrix.rank(axis=0, ascending=False, method="average")
    plt.figure(figsize=(max(7, len(ranks.columns) * 1.1), max(4, len(ranks) * .55)))
    sns.heatmap(ranks, annot=True, fmt=".1f", cmap="viridis_r", cbar_kws={"label": "PR-AUC rank (lower is better)"})
    plt.title(title); plt.tight_layout(); plt.savefig(output, dpi=220); plt.close()


def _critical_difference_plot(raw: pd.DataFrame, metric: str, output: Path) -> None:
    matrix = raw.loc[raw.status.eq("success")].groupby(["dataset", "model"], as_index=False)[metric].mean().pivot(index="dataset", columns="model", values=metric).dropna()
    plt.figure(figsize=(9, 3.5))
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        plt.text(.5, .5, "Critical difference requires >=2 complete datasets and >=3 models", ha="center", va="center"); plt.axis("off")
    else:
        ranks = matrix.rank(axis=1, ascending=False, method="average").mean().sort_values()
        k, n = len(ranks), len(matrix)
        q_alpha = float(studentized_range.ppf(.95, k, np.inf) / np.sqrt(2.0))
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
        y = np.arange(len(ranks)); plt.scatter(ranks.values, y, color="black")
        for index, (model, rank) in enumerate(ranks.items()): plt.text(rank + .04, index, model, va="center")
        plt.yticks([]); plt.xlabel("Average rank (lower is better)"); plt.xlim(.8, k + .8)
        plt.plot([1, 1 + cd], [len(ranks) + .2, len(ranks) + .2], color="tab:red", lw=4)
        plt.text(1 + cd / 2, len(ranks) + .35, f"CD={cd:.2f}", ha="center", color="tab:red")
        plt.ylim(-.7, len(ranks) + .8); plt.grid(axis="x", alpha=.25)
    plt.title(f"Nemenyi Critical Difference — {metric}"); plt.tight_layout(); plt.savefig(output, dpi=220); plt.close()


def analyze(output_root: Path, *, bootstrap_samples: int = 2000) -> dict[str, object]:
    raw_path = output_root / "multiseed/raw_results.csv"
    topology_path = output_root / "topology/dataset_topology_metrics.csv"
    raw, topology = pd.read_csv(raw_path), pd.read_csv(topology_path)
    tables, figures, statistics_dir = output_root / "tables", output_root / "figures", output_root / "statistics"
    for path in (tables, figures, statistics_dir): path.mkdir(parents=True, exist_ok=True)

    overall = _performance_table(raw, fraud_only=False); fraud = _performance_table(raw, fraud_only=True)
    overall.to_csv(tables / "table_a_overall_performance.csv", index=False); _latex(overall, tables / "table_a_overall_performance.tex")
    fraud.to_csv(tables / "table_b_fraud_performance.csv", index=False); _latex(fraud, tables / "table_b_fraud_performance.tex")
    topology.to_csv(tables / "table_d_topology.csv", index=False); _latex(topology, tables / "table_d_topology.tex")
    efficiency = raw.loc[raw.status.eq("success")].groupby("model", as_index=False).agg(runtime=("train_time_sec", "mean"), inference=("inference_time_sec", "mean"), RAM=("peak_ram_mb", "mean"), VRAM=("peak_vram_mb", "mean"))
    efficiency.to_csv(tables / "table_f_efficiency.csv", index=False); _latex(efficiency, tables / "table_f_efficiency.tex")

    stats, pairwise = [], []
    for metric in METRICS:
        try: stats.append(friedman_dataset_test(raw.loc[raw.status.eq("success")], metric=metric))
        except ValueError as exc: stats.append({"test": "friedman", "metric": metric, "status": "unsupported", "reason": str(exc)})
        try: pairwise.append(paired_model_tests(raw.loc[raw.status.eq("success")], metric=metric))
        except ValueError: pass
    (statistics_dir / "friedman.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pairwise_frame = pd.concat(pairwise, ignore_index=True) if pairwise else pd.DataFrame()
    pairwise_frame.to_csv(statistics_dir / "pairwise_holm.csv", index=False)
    pairwise_frame.to_csv(tables / "table_e_statistical_significance.csv", index=False)

    ablation_path = output_root / "ablation/ablation_summary.csv"
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path)
        ablation.to_csv(tables / "table_c_ablation.csv", index=False); _latex(ablation, tables / "table_c_ablation.tex")
        for metric, filename in (("pr_auc_mean", "07_ablation_pr_auc.png"), ("validation_f1_mean", "08_ablation_f1.png")):
            plt.figure(figsize=(7, 4)); sns.barplot(data=ablation, x="variant", y=metric, errorbar=None)
            plt.title(f"Empirical ablation — {metric}"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(figures / filename, dpi=220); plt.close()

    seed_mean = raw.loc[raw.status.eq("success")].groupby(["dataset", "model", "domain_group"], as_index=False)[list(METRICS)].mean()
    duplicate_metadata = [column for column in ("domain", "domain_group", "label_provenance") if column in topology.columns]
    joined = seed_mean.merge(topology.drop(columns=duplicate_metadata), on="dataset", how="inner")
    correlations = []
    for model, group in joined.groupby("model"):
        for topo_metric in TOPOLOGY:
            for performance_metric in METRICS:
                try:
                    result = spearman_with_bootstrap(group[topo_metric], group[performance_metric], iterations=bootstrap_samples)
                    correlations.append({"model": model, "topology_metric": topo_metric, "performance_metric": performance_metric, **result})
                except ValueError as exc:
                    correlations.append({"model": model, "topology_metric": topo_metric, "performance_metric": performance_metric, "rho": np.nan, "p_value": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n_datasets": len(group), "reason": str(exc)})
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(output_root / "topology/topology_performance_correlations.csv", index=False)

    _rank_heatmap(raw, figures / "01_rank_heatmap_all.png", fraud_only=False, title="Dataset-wise PR-AUC Rank")
    _rank_heatmap(raw, figures / "02_rank_heatmap_fraud.png", fraud_only=True, title="Fraud-oriented Dataset PR-AUC Rank")
    _critical_difference_plot(raw, "pr_auc", figures / "09_cd_pr_auc.png")
    _critical_difference_plot(raw, "validation_f1", figures / "10_cd_f1.png")
    plot_map = {"edge_homophily": "03_raw_homophily_correlation.png", "fraud_homophily": "04_fraud_homophily_correlation.png", "adjusted_homophily": "05_adjusted_homophily_correlation.png"}
    for metric, filename in plot_map.items():
        subset = correlation_frame.loc[correlation_frame.performance_metric.eq("pr_auc")].pivot(index="model", columns="topology_metric", values="rho")
        plt.figure(figsize=(8, max(4, len(subset) * .5))); sns.heatmap(subset[[metric]], annot=True, vmin=-1, vmax=1, cmap="coolwarm")
        plt.title(f"Association between {metric} and PR-AUC"); plt.tight_layout(); plt.savefig(figures / filename, dpi=220); plt.close()
    fraud_points = joined.loc[joined.domain_group.eq("fraud_oriented")]
    plt.figure(figsize=(7, 5))
    if fraud_points.empty:
        plt.text(.5, .5, "No fraud-oriented dataset in this run", ha="center", va="center"); plt.axis("off")
    else:
        sns.scatterplot(data=fraud_points, x="fraud_homophily", y="pr_auc", hue="model", style="dataset")
    plt.title("Association between Minority-Conditioned Homophily and PR-AUC"); plt.tight_layout(); plt.savefig(figures / "06_fraud_homophily_vs_pr_auc.png", dpi=220); plt.close()
    for y, filename in (("train_time_sec", "11_runtime_scaling.png"), ("peak_ram_mb", "12_memory_scaling.png")):
        plt.figure(figsize=(7, 5)); sns.scatterplot(data=raw.loc[raw.status.eq("success")], x="num_nodes", y=y, hue="model")
        plt.xscale("log"); plt.yscale("log"); plt.title(f"Observed {y} scaling"); plt.tight_layout(); plt.savefig(figures / filename, dpi=220); plt.close()
    return {"raw_rows": len(raw), "successful_rows": int(raw.status.eq("success").sum()), "topology_datasets": len(topology), "correlations": len(correlation_frame)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", default="outputs/sci"); parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(); summary = analyze(Path(args.output_root), bootstrap_samples=args.bootstrap_samples); print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
