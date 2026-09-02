"""Strengthen exact-name benchmark and strict-transfer tables with support and CIs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validation.sci_v3_final_common import atomic_csv


ROOT = Path("results/sci_v3_submission_r2")
TABLES = ROOT / "tables"
METRICS = ["roc_auc", "pr_auc", "f1", "precision", "fraud_recall", "fnr", "mcc", "balanced_accuracy"]


def save(name: str, frame: pd.DataFrame) -> None:
    path = TABLES / name
    atomic_csv(path.with_suffix(".csv"), frame)
    path.with_suffix(".tex").write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True),
        encoding="utf-8",
    )


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        n_seeds = int(group.seed.nunique())
        row.update({
            "n_seeds": n_seeds,
            "N": group.n_test.iloc[0] if "n_test" in group else pd.NA,
            "N_fraud": group.n_fraud.iloc[0] if "n_fraud" in group else pd.NA,
            "N_benign": group.n_benign.iloc[0] if "n_benign" in group else pd.NA,
        })
        for metric in METRICS:
            if metric in group:
                mean = float(group[metric].mean())
                sd = float(group[metric].std(ddof=1))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_ci95_low"] = mean - 1.96 * sd / np.sqrt(n_seeds) if n_seeds > 1 else np.nan
                row[f"{metric}_ci95_high"] = mean + 1.96 * sd / np.sqrt(n_seeds) if n_seeds > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    canonical = pd.read_csv("results/sci_v3_final/canonical/canonical_metrics.csv")
    supervised = canonical[
        canonical.evidence_family.isin(["tabular_baseline", "supervised_gnn_baseline"])
        | ((canonical.evidence_family == "legacy_main_raw_predictions") & canonical.model.isin(["DLG-L1", "DLG-L1-L2"]))
    ]
    save("table_main_supervised", summarize(supervised, ["chain", "model", "evidence_family"]))

    gad_models = ["DOMINANT", "AnomalyDAE", "CoLA", "CONAD", "DONE", "GAE"]
    gad = canonical[(canonical.evidence_family == "legacy_main_raw_predictions") & canonical.model.isin(gad_models)]
    save("table_unsupervised_gad", summarize(gad, ["chain", "model"]))

    cross = pd.read_csv("results/sci_v3_final/canonical/canonical_cross_chain.csv")
    strict = cross[cross.protocol == "strict_target_temporal_holdout"].copy()
    strict["target_interval"] = strict.target_test_start.astype("Int64").astype(str) + "--" + strict.target_test_end.astype("Int64").astype(str)
    summary = summarize(strict, ["train_chains", "target_chain", "target_interval"])
    undefined = strict.groupby(["train_chains", "target_chain", "target_interval"], dropna=False).metric_undefined_reason.first().reset_index()
    summary = summary.merge(undefined, on=["train_chains", "target_chain", "target_interval"], how="left")
    summary["metric_defined"] = summary.N_fraud.fillna(0) > 0
    summary.loc[~summary.metric_defined & summary.metric_undefined_reason.isna(), "metric_undefined_reason"] = "zero fraud-positive target support"
    save("table_cross_chain_strict", summary)
    print(f"wrote supervised={len(supervised)}, gad={len(gad)}, strict={len(strict)} source rows")


if __name__ == "__main__":
    main()
