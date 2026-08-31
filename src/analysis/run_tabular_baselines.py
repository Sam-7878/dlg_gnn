#!/usr/bin/env python3
"""
Classical Supervised Tabular Baselines (LogisticRegression, XGBoost, LightGBM).
Evaluates whether 11-D topological & chain summary features explain classification performance
without graph convolution under exact temporal splits and 5 random seeds (P2-A).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

from gog_fraud.pipelines.run_round4_experiments import (
    SciV2Records,
    _normalize,
)

CHAINS = ["ethereum", "bsc", "polygon", "pooled"]
SEEDS = [11, 22, 33, 44, 55]


def evaluate_model(y_true: np.ndarray, probas: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probas >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    both = len(np.unique(y_true)) == 2

    return {
        "roc_auc": float(roc_auc_score(y_true, probas)) if both else 0.0,
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, preds)) if both else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)) if both else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_tabular_baselines(dataset_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    dataset = SciV2Records(dataset_root)
    records: List[Dict[str, Any]] = []

    for chain in CHAINS:
        train_ids, val_ids, test_ids = (dataset.ids(chain, g) for g in ("train", "validation", "test"))
        tx, ty = dataset.arrays(train_ids)
        vx, vy = dataset.arrays(val_ids)
        qx, qy = dataset.arrays(test_ids)

        tx, vx, qx = _normalize(tx, vx, qx)

        # Scale weights for class imbalance
        pos_weight = float((len(ty) - np.sum(ty)) / max(1, np.sum(ty)))

        for seed in SEEDS:
            # 1. Logistic Regression
            lr = LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            )
            lr.fit(tx, ty)
            lr_val_p = lr.predict_proba(vx)[:, 1]
            lr_test_p = lr.predict_proba(qx)[:, 1]

            # Select optimal threshold on validation F1
            best_th = 0.5
            best_f1 = -1.0
            for th in np.linspace(0.1, 0.9, 81):
                f1_cand = f1_score(vy, (lr_val_p >= th).astype(int), zero_division=0)
                if f1_cand > best_f1:
                    best_f1 = f1_cand
                    best_th = float(th)

            lr_metrics = evaluate_model(qy, lr_test_p, threshold=best_th)
            records.append({
                "chain": chain, "seed": seed, "model": "LogisticRegression", "threshold": best_th, **lr_metrics
            })
            pd.DataFrame({"sample_id": test_ids, "label": qy, "score": lr_test_p}).to_csv(
                preds_dir / f"{chain}__LogisticRegression__seed{seed}.csv", index=False
            )

            # 2. XGBoost
            xgb_clf = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                tree_method="hist",
                eval_metric="logloss",
            )
            xgb_clf.fit(tx, ty, eval_set=[(vx, vy)], verbose=False)
            xgb_val_p = xgb_clf.predict_proba(vx)[:, 1]
            xgb_test_p = xgb_clf.predict_proba(qx)[:, 1]

            best_th = 0.5
            best_f1 = -1.0
            for th in np.linspace(0.1, 0.9, 81):
                f1_cand = f1_score(vy, (xgb_val_p >= th).astype(int), zero_division=0)
                if f1_cand > best_f1:
                    best_f1 = f1_cand
                    best_th = float(th)

            xgb_metrics = evaluate_model(qy, xgb_test_p, threshold=best_th)
            records.append({
                "chain": chain, "seed": seed, "model": "XGBoost", "threshold": best_th, **xgb_metrics
            })
            pd.DataFrame({"sample_id": test_ids, "label": qy, "score": xgb_test_p}).to_csv(
                preds_dir / f"{chain}__XGBoost__seed{seed}.csv", index=False
            )

            # 3. LightGBM
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
            )
            lgb_clf.fit(tx, ty, eval_set=[(vx, vy)], callbacks=[lgb.early_stopping(50, verbose=False)])
            lgb_val_p = lgb_clf.predict_proba(vx)[:, 1]
            lgb_test_p = lgb_clf.predict_proba(qx)[:, 1]

            best_th = 0.5
            best_f1 = -1.0
            for th in np.linspace(0.1, 0.9, 81):
                f1_cand = f1_score(vy, (lgb_val_p >= th).astype(int), zero_division=0)
                if f1_cand > best_f1:
                    best_f1 = f1_cand
                    best_th = float(th)

            lgb_metrics = evaluate_model(qy, lgb_test_p, threshold=best_th)
            records.append({
                "chain": chain, "seed": seed, "model": "LightGBM", "threshold": best_th, **lgb_metrics
            })
            pd.DataFrame({"sample_id": test_ids, "label": qy, "score": lgb_test_p}).to_csv(
                preds_dir / f"{chain}__LightGBM__seed{seed}.csv", index=False
            )

    df_out = pd.DataFrame(records)
    csv_path = output_dir / "tabular_baselines_metrics.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[Done] Evaluated {len(df_out)} tabular baseline experiments -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run classical tabular baselines across 5 seeds.")
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/_Work/_data/GoG_sci_v2")
    parser.add_argument("--output-dir", type=str, default="results/sci_v3/baselines/tabular")
    args = parser.parse_args()

    run_tabular_baselines(Path(args.dataset_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
