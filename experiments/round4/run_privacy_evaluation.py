"""Run an actual membership attack on pre-event observable representations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import DATASET_DIR, RESULTS_DIR, ensure_dirs


def main() -> int:
    ensure_dirs()
    frame = pd.read_parquet(DATASET_DIR / "transactions.parquet")
    frame = frame[frame.split.isin(["train", "test"])].copy()
    membership = (frame.split == "train").astype(int).to_numpy()
    dt = pd.to_datetime(frame.timestamp, unit="s", utc=True)
    chain = pd.get_dummies(frame.chain_id).reindex(columns=["ethereum", "bsc", "polygon"], fill_value=0)
    representations = {
        "observable_graph_shape": np.column_stack((np.log1p(frame.num_nodes), np.log1p(frame.num_edges))),
        "label_independent_context": np.column_stack((
            np.log1p(frame.num_nodes), np.log1p(frame.num_edges),
            np.sin(2 * np.pi * dt.dt.hour / 24), np.cos(2 * np.pi * dt.dt.hour / 24),
            dt.dt.dayofweek / 6, chain.to_numpy(float),
        )),
    }
    rows = []
    for name, values in representations.items():
        for seed in (7, 17, 27, 37, 47):
            x_train, x_test, y_train, y_test = train_test_split(
                values, membership, test_size=0.30, random_state=seed, stratify=membership
            )
            attack = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
            attack.fit(x_train, y_train)
            probability = attack.predict_proba(x_test)[:, 1]
            predicted = (probability >= 0.5).astype(int)
            rows.append({
                "representation": name, "seed": seed, "attack_type": "logistic_membership_inference",
                "attack_accuracy": accuracy_score(y_test, predicted),
                "attack_balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                "attack_macro_f1": f1_score(y_test, predicted, average="macro"),
                "attack_roc_auc": roc_auc_score(y_test, probability),
                "attack_pr_auc": average_precision_score(y_test, probability),
                "majority_baseline": max(y_test.mean(), 1 - y_test.mean()),
                "random_baseline": 0.5, "metric_source": "measured",
                "framework_claim": "Privacy-Aware",
            })
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS_DIR / "privacy_utility.csv", index=False)
    print(json.dumps({"rows": len(output), "representations": list(representations),
                      "metric_source": "measured"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
