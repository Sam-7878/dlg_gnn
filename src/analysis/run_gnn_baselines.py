#!/usr/bin/env python3
"""
Supervised Graph Neural Network Baselines (GCN, GIN, GraphSAGE, GATv2).
Evaluates canonical graph neural architectures under identical temporal splits,
reference graph construction, and 5 random seeds (P2-B).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv

from gog_fraud.pipelines.run_round4_experiments import (
    SciV2Records,
    _data,
    _normalize,
    _seed,
)

CHAINS = ["ethereum", "bsc", "polygon", "pooled"]
SEEDS = [11, 22, 33, 44, 55]


class StandardGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, gnn_type: str = "GCN", dropout: float = 0.25):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout

        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if gnn_type == "GCN":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == "GraphSAGE":
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif gnn_type == "GATv2":
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim // 2, heads=2)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 2, heads=2)
        elif gnn_type == "GIN":
            nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_mlp(x)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).view(-1)


def evaluate_preds(y_true: np.ndarray, probas: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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


def run_gnn_baselines(dataset_root: Path, output_dir: Path, device: torch.device, epochs: int = 50) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    dataset = SciV2Records(dataset_root)
    records: List[Dict[str, Any]] = []

    gnn_models = ["GCN", "GraphSAGE", "GIN", "GATv2"]

    for chain in CHAINS:
        train_ids, val_ids, test_ids = (dataset.ids(chain, g) for g in ("train", "validation", "test"))
        tx, ty = dataset.arrays(train_ids)
        vx, vy = dataset.arrays(val_ids)
        qx, qy = dataset.arrays(test_ids)

        tx, vx, qx = _normalize(tx, vx, qx)

        for seed in SEEDS:
            for gnn_type in gnn_models:
                _seed(seed)
                data_tr = _data(tx).to(device)
                y_tr = torch.from_numpy(ty).float().to(device)

                model = StandardGNN(tx.shape[1], hidden_dim=32, gnn_type=gnn_type).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                positives = max(1, int(ty.sum()))
                weight = torch.tensor([(len(ty) - positives) / positives], device=device)

                model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    logits = model(data_tr.x, data_tr.edge_index)
                    loss = F.binary_cross_entropy_with_logits(logits, y_tr, pos_weight=weight)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    # Validation inference
                    data_val = _data(tx, vx).to(device)
                    val_logits = model(data_val.x, data_val.edge_index)[len(tx) :]
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()

                    # Test inference
                    data_test = _data(tx, qx).to(device)
                    test_logits = model(data_test.x, data_test.edge_index)[len(tx) :]
                    test_probs = torch.sigmoid(test_logits).cpu().numpy()

                # Threshold selection on validation F1
                best_th = 0.5
                best_f1 = -1.0
                for th in np.linspace(0.1, 0.9, 81):
                    f1_cand = f1_score(vy, (val_probs >= th).astype(int), zero_division=0)
                    if f1_cand > best_f1:
                        best_f1 = f1_cand
                        best_th = float(th)

                gnn_metrics = evaluate_preds(qy, test_probs, threshold=best_th)
                records.append({
                    "chain": chain,
                    "seed": seed,
                    "model": f"Supervised-{gnn_type}",
                    "threshold": best_th,
                    **gnn_metrics,
                })

                # Save predictions
                pd.DataFrame({"sample_id": test_ids, "label": qy, "score": test_probs}).to_csv(
                    preds_dir / f"{chain}__{gnn_type}__seed{seed}.csv", index=False
                )

    df_out = pd.DataFrame(records)
    csv_path = output_dir / "supervised_gnn_baselines_metrics.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[Done] Evaluated {len(df_out)} supervised GNN experiments -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run supervised GNN baselines across 5 seeds.")
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/_Work/_data/GoG_sci_v2")
    parser.add_argument("--output-dir", type=str, default="results/sci_v3/baselines/gnn")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_gnn_baselines(Path(args.dataset_root), Path(args.output_dir), device)


if __name__ == "__main__":
    main()
