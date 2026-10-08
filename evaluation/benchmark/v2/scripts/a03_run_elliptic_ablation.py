#!/usr/bin/env python3
"""
a03_run_elliptic_ablation.py — Gate A03-7 Elliptic Mandatory 25-Run Ablation
Complies with Work Order A03 §9.

Runs on the actual frozen Elliptic artifact:
  - Variants (5): DLG-Base, DLG-Aug, DLG-Aug-Permuted, DLG-Aug-Zero, DLG-Base-70
  - Seeds (5): 42, 43, 44, 45, 46
  - Output: evaluation/benchmark/v2/paper_ready_a03/table_dlg_ablation_elliptic.csv
"""

import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDLGFull
from scripts import benchmark_8x10_pipeline as legacy

OUTPUT_CSV = REPO_ROOT / "evaluation/benchmark/v2/paper_ready_a03/table_dlg_ablation_elliptic.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

DATA_ROOT = "/mnt/d/_Work/_data/DLG"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
gpu_id = 0 if DEVICE.startswith("cuda") else -1


class SharedDLGAugZero(SharedDLGFull):
    """Ablation control: Features concatenated with zeros instead of L1 embeddings."""
    def process_graph(self, data):
        if hasattr(data, "_dlg_full_augmented") and data._dlg_full_augmented:
            data._reconstruction_backend = self.reconstruction_backend
            return
        self._orig_dim = data.x.size(1)
        data.dlg_original_x = data.x.clone()
        zeros = torch.zeros((data.x.size(0), self.l1_hid_dim), dtype=data.x.dtype, device=data.x.device)
        data.x = torch.cat([data.x, zeros], dim=-1)
        data._dlg_full_augmented = True
        data._reconstruction_backend = self.reconstruction_backend
        data._message_backend = self.message_backend


class SharedDLGAugPermuted(SharedDLGFull):
    """Ablation control: L1 embeddings randomly permuted across nodes prior to concatenation."""
    def __init__(self, perm_seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.perm_seed = perm_seed

    def process_graph(self, data):
        if hasattr(data, "_dlg_full_augmented") and data._dlg_full_augmented:
            data._reconstruction_backend = self.reconstruction_backend
            return
        self._orig_dim = data.x.size(1)
        data.dlg_original_x = data.x.clone()
        
        l1_embs = self._pretrain_level1(data)
        gen = torch.Generator().manual_seed(self.perm_seed)
        perm = torch.randperm(data.x.size(0), generator=gen)
        l1_embs_perm = l1_embs[perm]
        
        data.x = torch.cat([data.x, l1_embs_perm.to(data.x.device)], dim=-1)
        data._dlg_full_augmented = True
        data._reconstruction_backend = self.reconstruction_backend
        data._message_backend = self.message_backend


def load_frozen_elliptic() -> Data:
    legacy.DATA_ROOT = DATA_ROOT
    data = legacy.load_elliptic()
    return data


def compute_metrics(y_true, scores, seed: int):
    n = len(y_true)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    y_val, s_val = y_true[val_idx], scores[val_idx]
    y_test, s_test = y_true[test_idx], scores[test_idx]

    roc = float(roc_auc_score(y_test, s_test))
    pr = float(average_precision_score(y_test, s_test))

    # Threshold tuning on validation set
    thresholds = np.percentile(s_val, np.linspace(80, 99.5, 40))
    best_f1 = 0.0
    best_th = thresholds[0]
    for th in thresholds:
        pred_val = (s_val >= th).astype(int)
        f = f1_score(y_val, pred_val, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th

    pred_test = (s_test >= best_th).astype(int)
    val_f1 = float(f1_score(y_test, pred_test, zero_division=0))

    # Top 1% and 5% metrics
    n_test = len(y_test)
    k1 = max(1, int(0.01 * n_test))
    k5 = max(1, int(0.05 * n_test))

    topk1_idx = np.argsort(s_test)[-k1:]
    topk5_idx = np.argsort(s_test)[-k5:]

    total_pos = max(1, int(y_test.sum()))
    prec_1 = float(y_test[topk1_idx].sum() / k1)
    rec_1 = float(y_test[topk1_idx].sum() / total_pos)

    prec_5 = float(y_test[topk5_idx].sum() / k5)
    rec_5 = float(y_test[topk5_idx].sum() / total_pos)

    return roc, pr, val_f1, prec_1, rec_1, prec_5, rec_5


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=== Gate A03-7: Elliptic Mandatory 25-Run Ablation ===")
    raw_data = load_frozen_elliptic()
    print(f"Loaded frozen Elliptic: {raw_data.num_nodes:,} nodes, {raw_data.edge_index.size(1):,} edges, {int(raw_data.y.sum()):,} positives")

    variants = ["DLG-Base", "DLG-Aug", "DLG-Aug-Permuted", "DLG-Aug-Zero", "DLG-Base-70"]
    seeds = [42, 43, 44, 45, 46]
    records = []

    for var in variants:
        for s in seeds:
            print(f"\nRunning Elliptic | Variant: {var} | Seed: {s}...", end=" ", flush=True)
            data_copy = raw_data.clone()
            torch.manual_seed(s)
            np.random.seed(s)

            common_kwargs = {
                "gpu": gpu_id,
                "verbose": 0,
                "batch_size": 0,
            }

            if var == "DLG-Base":
                detector = SharedDLGBase(epoch=50, **common_kwargs)
                epoch_contract = "50_global"
            elif var == "DLG-Aug":
                detector = SharedDLGFull(epoch=50, l1_epochs=20, **common_kwargs)
                epoch_contract = "20_local_50_global"
            elif var == "DLG-Aug-Permuted":
                detector = SharedDLGAugPermuted(perm_seed=s, epoch=50, l1_epochs=20, **common_kwargs)
                epoch_contract = "20_local_permuted_50_global"
            elif var == "DLG-Aug-Zero":
                detector = SharedDLGAugZero(epoch=50, l1_epochs=0, **common_kwargs)
                epoch_contract = "zero_padded_50_global"
            elif var == "DLG-Base-70":
                detector = SharedDLGBase(epoch=70, **common_kwargs)
                epoch_contract = "70_global"
            else:
                raise ValueError(f"Unknown variant: {var}")

            t0 = time.time()
            detector.fit(data_copy)
            scores = detector.decision_function(data_copy).cpu().numpy().reshape(-1)
            wall_time = time.time() - t0

            # Count params
            param_count = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)

            roc, pr, val_f1, prec_1, rec_1, prec_5, rec_5 = compute_metrics(data_copy.y.cpu().numpy().reshape(-1), scores, s)
            print(f"DONE in {wall_time:.1f}s | ROC={roc:.4f}, PR={pr:.4f}, F1={val_f1:.4f}, P@1%={prec_1:.4f}, P@5%={prec_5:.4f}")

            h_str = f"Elliptic_{var}_{s}_{roc:.6f}_{pr:.6f}_{val_f1:.6f}"
            res_hash = hashlib.sha256(h_str.encode()).hexdigest()[:16]

            records.append({
                "dataset": "Elliptic",
                "variant": var,
                "seed": s,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4),
                "validation_f1": round(val_f1, 4),
                "precision_at_1pct": round(prec_1, 4),
                "recall_at_1pct": round(rec_1, 4),
                "precision_at_5pct": round(prec_5, 4),
                "recall_at_5pct": round(rec_5, 4),
                "parameter_count": param_count,
                "epoch_contract": epoch_contract,
                "result_hash": res_hash,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")

    # Summary table
    agg = df.groupby("variant")[["roc_auc", "pr_auc", "validation_f1", "precision_at_1pct", "recall_at_1pct", "precision_at_5pct", "recall_at_5pct"]].agg(["mean", "std"])
    print("\n--- Summary Performance Across 5 Seeds ---")
    print(agg)


if __name__ == "__main__":
    main()
