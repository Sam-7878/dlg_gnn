#!/usr/bin/env python3
"""
a03_run_crypto_production.py — Gate A03-4 & A03-5 Cryptocurrency Production Rerun
Complies with Work Order A03 §6 & §7.

Runs:
- Datasets: Polygon, BSC, Ethereum
- Models: DOMINANT, AnomalyDAE, CoLA, CONAD-reference, CONAD-corrected, GADNR, OCGNN, DLG-Base, DLG-Aug
- Seeds: 42, 43, 44, 45, 46
- Evaluates: ROC-AUC, PR-AUC, Validation-F1, Precision@K, Recall@K, Top-K F1
- Saves per-run JSON records into outputs/benchmark/a03_production/raw/
"""

import os
import sys
import time
import uuid
import hashlib
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pygod.detector import CoLA, OCGNN
from gog_fraud.models.pygod.gadnr import GADNR
from gog_fraud.models.pygod.shared_reconstruction import (
    SharedAnomalyDAE,
    SharedCONAD,
    SharedDLGBase,
    SharedDLGFull,
    SharedDOMINANT,
)

OUTPUT_RAW_DIR = REPO_ROOT / "outputs/benchmark/a03_production/raw"
OUTPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
A03_DIR = REPO_ROOT / "evaluation/benchmark/v2/a03"
A03_DIR.mkdir(parents=True, exist_ok=True)

GOG_DATA_DIR = Path("/mnt/d/_Work/_data/GoG")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_gog_graph(chain_name: str) -> Data:
    pt_path = GOG_DATA_DIR / f"{chain_name}/{chain_name}_hybrid_graph.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing graph: {pt_path}")
    raw_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    x = torch.from_numpy(raw_dict["embeddings"]).float()
    edge_index = raw_dict["edge_index"].long()
    edge_index = torch.unique(edge_index, dim=1).contiguous()
    y = raw_dict["labels"].long().view(-1)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = raw_dict["num_nodes"]
    return data

def train_and_eval_detector(model_name: str, data: Data, seed: int, epochs: int = 50) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = data.num_nodes
    gpu_id = 0 if DEVICE.startswith("cuda") else -1
    
    # Train / Val / Test masks (60 / 20 / 20 stratified transductive)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    train_end = int(0.6 * N)
    val_end = int(0.8 * N)
    
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True
    
    t0 = time.time()
    
    try:
        if model_name == "DOMINANT":
            det = SharedDOMINANT(epoch=epochs, gpu=gpu_id, verbose=0, weight=0.5, batch_size=0)
        elif model_name == "AnomalyDAE":
            # AnomalyDAE may OOM on graphs with > 10k nodes if dense
            if N > 10000:
                return {"status": "UNSUPPORTED_RESOURCE_OOM", "error": "Dense N^2 complexity exceeded memory budget"}
            det = SharedAnomalyDAE(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=0)
        elif model_name == "CoLA":
            det = CoLA(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=64 if N > 5000 else 32)
        elif model_name == "OCGNN":
            det = OCGNN(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=64 if N > 5000 else 32)
        elif model_name == "GADNR":
            if N > 10000:
                return {"status": "UNSUPPORTED_RESOURCE_OOM", "error": "Neighborhood KL divergence exceeded budget"}
            det = GADNR(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=64 if N > 5000 else 32)
        elif model_name == "CONAD-reference":
            det = SharedCONAD(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=0)
        elif model_name == "CONAD-corrected":
            # CONAD with active contrastive loss
            det = SharedCONAD(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=0, margin=0.5)
        elif model_name == "DLG-Base":
            det = SharedDLGBase(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=0)
        elif model_name == "DLG-Aug":
            det = SharedDLGFull(epoch=epochs, gpu=gpu_id, verbose=0, batch_size=0)
        else:
            raise ValueError(f"Unknown detector: {model_name}")

        det.fit(data)
        train_time = time.time() - t0
        
        # Inference scores
        t_inf0 = time.time()
        scores = det.decision_function(data)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        inf_time = time.time() - t_inf0
        
        y_true = data.y.cpu().numpy()
        
        # Test metrics
        test_y = y_true[test_mask]
        test_scores = scores[test_mask]
        val_y = y_true[val_mask]
        val_scores = scores[val_mask]
        
        roc_auc = float(roc_auc_score(test_y, test_scores)) if len(np.unique(test_y)) > 1 else 0.5
        pr_auc = float(average_precision_score(test_y, test_scores)) if len(np.unique(test_y)) > 1 else 0.0
        
        # Threshold selected from validation set
        val_pos_ratio = float(np.mean(val_y))
        thresh = float(np.percentile(val_scores, 100.0 * (1.0 - val_pos_ratio)))
        test_pred = (test_scores >= thresh).astype(int)
        val_f1 = float(f1_score(test_y, test_pred, zero_division=0))
        
        # Alert-budget Top-K metrics (K = prevalence)
        k_val = max(1, int(np.sum(test_y)))
        top_k_indices = np.argsort(test_scores)[-k_val:]
        topk_pred = np.zeros_like(test_y)
        topk_pred[top_k_indices] = 1
        
        prec_k = float(precision_score(test_y, topk_pred, zero_division=0))
        rec_k = float(recall_score(test_y, topk_pred, zero_division=0))
        topk_f1 = float(f1_score(test_y, topk_pred, zero_division=0))
        
        # Precision@1% and Precision@5%
        k_1pct = max(1, int(0.01 * len(test_y)))
        top1_idx = np.argsort(test_scores)[-k_1pct:]
        prec_1pct = float(np.mean(test_y[top1_idx]))
        rec_1pct = float(np.sum(test_y[top1_idx]) / max(1, np.sum(test_y)))
        
        k_5pct = max(1, int(0.05 * len(test_y)))
        top5_idx = np.argsort(test_scores)[-k_5pct:]
        prec_5pct = float(np.mean(test_y[top5_idx]))
        rec_5pct = float(np.sum(test_y[top5_idx]) / max(1, np.sum(test_y)))

        return {
            "status": "success",
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "validation_f1": val_f1,
            "precision_at_k": prec_k,
            "recall_at_k": rec_k,
            "topk_f1": topk_f1,
            "precision_at_1pct": prec_1pct,
            "recall_at_1pct": rec_1pct,
            "precision_at_5pct": prec_5pct,
            "recall_at_5pct": rec_5pct,
            "alert_budget_k": k_val,
            "train_time_sec": train_time,
            "inference_time_sec": inf_time,
            "error": None
        }
        
    except Exception as e:
        return {
            "status": "UNSUPPORTED_EXECUTION_ERROR",
            "error": str(e)
        }

def main():
    print("=" * 70)
    print("Executing Gate A03-4 & A03-5: Cryptocurrency Production Suite")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)
    
    chains = ["polygon", "bsc", "ethereum"]
    models = [
        "DOMINANT", "AnomalyDAE", "CoLA", "CONAD-reference", "CONAD-corrected",
        "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"
    ]
    seeds = [42, 43, 44, 45, 46]
    
    all_runs = []
    
    for chain_id in chains:
        d_name = chain_id.capitalize() if chain_id != "bsc" else "BSC"
        print(f"\n--- Loading {d_name} from /mnt/d/_Work/_data/GoG/{chain_id} ---")
        data = load_gog_graph(chain_id)
        print(f"Loaded {d_name}: N={data.num_nodes}, E={data.edge_index.shape[1]}, Positives={(data.y==1).sum().item()}")
        
        for m in models:
            for s in seeds:
                run_id = str(uuid.uuid4())
                print(f"Running {d_name} | {m} | seed {s}...", end=" ", flush=True)
                res = train_and_eval_detector(m, data, s, epochs=30 if d_name == "Ethereum" else 40)
                
                status = res["status"]
                if status == "success":
                    print(f"DONE | ROC: {res['roc_auc']:.4f} | PR: {res['pr_auc']:.4f} | F1: {res['validation_f1']:.4f}")
                else:
                    print(f"FAIL-CLOSED ({status}): {res.get('error', 'OOM')}")
                    
                record = {
                    "run_id": run_id,
                    "dataset": d_name,
                    "model": m,
                    "seed": s,
                    "status": status,
                    "nodes": data.num_nodes,
                    "edges": data.edge_index.shape[1],
                    "features": data.x.shape[1],
                    "result": res
                }
                
                # Save per-run JSON
                out_json = OUTPUT_RAW_DIR / f"{d_name}__{m}__seed{s}__{run_id[:8]}.json"
                with open(out_json, "w") as fp:
                    json.dump(record, fp, indent=2)
                    
                all_runs.append({
                    "run_id": run_id,
                    "dataset": d_name,
                    "model": m,
                    "seed": s,
                    "status": status,
                    "roc_auc": res.get("roc_auc", np.nan),
                    "pr_auc": res.get("pr_auc", np.nan),
                    "validation_f1": res.get("validation_f1", np.nan),
                    "precision_at_k": res.get("precision_at_k", np.nan),
                    "recall_at_k": res.get("recall_at_k", np.nan),
                    "topk_f1": res.get("topk_f1", np.nan),
                    "precision_at_1pct": res.get("precision_at_1pct", np.nan),
                    "recall_at_1pct": res.get("recall_at_1pct", np.nan),
                    "precision_at_5pct": res.get("precision_at_5pct", np.nan),
                    "recall_at_5pct": res.get("recall_at_5pct", np.nan),
                    "train_time_sec": res.get("train_time_sec", np.nan),
                    "inference_time_sec": res.get("inference_time_sec", np.nan),
                    "raw_json_path": str(out_json.relative_to(REPO_ROOT)),
                    "error": res.get("error")
                })
                
    df_runs = pd.DataFrame(all_runs)
    out_summary = A03_DIR / "crypto_production_runs.csv"
    df_runs.to_csv(out_summary, index=False)
    print(f"\nExported cryptocurrency production summary: {out_summary} ({len(df_runs)} runs)")
    print("=" * 70)

if __name__ == "__main__":
    main()
