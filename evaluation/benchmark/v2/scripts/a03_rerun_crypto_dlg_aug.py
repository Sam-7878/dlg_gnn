#!/usr/bin/env python3
"""
a03_rerun_crypto_dlg_aug.py — Rerun DLG-Aug on Polygon, BSC, Ethereum with fresh data.clone()
"""

import sys
import json
import time
import uuid
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gog_fraud.models.pygod.shared_reconstruction import SharedDLGFull

OUTPUT_RAW_DIR = REPO_ROOT / "outputs/benchmark/a03_production/raw"
GOG_DATA_DIR = Path("/mnt/d/_Work/_data/GoG")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
gpu_id = 0 if DEVICE.startswith("cuda") else -1


def load_gog_graph(chain_name: str) -> Data:
    pt_path = GOG_DATA_DIR / f"{chain_name}/{chain_name}_level2_graph.pt"
    if not pt_path.exists():
        pt_path = GOG_DATA_DIR / f"{chain_name}/{chain_name}_hybrid_graph.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing graph: {pt_path}")
    raw_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(raw_dict, Data):
        data = raw_dict
        if not hasattr(data, "num_nodes") or data.num_nodes is None:
            data.num_nodes = data.x.size(0)
        return data
    x = torch.from_numpy(raw_dict["embeddings"]).float() if isinstance(raw_dict["embeddings"], np.ndarray) else raw_dict["embeddings"].float()
    edge_index = raw_dict["edge_index"].long()
    edge_index = torch.unique(edge_index, dim=1).contiguous()
    y = raw_dict["labels"].long().view(-1)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = raw_dict["num_nodes"]
    return data



def evaluate_split(y_true, scores, seed: int):
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

    k = max(1, int(y_test.sum()))
    topk_idx = np.argsort(s_test)[-k:]
    p_k = float(precision_score(y_test, [1 if i in topk_idx else 0 for i in range(len(y_test))], zero_division=0))
    r_k = float(recall_score(y_test, [1 if i in topk_idx else 0 for i in range(len(y_test))], zero_division=0))
    topk_f1 = float(2 * p_k * r_k / max(p_k + r_k, 1e-9))

    return roc, pr, val_f1, p_k, r_k, topk_f1


def main():
    print("=== Rerunning DLG-Aug on Cryptocurrency Graphs ===")
    chains = ["Polygon", "BSC", "Ethereum"]
    seeds = [42, 43, 44, 45, 46]

    for c in chains:
        raw_graph = load_gog_graph(c)
        epochs = 30 if c == "Ethereum" else 40

        for s in seeds:
            print(f"Running {c} | DLG-Aug | seed {s}...", end=" ", flush=True)
            data_fresh = raw_graph.clone()
            torch.manual_seed(s)
            np.random.seed(s)

            det = SharedDLGFull(epoch=epochs, l1_epochs=20, gpu=gpu_id, verbose=0, batch_size=0)
            t0 = time.time()
            det.fit(data_fresh)
            scores = det.decision_function(data_fresh).cpu().numpy().reshape(-1)
            wall_time = time.time() - t0

            roc, pr, val_f1, p_k, r_k, topk_f1 = evaluate_split(data_fresh.y.cpu().numpy().reshape(-1), scores, s)
            print(f"DONE in {wall_time:.1f}s | ROC: {roc:.4f} | PR: {pr:.4f} | F1: {val_f1:.4f}")

            run_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c}_DLG-Aug_{s}"))
            record = {
                "run_id": run_id,
                "dataset": c,
                "model": "DLG-Aug",
                "seed": s,
                "status": "success",
                "roc_auc": round(roc, 6),
                "pr_auc": round(pr, 6),
                "validation_f1": round(val_f1, 6),
                "precision_at_k": round(p_k, 6),
                "recall_at_k": round(r_k, 6),
                "topk_f1": round(topk_f1, 6),
            }

            # Overwrite raw json
            out_json = OUTPUT_RAW_DIR / f"{c}__DLG-Aug__seed{s}__{run_id[:8]}.json"
            with open(out_json, "w") as fp:
                json.dump(record, fp, indent=2)

    # Update summary CSV
    print("Updating crypto_production_runs.csv...")
    csv_path = REPO_ROOT / "evaluation/benchmark/v2/a03/crypto_production_runs.csv"
    df = pd.read_csv(csv_path)

    for c in chains:
        for s in seeds:
            matches = list(OUTPUT_RAW_DIR.glob(f"{c}__DLG-Aug__seed{s}__*.json"))
            if matches:
                rec = json.loads(matches[0].read_text())
                mask = (df["dataset"] == c) & (df["model"] == "DLG-Aug") & (df["seed"] == s)
                if mask.any():
                    df.loc[mask, "status"] = "success"
                    roc = rec.get("roc_auc") or rec.get("result", {}).get("roc_auc")
                    pr = rec.get("pr_auc") or rec.get("result", {}).get("pr_auc")
                    f1 = rec.get("validation_f1") or rec.get("result", {}).get("validation_f1")
                    pk = rec.get("precision_at_k") or rec.get("result", {}).get("precision_at_k")
                    rk = rec.get("recall_at_k") or rec.get("result", {}).get("recall_at_k")
                    topf1 = rec.get("topk_f1") or rec.get("result", {}).get("topk_f1")

                    df.loc[mask, "roc_auc"] = roc
                    df.loc[mask, "pr_auc"] = pr
                    df.loc[mask, "validation_f1"] = f1
                    df.loc[mask, "precision_at_k"] = pk
                    df.loc[mask, "recall_at_k"] = rk
                    df.loc[mask, "topk_f1"] = topf1

    df.to_csv(csv_path, index=False)
    print("All DLG-Aug runs updated successfully!")


if __name__ == "__main__":
    main()
