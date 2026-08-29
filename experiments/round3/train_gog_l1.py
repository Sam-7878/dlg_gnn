"""
Phase D: Real GNN Training on GoG-MicroRAG-Stream-v1

Trains Level1 GNN with in_dim=8 on the temporal train split.
Selects best checkpoint on validation AUC-PR.
Evaluates once on test set (held-out).
Runs 5 seeds: 7, 17, 27, 37, 47.

Usage:
    python experiments/round3/train_gog_l1.py [--seeds 7,17,27,37,47] [--epochs 80]

Outputs:
    results/real_checkpoints/l1_seed{N}_best.pt
    results/checkpoint_manifests/l1_seed{N}.json
    results/real_train_log.jsonl
"""

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).parent.parent.parent
if not (ROOT / "data").exists():
    ROOT = Path(os.environ.get("DLG_GNN_ROOT", "/mnt/d/_Work/goat_bank/dlg_gnn"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train_gog_l1")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"
CKPT_DIR = ROOT / "results" / "real_checkpoints"
MANIFEST_DIR = ROOT / "results" / "checkpoint_manifests"
TRAIN_LOG = ROOT / "results" / "real_train_log.jsonl"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal L1 GNN (GIN encoder, self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Aggregate neighbor features
        N = x.size(0)
        row, col = edge_index
        # Sum neighbor features
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        out = self.net((1 + self.eps) * x + agg)
        return out


class Level1GNNDirect(nn.Module):
    """
    Level1 GNN for GoG-MicroRAG (in_dim=8).
    Graph-level binary fraud classifier.
    """
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(GINLayer(dims[i], dims[i + 1], dropout))

        # Node-level head: produces per-node fraud score
        head_in = hidden_dim * 2  # mean + max pooling
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward_node(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

    def forward(self, data: Data) -> torch.Tensor:
        """Returns per-node logits [N, 1]."""
        h = self.forward_node(data.x, data.edge_index)
        # Pool globally for graph-level, but we need node-level for streaming fraud
        # Use per-node prediction (each node = one transaction)
        return self.head(torch.cat([h, h], dim=-1)).squeeze(-1)

    def forward_mc(self, data: Data, T: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC Dropout inference: returns mean, variance, entropy."""
        # Enable dropout during inference
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()  # Keep BN in eval mode

        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                logits = self.forward(data)
                probs_list.append(torch.sigmoid(logits))

        self.eval()
        probs = torch.stack(probs_list, dim=0)  # [T, N]
        mean_prob = probs.mean(0)
        variance = probs.var(0)
        eps = 1e-8
        entropy = -(mean_prob * torch.log(mean_prob + eps)
                    + (1 - mean_prob) * torch.log(1 - mean_prob + eps))
        return mean_prob, variance, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Data loading utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_ids(path: Path) -> List[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def build_pyg_data(graph: dict, ids: List[int]) -> Data:
    """
    Build a PyG Data object from a subset of node IDs.
    We remap node IDs to contiguous indices.
    """
    id_set = set(ids)
    # Global node features and labels
    all_x = graph["embeddings"]          # [N, 8]
    all_y = graph["labels"]              # [N]
    edge_index = graph["edge_index"]     # [2, E]

    # Filter edges: only include edges where BOTH src and dst are in ids
    src, dst = edge_index[0], edge_index[1]
    src_np = src.numpy()
    dst_np = dst.numpy()
    mask = np.isin(src_np, ids) & np.isin(dst_np, ids)
    filtered_src = src[torch.from_numpy(mask)]
    filtered_dst = dst[torch.from_numpy(mask)]

    # Remap to contiguous indices
    id_to_new = {old: new for new, old in enumerate(ids)}
    new_src = torch.tensor([id_to_new[int(i)] for i in filtered_src], dtype=torch.long)
    new_dst = torch.tensor([id_to_new[int(i)] for i in filtered_dst], dtype=torch.long)
    new_edge_index = torch.stack([new_src, new_dst], dim=0)

    x = all_x[ids]   # [len(ids), 8]
    y = all_y[ids]   # [len(ids)]

    return Data(x=x, edge_index=new_edge_index, y=y)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    try:
        auc_pr = float(average_precision_score(y, probs))
        auc_roc = float(roc_auc_score(y, probs))
    except Exception:
        auc_pr = 0.0
        auc_roc = 0.0
    preds = (probs >= 0.5).astype(int)
    f1 = float(f1_score(y, preds, zero_division=0))
    return {"auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1}


def train_one_seed(
    graph: dict,
    train_ids: List[int],
    valid_ids: List[int],
    test_ids: List[int],
    seed: int,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    set_seed(seed)
    log.info(f"[Seed {seed}] Building datasets...")

    # Build PyG data objects
    train_data = build_pyg_data(graph, train_ids).to(device)
    valid_data = build_pyg_data(graph, valid_ids).to(device)
    test_data = build_pyg_data(graph, test_ids).to(device)

    log.info(f"  train nodes={len(train_ids)}, edges={train_data.edge_index.shape[1]}, fraud={int(train_data.y.sum())}")
    log.info(f"  valid nodes={len(valid_ids)}, fraud={int(valid_data.y.sum())}")
    log.info(f"  test  nodes={len(test_ids)}, fraud={int(test_data.y.sum())}")

    # Model
    model = Level1GNNDirect(in_dim=8, hidden_dim=128, num_layers=3, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class-weighted loss to handle imbalance
    n_pos = int(train_data.y.sum())
    n_neg = len(train_ids) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc_pr = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits, train_data.y.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(valid_data)
            val_loss = criterion(val_logits, valid_data.y.float())
        val_metrics = compute_metrics(val_logits, valid_data.y)

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(loss),
            "val_loss": float(val_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_log)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"  [Seed {seed}] Epoch {epoch:3d}: loss={float(loss):.4f} "
                f"val_auc_pr={val_metrics['auc_pr']:.4f} val_auc_roc={val_metrics['auc_roc']:.4f}"
            )

        # Early stopping on val AUC-PR
        if val_metrics["auc_pr"] > best_val_auc_pr:
            best_val_auc_pr = val_metrics["auc_pr"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  [Seed {seed}] Early stop at epoch {epoch} (no improve for {patience} epochs)")
                break

    # Load best checkpoint
    model.load_state_dict(best_state)
    log.info(f"  [Seed {seed}] Best val AUC-PR={best_val_auc_pr:.4f} at epoch {best_epoch}")

    # Test evaluation (ONE time, after training complete)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_data)
    test_metrics = compute_metrics(test_logits, test_data.y)

    # MC Dropout test evaluation (T=10)
    test_mean, test_var, test_entropy = model.forward_mc(test_data, T=10)
    test_mc_metrics = compute_metrics(
        (test_mean * 2 - 1),  # convert prob to logit-like for compute_metrics (sigmoid applied inside)
        test_data.y,
    )
    # Actually compute directly:
    y_np = test_data.y.cpu().numpy().astype(int)
    mean_np = test_mean.cpu().numpy()
    try:
        mc_auc_pr = float(average_precision_score(y_np, mean_np))
        mc_auc_roc = float(roc_auc_score(y_np, mean_np))
    except Exception:
        mc_auc_pr = 0.0
        mc_auc_roc = 0.0

    log.info(
        f"  [Seed {seed}] TEST: auc_pr={test_metrics['auc_pr']:.4f} "
        f"auc_roc={test_metrics['auc_roc']:.4f} f1={test_metrics['f1']:.4f}"
    )
    log.info(f"  [Seed {seed}] TEST MC(T=10): auc_pr={mc_auc_pr:.4f} auc_roc={mc_auc_roc:.4f}")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc_pr": best_val_auc_pr,
        "test_metrics": test_metrics,
        "test_mc_auc_pr": mc_auc_pr,
        "test_mc_auc_roc": mc_auc_roc,
        "history": history,
        "model_state": best_state,
        "model": model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint saving with manifest
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(seed: int, result: dict, model_cfg: dict, graph_sha256: str):
    ckpt_path = CKPT_DIR / f"l1_seed{seed}_best.pt"

    # Config hash
    cfg_str = json.dumps(model_cfg, sort_keys=True).encode()
    cfg_sha256 = hashlib.sha256(cfg_str).hexdigest()

    torch.save(
        {
            "model_state_dict": result["model_state"],
            "model_config": model_cfg,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "best_val_auc_pr": result["best_val_auc_pr"],
            "test_metrics": result["test_metrics"],
            "test_mc_auc_pr": result["test_mc_auc_pr"],
            "gnn_source": "real_checkpoint",
            "split_type": "chronological_real",
            "dataset": "GoG-MicroRAG-Stream-v1",
            "dataset_sha256": graph_sha256,
            "config_sha256": cfg_sha256,
            "git_commit": git_commit(),
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        ckpt_path,
    )

    ckpt_sha256 = sha256_file(ckpt_path)

    manifest = {
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_sha256": ckpt_sha256,
        "seed": seed,
        "model_config": model_cfg,
        "best_epoch": result["best_epoch"],
        "best_val_auc_pr": result["best_val_auc_pr"],
        "test_auc_pr": result["test_metrics"]["auc_pr"],
        "test_auc_roc": result["test_metrics"]["auc_roc"],
        "test_f1": result["test_metrics"]["f1"],
        "test_mc_auc_pr": result["test_mc_auc_pr"],
        "gnn_source": "real_checkpoint",
        "split_type": "chronological_real",
        "dataset_sha256": graph_sha256,
        "config_sha256": cfg_sha256,
        "git_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    manifest_path = MANIFEST_DIR / f"l1_seed{seed}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"  Saved: {ckpt_path.relative_to(ROOT)}")
    log.info(f"  Manifest: {manifest_path.relative_to(ROOT)}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="7,17,27,37,47")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # Load data
    graph_path = DATA_DIR / "polygon_hybrid_graph.pt"
    log.info(f"Loading graph from {graph_path.relative_to(ROOT)}...")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_sha256 = sha256_file(graph_path)

    train_ids = load_ids(DATA_DIR / "train_ids.txt")
    valid_ids = load_ids(DATA_DIR / "valid_ids.txt")
    test_ids = load_ids(DATA_DIR / "test_ids.txt")

    log.info(f"Dataset: nodes={graph['embeddings'].shape[0]}, "
             f"fraud={int(graph['labels'].sum())}, "
             f"split={len(train_ids)}/{len(valid_ids)}/{len(test_ids)}")

    model_cfg = {
        "in_dim": 8,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "encoder_backend": "gnn_direct",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "split_type": "chronological_real",
    }

    all_manifests = []
    all_test_auc_pr = []

    for seed in seeds:
        log.info(f"\n{'='*60}")
        log.info(f"Training seed {seed}")
        log.info(f"{'='*60}")
        t0 = time.time()
        result = train_one_seed(
            graph=graph,
            train_ids=train_ids,
            valid_ids=valid_ids,
            test_ids=test_ids,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        log.info(f"  Seed {seed} training time: {elapsed:.1f}s")

        manifest = save_checkpoint(seed, result, model_cfg, graph_sha256)
        all_manifests.append(manifest)
        all_test_auc_pr.append(result["test_metrics"]["auc_pr"])

        # Append to training log
        log_entry = {
            "seed": seed,
            "elapsed_s": round(elapsed, 2),
            **manifest,
        }
        with open(TRAIN_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # Summary
    mean_auc_pr = float(np.mean(all_test_auc_pr))
    std_auc_pr = float(np.std(all_test_auc_pr))
    log.info(f"\n{'='*60}")
    log.info(f"Multi-seed Summary ({len(seeds)} seeds)")
    log.info(f"  Test AUC-PR: {mean_auc_pr:.4f} ± {std_auc_pr:.4f}")
    log.info(f"  Per-seed: {[round(v,4) for v in all_test_auc_pr]}")
    log.info(f"{'='*60}")

    # Write summary manifest
    summary = {
        "gnn_source": "real_checkpoint",
        "split_type": "chronological_real",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "dataset_sha256": graph_sha256,
        "seeds": seeds,
        "seed_count": len(seeds),
        "mean_test_auc_pr": mean_auc_pr,
        "std_test_auc_pr": std_auc_pr,
        "all_test_auc_pr": all_test_auc_pr,
        "model_config": model_cfg,
        "per_seed_manifests": all_manifests,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    summary_path = MANIFEST_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
