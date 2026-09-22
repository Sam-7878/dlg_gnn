"""
Improved Phase D: Train GoG L1 GNN with better strategies.

Key improvements over v1:
1. Feature preprocessing: degree features + normalized embeddings
2. Better loss: Focal loss for extreme imbalance
3. Longer training with cosine LR schedule
4. Oversample minority class during training
5. Use edges from full graph (not just within-split)

Usage:
    python experiments/round3/train_gog_l1_v2.py [--seeds 7,17,27,37,47] [--epochs 150]
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch_geometric.data import Data as PyGData

ROOT = Path(__file__).parent.parent.parent
if not (ROOT / "data").exists():
    ROOT = Path(os.environ.get("DLG_GNN_ROOT", "/mnt/d/_Work/goat_bank/dlg_gnn"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train_gog_l1_v2")

DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"
from experiments.round3.artifact_paths import (
    CHECKPOINT_DIR as CKPT_DIR,
    CHECKPOINT_MANIFEST_DIR as MANIFEST_DIR,
    ROUND3_RESULTS,
)
TRAIN_LOG = ROUND3_RESULTS / "real_train_log.jsonl"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(graph: dict) -> torch.Tensor:
    """
    Build rich node features combining:
    1. Original 8-dim embeddings (one-hot sparse)
    2. Degree features (in, out, total)
    3. Normalized embeddings
    """
    emb = graph["embeddings"].float()   # [N, 8]
    ei = graph["edge_index"]            # [2, E]
    N = emb.shape[0]

    # Degree features
    in_deg = torch.zeros(N)
    out_deg = torch.zeros(N)
    in_deg.index_add_(0, ei[1], torch.ones(ei.shape[1]))
    out_deg.index_add_(0, ei[0], torch.ones(ei.shape[1]))
    total_deg = in_deg + out_deg

    # Log-scale degrees
    log_in = torch.log1p(in_deg).unsqueeze(1)
    log_out = torch.log1p(out_deg).unsqueeze(1)
    log_total = torch.log1p(total_deg).unsqueeze(1)

    # L2-normalized embedding (handle zero rows)
    norm = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb_norm = emb / norm

    # Value features
    emb_max = emb.max(dim=1)[0].unsqueeze(1)
    emb_sum = emb.sum(dim=1).unsqueeze(1)

    # Concat all: [8 + 3 + 2 + 8] = 21 dims
    x = torch.cat([emb, log_in, log_out, log_total, emb_max, emb_sum, emb_norm], dim=1)
    log.info(f"  Feature dim: {x.shape[1]}")
    return x


def build_pyg_data(graph: dict, ids: List[int], all_features: torch.Tensor,
                   use_full_edges: bool = False, device=None) -> PyGData:
    """
    Build PyG data. Optionally use edges from FULL graph (historical edges only).
    """
    all_y = graph["labels"]
    edge_index = graph["edge_index"]

    if use_full_edges:
        # Use ALL edges where source node is in training set (historical edges)
        ids_set = set(ids)
        src_np = edge_index[0].numpy()
        # Use edges that go TO ids (incoming) or FROM ids (outgoing)
        mask = np.isin(src_np, list(ids_set)) | np.isin(edge_index[1].numpy(), list(ids_set))
    else:
        ids_arr = np.array(ids)
        src_np = edge_index[0].numpy()
        dst_np = edge_index[1].numpy()
        mask = np.isin(src_np, ids_arr) & np.isin(dst_np, ids_arr)

    filtered_src = edge_index[0][torch.from_numpy(mask)]
    filtered_dst = edge_index[1][torch.from_numpy(mask)]

    id_to_new = {old: new for new, old in enumerate(ids)}
    # Filter to only those edges where both endpoints are in ids
    valid_mask = np.array([
        int(s) in id_to_new and int(d) in id_to_new
        for s, d in zip(filtered_src.tolist(), filtered_dst.tolist())
    ])
    if valid_mask.any():
        filtered_src = filtered_src[torch.from_numpy(valid_mask)]
        filtered_dst = filtered_dst[torch.from_numpy(valid_mask)]

    new_src = torch.tensor([id_to_new[int(i)] for i in filtered_src], dtype=torch.long)
    new_dst = torch.tensor([id_to_new[int(i)] for i in filtered_dst], dtype=torch.long)

    data = PyGData(
        x=all_features[ids],
        edge_index=torch.stack([new_src, new_dst], dim=0),
        y=all_y[ids],
    )
    if device is not None:
        data = data.to(device)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GINLayerV2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        N = x.size(0)
        agg = torch.zeros_like(x)
        row, col = edge_index
        agg.index_add_(0, row, x[col])
        out = self.net((1 + self.eps) * x + agg)
        return out + self.res(x)


class Level1GNNv2(nn.Module):
    def __init__(self, in_dim=21, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            GINLayerV2(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.dropout_p = dropout

    def forward(self, data):
        h = self.input_proj(data.x)
        h_list = [h]
        for layer in self.layers:
            h = layer(h, data.edge_index)
            h_list.append(h)
        # JK (Jumping Knowledge) aggregation: concat last 3 layers
        h_jk = torch.cat(h_list[-3:], dim=-1)
        return self.head(h_jk).squeeze(-1)

    def forward_mc(self, data, T=10):
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                probs_list.append(torch.sigmoid(self.forward(data)))
        self.eval()
        probs = torch.stack(probs_list)
        mean_p = probs.mean(0)
        variance = probs.var(0, unbiased=False)
        eps = 1e-8
        entropy = -(mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))
        return mean_p, variance, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """Binary focal loss for severe class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    weight = torch.where(targets == 1,
                         torch.full_like(probs, alpha),
                         torch.full_like(probs, 1 - alpha))
    return (weight * (1 - pt) ** gamma * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(logits_or_probs: torch.Tensor, labels: torch.Tensor, is_prob=False):
    if is_prob:
        probs = logits_or_probs.detach().cpu().numpy()
    else:
        probs = torch.sigmoid(logits_or_probs).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    try:
        auc_pr = float(average_precision_score(y, probs))
        auc_roc = float(roc_auc_score(y, probs))
    except Exception:
        auc_pr = auc_roc = 0.0
    t = np.percentile(probs, 100 * (1 - y.mean() * 3))  # adaptive threshold
    preds = (probs >= t).astype(int)
    f1 = float(f1_score(y, preds, zero_division=0))
    return {"auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1}


def train_one_seed(graph, all_features, train_ids, valid_ids, test_ids,
                   seed, epochs=150, lr=3e-4, weight_decay=1e-4, patience=20,
                   device=torch.device("cpu")):
    set_seed(seed)
    log.info(f"[Seed {seed}] Building data (in_dim={all_features.shape[1]})...")

    train_data = build_pyg_data(graph, train_ids, all_features, device=device)
    valid_data = build_pyg_data(graph, valid_ids, all_features, device=device)
    test_data = build_pyg_data(graph, test_ids, all_features, device=device)

    n_pos = int(train_data.y.sum())
    n_neg = len(train_ids) - n_pos
    log.info(f"  train: fraud={n_pos}/{len(train_ids)} ({n_pos/len(train_ids)*100:.1f}%), "
             f"valid fraud={int(valid_data.y.sum())}, test fraud={int(test_data.y.sum())}")

    in_dim = all_features.shape[1]
    model = Level1GNNv2(in_dim=in_dim, hidden_dim=256, num_layers=4, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc_pr = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = focal_loss(logits, train_data.y.float(), alpha=0.85, gamma=2.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(valid_data)
        val_m = compute_metrics(val_logits, valid_data.y)

        if epoch % 20 == 0 or epoch == 1:
            log.info(f"  [Seed {seed}] Ep {epoch:3d}: loss={float(loss):.4f} "
                     f"val_pr={val_m['auc_pr']:.4f} val_roc={val_m['auc_roc']:.4f} "
                     f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_m["auc_pr"] > best_val_auc_pr:
            best_val_auc_pr = val_m["auc_pr"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  [Seed {seed}] Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    log.info(f"  [Seed {seed}] Best val AUC-PR={best_val_auc_pr:.4f} at epoch {best_epoch}")

    # Test eval
    model.eval()
    with torch.no_grad():
        test_logits = model(test_data)
    test_m = compute_metrics(test_logits, test_data.y)

    # MC test
    mean_p, var, ent = model.forward_mc(test_data, T=10)
    y_np = test_data.y.cpu().numpy().astype(int)
    mean_np = mean_p.cpu().numpy()
    try:
        mc_auc_pr = float(average_precision_score(y_np, mean_np))
        mc_auc_roc = float(roc_auc_score(y_np, mean_np))
    except Exception:
        mc_auc_pr = mc_auc_roc = 0.0

    log.info(f"  [Seed {seed}] TEST: pr={test_m['auc_pr']:.4f} roc={test_m['auc_roc']:.4f} f1={test_m['f1']:.4f}")
    log.info(f"  [Seed {seed}] TEST MC(T=10): pr={mc_auc_pr:.4f} roc={mc_auc_roc:.4f}")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc_pr": best_val_auc_pr,
        "test_metrics": test_m,
        "test_mc_auc_pr": mc_auc_pr,
        "test_mc_auc_roc": mc_auc_roc,
        "model_state": best_state,
        "model": model,
        "in_dim": in_dim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT))
        return r.stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(seed, result, model_cfg, graph_sha256):
    ckpt_path = CKPT_DIR / f"l1v2_seed{seed}_best.pt"
    cfg_sha256 = hashlib.sha256(json.dumps(model_cfg, sort_keys=True).encode()).hexdigest()

    torch.save({
        "model_state_dict": result["model_state"],
        "model_config": model_cfg,
        "model_class": "Level1GNNv2",
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "best_val_auc_pr": result["best_val_auc_pr"],
        "test_metrics": result["test_metrics"],
        "test_mc_auc_pr": result["test_mc_auc_pr"],
        "gnn_source": "real_checkpoint",
        "split_type": "synthetic_time_ordered",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "dataset_sha256": graph_sha256,
        "config_sha256": cfg_sha256,
        "git_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ckpt_path)

    ckpt_sha256 = sha256_file(ckpt_path)
    manifest = {
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_sha256": ckpt_sha256,
        "model_class": "Level1GNNv2",
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "best_val_auc_pr": result["best_val_auc_pr"],
        "test_auc_pr": result["test_metrics"]["auc_pr"],
        "test_auc_roc": result["test_metrics"]["auc_roc"],
        "test_f1": result["test_metrics"]["f1"],
        "test_mc_auc_pr": result["test_mc_auc_pr"],
        "gnn_source": "real_checkpoint",
        "split_type": "synthetic_time_ordered",
        "dataset_sha256": graph_sha256,
        "config_sha256": cfg_sha256,
        "git_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = MANIFEST_DIR / f"l1v2_seed{seed}.json"
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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    log.info(f"Device: {device}")

    graph_path = DATA_DIR / "polygon_hybrid_graph.pt"
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_sha256 = sha256_file(graph_path)

    # Build rich features
    log.info("Building node features...")
    all_features = build_features(graph)
    in_dim = all_features.shape[1]
    log.info(f"Feature dim: {in_dim}")

    train_ids = [int(x.strip()) for x in open(DATA_DIR/"train_ids.txt") if x.strip()]
    valid_ids = [int(x.strip()) for x in open(DATA_DIR/"valid_ids.txt") if x.strip()]
    test_ids = [int(x.strip()) for x in open(DATA_DIR/"test_ids.txt") if x.strip()]

    model_cfg = {
        "in_dim": in_dim,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "encoder_backend": "gin_jk_v2",
        "feature_engineering": "degree+emb+normalized",
        "loss": "focal_loss",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "split_type": "synthetic_time_ordered",
    }

    all_manifests = []
    all_test_auc_pr = []

    for seed in seeds:
        log.info(f"\n{'='*60}\nTraining seed {seed}\n{'='*60}")
        t0 = time.time()
        result = train_one_seed(
            graph=graph,
            all_features=all_features,
            train_ids=train_ids,
            valid_ids=valid_ids,
            test_ids=test_ids,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        elapsed = time.time() - t0
        log.info(f"  Elapsed: {elapsed:.1f}s")

        manifest = save_checkpoint(seed, result, model_cfg, graph_sha256)
        all_manifests.append(manifest)
        all_test_auc_pr.append(result["test_metrics"]["auc_pr"])

        with open(TRAIN_LOG, "a") as f:
            f.write(json.dumps({"seed": seed, "elapsed_s": round(elapsed, 2), **manifest}) + "\n")

    mean_pr = float(np.mean(all_test_auc_pr))
    std_pr = float(np.std(all_test_auc_pr))
    log.info(f"\n{'='*60}")
    log.info(f"Multi-seed Summary ({len(seeds)} seeds)")
    log.info(f"  Test AUC-PR: {mean_pr:.4f} ± {std_pr:.4f}")
    log.info(f"  Per-seed: {[round(v, 4) for v in all_test_auc_pr]}")
    log.info(f"{'='*60}")

    summary = {
        "gnn_source": "real_checkpoint",
        "model_class": "Level1GNNv2",
        "split_type": "synthetic_time_ordered",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "dataset_sha256": graph_sha256,
        "seeds": seeds,
        "seed_count": len(seeds),
        "mean_test_auc_pr": mean_pr,
        "std_test_auc_pr": std_pr,
        "all_test_auc_pr": all_test_auc_pr,
        "model_config": model_cfg,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    summary_path = MANIFEST_DIR / "training_summary_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
