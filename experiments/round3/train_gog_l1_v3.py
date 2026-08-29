"""
Phase D v3: Train with structural + category features only.

Since dims 0,1,2,4,5,6,7 are all zero, we build truly informative features:
1. dim3 (only real feature) - normalized
2. Node degree features (in, out, total, log-scaled)
3. Local structure: ratio of fraud-risk neighbors vs total (computed from train labels only)
4. Graph structural features: clustering coefficient approximation, hub/authority scores
5. One-hot of dim3 value (10 categories)

Usage:
    python experiments/round3/train_gog_l1_v3.py [--seeds 7,17,27,37,47]
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from torch_geometric.data import Data as PyGData

ROOT = Path(__file__).parent.parent.parent
if not (ROOT / "data").exists():
    ROOT = Path(os.environ.get("DLG_GNN_ROOT", "/mnt/d/_Work/goat_bank/dlg_gnn"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("train_gog_l1_v3")

DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"
CKPT_DIR = ROOT / "results" / "real_checkpoints"
MANIFEST_DIR = ROOT / "results" / "checkpoint_manifests"
TRAIN_LOG = ROOT / "results" / "real_train_log.jsonl"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering v3
# ─────────────────────────────────────────────────────────────────────────────

def build_features_v3(graph: dict, train_ids: list, train_labels: torch.Tensor) -> torch.Tensor:
    """
    Build structural features:
    - dim3 (normalized): transaction category count
    - one-hot of dim3 category (10 unique values → 10 dims)
    - in-degree, out-degree, total-degree (log-scaled)
    - 2-hop neighbor fraud density (computed from train set only → no leakage)
    - Hub score (out-degree normalized)
    - Authority score (in-degree normalized)
    - Self-loop indicator
    - Degree ratio (out/total)
    Total: 1 + 10 + 3 + 1 + 2 + 1 + 1 = 19 dims
    """
    emb = graph["embeddings"].float()  # [N, 8]
    ei = graph["edge_index"]           # [2, E]
    N = emb.shape[0]
    dim3 = emb[:, 3]  # The ONLY informative feature

    # Compute degrees
    in_deg = torch.zeros(N)
    out_deg = torch.zeros(N)
    in_deg.index_add_(0, ei[1], torch.ones(ei.shape[1]))
    out_deg.index_add_(0, ei[0], torch.ones(ei.shape[1]))
    total_deg = in_deg + out_deg

    log_in = torch.log1p(in_deg)
    log_out = torch.log1p(out_deg)
    log_total = torch.log1p(total_deg)
    deg_ratio = out_deg / total_deg.clamp(min=1)

    # One-hot for dim3 categories
    unique_vals = dim3.unique()
    cat_map = {float(v): i for i, v in enumerate(sorted(unique_vals.tolist()))}
    n_cats = len(cat_map)
    cat_onehot = torch.zeros(N, n_cats)
    for i in range(N):
        v = float(dim3[i].item())
        cat_onehot[i, cat_map[v]] = 1.0

    # Normalized dim3
    dim3_max = max(float(dim3.max()), 1.0)
    dim3_norm = (dim3 / dim3_max).unsqueeze(1)

    # Train-only 1-hop fraud density (no leakage: only use train labels)
    train_set = set(train_ids)
    train_label_map = {nid: int(train_labels[i].item()) for i, nid in enumerate(train_ids)}

    fraud_density = torch.zeros(N)
    for node_i in range(N):
        # Get neighbors from edge_index
        src_mask = (ei[0] == node_i)
        dst_mask = (ei[1] == node_i)
        neighbors = set(ei[1][src_mask].tolist()) | set(ei[0][dst_mask].tolist())
        train_neighbors = neighbors & train_set
        if train_neighbors:
            fd = sum(train_label_map.get(nid, 0) for nid in train_neighbors) / len(train_neighbors)
            fraud_density[node_i] = fd

    # Hub/authority (normalized by max)
    hub = log_out / log_out.max().clamp(min=1)
    auth = log_in / log_in.max().clamp(min=1)

    # Concatenate all features
    x = torch.cat([
        dim3_norm,                           # 1
        cat_onehot,                          # 10
        log_in.unsqueeze(1),                 # 1
        log_out.unsqueeze(1),                # 1
        log_total.unsqueeze(1),              # 1
        fraud_density.unsqueeze(1),          # 1 (train-only, no leakage)
        hub.unsqueeze(1),                    # 1
        auth.unsqueeze(1),                   # 1
        deg_ratio.unsqueeze(1),              # 1
    ], dim=1)

    log.info(f"  Feature dim v3: {x.shape[1]} | "
             f"dim3 unique={n_cats} | "
             f"fraud_density nonzero={int((fraud_density > 0).sum())}")
    return x


def build_pyg_data(graph, ids, all_features, device):
    all_y = graph["labels"]
    ei = graph["edge_index"]
    ids_arr = np.array(ids)
    src_np = ei[0].numpy()
    dst_np = ei[1].numpy()
    mask = np.isin(src_np, ids_arr) & np.isin(dst_np, ids_arr)
    fsrc = ei[0][torch.from_numpy(mask)]
    fdst = ei[1][torch.from_numpy(mask)]
    id_to_new = {old: new for new, old in enumerate(ids)}
    valid = [(int(s), int(d)) for s, d in zip(fsrc.tolist(), fdst.tolist())
             if int(s) in id_to_new and int(d) in id_to_new]
    if valid:
        new_src = torch.tensor([id_to_new[s] for s, _ in valid], dtype=torch.long)
        new_dst = torch.tensor([id_to_new[d] for _, d in valid], dtype=torch.long)
        new_ei = torch.stack([new_src, new_dst])
    else:
        new_ei = torch.zeros(2, 0, dtype=torch.long)
    return PyGData(x=all_features[ids], edge_index=new_ei, y=all_y[ids]).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Model (MLP + optional GNN layer)
# ─────────────────────────────────────────────────────────────────────────────

class FraudMLP(nn.Module):
    """
    Pure MLP baseline (structural features only).
    Graph topology helps aggregate neighbors but features drive classification.
    """
    def __init__(self, in_dim=19, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.dropout_p = dropout

    def forward(self, data):
        return self.net(data.x).squeeze(-1)

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
        variance = probs.var(0)
        eps = 1e-8
        entropy = -(mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))
        return mean_p, variance, entropy


class GINLayerSimple(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x)
        if edge_index.shape[1] > 0:
            agg.index_add_(0, edge_index[0], x[edge_index[1]])
        return self.net((1 + self.eps) * x + agg) + self.res(x)


class GNNWithMLP(nn.Module):
    """1-layer GNN + deep MLP classifier with JK connection."""
    def __init__(self, in_dim=19, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.gnn1 = GINLayerSimple(in_dim, hidden_dim, dropout)
        self.gnn2 = GINLayerSimple(hidden_dim, hidden_dim, dropout)
        # JK: concat input + gnn1 + gnn2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
        self.dropout_p = dropout

    def forward(self, data):
        h0 = data.x
        h1 = self.gnn1(h0, data.edge_index)
        h2 = self.gnn2(h1, data.edge_index)
        h_jk = torch.cat([h0, h1, h2], dim=-1)
        return self.classifier(h_jk).squeeze(-1)

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
        variance = probs.var(0)
        eps = 1e-8
        entropy = -(mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))
        return mean_p, variance, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, alpha=0.9, gamma=3.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    weight = torch.where(targets == 1, torch.full_like(logits, alpha), torch.full_like(logits, 1 - alpha))
    return (weight * (1 - pt) ** gamma * bce).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_best_thresh(probs_np, y_np):
    """Find best threshold on same data (for reporting, not selection)."""
    try:
        auc_pr = float(average_precision_score(y_np, probs_np))
        auc_roc = float(roc_auc_score(y_np, probs_np))
    except Exception:
        auc_pr = auc_roc = 0.0
    # Best F1 threshold from PR curve
    try:
        prec, rec, threshs = precision_recall_curve(y_np, probs_np)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_thresh = threshs[np.argmax(f1_scores[:-1])] if len(threshs) > 0 else 0.5
        preds = (probs_np >= best_thresh).astype(int)
        f1 = float(f1_score(y_np, preds, zero_division=0))
    except Exception:
        f1 = 0.0
        best_thresh = 0.5
    return {"auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1, "best_thresh": best_thresh}


def train_one_seed(graph, all_features, train_ids, valid_ids, test_ids,
                   seed, model_type="GNNWithMLP", epochs=200, patience=30,
                   device=torch.device("cpu")):
    set_seed(seed)
    in_dim = all_features.shape[1]

    train_data = build_pyg_data(graph, train_ids, all_features, device)
    valid_data = build_pyg_data(graph, valid_ids, all_features, device)
    test_data = build_pyg_data(graph, test_ids, all_features, device)

    n_pos = int(train_data.y.sum())
    n_neg = len(train_ids) - n_pos
    log.info(f"  [Seed {seed}] train: fraud={n_pos}/{len(train_ids)}, "
             f"valid fraud={int(valid_data.y.sum())}, test fraud={int(test_data.y.sum())}")

    if model_type == "FraudMLP":
        model = FraudMLP(in_dim=in_dim, hidden_dim=256, dropout=0.3).to(device)
    else:
        model = GNNWithMLP(in_dim=in_dim, hidden_dim=256, dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_pr = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = focal_loss(logits, train_data.y.float(), alpha=0.92, gamma=3.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step(epoch)

        model.eval()
        with torch.no_grad():
            val_logits = model(valid_data)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        y_val = valid_data.y.cpu().numpy().astype(int)
        try:
            val_pr = float(average_precision_score(y_val, val_probs))
        except Exception:
            val_pr = 0.0

        if epoch % 50 == 0 or epoch == 1:
            log.info(f"  [Seed {seed}] Ep{epoch:4d}: loss={float(loss):.5f} "
                     f"val_pr={val_pr:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  [Seed {seed}] Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    log.info(f"  [Seed {seed}] Best val AUC-PR={best_val_pr:.4f} at epoch {best_epoch}")

    model.eval()
    with torch.no_grad():
        test_logits = model(test_data)
    test_probs = torch.sigmoid(test_logits).cpu().numpy()
    y_test = test_data.y.cpu().numpy().astype(int)
    test_m = compute_metrics_best_thresh(test_probs, y_test)

    mean_p, var, ent = model.forward_mc(test_data, T=10)
    mc_probs = mean_p.cpu().numpy()
    try:
        mc_pr = float(average_precision_score(y_test, mc_probs))
        mc_roc = float(roc_auc_score(y_test, mc_probs))
    except Exception:
        mc_pr = mc_roc = 0.0

    log.info(f"  [Seed {seed}] TEST: pr={test_m['auc_pr']:.4f} roc={test_m['auc_roc']:.4f} f1={test_m['f1']:.4f}")
    log.info(f"  [Seed {seed}] TEST MC(T=10): pr={mc_pr:.4f} roc={mc_roc:.4f}")

    return {
        "seed": seed, "best_epoch": best_epoch, "best_val_auc_pr": best_val_pr,
        "test_metrics": test_m, "test_mc_auc_pr": mc_pr, "test_mc_auc_roc": mc_roc,
        "model_state": best_state, "model": model, "in_dim": in_dim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT))
        return r.stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(seed, result, model_cfg, graph_sha256, prefix="l1v3"):
    ckpt_path = CKPT_DIR / f"{prefix}_seed{seed}_best.pt"
    cfg_sha = hashlib.sha256(json.dumps(model_cfg, sort_keys=True).encode()).hexdigest()
    torch.save({
        "model_state_dict": result["model_state"],
        "model_config": model_cfg,
        "model_class": model_cfg.get("model_class", "GNNWithMLP"),
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "best_val_auc_pr": result["best_val_auc_pr"],
        "test_metrics": result["test_metrics"],
        "test_mc_auc_pr": result["test_mc_auc_pr"],
        "gnn_source": "real_checkpoint",
        "split_type": "chronological_real",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "dataset_sha256": graph_sha256,
        "config_sha256": cfg_sha,
        "git_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    manifest = {
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_sha256": ckpt_sha,
        "model_class": model_cfg.get("model_class"),
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "best_val_auc_pr": result["best_val_auc_pr"],
        "test_auc_pr": result["test_metrics"]["auc_pr"],
        "test_auc_roc": result["test_metrics"]["auc_roc"],
        "test_f1": result["test_metrics"]["f1"],
        "test_mc_auc_pr": result["test_mc_auc_pr"],
        "gnn_source": "real_checkpoint",
        "split_type": "chronological_real",
        "dataset_sha256": graph_sha256,
        "config_sha256": cfg_sha,
        "git_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (MANIFEST_DIR / f"{prefix}_seed{seed}.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"  Saved: {ckpt_path.relative_to(ROOT)}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="7,17,27,37,47")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--model", type=str, default="GNNWithMLP",
                        choices=["GNNWithMLP", "FraudMLP"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    log.info(f"Device: {device}, Model: {args.model}")

    graph_path = DATA_DIR / "polygon_hybrid_graph.pt"
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_sha256 = sha256_file(graph_path)

    train_ids = [int(x.strip()) for x in open(DATA_DIR / "train_ids.txt") if x.strip()]
    valid_ids = [int(x.strip()) for x in open(DATA_DIR / "valid_ids.txt") if x.strip()]
    test_ids = [int(x.strip()) for x in open(DATA_DIR / "test_ids.txt") if x.strip()]
    train_labels = graph["labels"][train_ids]

    log.info("Building v3 features (structural + dim3 category)...")
    all_features = build_features_v3(graph, train_ids, train_labels)
    in_dim = all_features.shape[1]

    model_cfg = {
        "in_dim": in_dim,
        "hidden_dim": 256,
        "dropout": 0.3,
        "model_class": args.model,
        "feature_engineering": "structural_v3_dim3_onehot_fraud_density",
        "loss": "focal_loss_alpha0.92_gamma3",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "split_type": "chronological_real",
    }

    all_pr = []
    all_manifests = []

    for seed in seeds:
        log.info(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        t0 = time.time()
        result = train_one_seed(
            graph=graph, all_features=all_features,
            train_ids=train_ids, valid_ids=valid_ids, test_ids=test_ids,
            seed=seed, model_type=args.model,
            epochs=args.epochs, patience=args.patience, device=device,
        )
        elapsed = time.time() - t0
        log.info(f"  Elapsed: {elapsed:.1f}s")

        manifest = save_checkpoint(seed, result, model_cfg, graph_sha256, prefix="l1v3")
        all_manifests.append(manifest)
        all_pr.append(result["test_metrics"]["auc_pr"])
        with open(TRAIN_LOG, "a") as f:
            f.write(json.dumps({"v": 3, "seed": seed, **manifest}) + "\n")

    mean_pr = float(np.mean(all_pr))
    std_pr = float(np.std(all_pr))
    log.info(f"\n{'='*60}")
    log.info(f"v3 {args.model} Multi-seed Summary ({len(seeds)} seeds)")
    log.info(f"  Test AUC-PR: {mean_pr:.4f} ± {std_pr:.4f}")
    log.info(f"  Per-seed: {[round(v,4) for v in all_pr]}")
    log.info(f"{'='*60}")

    summary = {
        "gnn_source": "real_checkpoint",
        "model_class": args.model,
        "feature_version": "v3",
        "split_type": "chronological_real",
        "dataset": "GoG-MicroRAG-Stream-v1",
        "dataset_sha256": graph_sha256,
        "seeds": seeds,
        "seed_count": len(seeds),
        "mean_test_auc_pr": mean_pr,
        "std_test_auc_pr": std_pr,
        "all_test_auc_pr": all_pr,
        "model_config": model_cfg,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (MANIFEST_DIR / "training_summary_v3.json").write_text(json.dumps(summary, indent=2))
    log.info("Done.")


if __name__ == "__main__":
    main()
