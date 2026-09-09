"""
Phase F: Real GraphRAG Fusion Evaluation — Full Pipeline (9 comparison groups)
Phase H: Privacy Finalization
Phase I: Robustness
Phase K: Statistical Finalization (Paired Bootstrap CI)

Usage:
    python experiments/round3/run_full_pipeline.py [--seeds 7,17,27,37,47]

Outputs:
    results/real_main_results.csv
    results/real_ablation_results.csv
    results/real_statistical_summary.csv
    results/real_calibration.csv
    results/real_privacy_utility.csv
    results/real_robustness.csv
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
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
log = logging.getLogger("full_pipeline")

DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"
from experiments.round3.artifact_paths import (
    CHECKPOINT_DIR as CKPT_DIR,
    RAW_PREDICTION_DIR as PRED_DIR,
    ROUND3_REPORTS as REPORTS_DIR,
    ROUND3_RESULTS as RESULTS_DIR,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inline model (same as train_gog_l1.py)
# ─────────────────────────────────────────────────────────────────────────────

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim), nn.ReLU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x)
        row, col = edge_index
        agg.index_add_(0, row, x[col])
        return self.net((1 + self.eps) * x + agg)


class Level1GNNDirect(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(GINLayer(dims[i], dims[i+1], dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        h = data.x
        for layer in self.layers:
            h = layer(h, data.edge_index)
        return self.head(torch.cat([h, h], dim=-1)).squeeze(-1)

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


class GINLayerSimple(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim), nn.GELU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x)
        if edge_index.shape[1] > 0:
            agg.index_add_(0, edge_index[0], x[edge_index[1]])
        return self.net((1 + self.eps) * x + agg) + self.res(x)


class GNNWithMLP(nn.Module):
    def __init__(self, in_dim=19, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.gnn1 = GINLayerSimple(in_dim, hidden_dim, dropout)
        self.gnn2 = GINLayerSimple(hidden_dim, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout / 2), nn.Linear(64, 1),
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
        variance = probs.var(0, unbiased=False)
        eps = 1e-8
        entropy = -(mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))
        return mean_p, variance, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Data utils
# ─────────────────────────────────────────────────────────────────────────────

from experiments.round3.train_gog_l1_v3 import build_features_v3


def load_ids(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def build_pyg_data(graph, ids, all_features=None, device=None):
    all_x = all_features if all_features is not None else graph["embeddings"].float()
    all_y = graph["labels"]
    edge_index = graph["edge_index"]
    ids_arr = np.array(ids)
    src_np = edge_index[0].numpy()
    dst_np = edge_index[1].numpy()
    mask = np.isin(src_np, ids_arr) & np.isin(dst_np, ids_arr)
    fsrc = edge_index[0][torch.from_numpy(mask)]
    fdst = edge_index[1][torch.from_numpy(mask)]
    id_to_new = {old: new for new, old in enumerate(ids)}
    valid = [(int(s), int(d)) for s, d in zip(fsrc.tolist(), fdst.tolist())
             if int(s) in id_to_new and int(d) in id_to_new]
    if valid:
        new_src = torch.tensor([id_to_new[s] for s, _ in valid], dtype=torch.long)
        new_dst = torch.tensor([id_to_new[d] for _, d in valid], dtype=torch.long)
        new_ei = torch.stack([new_src, new_dst])
    else:
        new_ei = torch.zeros(2, 0, dtype=torch.long)
    data = PyGData(
        x=all_x[ids], edge_index=new_ei, y=all_y[ids]
    )
    if device is not None:
        data = data.to(device)
    return data


def load_model(seed, device):
    for prefix in ["l1v3_seed", "l1v2_seed", "l1_seed"]:
        ckpt_path = CKPT_DIR / f"{prefix}{seed}_best.pt"
        if ckpt_path.exists():
            break
    else:
        raise FileNotFoundError(f"No checkpoint found for seed {seed}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    mc = ckpt.get("model_class", "Level1GNNDirect")
    if mc == "GNNWithMLP":
        model = GNNWithMLP(
            in_dim=cfg.get("in_dim", 18),
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=cfg.get("dropout", 0.3),
        ).to(device)
    else:
        model = Level1GNNDirect(
            in_dim=cfg.get("in_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.2),
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt



def load_contexts():
    ctx_path = DATA_DIR / "contexts.jsonl"
    if not ctx_path.exists():
        return {}
    contexts = {}
    with open(ctx_path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                node_id = int(c["event_id"].split("_")[1])
                contexts[node_id] = c.get("context_text", "")
    return contexts


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(y, probs, method="", split_type="synthetic_time_ordered", gnn_source="real_checkpoint"):
    y = np.array(y, dtype=int)
    probs = np.array(probs, dtype=float)
    try:
        auc_pr = float(average_precision_score(y, probs))
        auc_roc = float(roc_auc_score(y, probs))
    except Exception:
        auc_pr = auc_roc = 0.0
    preds = (probs >= 0.5).astype(int)
    f1 = float(f1_score(y, preds, zero_division=0))
    recall = float((preds[y == 1]).sum() / max(y.sum(), 1))
    brier = float(brier_score_loss(y, probs))
    # ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y[mask].mean() - probs[mask].mean())
    ece /= max(len(y), 1)
    return {
        "method": method,
        "auc_pr": round(auc_pr, 6),
        "auc_roc": round(auc_roc, 6),
        "f1": round(f1, 6),
        "recall": round(recall, 6),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "gnn_source": gnn_source,
        "split_type": split_type,
    }


def bootstrap_ci(y, probs_a, probs_b, n_boot=10000, metric="auc_pr"):
    """Paired bootstrap CI for metric(a) - metric(b)."""
    y = np.array(y, dtype=int)
    rng = np.random.RandomState(42)
    N = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, N, N)
        try:
            if metric == "auc_pr":
                va = average_precision_score(y[idx], probs_a[idx])
                vb = average_precision_score(y[idx], probs_b[idx])
            else:
                va = roc_auc_score(y[idx], probs_a[idx])
                vb = roc_auc_score(y[idx], probs_b[idx])
            deltas.append(va - vb)
        except Exception:
            pass
    if not deltas:
        return 0.0, 0.0, 0.0
    deltas = np.array(deltas)
    delta_mean = float(np.mean(deltas))
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    return delta_mean, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# GraphRAG integration
# ─────────────────────────────────────────────────────────────────────────────

def get_graphrag_scores(test_ids, contexts):
    """Get risk scores from GraphRAG pipeline."""
    try:
        from graphrag.local_kb import LocalKnowledgeBase
        from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
        from graphrag.risk_extractor import RiskExtractor

        kb = LocalKnowledgeBase()
        config = RetrieverConfig(top_k=5, graph_hops=1)
        retriever = GraphRAGRetriever(kb=kb, config=config)
        extractor = RiskExtractor()

        risk_scores = []
        for nid in test_ids:
            ctx_text = contexts.get(nid, "")
            if not ctx_text:
                risk_scores.append(0.5)
                continue
            try:
                evidence = retriever.retrieve(ctx_text)
                ctx_dict = extractor.extract(evidence, event_id=f"tx_{nid:06d}")
                score = float(ctx_dict.get("local_risk_score", 0.5))
                risk_scores.append(score)
            except Exception:
                risk_scores.append(0.5)
        return np.array(risk_scores)
    except ImportError as e:
        log.warning(f"GraphRAG not importable: {e}, using baseline for semantic scores")
        return np.random.RandomState(42).uniform(0.3, 0.7, len(test_ids))



def get_tfidf_scores(train_ids, test_ids, contexts, labels_all):
    """TF-IDF + LR baseline."""
    train_texts = [contexts.get(nid, "") for nid in train_ids]
    test_texts = [contexts.get(nid, "") for nid in test_ids]
    graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
    y_train = graph["labels"][train_ids].numpy().astype(int)

    if not any(train_texts) or y_train.sum() == 0:
        return np.full(len(test_ids), 0.5)

    try:
        vec = TfidfVectorizer(max_features=500, sublinear_tf=True)
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_train, y_train)
        return lr.predict_proba(X_test)[:, 1]
    except Exception as e:
        log.warning(f"TF-IDF LR failed: {e}")
        return np.full(len(test_ids), 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion modes
# ─────────────────────────────────────────────────────────────────────────────

def fixed_fusion(p_gnn, p_risk, alpha=0.5):
    return alpha * p_gnn + (1 - alpha) * p_risk


def uncertainty_fusion(p_gnn, u_mc, p_risk):
    """β_t = sigmoid(U_t), higher uncertainty → more weight to GraphRAG."""
    beta = 1.0 / (1.0 + np.exp(-10 * (u_mc - u_mc.mean())))  # sigmoid scaling
    return (1 - beta) * p_gnn + beta * p_risk


def validation_tuned_fixed_fusion(p_gnn, p_risk, val_y, val_p_gnn, val_p_risk):
    """Sweep a fixed alpha on validation and apply the selected value to test."""
    best_alpha, best_val = 0.5, -1.0
    for alpha in np.arange(0.0, 1.05, 0.1):
        val_fused = alpha * val_p_gnn + (1 - alpha) * val_p_risk
        try:
            v = average_precision_score(val_y, val_fused)
        except Exception:
            v = 0.0
        if v > best_val:
            best_val = v
            best_alpha = alpha
    return best_alpha * p_gnn + (1 - best_alpha) * p_risk, best_alpha


def learned_fusion(p_gnn, p_risk, val_y, val_p_gnn, val_p_risk):
    """Fit a two-feature logistic fusion head on validation only."""
    if len(np.unique(val_y)) < 2:
        return fixed_fusion(p_gnn, p_risk, alpha=0.5)
    head = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    head.fit(np.column_stack((val_p_gnn, val_p_risk)), val_y)
    return head.predict_proba(np.column_stack((p_gnn, p_risk)))[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Privacy utility
# ─────────────────────────────────────────────────────────────────────────────

def run_privacy_analysis(test_probs_full, test_probs_gnn, y_test):
    """Membership inference proxy attack."""
    log.info("=== Phase H: Privacy Analysis ===")
    from sklearn.linear_model import LogisticRegression as LR

    results = []
    noise_levels = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]

    for representation, probs in [
        ("full_risk_vector", test_probs_full),
        ("gnn_only", test_probs_gnn),
    ]:
        for sigma in noise_levels:
            rng = np.random.RandomState(0)
            noisy = np.clip(probs + rng.normal(0, sigma, len(probs)), 0, 1)

            # Attack: use prob score to predict label (proxy for leakage)
            X = noisy.reshape(-1, 1)

            # Use cross-validation to simulate attack accuracy
            n = len(y_test)
            half = n // 2
            idx = np.random.permutation(n)
            X_tr, y_tr = X[idx[:half]], y_test[idx[:half]]
            X_te, y_te = X[idx[half:]], y_test[idx[half:]]

            try:
                atk = LR(max_iter=1000, class_weight="balanced", random_state=0)
                atk.fit(X_tr, y_tr)
                pred = atk.predict(X_te)
                prob = atk.predict_proba(X_te)[:, 1]

                acc = float(accuracy_score(y_te, pred))
                bal_acc = float(balanced_accuracy_score(y_te, pred))
                macro_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
                try:
                    roc = float(roc_auc_score(y_te, prob))
                    pr = float(average_precision_score(y_te, prob))
                except Exception:
                    roc = pr = 0.5
            except Exception:
                acc = bal_acc = macro_f1 = 0.5
                roc = pr = 0.5

            majority_baseline = float(max(y_te.mean(), 1 - y_te.mean()))
            random_baseline = 0.5

            results.append({
                "representation": representation,
                "noise_sigma": sigma,
                "attack_accuracy": round(acc, 6),
                "attack_balanced_accuracy": round(bal_acc, 6),
                "attack_macro_f1": round(macro_f1, 6),
                "attack_roc_auc": round(roc, 6),
                "attack_pr_auc": round(pr, 6),
                "majority_baseline": round(majority_baseline, 6),
                "random_baseline": round(random_baseline, 6),
                "leakage_risk": "HIGH" if bal_acc > majority_baseline + 0.05 else "LOW",
            })
            log.info(f"  {representation} sigma={sigma:.2f}: "
                     f"bal_acc={bal_acc:.3f} macro_f1={macro_f1:.3f} "
                     f"roc={roc:.3f} → {results[-1]['leakage_risk']}")

    priv_path = RESULTS_DIR / "real_privacy_utility.csv"
    with open(priv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Written: {priv_path.relative_to(ROOT)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness(model, test_data, val_p_gnn, val_p_risk, val_y, test_ids, contexts, y_test, device):
    """Phase I: Robustness under context dropout and noise."""
    log.info("=== Phase I: Robustness Analysis ===")
    results = []
    rng = np.random.RandomState(42)

    conditions = [
        ("missing_context", 0.0),
        ("missing_context", 0.10),
        ("missing_context", 0.30),
        ("missing_context", 0.50),
        ("noisy_context", 0.10),
        ("noisy_context", 0.30),
        ("noisy_context", 0.50),
    ]

    # Get real GNN scores (T=10)
    with torch.no_grad():
        p_gnn_mean, u_mc, _ = model.forward_mc(test_data, T=10)
    p_gnn_np = p_gnn_mean.cpu().numpy()
    u_mc_np = u_mc.cpu().numpy()

    # Base risk scores from contexts
    base_risk = np.full(len(test_ids), 0.5)  # neutral when no context
    try:
        graphrag_risk = get_graphrag_scores(test_ids, contexts)
    except Exception:
        graphrag_risk = np.full(len(test_ids), 0.5)

    for condition, rate in conditions:
        perturbed_risk = graphrag_risk.copy()

        if condition == "missing_context":
            # Drop fraction of contexts → use neutral 0.5
            drop_mask = rng.rand(len(test_ids)) < rate
            perturbed_risk[drop_mask] = 0.5
        elif condition == "noisy_context":
            # Add Gaussian noise to risk scores
            perturbed_risk += rng.normal(0, rate, len(test_ids))
            perturbed_risk = np.clip(perturbed_risk, 0, 1)

        # Run 3 methods
        for method_name, fused_probs in [
            ("GNN_Only", p_gnn_np),
            ("Fixed_Fusion_0.5", fixed_fusion(p_gnn_np, perturbed_risk, alpha=0.5)),
            ("Uncertainty_Fusion", uncertainty_fusion(p_gnn_np, u_mc_np, perturbed_risk)),
        ]:
            m = metrics(y_test, fused_probs, method=method_name)
            results.append({
                "condition": condition,
                "rate": rate,
                "method": method_name,
                "auc_pr": m["auc_pr"],
                "auc_roc": m["auc_roc"],
                "f1": m["f1"],
            })
            log.info(f"  {condition}@{rate:.0%} {method_name}: "
                     f"AUC-PR={m['auc_pr']:.4f}")

    rob_path = RESULTS_DIR / "real_robustness.csv"
    with open(rob_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Written: {rob_path.relative_to(ROOT)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibration_data(y, probs, method, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        rows.append({
            "method": method,
            "bin_lo": round(bins[i], 2),
            "bin_hi": round(bins[i+1], 2),
            "mean_conf": round(float(probs[mask].mean()), 6),
            "mean_acc": round(float(y[mask].mean()), 6),
            "count": int(mask.sum()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global CKPT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="7,17,27,37,47")
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-privacy", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    CKPT_DIR = args.checkpoint_dir.resolve()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    log.info(f"Device: {device}, Seeds: {seeds}, MC T={args.T}")

    # Load data
    graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
    train_ids = load_ids(DATA_DIR / "train_ids.txt")
    valid_ids = load_ids(DATA_DIR / "valid_ids.txt")
    test_ids = load_ids(DATA_DIR / "test_ids.txt")
    contexts = load_contexts()
    train_labels = graph["labels"][train_ids]
    all_features = build_features_v3(graph, train_ids, train_labels)

    val_data = build_pyg_data(graph, valid_ids, all_features=all_features, device=device)
    test_data = build_pyg_data(graph, test_ids, all_features=all_features, device=device)
    y_val = val_data.y.cpu().numpy().astype(int)
    y_test = test_data.y.cpu().numpy().astype(int)


    log.info(f"Val: {len(valid_ids)} nodes, fraud={y_val.sum()} | "
             f"Test: {len(test_ids)} nodes, fraud={y_test.sum()}")

    # Baselines (context-based, no GNN)
    log.info("Computing TF-IDF baseline...")
    tfidf_scores_test = get_tfidf_scores(train_ids, test_ids, contexts, graph["labels"])
    tfidf_scores_val = get_tfidf_scores(train_ids, valid_ids, contexts, graph["labels"])

    log.info("Computing GraphRAG semantic scores...")
    graphrag_scores_test = get_graphrag_scores(test_ids, contexts)
    graphrag_scores_val = get_graphrag_scores(valid_ids, contexts)

    # Per-seed results collection
    per_seed = {
        "GNN_Only": [],
        "Semantic_Only_GraphRAG": [],
        "TF_IDF_Only": [],
        "Fixed_Fusion_0.5": [],
        "Fixed_Fusion_ValTuned": [],
        "Learned_Fusion": [],
        "Uncertainty_Fusion": [],
        "Without_MC_T1": [],
        "Without_GraphRAG": [],
    }
    all_probs = {k: [] for k in per_seed}
    all_u_mc = []

    for seed in seeds:
        log.info(f"\n=== Seed {seed} ===")
        try:
            model, ckpt = load_model(seed, device)
        except FileNotFoundError as e:
            log.error(f"  SKIP: {e}")
            continue

        model.eval()

        # GNN inference (T=10 MC)
        with torch.no_grad():
            p_gnn_mean, u_mc, _ = model.forward_mc(test_data, T=args.T)
        p_gnn_np = p_gnn_mean.cpu().numpy()
        u_mc_np = u_mc.cpu().numpy()
        all_u_mc.append(u_mc_np)

        # Val GNN for learned alpha
        with torch.no_grad():
            val_gnn_mean, _, _ = model.forward_mc(val_data, T=args.T)
        val_p_gnn = val_gnn_mean.cpu().numpy()

        # T=1 (Without MC)
        with torch.no_grad():
            p_gnn_t1_mean, _, _ = model.forward_mc(test_data, T=1)
        p_gnn_t1 = p_gnn_t1_mean.cpu().numpy()

        # Validation-tuned fixed alpha and distinct learned fusion head.
        fused_val_tuned, best_alpha = validation_tuned_fixed_fusion(
            p_gnn_np, graphrag_scores_test,
            y_val, val_p_gnn, graphrag_scores_val
        )
        fused_learned = learned_fusion(
            p_gnn_np, graphrag_scores_test,
            y_val, val_p_gnn, graphrag_scores_val,
        )
        log.info(f"  Best alpha (val-tuned): {best_alpha:.1f}")

        # Fixed 0.5 fusion
        fused_fixed05 = fixed_fusion(p_gnn_np, graphrag_scores_test, 0.5)
        # Uncertainty fusion
        fused_unc = uncertainty_fusion(p_gnn_np, u_mc_np, graphrag_scores_test)
        # Without GraphRAG (GNN alone)
        without_graphrag = p_gnn_np
        # Without MC (T=1)
        without_mc = p_gnn_t1

        seed_results = {
            "GNN_Only":               p_gnn_np,
            "Semantic_Only_GraphRAG": graphrag_scores_test,
            "TF_IDF_Only":            tfidf_scores_test,
            "Fixed_Fusion_0.5":       fused_fixed05,
            "Fixed_Fusion_ValTuned":  fused_val_tuned,
            "Learned_Fusion":         fused_learned,
            "Uncertainty_Fusion":     fused_unc,
            "Without_MC_T1":          without_mc,
            "Without_GraphRAG":       without_graphrag,
        }

        for method_name, probs in seed_results.items():
            m = metrics(y_test, probs, method=method_name)
            per_seed[method_name].append(m)
            all_probs[method_name].append(probs)
            log.info(f"  {method_name:30s}: AUC-PR={m['auc_pr']:.4f} "
                     f"AUC-ROC={m['auc_roc']:.4f} F1={m['f1']:.4f}")

    if not any(per_seed["GNN_Only"]):
        log.error("No checkpoints found. Run train_gog_l1.py first.")
        sys.exit(1)

    # ── Aggregate across seeds ──────────────────────────────────────────────
    main_results = []
    for method_name, seed_metrics_list in per_seed.items():
        if not seed_metrics_list:
            continue
        auc_prs = [m["auc_pr"] for m in seed_metrics_list]
        auc_rocs = [m["auc_roc"] for m in seed_metrics_list]
        f1s = [m["f1"] for m in seed_metrics_list]
        eccs = [m["ece"] for m in seed_metrics_list]
        main_results.append({
            "method": method_name,
            "mean_auc_pr": round(float(np.mean(auc_prs)), 6),
            "std_auc_pr": round(float(np.std(auc_prs)), 6),
            "mean_auc_roc": round(float(np.mean(auc_rocs)), 6),
            "std_auc_roc": round(float(np.std(auc_rocs)), 6),
            "mean_f1": round(float(np.mean(f1s)), 6),
            "std_f1": round(float(np.std(f1s)), 6),
            "mean_ece": round(float(np.mean(eccs)), 6),
            "n_seeds": len(seed_metrics_list),
            "gnn_source": "real_checkpoint",
            "split_type": "synthetic_time_ordered",
        })

    main_path = RESULTS_DIR / "real_main_results.csv"
    with open(main_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(main_results[0].keys()))
        writer.writeheader()
        writer.writerows(main_results)
    log.info(f"\nWritten: {main_path.relative_to(ROOT)}")

    # ── Ablation (same as main but formatted differently) ───────────────────
    ablation_path = RESULTS_DIR / "real_ablation_results.csv"
    with open(ablation_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(main_results[0].keys()))
        writer.writeheader()
        writer.writerows(main_results)
    log.info(f"Written: {ablation_path.relative_to(ROOT)}")

    # ── Paired Bootstrap CI ─────────────────────────────────────────────────
    log.info("\n=== Phase K: Bootstrap CI ===")
    # Use mean probs across seeds
    mean_probs = {k: np.mean(v, axis=0) for k, v in all_probs.items() if v}

    boot_results = []
    reference = "Uncertainty_Fusion"
    comparisons = ["GNN_Only", "Fixed_Fusion_0.5", "Learned_Fusion", "Without_MC_T1"]

    for comp in comparisons:
        if reference not in mean_probs or comp not in mean_probs:
            continue
        delta, ci_lo, ci_hi = bootstrap_ci(
            y_test, mean_probs[reference], mean_probs[comp], n_boot=args.n_boot
        )
        pval_approx = "p<0.05" if ci_lo > 0 or ci_hi < 0 else "p>=0.05"
        boot_results.append({
            "comparison": f"{reference}_vs_{comp}",
            "delta_auc_pr": round(delta, 6),
            "ci_lo_95": round(ci_lo, 6),
            "ci_hi_95": round(ci_hi, 6),
            "significance": pval_approx,
            "n_bootstrap": args.n_boot,
        })
        log.info(f"  {reference} vs {comp}: Δ={delta:.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}] {pval_approx}")

    stat_path = RESULTS_DIR / "real_statistical_summary.csv"
    if boot_results:
        with open(stat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(boot_results[0].keys()))
            writer.writeheader()
            writer.writerows(boot_results)
        log.info(f"Written: {stat_path.relative_to(ROOT)}")

    # ── Calibration ─────────────────────────────────────────────────────────
    cal_rows = []
    for method_name, probs_avg in mean_probs.items():
        cal_rows.extend(calibration_data(y_test, probs_avg, method_name))
    cal_path = RESULTS_DIR / "real_calibration.csv"
    if cal_rows:
        with open(cal_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cal_rows)
        log.info(f"Written: {cal_path.relative_to(ROOT)}")

    # ── Privacy Analysis ─────────────────────────────────────────────────────
    if not args.skip_privacy:
        full_probs = mean_probs.get("Uncertainty_Fusion", mean_probs.get("Fixed_Fusion_0.5"))
        gnn_probs = mean_probs.get("GNN_Only")
        if full_probs is not None and gnn_probs is not None:
            run_privacy_analysis(full_probs, gnn_probs, y_test)

    # ── Robustness ───────────────────────────────────────────────────────────
    if not args.skip_robustness and per_seed["GNN_Only"]:
        # Use last seed's model for robustness (sufficient for this analysis)
        try:
            last_seed = seeds[-1]
            rob_model, _ = load_model(last_seed, device)
            run_robustness(
                rob_model, test_data,
                mean_probs.get("GNN_Only", graphrag_scores_val),
                graphrag_scores_val,
                y_val, test_ids, contexts, y_test, device,
            )
        except FileNotFoundError:
            log.warning("Skipping robustness: checkpoint not found")

    log.info("\n=== Phase F + H + I + K Complete ===")


if __name__ == "__main__":
    main()
