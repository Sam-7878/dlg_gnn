"""
Run Phase E capacity and training-budget controls:
1. DLG-Aug-Zero: Zero-padded local feature augmentation [X || 0_64] (isolating input capacity from learned representation)
2. DLG-Base-70: DLG-Base with 70 global epochs (isolating training budget sensitivity)
Matrix: 4 datasets (Elliptic, DGraphFin, LANL-RedTeam, Reddit-Syn) × 2 controls × 5 seeds (42..46).
Results are persisted separately from primary benchmark tables.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure gog_fraud and scripts are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
DLG_SRC = REPO_ROOT / "dlg_gnn/src"
DLG_ROOT = REPO_ROOT / "dlg_gnn"
if str(DLG_SRC) not in sys.path:
    sys.path.insert(0, str(DLG_SRC))
if str(DLG_ROOT) not in sys.path:
    sys.path.insert(0, str(DLG_ROOT))

from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDLGFull
from gog_fraud.pipelines.run_sci_round1_ablation import _eligible_labels
from gog_fraud.pipelines.run_sci_round1_benchmark import _legacy_registries, _validation_test_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("capacity_controls")

OUTPUT_DIR = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/capacity_controls"


class SharedDLGAugZero(SharedDLGFull):
    """
    Capacity-control model: identical architecture and global input dimension (F + 64)
    to DLG-Aug, but local embedding features are replaced with zero vectors.
    """
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


def load_dataset(name: str, data_root: Path):
    if name == "DGraphFin":
        return load_dgraphfin_aligned(data_root / "DGraphFin/dgraphfin.npz")
    elif name == "LANL-RedTeam":
        lanl_pt = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_defense_extension_real/graphs/lanl_graph.pt"
        if not lanl_pt.exists():
            raise FileNotFoundError(f"LANL graph not found: {lanl_pt}")
        return torch.load(lanl_pt, weights_only=False)
    else:
        registry, _ = _legacy_registries(str(data_root), 42)
        if name in registry:
            return registry[name]()
        raise KeyError(f"Unknown dataset: {name}")


def get_split(data, dataset_name: str, seed: int):
    y_all = data.y.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if dataset_name == "DGraphFin":
        val_nodes = np.flatnonzero(data.val_mask.detach().cpu().numpy())
        test_nodes = np.flatnonzero(data.test_mask.detach().cpu().numpy())
        return val_nodes, test_nodes, y_all[val_nodes], y_all[test_nodes]
    elif dataset_name == "LANL-RedTeam":
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        rng = np.random.RandomState(seed)
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_val_pos, n_val_neg = int(len(pos_idx) * 0.2), int(len(neg_idx) * 0.2)
        n_test_pos, n_test_neg = int(len(pos_idx) * 0.2), int(len(neg_idx) * 0.2)
        val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
        test_idx = np.concatenate([pos_idx[n_val_pos:n_val_pos + n_test_pos], neg_idx[n_val_neg:n_val_neg + n_test_neg]])
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        return val_idx, test_idx, y_all[val_idx], y_all[test_idx]
    else:
        eligible, y = _eligible_labels(data)
        val_local, test_local = _validation_test_indices(y, seed, 0.2, 0.2)
        return eligible[val_local], eligible[test_local], y[val_local], y[test_local]


def run_single_cell(dataset_name: str, model_name: str, seed: int, data_root: Path, gpu: int = 0):
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_json = raw_dir / f"{dataset_name}__{model_name}__seed{seed}.json"

    if out_json.exists():
        try:
            cached = json.loads(out_json.read_text(encoding="utf-8"))
            if cached.get("status") == "success":
                log.info(f"[CACHED] {dataset_name}/{model_name}/seed={seed}")
                return cached
        except Exception:
            pass

    log.info(f"[RUNNING] {dataset_name}/{model_name}/seed={seed}")
    seed_everything(seed)
    data = load_dataset(dataset_name, data_root).clone()

    val_idx, test_idx, y_val, y_test = get_split(data, dataset_name, seed)

    common = {
        "gpu": gpu if torch.cuda.is_available() and gpu >= 0 else -1,
        "batch_size": 0,
        "verbose": 0,
        "message_backend": "sparse_fused",
        "reconstruction_backend": "exact_sparse",
        "gradient_checkpointing": False,
        "score_chunk_size": 8192,
    }

    if model_name == "DLG-Aug-Zero":
        detector = SharedDLGAugZero(epoch=50, l1_epochs=0, **common)
    elif model_name == "DLG-Base-70":
        detector = SharedDLGBase(epoch=70, **common)
    else:
        raise ValueError(f"Unknown control model: {model_name}")

    started = time.perf_counter()
    detector.fit(data)
    train_sec = time.perf_counter() - started

    started = time.perf_counter()
    score = detector.decision_function(data).cpu().numpy().reshape(-1)
    infer_sec = time.perf_counter() - started

    score_val = score[val_idx]
    score_test = score[test_idx]

    roc = float(roc_auc_score(y_test, score_test))
    pr = float(average_precision_score(y_test, score_test))
    threshold_eval = evaluate_threshold_protocol(y_val, score_val, y_test, score_test)
    val_f1 = float(threshold_eval.validation_f1)
    oracle_f1 = float(threshold_eval.oracle_best_f1)

    record = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "status": "success",
        "roc_auc": roc,
        "pr_auc": pr,
        "validation_f1": val_f1,
        "oracle_best_f1": oracle_f1,
        "train_sec": train_sec,
        "infer_sec": infer_sec,
        "runtime": train_sec + infer_sec,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    log.info(f"[DONE] {dataset_name}/{model_name}/seed={seed} ROC={roc:.4f} PR={pr:.4f} F1={val_f1:.4f}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="Elliptic,DGraphFin,LANL-RedTeam",
                        help="Comma-separated list of datasets to run")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    data_root = Path("/mnt/d/_Work/_data/DLG")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = ["DLG-Aug-Zero", "DLG-Base-70"]

    records = []
    for ds in datasets:
        for m in models:
            for s in seeds:
                rec = run_single_cell(ds, m, s, data_root, gpu=args.gpu)
                records.append(rec)

    df = pd.DataFrame(records)
    summary_csv = OUTPUT_DIR / "capacity_controls_summary.csv"
    df.to_csv(summary_csv, index=False)
    log.info(f"Wrote summary of {len(df)} runs to {summary_csv}")


if __name__ == "__main__":
    main()
