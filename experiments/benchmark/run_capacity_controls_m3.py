#!/usr/bin/env python3
"""
run_capacity_controls_m3.py

Executes Round M3 sensitivity controls with full provenance qualification schema:
1. DLG-Aug-Zero: Zero-padded local feature augmentation [X || 0_64]
2. DLG-Base-70: DLG-Base with 70 global epochs (budget sensitivity)
3. DLG-Aug-Permuted: Detached L1 embedding randomly permuted across nodes prior to concatenation

Matrix: 3 datasets (Elliptic, DGraphFin, LANL-RedTeam) x 3 models x 5 seeds (42..46) = 45 runs.
Note on Reddit-Syn: Option B is selected (excluded; audited as NOT_RUN).

Enriched Schema guarantees:
- Complete provenance: python_class, source commit, file SHA-256s, dataset graph SHA-256s, split hashes
- Exact same frozen graph, split strategy, score orientation, and threshold protocol as frozen primary.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

# Ensure gog_fraud and scripts are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
DLG_SRC = REPO_ROOT / "dlg_gnn" / "src"
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
log = logging.getLogger("capacity_controls_m3")

OUTPUT_DIR = DLG_ROOT / "outputs" / "benchmark" / "manuscript_m3" / "controls"
RAW_DIR = OUTPUT_DIR / "raw"
SHARED_RECON_PATH = DLG_SRC / "gog_fraud" / "models" / "pygod" / "shared_reconstruction.py"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "git_commit_unavailable"


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


class SharedDLGAugPermuted(SharedDLGFull):
    """
    Permuted-control model: local pretraining is run normally for 20 epochs,
    but the extracted L1 embedding is randomly permuted across node indices
    prior to concatenation [X || H_permuted].
    """
    def __init__(self, perm_seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.perm_seed = perm_seed

    def process_graph(self, data):
        if hasattr(data, "_dlg_full_augmented") and data._dlg_full_augmented:
            data._reconstruction_backend = self.reconstruction_backend
            return
        self._orig_dim = data.x.size(1)
        data.dlg_original_x = data.x.clone()
        
        # Run standard L1 pretraining
        l1_embs = self._pretrain_level1(data)
        
        # Permute nodes deterministically using perm_seed
        gen = torch.Generator().manual_seed(self.perm_seed)
        perm = torch.randperm(data.x.size(0), generator=gen)
        l1_embs_perm = l1_embs[perm]
        
        # Augment features
        data.x = torch.cat([data.x, l1_embs_perm.to(data.x.device)], dim=-1)
        data._dlg_full_augmented = True
        data._reconstruction_backend = self.reconstruction_backend
        data._message_backend = self.message_backend


def stratified_split_indices(y: np.ndarray, seed: int, val_ratio: float = 0.2, test_ratio: float = 0.2):
    """Deterministic stratified node transductive train/val/test split matching canonical defense protocol."""
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = np.arange(n)

    pos_idx = indices[y == 1]
    neg_idx = indices[y == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(len(pos_idx) * val_ratio))
    n_test_pos = max(1, int(len(pos_idx) * test_ratio))

    n_val_neg = max(1, int(len(neg_idx) * val_ratio))
    n_test_neg = max(1, int(len(neg_idx) * test_ratio))

    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    test_idx = np.concatenate([pos_idx[n_val_pos:n_val_pos + n_test_pos], neg_idx[n_val_neg:n_val_neg + n_test_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos + n_test_pos:], neg_idx[n_val_neg + n_test_neg:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def load_dataset(name: str, data_root: Path):
    if name == "DGraphFin":
        return load_dgraphfin_aligned(data_root / "DGraphFin/dgraphfin.npz")
    elif name == "LANL-RedTeam":
        lanl_pt = DLG_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real" / "graphs" / "lanl_graph.pt"
        if not lanl_pt.exists():
            raise FileNotFoundError(f"Canonical LANL graph not found: {lanl_pt}")
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
        train_idx, val_idx, test_idx = stratified_split_indices(y_all, seed, 0.2, 0.2)
        return val_idx, test_idx, y_all[val_idx], y_all[test_idx]
    else:
        eligible, y = _eligible_labels(data)
        val_local, test_local = _validation_test_indices(y, seed, 0.2, 0.2)
        return eligible[val_local], eligible[test_local], y[val_local], y[test_local]


def run_single_cell(dataset_name: str, model_name: str, seed: int, data_root: Path, gpu: int = 0, force: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RAW_DIR / f"{dataset_name}__{model_name}__seed{seed}.json"

    if out_json.exists() and not force:
        try:
            cached = json.loads(out_json.read_text(encoding="utf-8"))
            if cached.get("status") == "success" and "dataset_graph_sha256" in cached:
                log.info(f"[CACHED PROVENANCE] {dataset_name}/{model_name}/seed={seed}")
                return cached
        except Exception:
            pass

    log.info(f"[RUNNING M3 CONTROL] {dataset_name}/{model_name}/seed={seed}")
    seed_everything(seed)
    raw_data = load_dataset(dataset_name, data_root)
    data = raw_data.clone()

    val_idx, test_idx, y_val, y_test = get_split(data, dataset_name, seed)

    # Compute provenance hashes
    x_sha = sha256_tensor(data.x)
    edge_sha = sha256_tensor(data.edge_index)
    y_sha = sha256_tensor(data.y)
    split_sha = sha256_bytes(np.concatenate([val_idx, test_idx]).tobytes())
    graph_composite_sha = sha256_bytes(f"{x_sha}:{edge_sha}:{y_sha}:{data.num_nodes}:{data.edge_index.size(1)}".encode())

    git_commit = get_git_commit()
    script_sha = sha256_file(Path(__file__).resolve())
    source_file_sha = sha256_file(SHARED_RECON_PATH) if SHARED_RECON_PATH.exists() else "unknown"

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
        py_cls = "gog_fraud.models.pygod.shared_reconstruction.SharedDLGAugZero"
        base_cls = "SharedDLGFull"
        detector = SharedDLGAugZero(epoch=50, l1_epochs=0, **common)
        g_epochs = 50
        l_epochs = 0
    elif model_name == "DLG-Base-70":
        py_cls = "gog_fraud.models.pygod.shared_reconstruction.SharedDLGBase"
        base_cls = "SharedDLGBase"
        detector = SharedDLGBase(epoch=70, **common)
        g_epochs = 70
        l_epochs = 0
    elif model_name == "DLG-Aug-Permuted":
        py_cls = "gog_fraud.models.pygod.shared_reconstruction.SharedDLGAugPermuted"
        base_cls = "SharedDLGFull"
        detector = SharedDLGAugPermuted(epoch=50, l1_epochs=20, perm_seed=seed, **common)
        g_epochs = 50
        l_epochs = 20
    else:
        raise ValueError(f"Unknown control model: {model_name}")

    config_dict = {
        "model": model_name,
        "global_epochs": g_epochs,
        "local_epochs": l_epochs,
        "hidden_dim": 64,
        "num_layers": 4,
        "weight": 0.5,
        "alpha": 0.5,
        "message_backend": "sparse_fused",
        "reconstruction_backend": "exact_sparse",
    }
    config_sha = sha256_bytes(json.dumps(config_dict, sort_keys=True).encode())

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

    backend_meta = getattr(detector, "reconstruction_metadata", {})
    backend_sha = sha256_bytes(json.dumps(backend_meta, sort_keys=True).encode())

    record = {
        "dataset": dataset_name,
        "display_name": dataset_name,
        "model": model_name,
        "seed": seed,
        "status": "success",
        "python_class": py_cls,
        "control_base_class": base_cls,
        "source_git_commit": git_commit,
        "source_file_sha256": source_file_sha,
        "control_script_sha256": script_sha,
        "dataset_graph_sha256": graph_composite_sha,
        "x_sha256": x_sha,
        "edge_index_sha256": edge_sha,
        "y_sha256": y_sha,
        "split_indices_sha256": split_sha,
        "config_sha256": config_sha,
        "backend_sha256": backend_sha,
        "message_backend": "sparse_fused",
        "reconstruction_backend": "exact_sparse",
        "global_epochs": g_epochs,
        "local_epochs": l_epochs,
        "hidden_dim": 64,
        "num_layers": 4,
        "weight": 0.5,
        "threshold_protocol": "validation_selected_f1",
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

    log.info(f"[M3 DONE] {dataset_name}/{model_name}/seed={seed} ROC={roc:.4f} PR={pr:.4f} F1={val_f1:.4f}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="Elliptic,DGraphFin,LANL-RedTeam")
    parser.add_argument("--models", type=str, default="DLG-Aug-Zero,DLG-Base-70,DLG-Aug-Permuted")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Force rerun even if cached")
    args = parser.parse_args()

    data_root = Path("/mnt/d/_Work/_data/DLG")
    if not data_root.exists():
        data_root = REPO_ROOT.parents[0] / "_data" / "DLG"

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    records = []
    for ds in datasets:
        for m in models:
            for s in seeds:
                rec = run_single_cell(ds, m, s, data_root, gpu=args.gpu, force=args.force)
                records.append(rec)

    df = pd.DataFrame(records)
    summary_raw_csv = OUTPUT_DIR / "capacity_controls_summary_m3_raw.csv"
    df.to_csv(summary_raw_csv, index=False)
    log.info(f"Wrote raw summary of {len(df)} runs to {summary_raw_csv}")

    agg = df.groupby(["dataset", "model"]).agg(
        roc_mean=("roc_auc", "mean"),
        roc_std=("roc_auc", "std"),
        pr_mean=("pr_auc", "mean"),
        pr_std=("pr_auc", "std"),
        f1_mean=("validation_f1", "mean"),
        f1_std=("validation_f1", "std"),
        count=("seed", "count")
    ).reset_index()

    summary_csv = OUTPUT_DIR / "capacity_controls_summary_m3.csv"
    agg.to_csv(summary_csv, index=False)
    log.info(f"Wrote aggregated summary to {summary_csv}")


if __name__ == "__main__":
    main()
