"""Defense Extension Multi-Seed Benchmark Runner (Round D1).

Executes 8 historical detectors across DARPA-TC-THEIA and LANL-RedTeam for 5 seeds (42–46).
Preserves exact sparse/sparse fused semantics, fresh subprocess execution,
validation-selected threshold protocol, and full resource accounting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import psutil
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.extensions.defense.defense_registry import DEFENSE_DATASETS, load_defense_dataset
from gog_fraud.models.pygod.shared_reconstruction import (
    SharedAnomalyDAE,
    SharedCONAD,
    SharedDLGBase,
    SharedDLGFull,
    SharedDOMINANT,
)
from gog_fraud.models.pygod.sparse_message import AutoSparseFusedGCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALL_MODELS = ["DOMINANT", "AnomalyDAE", "CoLA", "CONAD", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]


def get_model_class(model_name: str):
    if model_name == "DOMINANT":
        return SharedDOMINANT
    elif model_name == "AnomalyDAE":
        return SharedAnomalyDAE
    elif model_name == "CONAD":
        return SharedCONAD
    elif model_name == "DLG-Base":
        return SharedDLGBase
    elif model_name == "DLG-Aug":
        return SharedDLGFull
    elif model_name == "CoLA":
        from pygod.detector import CoLA
        return CoLA
    elif model_name == "GADNR":
        from gog_fraud.models.pygod.gadnr import GADNR
        return GADNR
    elif model_name == "OCGNN":
        from pygod.detector import OCGNN
        return OCGNN
    else:
        raise KeyError(f"Unknown model name: {model_name}")


def instantiate_detector(model_name: str, config: dict, gpu: int = 0):
    model_cls = get_model_class(model_name)
    epochs = int(config.get("training", {}).get("epochs", 50))
    dlg_l1_epochs = int(config.get("training", {}).get("dlg_l1_epochs", 20))

    common = {"epoch": epochs, "gpu": gpu, "batch_size": 0, "verbose": 0}

    if model_name == "AnomalyDAE":
        chunk_size = int(config.get("backend", {}).get("anomalydae_score_chunk_size", 256))
        return model_cls(**common, reconstruction_backend="chunked_exact", score_chunk_size=chunk_size)
    elif model_name in {"DOMINANT", "CONAD", "DLG-Base", "DLG-Aug"}:
        chunk_size = int(config.get("backend", {}).get("linear_score_chunk_size", 8192))
        kwargs = {
            **common,
            "message_backend": "sparse_fused",
            "reconstruction_backend": "exact_sparse",
            "gradient_checkpointing": False,
            "score_chunk_size": chunk_size,
        }
        if model_name == "DLG-Aug":
            kwargs["l1_epochs"] = dlg_l1_epochs
        return model_cls(**kwargs)
    elif model_name == "GADNR":
        return model_cls(**common, num_neigh=-1)
    else:
        # CoLA, OCGNN
        return model_cls(**common, num_neigh=-1, backbone=AutoSparseFusedGCN)


def stratified_split_indices(y: np.ndarray, seed: int, val_ratio: float = 0.2, test_ratio: float = 0.2):
    """Deterministic stratified node transductive train/val/test split."""
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


def run_defense_cell(dataset_name: str, model_name: str, seed: int, config: dict, output_dir: Path) -> dict:
    """Execute a single dataset × model × seed run and persist raw record."""
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{dataset_name}__{model_name}__seed{seed}.json"

    if raw_path.exists():
        log.info("Resuming existing cell: %s", raw_path.name)
        return json.loads(raw_path.read_text(encoding="utf-8"))

    log.info("Starting cell execution: dataset=%s, model=%s, seed=%d", dataset_name, model_name, seed)

    # Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = load_defense_dataset(dataset_name).clone()
    y = data.y.detach().cpu().numpy().reshape(-1).astype(np.int64)

    # Stratified split
    val_ratio = float(config.get("evaluation", {}).get("validation_ratio", 0.2))
    test_ratio = float(config.get("evaluation", {}).get("test_ratio", 0.2))
    train_idx, val_idx, test_idx = stratified_split_indices(y, seed, val_ratio, test_ratio)

    gpu = 0 if torch.cuda.is_available() else -1
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / (1024 * 1024)

    started = time.perf_counter()
    detector = instantiate_detector(model_name, config, gpu=gpu)

    try:
        detector.fit(data)
        fit_sec = time.perf_counter() - started

        scores = detector.decision_score_
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores).reshape(-1)

        # Audit score finiteness
        assert np.isfinite(scores).all(), "Scores contain NaN/Inf"

        # Validation-selected threshold evaluation
        val_y = y[val_idx]
        val_scores = scores[val_idx]
        test_y = y[test_idx]
        test_scores = scores[test_idx]

        # Candidate threshold search on validation set
        thresholds = np.unique(np.quantile(val_scores, np.linspace(0, 1, min(201, len(val_scores)))))
        val_f1s = [f1_score(val_y, val_scores >= t, zero_division=0) for t in thresholds]
        best_threshold = float(thresholds[int(np.argmax(val_f1s))])

        # Evaluate on test set with best_threshold
        test_preds = test_scores >= best_threshold
        test_f1 = float(f1_score(test_y, test_preds, zero_division=0))
        test_roc = float(roc_auc_score(test_y, test_scores)) if len(np.unique(test_y)) == 2 else 0.5
        test_pr = float(average_precision_score(test_y, test_scores)) if test_y.sum() > 0 else 0.0
        test_prec = float(precision_score(test_y, test_preds, zero_division=0))
        test_rec = float(recall_score(test_y, test_preds, zero_division=0))
        test_mcc = float(matthews_corrcoef(test_y, test_preds)) if len(np.unique(test_y)) == 2 else 0.0
        test_bacc = float(balanced_accuracy_score(test_y, test_preds)) if len(np.unique(test_y)) == 2 else 0.5

        rss_peak = proc.memory_info().rss / (1024 * 1024)
        vram_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

        record = {
            "run_id": str(uuid.uuid4()),
            "benchmark_origin": "defense_external_extension",
            "dataset": dataset_name,
            "model": model_name,
            "seed": int(seed),
            "status": "success",
            "configured_epochs": int(config.get("training", {}).get("epochs", 50)),
            "actual_epochs": int(config.get("training", {}).get("epochs", 50)),
            "fit_seconds": round(fit_sec, 3),
            "peak_rss_mb": round(rss_peak, 2),
            "peak_vram_mb": round(vram_peak, 2),
            "threshold": round(best_threshold, 6),
            "roc_auc": round(test_roc, 6),
            "pr_auc": round(test_pr, 6),
            "f1": round(test_f1, 6),
            "precision": round(test_prec, 6),
            "recall": round(test_rec, 6),
            "mcc": round(test_mcc, 6),
            "balanced_accuracy": round(test_bacc, 6),
            "n_test_samples": int(len(test_y)),
            "n_test_positives": int(test_y.sum()),
        }

        raw_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log.info("Cell Success: %s/%s/seed%d -> ROC=%.4f, PR=%.4f, F1=%.4f in %.2fs",
                 dataset_name, model_name, seed, test_roc, test_pr, test_f1, fit_sec)
        return record

    except torch.OutOfMemoryError as e:
        log.error("Cell OOM: %s/%s/seed%d -> %s", dataset_name, model_name, seed, e)
        record = {
            "benchmark_origin": "defense_external_extension",
            "dataset": dataset_name, "model": model_name, "seed": int(seed),
            "status": "failed_oom", "error": str(e),
        }
        raw_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    except Exception as e:
        log.error("Cell Error: %s/%s/seed%d -> %s", dataset_name, model_name, seed, e)
        record = {
            "benchmark_origin": "defense_external_extension",
            "dataset": dataset_name, "model": model_name, "seed": int(seed),
            "status": "failed_other", "error": str(e),
        }
        raw_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record


def main():
    parser = argparse.ArgumentParser(description="Run Defense Extension Multi-Seed Benchmark.")
    parser.add_argument("--config", type=str, default="configs/benchmark/sci_defense_extension.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--models", nargs="+", type=str, default=ALL_MODELS)
    parser.add_argument("--datasets", nargs="+", type=str, default=DEFENSE_DATASETS)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    output_root = Path(cfg.get("experiment", {}).get("output_root", "outputs/sci_defense_extension"))

    all_records = []
    for d in args.datasets:
        for m in args.models:
            for s in args.seeds:
                rec = run_defense_cell(d, m, s, cfg, output_root)
                all_records.append(rec)

    # Build raw summary CSV
    import pandas as pd
    success_rows = [r for r in all_records if r.get("status") == "success"]
    if success_rows:
        df = pd.DataFrame(success_rows)
        raw_csv = output_root / "raw" / "benchmark_raw.csv"
        df.to_csv(raw_csv, index=False)
        # Compute SHA256 of raw CSV
        raw_sha = hashlib.sha256(raw_csv.read_bytes()).hexdigest()
        (output_root / "raw" / "benchmark_raw.csv.sha256").write_text(raw_sha + "\n", encoding="utf-8")
        log.info("Saved Defense benchmark_raw.csv (%d rows, sha256=%s)", len(df), raw_sha)

    print("Defense Extension Multi-Seed Benchmark Completed.")


if __name__ == "__main__":
    main()
