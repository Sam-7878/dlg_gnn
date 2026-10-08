#!/usr/bin/env python3
"""Targeted A05 BitcoinOTC rerun on the frozen A04 constructed graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.pipelines.run_sci_round4c import _instantiate, _models, _split

OUT = ROOT / "outputs/benchmark/a05_bitcoinotc"
GRAPH = ROOT / "outputs/benchmark/a04_constructed_graphs/BitcoinOTC.pt"
MANIFEST = ROOT / "evaluation/benchmark/v2/paper_ready_final/dataset_manifest_canonical.json"
CONFIG = ROOT / "configs/benchmark/sci_round5_final.yaml"
LOCK = ROOT / "environment/locks/benchmark-a04-cuda.hashed.txt"
MODELS = ("DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug")
SEEDS = (42, 43, 44, 45, 46)


def filehash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arrayhash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def run_cell(model: str, seed: int) -> int:
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("scores").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    path = OUT / "raw" / f"BitcoinOTC__{model}__seed{seed}.json"
    if path.exists() and json.loads(path.read_text()).get("status") == "success":
        print(f"[SKIP] {model} seed{seed}", flush=True)
        return 0
    config = yaml.safe_load(CONFIG.read_text())
    config["data"]["root"] = "data"
    manifest = next(row for row in json.loads(MANIFEST.read_text()) if row["dataset_id"] == "BitcoinOTC")
    if filehash(GRAPH) != manifest["constructed_artifact_sha256"]:
        raise RuntimeError("A04 BitcoinOTC graph artifact changed")
    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    data = Data(x=payload["x"], edge_index=payload["edge_index"], y=payload["y"], num_nodes=payload["num_nodes"])
    seed_everything(seed, deterministic=True)
    val_nodes, test_nodes, val_y, test_y, split_strategy = _split(data, "BitcoinOTC", seed, config)
    split_hash = digest({"val_nodes": arrayhash(val_nodes), "test_nodes": arrayhash(test_nodes),
                         "val_y": arrayhash(val_y), "test_y": arrayhash(test_y)})
    cls = _models(config)[model]
    effective = {"runner": "a05_run_bitcoinotc.py", "model": model,
                 "constructor": "run_sci_round4c._instantiate", "round5_config": config,
                 "model_class": f"{cls.__module__}.{cls.__qualname__}"}
    source_files = [Path(__file__), ROOT / "src/gog_fraud/pipelines/run_sci_round4c.py",
                    ROOT / "src/gog_fraud/models/pygod/shared_reconstruction.py",
                    ROOT / "src/gog_fraud/models/pygod/exact_reconstruction.py"]
    record = {"run_id": f"A05-BitcoinOTC-{model}-seed{seed}", "dataset": "BitcoinOTC",
              "model": model, "seed": seed, "status": "running", "configured_epochs": 50,
              "dataset_artifact_path": str(GRAPH.relative_to(ROOT)),
              "dataset_artifact_sha256": filehash(GRAPH),
              "constructed_tensor_sha256": manifest["constructed_tensor_sha256"],
              "feature_hash": manifest["feature_hash"], "edge_hash": manifest["edge_hash"],
              "label_hash": manifest["label_hash"], "split_hash": split_hash,
              "split_strategy": split_strategy, "model_config_hash": digest(effective),
              "model_config": effective, "environment_lock_sha256": filehash(LOCK),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_file_sha256": {str(p.relative_to(ROOT)): filehash(p) for p in source_files},
              "python": sys.version.split()[0], "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "full_shared_graph": True, "approximation_used": False,
              "validation_count": len(val_nodes), "test_count": len(test_nodes),
              "test_positive_count": int(np.sum(test_y))}
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    started = time.perf_counter()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        detector = _instantiate(config, model, cls, 0)
        if model == "AnomalyDAE":
            detector.training_checkpoint_path = OUT / "checkpoints" / f"{model}__seed{seed}.pt"
            detector.training_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        train_started = time.perf_counter()
        detector.fit(data)
        train_sec = time.perf_counter() - train_started
        inference_started = time.perf_counter()
        raw = detector.decision_function(data)
        torch.cuda.synchronize(0)
        inference_sec = time.perf_counter() - inference_started
        score = raw.detach().cpu().numpy().reshape(-1) if torch.is_tensor(raw) else np.asarray(raw).reshape(-1)
        if score.size != data.num_nodes or not np.isfinite(score).all():
            raise FloatingPointError("score length or finite check failed")
        score_path = OUT / "scores" / f"BitcoinOTC__{model}__seed{seed}.npy"
        np.save(score_path, score)
        threshold = evaluate_threshold_protocol(val_y, score[val_nodes], test_y, score[test_nodes], fixed_05_applicable=False)
        record.update({"status": "success", "actual_epochs": int(getattr(detector, "actual_epochs_", len(getattr(detector, "loss_history_", [])) or detector.epoch)),
                       "roc_auc": float(roc_auc_score(test_y, score[test_nodes])),
                       "pr_auc": float(average_precision_score(test_y, score[test_nodes])),
                       **threshold.to_dict(), "raw_score_path": str(score_path.relative_to(ROOT)),
                       "raw_score_sha256": arrayhash(score), "raw_score_file_sha256": filehash(score_path),
                       "train_time_sec": train_sec, "inference_time_sec": inference_sec,
                       "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20})
    except BaseException as exc:
        record.update({"status": "failed_oom" if isinstance(exc, torch.OutOfMemoryError) else "failed_other",
                       "failure_type": type(exc).__name__, "failure_message": str(exc),
                       "traceback": traceback.format_exc()})
    record["total_wall_sec"] = time.perf_counter() - started
    path.write_text(json.dumps(record, indent=2, default=str, allow_nan=True) + "\n")
    print(f"[DONE] {model} seed{seed} {record['status']} {record['total_wall_sec']:.1f}s", flush=True)
    return 0 if record["status"] == "success" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if (args.model is None) != (args.seed is None):
        parser.error("--model and --seed must be supplied together")
    if args.model:
        return run_cell(args.model, args.seed)
    for model in MODELS:
        for seed in SEEDS:
            command = [sys.executable, str(Path(__file__)), "--model", model, "--seed", str(seed)]
            log = OUT / "logs" / f"BitcoinOTC__{model}__seed{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                try:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=24 * 3600, check=False)
                    print(f"{model} seed{seed}: exit {result.returncode}", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"{model} seed{seed}: 24-hour guard exceeded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
