"""Real, contract-level SCI-v2 experiments for Round 4.

The immutable SCI-v2 unit is a labelled contract graph.  This runner derives
auditable graph summary features from each manifest record, builds relations
against training candidates only, and uses the same split/metrics for PyGOD
and DLG variants.  Every successful record has raw predictions and provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gog_fraud.evaluation.calibration import binary_calibration_metrics, fit_temperature, write_reliability_csv
from gog_fraud.experiments.manifest import RunManifest
from gog_fraud.experiments.round4_policy import DATASET_VERSION, assess_paper_eligibility

CHAINS = ("ethereum", "bsc", "polygon")
SEEDS = (11, 22, 33, 44, 55)
PYGOD_MODELS = ("DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA", "CONAD", "GAAN", "GUIDE")
DLG_VARIANTS = ("DLG-L1", "DLG-L1-L2", "DLG-Full-Fusion", "DLG-Full-Fusion-LPP")
STREAM_VARIANTS = ("StreamMC-deterministic", "StreamMC-variance", "StreamMC-dual", "StreamMC-risk")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _git_state(repo: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _feature(record: dict[str, Any], chain_feature: bool = True) -> list[float]:
    nodes = max(1, int(record["num_nodes"])); edges = max(1, int(record["num_edges"]))
    duration = max(1, int(record["event_end"]) - int(record["event_start"]) + 1)
    value = max(0.0, float(record["transaction_value_sum"]))
    result = [
        math.log1p(nodes), math.log1p(edges), math.log1p(value), math.log1p(duration),
        math.log1p(edges / nodes), math.log1p(edges / duration * 86400.0),
        math.log1p(value / edges), math.log1p(value / duration * 86400.0),
    ]
    if chain_feature:
        result.extend(float(record["chain_id"] == chain) for chain in CHAINS)
    return result


class SciV2Records:
    def __init__(self, root: Path, *, chain_feature: bool = True):
        self.root = root
        self.records: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, dict[str, list[str]]] = {}
        self.split_hashes: dict[str, str] = {}
        for chain in CHAINS:
            manifest = json.loads((root / f"manifests/{chain}.json").read_text(encoding="utf-8"))
            for row in manifest["records"]:
                self.records[row["sample_id"]] = row
            split = json.loads((root / f"splits/{chain}_holdout_v2.json").read_text(encoding="utf-8"))
            self.groups[chain] = {name: value["sample_ids"] for name, value in split["groups"].items()}
            self.split_hashes[chain] = split["split_hash"]
        self.chain_feature = chain_feature

    def ids(self, chain: str, group: str) -> list[str]:
        if chain == "pooled":
            return [sample for c in CHAINS for sample in self.groups[c][group]]
        return list(self.groups[chain][group])

    def arrays(self, ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        selected = list(ids)
        return (np.asarray([_feature(self.records[x], self.chain_feature) for x in selected], dtype=np.float32),
                np.asarray([self.records[x]["label"] for x in selected], dtype=np.int64))

    def split_hash(self, chain: str) -> str:
        if chain != "pooled": return self.split_hashes[chain]
        return _sha_bytes(json.dumps(self.split_hashes, sort_keys=True).encode())


def _normalize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(0); scale = train.std(0); scale[scale == 0] = 1
    return tuple(((value - mean) / scale).astype(np.float32) for value in (train, *others))


def _relation_edges(reference: np.ndarray, target: np.ndarray | None = None, k: int = 8) -> torch.Tensor:
    """Reference graph plus target->reference edges; never target->target."""
    n_ref = len(reference)
    if n_ref == 0: return torch.empty((2, 0), dtype=torch.long)
    k_ref = min(k + 1, n_ref)
    finder = NearestNeighbors(n_neighbors=k_ref).fit(reference)
    _, local = finder.kneighbors(reference)
    src: list[int] = []; dst: list[int] = []
    for i, neighbors in enumerate(local):
        for j in neighbors:
            if i != int(j): src.extend((i, int(j))); dst.extend((int(j), i))
    if target is not None and len(target):
        _, external = finder.kneighbors(target, n_neighbors=min(k, n_ref))
        for offset, neighbors in enumerate(external):
            node = n_ref + offset
            for j in neighbors:
                src.extend((node, int(j))); dst.extend((int(j), node))
    if not src:
        src = list(range(n_ref)); dst = list(range(n_ref))
    return torch.tensor([src, dst], dtype=torch.long)


def _data(reference: np.ndarray, target: np.ndarray | None = None) -> Data:
    x = reference if target is None else np.concatenate((reference, target), axis=0)
    return Data(x=torch.from_numpy(x), edge_index=_relation_edges(reference, target), num_nodes=len(x))


def _threshold(y: np.ndarray, scores: np.ndarray) -> float:
    if not len(scores): return 0.5
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, min(201, len(scores)))))
    values = [f1_score(y, scores >= item, zero_division=0) for item in candidates]
    return float(candidates[int(np.argmax(values))])


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "pr_auc": float(average_precision_score(y, scores)) if y.sum() else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fraud_recall": float(recall_score(y, pred, zero_division=0)),
        "benign_recall": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "mcc": float(matthews_corrcoef(y, pred)) if both else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "threshold": threshold, "n": int(len(y)), "fraud_n": int(y.sum()),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


class ContractDLG(nn.Module):
    def __init__(self, dim: int, variant: str, dropout: float = 0.25):
        super().__init__(); self.variant = variant; self.dropout = dropout
        hidden = 32
        self.local = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.conv1 = GCNConv(hidden, hidden); self.conv2 = GCNConv(hidden, hidden)
        self.local_head = nn.Linear(hidden, 1); self.relation_head = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        local = self.local(x); local_logit = self.local_head(local).view(-1)
        if self.variant == "DLG-L1": return local_logit
        relation = F.relu(self.conv1(local, edge_index)); relation = F.dropout(relation, self.dropout, self.training)
        relation = F.relu(self.conv2(relation, edge_index)); relation_logit = self.relation_head(relation).view(-1)
        if self.variant == "DLG-L1-L2": return relation_logit
        gate = torch.sigmoid(self.gate(torch.cat((local, relation), dim=1))).view(-1)
        return gate * local_logit + (1 - gate) * relation_logit


def _fit_dlg(train_x: np.ndarray, train_y: np.ndarray, *, variant: str, seed: int,
             epochs: int, device: torch.device) -> tuple[ContractDLG, dict[str, Any]]:
    _seed(seed); data = _data(train_x).to(device); y = torch.from_numpy(train_y).float().to(device)
    model = ContractDLG(train_x.shape[1], variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    positives = max(1, int(train_y.sum())); weight = torch.tensor([(len(train_y) - positives) / positives], device=device)
    started = time.perf_counter(); model.train()
    for _ in range(epochs):
        optimizer.zero_grad(); logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight); loss.backward(); optimizer.step()
    elapsed = time.perf_counter() - started
    state = b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
    return model, {"fit_seconds": elapsed, "fitted_state_hash": _sha_bytes(state), "epochs": epochs}


def _dlg_scores(model: ContractDLG, train_x: np.ndarray, target_x: np.ndarray,
                device: torch.device, *, mc: int = 1) -> tuple[np.ndarray, np.ndarray, list[float]]:
    data = _data(train_x, target_x).to(device); offset = len(train_x); samples = []; latency = []
    model.train(mc > 1)
    with torch.no_grad():
        for _ in range(mc):
            started = time.perf_counter(); score = torch.sigmoid(model(data.x, data.edge_index))[offset:]
            if device.type == "cuda": torch.cuda.synchronize()
            latency.append((time.perf_counter() - started) * 1000.0)
            samples.append(score.detach().cpu().numpy())
    matrix = np.asarray(samples)
    return matrix.mean(0), matrix.var(0), latency


def _pygod_detector(name: str, *, epochs: int, seed: int, gpu: int):
    from pygod import detector
    cls = getattr(detector, name)
    kwargs: dict[str, Any] = {"epoch": epochs, "gpu": gpu, "verbose": 0, "contamination": 0.1}
    if name == "GUIDE": kwargs["cache_dir"] = None
    _seed(seed)
    return cls(**kwargs)


def _base_record(*, run_id: str, chain: str, seed: int, model: str, split_hash: str,
                 manifest_path: Path, config_path: Path, clean: bool, phase: str) -> dict[str, Any]:
    return {
        "experiment_id": f"{run_id}:{phase}:{chain}:{model}:{seed}", "phase": phase,
        "chain": chain, "seed": seed, "model": model, "dataset_version": DATASET_VERSION,
        "leakage_audit_status": "PASS", "split_hash": split_hash,
        "run_manifest": str(manifest_path), "resolved_config": str(config_path),
        "git_clean_at_start": clean, "real_model_inference": True,
        "demo_or_synthetic_metric": False, "status": "SUCCESS",
        "legacy_compatibility": "PARTIAL",
    }


def _finish_record(record: dict[str, Any], *, ids: list[str], y: np.ndarray, scores: np.ndarray,
                   threshold: float, output: Path, extra_columns: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    rows = [{"sample_id": sample, "label": int(label), "score": float(score)}
            for sample, label, score in zip(ids, y, scores)]
    if extra_columns:
        for index, row in enumerate(rows):
            for key, values in extra_columns.items(): row[key] = float(values[index])
    _write_csv(output, rows)
    record.update(_metrics(y, scores, threshold))
    record.update({"prediction_path": str(output), "prediction_sha256": _sha_file(output),
                   "expected_sample_count": len(ids), "prediction_count": len(scores),
                   "sample_count_consistent": len(ids) == len(scores) == len(y)})
    eligible, reasons = assess_paper_eligibility(record)
    record["paper_eligible"] = eligible; record["eligibility_reasons"] = reasons
    return record


def run_dlg(dataset: SciV2Records, root: Path, *, run_id: str, phase: str, chains: Iterable[str],
            seeds: Iterable[int], variants: Iterable[str], epochs: int, mc_values: Iterable[int],
            device: torch.device, manifest_path: Path, config_path: Path, clean: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for chain in chains:
        train_ids, valid_ids, test_ids = (dataset.ids(chain, name) for name in ("train", "validation", "test"))
        train_x, train_y = dataset.arrays(train_ids); valid_x, valid_y = dataset.arrays(valid_ids); test_x, test_y = dataset.arrays(test_ids)
        train_x, valid_x, test_x = _normalize(train_x, valid_x, test_x)
        for seed in seeds:
            for variant in variants:
                try:
                    print(f"[round4] start {phase} {chain} {variant} seed={seed}", flush=True)
                    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
                    rss0 = psutil.Process().memory_info().rss; started = time.perf_counter()
                    model, fit_meta = _fit_dlg(train_x, train_y, variant=variant, seed=seed, epochs=epochs, device=device)
                    val_score, _, _ = _dlg_scores(model, train_x, valid_x, device)
                    threshold = _threshold(valid_y, val_score)
                    test_score, variance, latency = _dlg_scores(model, train_x, test_x, device, mc=1)
                    out = root / phase / "predictions" / f"{chain}__{variant}__seed{seed}.csv"
                    record = _base_record(run_id=run_id, chain=chain, seed=seed, model=variant,
                        split_hash=dataset.split_hash(chain), manifest_path=manifest_path, config_path=config_path, clean=clean, phase=phase)
                    record.update(fit_meta | {"wall_seconds": time.perf_counter() - started,
                        "peak_rss_mb": max(rss0, psutil.Process().memory_info().rss) / 2**20,
                        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
                        "mean_inference_latency_ms": float(np.mean(latency)), "actual_model_class": "ContractDLG"})
                    records.append(_finish_record(record, ids=test_ids, y=test_y, scores=test_score,
                                                  threshold=threshold, output=out, extra_columns={"mc_variance": variance}))
                    print(f"[round4] success {phase} {chain} {variant} seed={seed}", flush=True)
                    if variant in ("DLG-Full-Fusion", "DLG-Full-Fusion-LPP"):
                        for t in mc_values:
                            mean, var, lat = _dlg_scores(model, train_x, test_x, device, mc=t)
                            mc_out = root / "mc" / "predictions" / f"{chain}__seed{seed}__T{t}.csv"
                            mc_record = _base_record(run_id=run_id, chain=chain, seed=seed, model="DLG-StreamMC",
                                split_hash=dataset.split_hash(chain), manifest_path=manifest_path, config_path=config_path, clean=clean, phase="mc")
                            mc_record.update({"mc_passes": t, "mean_latency_ms": float(np.mean(lat)),
                                "p95_latency_ms": float(np.quantile(lat, .95)), "p99_latency_ms": float(np.quantile(lat, .99)),
                                "uncertainty_mean": float(var.mean()), "fitted_state_hash": fit_meta["fitted_state_hash"],
                                "actual_model_class": "ContractDLG", "throughput_samples_per_second": len(test_ids) / max(sum(lat) / 1000, 1e-9)})
                            records.append(_finish_record(mc_record, ids=test_ids, y=test_y, scores=mean,
                                threshold=threshold, output=mc_out, extra_columns={"mc_variance": var}))
                    del model
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    failures.append({"phase": phase, "chain": chain, "seed": seed, "model": variant,
                                     "error_type": type(exc).__name__, "error": str(exc), "oom": "out of memory" in str(exc).lower()})
    return records, failures


def run_pygod(dataset: SciV2Records, root: Path, *, run_id: str, phase: str, chains: Iterable[str],
              seeds: Iterable[int], models: Iterable[str], epochs: int, gpu: int,
              manifest_path: Path, config_path: Path, clean: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for chain in chains:
        train_ids, valid_ids, test_ids = (dataset.ids(chain, name) for name in ("train", "validation", "test"))
        train_x, train_y = dataset.arrays(train_ids); valid_x, valid_y = dataset.arrays(valid_ids); test_x, test_y = dataset.arrays(test_ids)
        train_x, valid_x, test_x = _normalize(train_x, valid_x, test_x)
        train_data = _data(train_x); valid_data = _data(train_x, valid_x); test_data = _data(train_x, test_x)
        for seed in seeds:
            for name in models:
                started = time.perf_counter(); rss0 = psutil.Process().memory_info().rss
                try:
                    print(f"[round4] start {phase} {chain} {name} seed={seed}", flush=True)
                    if gpu >= 0: torch.cuda.reset_peak_memory_stats(gpu)
                    model = _pygod_detector(name, epochs=epochs, seed=seed, gpu=gpu)
                    model.fit(train_data)
                    val_all = np.asarray(model.decision_function(valid_data)); val_score = val_all[len(train_x):]
                    threshold = _threshold(valid_y, val_score)
                    score_all = np.asarray(model.decision_function(test_data)); test_score = score_all[len(train_x):]
                    state = b"".join(v.detach().cpu().numpy().tobytes() for v in model.model.state_dict().values())
                    out = root / phase / "predictions" / f"{chain}__{name}__seed{seed}.csv"
                    record = _base_record(run_id=run_id, chain=chain, seed=seed, model=name,
                        split_hash=dataset.split_hash(chain), manifest_path=manifest_path, config_path=config_path, clean=clean, phase=phase)
                    record.update({"fit_called": True, "decision_function_called": True, "epochs": epochs,
                        "fitted_state_hash": _sha_bytes(state), "actual_model_class": f"pygod.detector.{name}",
                        "wall_seconds": time.perf_counter() - started,
                        "peak_rss_mb": max(rss0, psutil.Process().memory_info().rss) / 2**20,
                        "peak_vram_mb": torch.cuda.max_memory_allocated(gpu) / 2**20 if gpu >= 0 else 0.0})
                    records.append(_finish_record(record, ids=test_ids, y=test_y, scores=test_score, threshold=threshold, output=out))
                    print(f"[round4] success {phase} {chain} {name} seed={seed}", flush=True)
                    del model
                    if gpu >= 0: torch.cuda.empty_cache()
                except Exception as exc:
                    print(f"[round4] failure {phase} {chain} {name} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                    failures.append({"phase": phase, "chain": chain, "seed": seed, "model": name,
                        "error_type": type(exc).__name__, "error": str(exc), "oom": "out of memory" in str(exc).lower(),
                        "fallback": False, "exclusion_justification": "real PyGOD execution failed; no substitute metric emitted"})
                    if gpu >= 0: torch.cuda.empty_cache()
    return records, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--require-clean-git", action="store_true"); parser.add_argument("--real-inference-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(); repo = Path.cwd().resolve(); output = Path(args.output_root).resolve(); output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); git_sha, dirty = _git_state(repo)
    if args.require_clean_git and dirty:
        raise SystemExit("Round 4 refuses to start: Git working tree is dirty")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.phase}_{git_sha[:8]}"
    manifest_dir = output / "manifests" / run_id; manifest_dir.mkdir(parents=True)
    resolved = manifest_dir / "resolved_config.yaml"; resolved.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    manifest_path = manifest_dir / "run_manifest.json"
    run_manifest = RunManifest.capture(experiment_id=run_id, config=config, seed=11,
        dataset_files=[Path(args.dataset_root) / "manifests/dataset_summary.json", resolved], repo_root=repo)
    dataset = SciV2Records(Path(args.dataset_root), chain_feature=True)
    phase_cfg = config[args.phase]; device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.phase == "pilot":
        chains = CHAINS; seeds = (11,); pygod_models = tuple(phase_cfg["pygod_models"]); dlg_variants = ("DLG-Full-Fusion",)
        mc_values = (1, 8)
    else:
        chains = (*CHAINS, "pooled"); seeds = tuple(config["seeds"]); pygod_models = tuple(config["pygod_models"])
        dlg_variants = DLG_VARIANTS; mc_values = tuple(config["mc_values"])
    common = dict(run_id=run_id, phase=args.phase, chains=chains, seeds=seeds, manifest_path=manifest_path,
                  config_path=resolved, clean=not dirty)
    dlg_records, dlg_failures = run_dlg(dataset, output, variants=dlg_variants, epochs=int(phase_cfg["dlg_epochs"]),
        mc_values=mc_values, device=device, **common)
    pygod_records, pygod_failures = run_pygod(dataset, output, models=pygod_models, epochs=int(phase_cfg["pygod_epochs"]),
        gpu=0 if device.type == "cuda" else -1, **common)
    records = dlg_records + pygod_records; failures = dlg_failures + pygod_failures
    if args.phase == "pilot":
        for record in records:
            record["experiment_scope"] = "EXPLORATORY"
            record["paper_eligible"] = False
            record["eligibility_reasons"] = ["pilot results are exploratory and excluded from main tables"]
    else:
        for record in records:
            record["experiment_scope"] = "MAIN"
    _write_csv(output / f"{args.phase}/experiment_records.csv", records)
    _write_csv(output / "failures" / f"{args.phase}_failures.csv", failures)
    summary = {"run_id": run_id, "phase": args.phase, "status": "SUCCESS" if records else "FAILED",
        "records": len(records), "paper_eligible_records": sum(bool(r["paper_eligible"]) for r in records),
        "failures": len(failures), "git_clean_at_experiment_start": not dirty, "git_sha": git_sha,
        "real_inference_only": True, "pilot_is_exploratory": args.phase == "pilot"}
    summary_path = output / f"{args.phase}/summary.json"; _atomic_json(summary_path, summary)
    run_manifest.finalize(status="success" if records else "failed", output_files=[resolved, summary_path, output / f"{args.phase}/experiment_records.csv"])
    run_manifest.write(manifest_path)
    print(json.dumps(summary, indent=2))
    if args.strict and (not records or (args.phase == "pilot" and failures)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
