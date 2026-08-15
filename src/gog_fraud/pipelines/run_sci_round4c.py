"""Round 4C production-equivalent representative benchmark runner."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.score_semantics import audit_score_orientation, get_score_semantics
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.experiments.round4c_policy import (
    canonical_hash, cell_key, classify_timeout, should_skip, validate_status,
)
from gog_fraud.models.pygod.shared_reconstruction import (
    SharedAnomalyDAE, SharedCONAD, SharedDLGBase, SharedDLGFull, SharedDOMINANT,
)
from gog_fraud.models.pygod.sparse_message import AutoSparseFusedGCN
from gog_fraud.pipelines.run_sci_round1_benchmark import (
    _eligible_labels, _legacy_registries, _resolve_gpu, _validation_test_indices,
)
from gog_fraud.pipelines.run_sci_round4b import ResourceMonitor

LAYOUT = ("raw", "checkpoints", "logs", "resources", "tables", "figures", "manifests", "freeze")
STATUS_FROM_EXCEPTION = {
    torch.OutOfMemoryError: "failed_oom",
    FloatingPointError: "failed_numerical",
    FileNotFoundError: "failed_data",
    KeyError: "failed_data",
}


def ensure_layout(root: Path) -> None:
    for name in LAYOUT:
        (root / name).mkdir(parents=True, exist_ok=True)


def hashes(config: dict) -> tuple[str, str]:
    config_hash = canonical_hash(config)
    backend_hash = canonical_hash({
        "backend": config["backend"], "full_shared_graph": True,
        "partition_fallback": False, "cpu_fallback": False,
    })
    return config_hash, backend_hash


def result_path(config: dict, output: Path, dataset: str, model: str, seed: int) -> Path:
    config_hash, backend_hash = hashes(config)
    key = cell_key(dataset, model, seed, config_hash, backend_hash)
    return output / "raw" / f"{dataset}__{model}__seed{seed}__{key}.json"


def _datasets(config: dict):
    registry, _ = _legacy_registries(config["data"]["root"], int(config["data"]["fixed_dataset_seed"]))
    root = Path(config["data"]["root"])
    registry["DGraphFin"] = lambda: load_dgraphfin_aligned(root / "DGraphFin/dgraphfin.npz")
    return registry


def _models(config: dict):
    _, legacy = _legacy_registries(config["data"]["root"], int(config["data"]["fixed_dataset_seed"]))
    return {
        "DOMINANT": SharedDOMINANT,
        "AnomalyDAE": SharedAnomalyDAE,
        "CoLA": legacy["CoLA"],
        "CONAD": SharedCONAD,
        "GADNR": legacy["GADNR"],
        "OCGNN": legacy["OCGNN"],
        "DLG-Base": SharedDLGBase,
        "DLG-Aug": SharedDLGFull,
    }


def _instantiate(config: dict, model_name: str, model_class, gpu: int):
    epochs = int(config["training"]["epochs"])
    common = {"epoch": epochs, "gpu": gpu, "batch_size": 0, "verbose": 0}
    if model_name == "AnomalyDAE":
        return model_class(
            **common, reconstruction_backend="chunked_exact",
            score_chunk_size=int(config["backend"]["anomalydae_score_chunk_size"]),
        )
    if model_name in {"DOMINANT", "CONAD", "DLG-Base", "DLG-Aug"}:
        kwargs = {
            **common, "message_backend": "sparse_fused",
            "reconstruction_backend": "exact_sparse", "gradient_checkpointing": False,
            "score_chunk_size": int(config["backend"]["linear_score_chunk_size"]),
        }
        if model_name == "DLG-Aug":
            kwargs["l1_epochs"] = int(config["training"]["dlg_l1_epochs"])
        return model_class(**kwargs)
    # These historical detectors keep their native objective and full-batch
    # policy; only their GCN backbone aggregation is replaced exactly.
    kwargs = {**common, "num_neigh": -1, "backbone": AutoSparseFusedGCN}
    return model_class(**kwargs)


def _split(data, dataset: str, seed: int, config: dict):
    y_all = data.y.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if dataset == "DGraphFin":
        val_nodes = np.flatnonzero(data.val_mask.detach().cpu().numpy())
        test_nodes = np.flatnonzero(data.test_mask.detach().cpu().numpy())
        return val_nodes, test_nodes, y_all[val_nodes], y_all[test_nodes], "official_random_70_15_15"
    eligible, y = _eligible_labels(data)
    val_local, test_local = _validation_test_indices(
        y, seed, float(config["evaluation"]["validation_ratio"]),
        float(config["evaluation"]["test_ratio"]),
    )
    return eligible[val_local], eligible[test_local], y[val_local], y[test_local], "stratified_node_transductive"


def _label_provenance(dataset: str) -> str:
    return {
        "Elliptic": "real_illicit_labels_unknown_excluded",
        "DGraphFin": "real_labels_official_random_split",
        "Yelp": "synthetic_injection_fixed_dataset_seed_42",
        "Cora": "synthetic_injection_fixed_dataset_seed_42",
        "Reddit": "synthetic_injection_fixed_dataset_seed_42",
    }[dataset]


def _exception_status(exc: BaseException) -> str:
    for kind, status in STATUS_FROM_EXCEPTION.items():
        if isinstance(exc, kind):
            return status
    message = str(exc).lower()
    if "cuda" in message:
        return "failed_cuda"
    if "nan" in message or "inf" in message or "non-finite" in message:
        return "failed_numerical"
    return "failed_other"


def _base_record(config, dataset, model_name, model_class, seed, data):
    config_hash, backend_hash = hashes(config)
    display = config["display_names"][dataset]
    return {
        "run_id": str(uuid.uuid4()), "cell_key": cell_key(dataset, model_name, seed, config_hash, backend_hash),
        "config_hash": config_hash, "backend_hash": backend_hash,
        "dataset": dataset, "display_name": display, "label_provenance": _label_provenance(dataset),
        "model": model_name, "paper_model_name": "DLG" if model_name == "DLG-Aug" else model_name,
        "python_class": f"{model_class.__module__}.{model_class.__qualname__}", "seed": int(seed),
        "configured_epochs": int(config["training"]["epochs"]), "actual_epochs": 0,
        "early_stopped": False, "best_validation_epoch": None,
        "message_backend": "none" if model_name == "AnomalyDAE" else "sparse_fused",
        "reconstruction_backend": (
            "chunked_exact" if model_name == "AnomalyDAE" else
            "exact_sparse" if model_name in {"DOMINANT","CONAD","DLG-Base","DLG-Aug"} else
            "not_applicable_native_objective"
        ),
        "full_shared_graph": True, "approximation_used": False,
        "partition_fallback": False, "cpu_fallback": False,
        "nodes": int(data.num_nodes), "edges": int(data.num_edges),
        "status": "failed_other", "failure_type": None, "failure_message": None,
    }


def run_cell(config: dict, output: Path, dataset: str, model_name: str, seed: int) -> dict:
    ensure_layout(output)
    datasets, models = _datasets(config), _models(config)
    seed_everything(int(config["data"]["fixed_dataset_seed"]), deterministic=True)
    data = datasets[dataset]()
    model_class = models[model_name]
    record = _base_record(config, dataset, model_name, model_class, seed, data)
    path = result_path(config, output, dataset, model_name, seed)
    started_total = time.perf_counter()
    try:
        seed_everything(seed, deterministic=bool(config["execution"]["deterministic"]))
        gpu = _resolve_gpu(int(config["execution"]["gpu"]))
        if gpu < 0:
            raise RuntimeError("silent CPU fallback is forbidden")
        detector = _instantiate(config, model_name, model_class, gpu)
        checkpoint = output / "checkpoints" / f"{record['cell_key']}.pt"
        if model_name == "AnomalyDAE":
            detector.training_checkpoint_path = checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(gpu)
        val_nodes, test_nodes, val_y, test_y, split_strategy = _split(data, dataset, seed, config)
        with ResourceMonitor() as monitor:
            started = time.perf_counter(); detector.fit(data); train_sec = time.perf_counter()-started
            started = time.perf_counter(); raw_score = detector.decision_function(data); inference_sec = time.perf_counter()-started
            torch.cuda.synchronize(gpu)
        score = raw_score.detach().cpu().numpy().reshape(-1) if torch.is_tensor(raw_score) else np.asarray(raw_score).reshape(-1)
        if score.size != data.num_nodes or not np.isfinite(score).all():
            raise FloatingPointError("non-finite score or score-count mismatch")
        val_score, test_score = score[val_nodes], score[test_nodes]
        semantics_name = "DLG" if model_name == "DLG-Aug" else model_name
        semantics = get_score_semantics(semantics_name)
        threshold = evaluate_threshold_protocol(
            val_y, val_score, test_y, test_score, fixed_05_applicable=False,
        )
        orientation = audit_score_orientation(test_y, test_score, expected_higher=True)
        metadata = detector.backend_metadata() if hasattr(detector, "backend_metadata") else {}
        actual_epochs = int(getattr(detector, "actual_epochs_", len(getattr(detector, "loss_history_", [])) or detector.epoch))
        record.update({
            "split_strategy": split_strategy, "actual_epochs": actual_epochs,
            "resumed_from_epoch": int(getattr(detector, "resumed_from_epoch_", 0)),
            "roc_auc": float(roc_auc_score(test_y, test_score)),
            "pr_auc": float(average_precision_score(test_y, test_score)),
            **threshold.to_dict(),
            "mcc": threshold.validation_mcc,
            "balanced_accuracy": threshold.validation_balanced_accuracy,
            "topk_precision": threshold.precision_at_k, "topk_recall": threshold.recall_at_k,
            "raw_roc_auc": orientation["raw_roc_auc"],
            "inverted_roc_auc_diagnostic": orientation["inverted_roc_auc"],
            "orientation_status": "warning_inverse_association" if orientation["orientation_warning"] else "expected_direction",
            "orientation_action": "report_only_no_silent_inversion",
            "train_time_sec": train_sec, "inference_time_sec": inference_sec,
            "total_wall_sec": time.perf_counter()-started_total,
            "rss_peak_mb": monitor.to_dict()["peak_process_rss_mb"],
            "nvidia_smi_peak_mb": monitor.to_dict()["peak_nvidia_smi_used_mb"],
            "cuda_allocated_peak_mb": torch.cuda.max_memory_allocated(gpu)/2**20,
            "cuda_reserved_peak_mb": torch.cuda.max_memory_reserved(gpu)/2**20,
            "score_min": float(score.min()), "score_max": float(score.max()),
            "fixed_05_applicable": False, "f1_at_05": float("nan"),
            "backend_metadata": metadata, "status": "success",
        })
    except BaseException as exc:
        record.update({
            "total_wall_sec": time.perf_counter()-started_total,
            "status": _exception_status(exc), "failure_type": type(exc).__name__,
            "failure_message": str(exc), "traceback": traceback.format_exc(),
        })
    validate_status(record["status"])
    path.write_text(json.dumps(record, indent=2, default=str, allow_nan=True)+"\n", encoding="utf-8")
    return record


def _timeout_record(config, output, dataset, model, seed, timeout_sec):
    config_hash, backend_hash = hashes(config)
    key = cell_key(dataset,model,seed,config_hash,backend_hash)
    progress_path = output / "checkpoints" / f"{key}.pt.progress.json"
    actual_epochs = None
    if progress_path.exists():
        try:
            actual_epochs = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed_epochs"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            actual_epochs = None
    record = {
        "run_id":str(uuid.uuid4()), "cell_key":key,
        "config_hash":config_hash, "backend_hash":backend_hash, "dataset":dataset,
        "display_name":config["display_names"][dataset], "model":model, "seed":seed,
        "configured_epochs":int(config["training"]["epochs"]), "actual_epochs":actual_epochs,
        "status":classify_timeout(model), "failure_type":"WallClockTimeout",
        "failure_message":f"exceeded predeclared {timeout_sec/3600:.1f} hour wall-clock guard",
        "total_wall_sec":float(timeout_sec), "full_shared_graph":True,
        "approximation_used":False, "partition_fallback":False, "cpu_fallback":False,
    }
    result_path(config,output,dataset,model,seed).write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


def _external_failure_record(config, output, dataset, model, seed, returncode, elapsed):
    config_hash, backend_hash = hashes(config)
    record={
        "run_id":str(uuid.uuid4()),"cell_key":cell_key(dataset,model,seed,config_hash,backend_hash),
        "config_hash":config_hash,"backend_hash":backend_hash,"dataset":dataset,
        "display_name":config["display_names"][dataset],"model":model,"seed":seed,
        "configured_epochs":int(config["training"]["epochs"]),"actual_epochs":None,
        "status":"failed_other","failure_type":"ExternalProcessExit",
        "failure_message":f"fresh subprocess exited with code {returncode}","total_wall_sec":float(elapsed),
        "full_shared_graph":True,"approximation_used":False,"partition_fallback":False,"cpu_fallback":False,
    }
    result_path(config,output,dataset,model,seed).write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


def run_matrix(config: dict, config_path: Path, output: Path, *, resume: bool, retry_failed: bool,
               datasets: list[str] | None=None, models: list[str] | None=None) -> int:
    ensure_layout(output)
    timeout_sec = float(config["execution"]["max_run_wall_hours"])*3600
    attempted = 0
    selected_datasets=datasets or config["datasets"];selected_models=models or config["models"]
    for dataset in selected_datasets:
        for seed in config["seeds"]:
            for model in selected_models:
                path = result_path(config, output, dataset, model, int(seed))
                if should_skip(path, resume=resume):
                    print(f"[SKIP] {dataset}/{model}/seed={seed}", flush=True); continue
                if path.exists() and not retry_failed:
                    print(f"[KEEP-FAILED] {dataset}/{model}/seed={seed}", flush=True); continue
                log_path = output/"logs"/f"{dataset}__{model}__seed{seed}.log"
                command = [sys.executable,"-m","gog_fraud.pipelines.run_sci_round4c",
                           "--config",str(config_path),"--stage","cell","--dataset",dataset,
                           "--model",model,"--seed",str(seed)]
                print(f"[RUN] {dataset}/{model}/seed={seed}", flush=True)
                started=time.perf_counter()
                with log_path.open("w",encoding="utf-8") as log:
                    try:
                        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                                 timeout=timeout_sec,check=False)
                        if completed.returncode != 0 and not path.exists():
                            _external_failure_record(config,output,dataset,model,int(seed),completed.returncode,time.perf_counter()-started)
                    except subprocess.TimeoutExpired:
                        _timeout_record(config,output,dataset,model,int(seed),timeout_sec)
                attempted += 1
                result=json.loads(path.read_text(encoding="utf-8"))
                print(f"[DONE] {dataset}/{model}/seed={seed} status={result['status']} wall={result.get('total_wall_sec')}",flush=True)
    return attempted


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    parser.add_argument("--stage",choices=("cell","matrix"),required=True)
    parser.add_argument("--dataset"); parser.add_argument("--model"); parser.add_argument("--seed",type=int)
    parser.add_argument("--datasets",nargs="+"); parser.add_argument("--models",nargs="+")
    parser.add_argument("--resume",action="store_true"); parser.add_argument("--retry-failed",action="store_true")
    args=parser.parse_args(); config_path=Path(args.config).resolve()
    config=yaml.safe_load(config_path.read_text(encoding="utf-8")); output=Path(config["experiment"]["output_root"])
    ensure_layout(output)
    if args.stage=="cell":
        if args.dataset is None or args.model is None or args.seed is None: parser.error("cell requires dataset/model/seed")
        record=run_cell(config,output,args.dataset,args.model,args.seed)
        print(json.dumps({"status":record["status"],"cell_key":record["cell_key"]}))
        return 0
    run_matrix(config,config_path,output,resume=args.resume,retry_failed=args.retry_failed,
               datasets=args.datasets,models=args.models)
    (output/"manifests/config_snapshot.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    (output/"manifests/environment.json").write_text(json.dumps({
        "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform":platform.platform(),
    },indent=2)+"\n",encoding="utf-8")
    return 0


if __name__=="__main__": raise SystemExit(main())
