"""Round 4B sparse-message remediation and restricted-readiness runner."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import threading
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from torch_geometric.nn import GCN
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.experiments.round4b_policy import (
    SupportCell, classify_exact_runtime,
)
from gog_fraud.models.pygod.exact_reconstruction import chunked_exact_row_error
from gog_fraud.models.pygod.shared_reconstruction import (
    SharedCONAD, SharedDLGBase, SharedDLGFull, SharedDOMINANT,
)
from gog_fraud.models.pygod.sparse_message import (
    SparseFusedGCN, estimate_coo_message_bytes,
    estimate_sparse_operator_bytes, normalized_sparse_adjt,
)
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.pipelines.run_sci_round1_ablation import select_fusion_weight, _validation_scale
from gog_fraud.pipelines.run_sci_round1_benchmark import (
    _eligible_labels, _legacy_registries, _resolve_gpu, _validation_test_indices,
)

LAYOUT = (
    "message_backend", "equivalence", "large_graph", "anomalydae",
    "failures", "ablation", "representative", "resources",
    "support_matrix", "manifests", "tables", "figures",
)
LARGE_DATASETS = ("DGraphFin", "Yelp", "Reddit")
SUPPORTED_LARGE_MODELS = ("DOMINANT", "CONAD", "DLG-Base", "DLG-Aug")


def ensure_layout(output: Path):
    for name in LAYOUT:
        (output / name).mkdir(parents=True, exist_ok=True)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected)), torch.finfo(expected.dtype).eps)
    return float(torch.linalg.vector_norm(actual - expected)) / denominator


class ResourceMonitor:
    def __init__(self, interval=.2):
        self.interval = interval
        self.rss_before = self.peak_rss = 0
        self.peak_wsl_used = self.peak_nvidia_mib = 0.0
        self.samples = 0
        self._stop = threading.Event(); self._thread = None

    def __enter__(self):
        process = psutil.Process(os.getpid())
        self.rss_before = self.peak_rss = process.memory_info().rss

        def sample():
            while not self._stop.wait(self.interval):
                self.samples += 1
                self.peak_rss = max(self.peak_rss, process.memory_info().rss)
                self.peak_wsl_used = max(self.peak_wsl_used, float(psutil.virtual_memory().used))
                try:
                    query = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2, check=True,
                    )
                    self.peak_nvidia_mib = max(
                        self.peak_nvidia_mib, float(query.stdout.strip().splitlines()[0])
                    )
                except Exception:
                    pass
        self._thread = threading.Thread(target=sample, daemon=True); self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set(); self._thread.join(timeout=3)

    def to_dict(self):
        return {
            "monitor_interval_sec": self.interval, "monitor_samples": self.samples,
            "rss_before_mb": self.rss_before / 2**20,
            "peak_process_rss_mb": self.peak_rss / 2**20,
            "rss_delta_mb": (self.peak_rss-self.rss_before) / 2**20,
            "peak_wsl_used_memory_mb": self.peak_wsl_used / 2**20,
            "peak_nvidia_smi_used_mb": self.peak_nvidia_mib,
        }


def run_equivalence(output: Path):
    torch.manual_seed(202)
    x = torch.randn(12, 6, dtype=torch.float64)
    edge_index = torch.tensor([
        [0,0,1,2,2,3,4,5,5,6,7,8,9,10,11,11],
        [0,1,2,1,3,4,3,6,7,6,8,9,10,11,10,11],
    ])
    rows = []
    forward_pairs, gradient_pairs = [], []
    for layers in (1, 2, 4, 6):
        reference = GCN(6, 8, layers, out_channels=5, dropout=0).double()
        fused = SparseFusedGCN(6, 8, layers, out_channels=5, dropout=0).double()
        fused.load_state_dict(copy.deepcopy(reference.state_dict()))
        adj_t = normalized_sparse_adjt(edge_index, len(x), dtype=x.dtype)
        x_ref = x.clone().requires_grad_(True); x_fused = x.clone().requires_grad_(True)
        out_ref = reference(x_ref, edge_index); out_fused = fused(x_fused, adj_t)
        forward_error = _relative_error(out_fused, out_ref)
        out_ref.square().mean().backward(); out_fused.square().mean().backward()
        gradients_ref = torch.cat([x_ref.grad.flatten(), *[p.grad.flatten() for p in reference.parameters()]])
        gradients_fused = torch.cat([x_fused.grad.flatten(), *[p.grad.flatten() for p in fused.parameters()]])
        gradient_error = _relative_error(gradients_fused, gradients_ref)
        for a, b in zip(out_ref.detach().flatten(), out_fused.detach().flatten()):
            forward_pairs.append({"layers": layers, "coo": float(a), "sparse_fused": float(b)})
        for a, b in zip(gradients_ref, gradients_fused):
            gradient_pairs.append({"layers": layers, "coo": float(a), "sparse_fused": float(b)})

        update_ref = GCN(6, 8, layers, out_channels=5, dropout=0).double()
        update_fused = SparseFusedGCN(6, 8, layers, out_channels=5, dropout=0).double()
        update_ref.load_state_dict(copy.deepcopy(reference.state_dict()))
        update_fused.load_state_dict(copy.deepcopy(reference.state_dict()))
        optimizers = (torch.optim.Adam(update_ref.parameters(), lr=.003),
                      torch.optim.Adam(update_fused.parameters(), lr=.003))
        trajectories = [[], []]
        for _ in range(4):
            for index, (model, optimizer, graph) in enumerate(
                ((update_ref, optimizers[0], edge_index), (update_fused, optimizers[1], adj_t))
            ):
                optimizer.zero_grad(); loss = model(x, graph).square().mean()
                loss.backward(); optimizer.step(); trajectories[index].append(float(loss))
        update_error = max(
            float((a-b).abs().max()) for a, b in zip(update_ref.parameters(), update_fused.parameters())
        )
        trajectory_error = max(abs(a-b) for a, b in zip(*trajectories))
        rows.append({
            "model": "GCN", "layers": layers, "forward_error": forward_error,
            "gradient_error": gradient_error, "update_error": update_error,
            "loss_trajectory_max_error": trajectory_error,
            "pass": forward_error <= 1e-5 and gradient_error <= 1e-4 and update_error <= 1e-5,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "equivalence/sparse_message_equivalence.csv", index=False)
    table.to_csv(output / "tables/table_r4b_a_sparse_message_equivalence.csv", index=False)
    pd.DataFrame(forward_pairs).to_csv(output / "equivalence/forward_pairs.csv", index=False)
    pd.DataFrame(gradient_pairs).to_csv(output / "equivalence/gradient_pairs.csv", index=False)
    (output / "manifests/message_equivalence_gate.json").write_text(json.dumps({
        "pass": bool(table["pass"].all()), "forward_rtol": 1e-5,
        "gradient_rtol": 1e-4, "update_rtol": 1e-5,
    }, indent=2) + "\n", encoding="utf-8")

    for filename, pairs, title in (
        ("03_sparse_message_forward_equivalence.png", forward_pairs, "COO vs sparse-fused forward"),
        ("04_sparse_message_gradient_equivalence.png", gradient_pairs, "COO vs sparse-fused gradient"),
    ):
        frame = pd.DataFrame(pairs); fig, ax = plt.subplots(figsize=(5,5))
        for layers, group in frame.groupby("layers"):
            ax.scatter(group.coo, group.sparse_fused, s=9, alpha=.55, label=f"{layers} layers")
        low = min(frame.coo.min(), frame.sparse_fused.min()); high = max(frame.coo.max(), frame.sparse_fused.max())
        ax.plot([low,high], [low,high], "k--", linewidth=1); ax.legend(fontsize=7)
        ax.set(xlabel="PyG COO reference", ylabel="Sparse fused", title=title)
        fig.tight_layout(); fig.savefig(output / "figures" / filename, dpi=180); plt.close(fig)


def _registries(config):
    datasets, _ = _legacy_registries(config["data"]["root"], 42)
    root = Path(config["data"]["root"])
    datasets["DGraphFin"] = lambda: load_dgraphfin_aligned(root / "DGraphFin/dgraphfin.npz")
    models = {
        "DOMINANT": SharedDOMINANT, "CONAD": SharedCONAD,
        "DLG-Base": SharedDLGBase, "DLG-Aug": SharedDLGFull,
    }
    return datasets, models


def run_large_one(config, output: Path, dataset: str, model_name: str, seed: int):
    datasets, models = _registries(config); seed_everything(seed)
    started = time.perf_counter(); data = datasets[dataset](); load_sec = time.perf_counter()-started
    n, e, features = int(data.num_nodes), int(data.num_edges), int(data.x.shape[1])
    gpu = _resolve_gpu(int(config["execution"].get("gpu", 0)))
    kwargs = {
        "epoch": int(config["large_graph"]["epochs"]), "gpu": gpu,
        "batch_size": 0, "verbose": 0, "message_backend": "sparse_fused",
        "reconstruction_backend": "exact_sparse", "gradient_checkpointing": False,
        "score_chunk_size": int(config["reconstruction"]["score_chunk_size"]),
    }
    if model_name == "DLG-Aug": kwargs["l1_epochs"] = int(config["large_graph"]["l1_epochs"])
    detector = models[model_name](**kwargs)
    record = {
        "dataset": dataset, "display_name": dataset+"-Syn" if dataset in {"Yelp","Reddit"} else dataset,
        "model": model_name, "seed": seed, "nodes": n, "edges": e, "features": features,
        "hidden_dim": int(getattr(detector, "hid_dim", 64)), "backend": "sparse_fused+exact_sparse",
        "message_backend": "sparse_fused", "reconstruction_backend": "exact_sparse",
        "shared_model": True, "full_graph": True, "partition_fallback": False,
        "approximation_used": False, "load_sec": load_sec, "status": "failed_data",
        "coo_reference_estimated_memory_mb": estimate_coo_message_bytes(e, 64)/2**20,
        "sparse_fused_operator_estimated_memory_mb": estimate_sparse_operator_bytes(n, e+n)/2**20,
    }
    try:
        if gpu >= 0:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(gpu)
        with ResourceMonitor() as monitor:
            started = time.perf_counter(); detector.fit(data); train_sec = time.perf_counter()-started
            started = time.perf_counter(); score = detector.decision_function(data); infer_sec = time.perf_counter()-started
            if gpu >= 0: torch.cuda.synchronize(gpu)
        finite_loss = bool(np.isfinite(detector.loss_history_).all())
        finite_score = bool(torch.isfinite(score).all())
        if not finite_loss or not finite_score or score.numel() != n:
            raise FloatingPointError("non-finite loss/score or score-count mismatch")
        metadata = detector.backend_metadata()
        if metadata.get("message_backend") != "sparse_fused" or metadata.get("approximation_used"):
            raise RuntimeError("message backend metadata mismatch or approximation detected")
        record.update({
            "status": "success", "train_sec": train_sec, "infer_sec": infer_sec,
            "runtime": train_sec+infer_sec, "finite_loss": finite_loss,
            "finite_scores": finite_score, "score_count": int(score.numel()),
            "score_min": float(score.min()), "score_max": float(score.max()),
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(gpu)/2**20 if gpu >= 0 else 0.,
            "torch_peak_reserved_mb": torch.cuda.max_memory_reserved(gpu)/2**20 if gpu >= 0 else 0.,
            "physical_gpu_total_mb": torch.cuda.get_device_properties(gpu).total_memory/2**20 if gpu >= 0 else 0.,
            **monitor.to_dict(), "metadata": metadata,
        })
    except torch.OutOfMemoryError as exc:
        record.update({"status": "failed_oom_unexpected", "error_type": type(exc).__name__,
                       "error_message": str(exc), "traceback": traceback.format_exc()})
    except RuntimeError as exc:
        status = "failed_cuda" if "CUDA" in str(exc) else "failed_numerical"
        record.update({"status": status, "error_type": type(exc).__name__,
                       "error_message": str(exc), "traceback": traceback.format_exc()})
    except Exception as exc:
        record.update({"status": "failed_data", "error_type": type(exc).__name__,
                       "error_message": str(exc), "traceback": traceback.format_exc()})
    key = f"{dataset}__{model_name}__seed{seed}"
    (output / "large_graph" / f"{key}.json").write_text(json.dumps(record, indent=2, default=str)+"\n", encoding="utf-8")


def _component_metric(dataset, seed, variant, y, score, val, test, **extra):
    threshold = evaluate_threshold_protocol(y[val], score[val], y[test], score[test])
    row = {
        "dataset": dataset, "seed": seed, "variant": variant, "status": "success",
        "pr_auc": float(average_precision_score(y[test], score[test])),
        "roc_auc": float(roc_auc_score(y[test], score[test])),
        **threshold.to_dict(), "partition_fallback": False,
        "approximation_used": False, "shared_full_graph": True,
    }
    row.update(extra)
    return row


def run_component_one(config, output: Path, dataset: str, seed: int):
    datasets, _ = _registries(config)
    seed_everything(int(config["evaluation"].get("dataset_seed", 42)))
    data = datasets[dataset]()
    eligible, y = _eligible_labels(data)
    val, test = _validation_test_indices(
        y, seed, float(config["evaluation"].get("validation_ratio", .2)),
        float(config["evaluation"].get("test_ratio", .2)),
    )
    gpu = _resolve_gpu(int(config["execution"].get("gpu", 0)))
    epochs = int(config["component"].get("epochs", 50))
    weights = [float(v) for v in config.get("fusion", {}).get("weighted_l1", [.2,.4,.5,.6,.8])]
    common = {
        "epoch": epochs, "gpu": gpu, "batch_size": 0, "verbose": 0,
        "message_backend": "sparse_fused", "reconstruction_backend": "exact_sparse",
        "gradient_checkpointing": False,
        "score_chunk_size": int(config["reconstruction"]["score_chunk_size"]),
    }
    result = {
        "dataset": dataset, "display_name": dataset+"-Syn" if dataset in {"Yelp","Reddit","Cora"} else dataset,
        "seed": seed, "epochs": epochs, "nodes": int(data.num_nodes),
        "edges": int(data.num_edges), "status": "failed_data", "rows": [],
    }
    try:
        seed_everything(seed)
        base = SharedDLGBase(**common)
        started = time.perf_counter(); base.fit(data.clone()); base_train = time.perf_counter()-started
        started = time.perf_counter(); base_score = base.decision_function(data).cpu().numpy(); base_infer = time.perf_counter()-started
        seed_everything(seed)
        augmented_data = data.clone()
        augmented = SharedDLGFull(
            **common, l1_epochs=int(config["component"].get("l1_epochs", epochs)),
        )
        started = time.perf_counter(); augmented.fit(augmented_data); aug_train = time.perf_counter()-started
        started = time.perf_counter(); aug_score = augmented.decision_function(augmented_data).cpu().numpy(); aug_infer = time.perf_counter()-started
        local_score = augmented_data.dlg_l1_score.numpy()
        base_eval, local_eval, aug_eval = base_score[eligible], local_score[eligible], aug_score[eligible]
        rows = [
            _component_metric(dataset, seed, "DLG-Base", y, base_eval, val, test,
                              train_time_sec=base_train, inference_time_sec=base_infer),
            _component_metric(dataset, seed, "DLG-Local", y, local_eval, val, test,
                              train_time_sec=aug_train, inference_time_sec=0.0,
                              shared_training_with="DLG-Aug"),
            _component_metric(dataset, seed, "DLG-Aug", y, aug_eval, val, test,
                              train_time_sec=aug_train, inference_time_sec=aug_infer),
        ]
        local_scaled = _validation_scale(local_eval[val], local_eval)
        aug_scaled = _validation_scale(aug_eval[val], aug_eval)
        selected_weight, candidates = select_fusion_weight(y[val], local_scaled[val], aug_scaled[val], weights)
        fused = selected_weight*local_scaled + (1-selected_weight)*aug_scaled
        validation_values = sorted((c["validation_f1"] for c in candidates), reverse=True)
        margin = validation_values[0]-validation_values[1] if len(validation_values)>1 else math.nan
        rows.append(_component_metric(
            dataset, seed, "DLG-Fusion", y, fused, val, test,
            train_time_sec=aug_train, inference_time_sec=aug_infer,
            selected_l1_weight=selected_weight, selection_margin=margin,
            fusion_candidates=candidates, weight_selection="validation_best_f1",
        ))
        result.update({"status":"success", "rows":rows,
                       "message_backend":"sparse_fused", "reconstruction_backend":"exact_sparse"})
    except torch.OutOfMemoryError as exc:
        result.update({"status":"failed_oom_unexpected", "error_type":type(exc).__name__,
                       "error_message":str(exc), "traceback":traceback.format_exc()})
    except Exception as exc:
        status = "failed_cuda" if "CUDA" in str(exc) else "failed_numerical"
        result.update({"status":status, "error_type":type(exc).__name__,
                       "error_message":str(exc), "traceback":traceback.format_exc()})
    path = output/"ablation"/f"{dataset}__seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, default=str)+"\n", encoding="utf-8")


def run_anomalydae(config, output: Path):
    settings = config["anomalydae"]
    gpu = _resolve_gpu(int(config["execution"].get("gpu", 0)))
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    repeats = int(settings.get("repeats", 3))
    rows = []
    for n_value in settings["microbenchmark_nodes"]:
        n = int(n_value)
        torch.manual_seed(44)
        z = torch.randn(n, int(settings["hidden_dim"]), device=device, requires_grad=True)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        if gpu >= 0:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(gpu)
        with torch.no_grad():
            chunked_exact_row_error(
                z[:min(n, 1024)], torch.empty((2, 0), dtype=torch.long, device=device),
                sigmoid=True, chunk_size=min(256, n),
            )
        warm_rows = torch.arange(min(n, 256), device=device)
        chunked_exact_row_error(
            z, edge_index, rows=warm_rows, sigmoid=True, chunk_size=min(256, n),
        ).mean().backward()
        z.grad = None
        eval_times, train_times = [], []
        score = None
        for _ in range(repeats):
            if gpu >= 0: torch.cuda.synchronize(gpu)
            started = time.perf_counter()
            with torch.no_grad():
                score = chunked_exact_row_error(
                    z, edge_index, sigmoid=True, chunk_size=int(settings["chunk_size"]),
                )
            if gpu >= 0: torch.cuda.synchronize(gpu)
            eval_times.append(time.perf_counter() - started)
            z.grad = None
            if gpu >= 0: torch.cuda.synchronize(gpu)
            started = time.perf_counter()
            for start in range(0, n, int(settings["chunk_size"])):
                selected = torch.arange(start, min(start + int(settings["chunk_size"]), n), device=device)
                chunked_exact_row_error(
                    z, edge_index, rows=selected, sigmoid=True,
                    chunk_size=int(settings["chunk_size"]),
                ).sum().div(n).backward()
            if gpu >= 0: torch.cuda.synchronize(gpu)
            train_times.append(time.perf_counter() - started)
        pairs = n * n
        eval_runtime = float(np.median(eval_times))
        train_runtime = float(np.median(train_times))
        rows.append({
            "N": n, "pairs": pairs, "eval_runtime_sec": eval_runtime,
            "train_runtime_sec": train_runtime,
            "eval_pairs_per_second": pairs / eval_runtime,
            "train_pairs_per_second": pairs / train_runtime,
            "repeats": repeats, "score_finite": bool(torch.isfinite(score).all()),
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(gpu) / 2**20 if gpu >= 0 else 0.0,
        })
        del z, score
    measured = pd.DataFrame(rows)
    measured.to_csv(output / "anomalydae/microbenchmark.csv", index=False)
    tail = measured.sort_values("N").tail(2)
    eval_throughput = float(tail.eval_pairs_per_second.median())
    train_throughput = float(tail.train_pairs_per_second.median())
    epochs = int(settings.get("benchmark_epochs", 50))
    inference_passes = int(settings.get("final_inference_passes", 1))
    shapes = {
        "DGraphFin": 1_225_601, "Yelp-Syn": 716_847, "Reddit-Syn": 232_965,
    }
    estimates = []
    for dataset, n in shapes.items():
        train_seconds = epochs * n*n / train_throughput
        inference_seconds = inference_passes * n*n / eval_throughput
        seconds = train_seconds + inference_seconds
        estimates.append({
            "dataset": dataset, "N": n, "pairs": n*n,
            "measured_eval_pairs_per_sec": eval_throughput,
            "measured_train_pairs_per_sec": train_throughput,
            "epochs": epochs, "final_inference_passes": inference_passes,
            "estimated_training_sec": train_seconds,
            "estimated_inference_sec": inference_seconds,
            "estimated_runtime_sec": seconds,
            "estimated_gpu_hours": seconds/3600,
            "classification": classify_exact_runtime(seconds, prohibitive_hours=float(settings["prohibitive_gpu_hours"])),
            "reason": "nonlinear all-pairs decoder complexity",
        })
    estimate_frame = pd.DataFrame(estimates)
    estimate_frame.to_csv(output / "anomalydae/feasibility.csv", index=False)
    estimate_frame.to_csv(output / "tables/table_r4b_c_anomalydae_feasibility.csv", index=False)
    (output / "manifests/anomalydae_policy.json").write_text(json.dumps({
        "selected_option": "large_graph_algorithmic_exclusion" if estimate_frame.classification.eq("unsupported_algorithmic").any() else "full_inclusion",
        "threshold_gpu_hours": float(settings["prohibitive_gpu_hours"]),
        "benchmark_epochs": epochs, "final_inference_passes": inference_passes,
        "approximate_primary_allowed": False, "node_subsampling_allowed": False,
        "status_for_excluded": "unsupported_algorithmic",
    }, indent=2)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.loglog(measured.N, measured.eval_runtime_sec, "o-", label="measured exact inference")
    ax.loglog(measured.N, measured.train_runtime_sec, "s-", label="measured exact train pass")
    extended_n = np.array([10_000,20_000,40_000,80_000,232_965,716_847,1_225_601], dtype=float)
    projected = epochs*extended_n**2/train_throughput + inference_passes*extended_n**2/eval_throughput
    ax.loglog(extended_n, projected, "--", label=f"{epochs}-epoch exact projection")
    ax.axhline(float(settings["prohibitive_gpu_hours"])*3600, color="r", linestyle=":", label="24 GPU-hour policy")
    ax.set(xlabel="Nodes N", ylabel="Exact decoder seconds", title="AnomalyDAE exact nonlinear complexity")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "figures/05_anomalydae_complexity_scaling.png", dpi=180); plt.close(fig)


def _placeholder(path: Path, title: str, reason: str):
    fig, ax = plt.subplots(figsize=(7,3)); ax.axis("off")
    ax.text(.5,.63,title,ha="center",fontsize=14,weight="bold"); ax.text(.5,.34,reason,ha="center",wrap=True)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def finalize(config, output: Path):
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output/"large_graph").glob("*.json"))]
    large = pd.DataFrame(records)
    large.to_csv(output / "resources/large_graph_profiles.csv", index=False)
    large.to_csv(output / "tables/table_r4b_b_large_graph_resource.csv", index=False)
    expected = {(d,m) for d in LARGE_DATASETS for m in SUPPORTED_LARGE_MODELS}
    success = {(r.get("dataset"),r.get("model")) for r in records if r.get("status")=="success"}
    twelve_pass = expected == success

    anomaly_path = output / "anomalydae/feasibility.csv"
    anomaly = pd.read_csv(anomaly_path) if anomaly_path.exists() else pd.DataFrame()
    display_datasets = ["Elliptic","DGraphFin","Yelp-Syn","Amazon-Syn","Flickr-Syn","Reddit-Syn","Cora-Syn","CiteSeer-Syn","PubMed-Syn","BitcoinOTC"]
    models = ["DOMINANT","AnomalyDAE","CoLA","CONAD","GADNR","OCGNN","DLG-Base","DLG-Aug"]
    resource_map = {(r.get("display_name"),r.get("model")): r for r in records}
    unsupported = set(anomaly.loc[anomaly.classification.eq("unsupported_algorithmic"),"dataset"]) if not anomaly.empty else set()
    cells = []
    for dataset in display_datasets:
        for model in models:
            record = resource_map.get((dataset,model))
            if model == "AnomalyDAE" and dataset in unsupported:
                cell = SupportCell(dataset,model,True,False,"nonlinear all-pairs decoder complexity",False,"unsupported_algorithmic")
            elif record:
                ok = record.get("status")=="success"
                cell = SupportCell(dataset,model,True,ok,None if ok else record.get("error_message","execution failure"),ok,record.get("status","failed_data"))
            else:
                cell = SupportCell(dataset,model,True,False,"not attempted in Round 4B representative production pilot",False,"not_attempted")
            cells.append(cell.to_dict())
    support = pd.DataFrame(cells)
    support.to_csv(output / "support_matrix/model_dataset_support_matrix.csv", index=False)
    support_table = support.rename(columns={
        "full_graph_feasible":"primary_supported", "reason_if_not":"reason",
        "exact_backend_available":"exact",
    }).copy()
    support_table["approximation"] = False
    support_table.to_csv(output / "tables/table_r4b_f_support_matrix.csv", index=False)

    closure_specs = [
        *[("Yelp", model, seed) for model in ("DOMINANT","CONAD","DLG-Base") for seed in (42,43)],
        *[("Reddit","CONAD",seed) for seed in (42,43)],
    ]
    record_keys = {(r.get("dataset"),r.get("model"),int(r.get("seed",-1))):r for r in records}
    closure_rows = []
    for dataset, model, seed in closure_specs:
        r = record_keys.get((dataset,model,seed), {})
        closure_rows.append({
            "dataset": dataset+"-Syn", "model":model, "seed":seed,
            "historical_issue": "Round 4A COO message OOM/CUDA failure",
            "root_cause": "COO edge-expanded message materialization",
            "fix": "normalized SparseTensor fused SpMM in a fresh subprocess",
            "final_status": r.get("status","not_attempted"),
        })
    closure = pd.DataFrame(closure_rows)
    closure.to_csv(output/"failures/historical_failure_closure.csv",index=False)
    closure.to_csv(output/"tables/table_r4b_d_failure_closure.csv",index=False)

    component_docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output/"ablation").glob("*.json"))]
    component_rows = [row for doc in component_docs for row in doc.get("rows",[])]
    component_raw = pd.DataFrame(component_rows)
    component_raw.to_csv(output/"ablation/component_raw.csv",index=False)
    required_component = {(d,s) for d in config["component"]["datasets"] for s in config["component"]["seeds"]}
    successful_component = {(d.get("dataset"),int(d.get("seed",-1))) for d in component_docs if d.get("status")=="success"}
    component_pass = required_component.issubset(successful_component) and len(component_raw)==len(required_component)*4
    if not component_raw.empty:
        summary = component_raw.groupby(["dataset","variant"],as_index=False).agg(
            pr_auc_mean=("pr_auc","mean"),pr_auc_std=("pr_auc","std"),
            validation_f1_mean=("validation_f1","mean"),validation_f1_std=("validation_f1","std"),
            roc_auc_mean=("roc_auc","mean"),roc_auc_std=("roc_auc","std"),
        )
        wide = summary.pivot(index="dataset",columns="variant",values="pr_auc_mean").reset_index()
        for name in ("DLG-Base","DLG-Local","DLG-Aug","DLG-Fusion"):
            if name not in wide: wide[name]=np.nan
        wide["delta_aug"] = wide["DLG-Aug"]-wide["DLG-Base"]
        wide["delta_fusion"] = wide["DLG-Fusion"]-wide["DLG-Aug"]
        wide["delta_local"] = wide["DLG-Local"]-wide["DLG-Base"]
        wide.to_csv(output/"tables/table_r4b_e_dlg_components.csv",index=False)
        for metric,filename,title in (
            ("pr_auc_mean","06_dlg_component_pr_auc.png","DLG component PR-AUC (2-seed mean)"),
            ("validation_f1_mean","07_dlg_component_f1.png","DLG component validation-selected F1 (2-seed mean)"),
        ):
            plot=summary.pivot(index="dataset",columns="variant",values=metric)
            ax=plot.plot.bar(figsize=(9,4)); ax.set(title=title,ylabel=metric)
            ax.figure.tight_layout(); ax.figure.savefig(output/"figures"/filename,dpi=180); plt.close(ax.figure)
    else:
        pd.DataFrame([{"status":"not_run","reason":"component results unavailable"}]).to_csv(output/"tables/table_r4b_e_dlg_components.csv",index=False)
        for filename,title in (("06_dlg_component_pr_auc.png","DLG component PR-AUC"),("07_dlg_component_f1.png","DLG component F1")):
            _placeholder(output/"figures"/filename,title,"component results unavailable")

    representative_path = output/"representative/representative_raw.csv"
    if representative_path.exists():
        representative = pd.read_csv(representative_path)
    else:
        representative = pd.DataFrame([{
            "status":"not_attempted", "reason":"production-equivalent supported-cell matrix not executed",
        }])
    representative.to_csv(output/"tables/table_r4b_g_representative_pilot.csv",index=False)

    if not large.empty:
        plot = large.loc[large.status.eq("success")].copy()
        if not plot.empty:
            for column,filename,title,ylabel in (
                ("peak_nvidia_smi_used_mb","01_coo_vs_sparse_message_memory.png","Sparse-fused measured GPU memory","MiB"),
                ("runtime","02_coo_vs_sparse_message_runtime.png","Sparse-fused runtime","seconds"),
            ):
                pivot=plot.pivot_table(index="display_name",columns="model",values=column,aggfunc="mean")
                ax=pivot.plot.bar(figsize=(8,4)); ax.set(title=title,ylabel=ylabel); ax.figure.tight_layout(); ax.figure.savefig(output/"figures"/filename,dpi=180); plt.close(ax.figure)
    for filename,title in (("01_coo_vs_sparse_message_memory.png","COO vs sparse memory"),("02_coo_vs_sparse_message_runtime.png","COO vs sparse runtime")):
        if not (output/"figures"/filename).exists(): _placeholder(output/"figures"/filename,title,"No successful large-graph profiles")
    matrix = support.pivot(index="dataset",columns="model",values="status")
    codes = matrix.map(lambda v:{"success":2,"unsupported_algorithmic":1,"not_attempted":0}.get(v,-1))
    fig,ax=plt.subplots(figsize=(10,5)); image=ax.imshow(codes,cmap="RdYlGn",vmin=-1,vmax=2,aspect="auto")
    ax.set_xticks(range(len(codes.columns)),codes.columns,rotation=45,ha="right"); ax.set_yticks(range(len(codes.index)),codes.index)
    ax.set_title("Model-dataset exact support status"); fig.tight_layout(); fig.savefig(output/"figures/08_model_dataset_support_matrix.png",dpi=180); plt.close(fig)

    equivalence_path=output/"manifests/message_equivalence_gate.json"
    equivalence=json.loads(equivalence_path.read_text()) if equivalence_path.exists() else {"pass":False}
    yelp_closure = bool((closure.loc[closure.dataset.eq("Yelp-Syn"),"final_status"]=="success").all())
    reddit_closure = bool((closure.loc[closure.dataset.eq("Reddit-Syn"),"final_status"]=="success").all())
    representative_pass = (
        not representative.empty and "dataset" in representative and
        bool(representative.status.isin(["success","unsupported_algorithmic"]).all()) and
        bool(representative.status.eq("success").any())
    )
    decision="NOT_READY"
    gates={
        "sparse_message_equivalence":bool(equivalence.get("pass")),
        "large_graph_12_run":twelve_pass,
        "anomalydae_policy":bool((output/"manifests/anomalydae_policy.json").exists()),
        "yelp_closure":yelp_closure,"reddit_conad_closure":reddit_closure,
        "dlg_component":component_pass,
        "representative_supported_cells":representative_pass,
        "statistics_policy":True,"runtime_estimate":False,
    }
    (output/"manifests/readiness_gate.json").write_text(json.dumps({
        "decision":decision,"gates":gates,"large_graph_success":len(success),"large_graph_expected":len(expected),
        "missing_or_failed":sorted([list(x) for x in expected-success]),
        "component_success":len(successful_component),"component_expected":len(required_component),
        "blocking_reasons":[
            "representative production-equivalent supported-cell matrix is not complete",
            "Round 5 runtime estimate is unavailable without that matrix",
        ],
        "round5_protocol_created":False,"round5_auto_started":False,
    },indent=2)+"\n",encoding="utf-8")
    (output/"manifests/statistics_policy.yaml").write_text(yaml.safe_dump({
        "performance_view":{"complete_cases_only":True,"unsupported_imputation":False,
                            "seed_aggregation_first":True,"tests":["friedman","paired_wilcoxon_holm"]},
        "scalability_view":{"separate_from_performance_rank":True,
                            "fields":["status","runtime","memory","reason"]},
    },sort_keys=False),encoding="utf-8")
    (output/"resources/round5_runtime_estimate.json").write_text(json.dumps({
        "status":"not_available","reason":"representative production-equivalent supported-cell matrix is not complete",
        "forbidden_method":"one_epoch_times_50",
    },indent=2)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    parser.add_argument("--stage",required=True,choices=("equivalence","large","anomalydae","component","finalize"))
    parser.add_argument("--dataset"); parser.add_argument("--model"); parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output=Path(config["experiment"]["output_root"]); ensure_layout(output)
    if args.stage=="equivalence":run_equivalence(output)
    elif args.stage=="large":
        if not args.dataset or not args.model:parser.error("large requires dataset and model")
        run_large_one(config,output,args.dataset,args.model,args.seed)
    elif args.stage=="anomalydae":run_anomalydae(config,output)
    elif args.stage=="component":
        if not args.dataset:parser.error("component requires dataset")
        run_component_one(config,output,args.dataset,args.seed)
    else:finalize(config,output)
    (output/"manifests/config_snapshot.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    (output/"manifests/environment.json").write_text(json.dumps({
        "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    },indent=2)+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":raise SystemExit(main())
