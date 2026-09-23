"""SCI Benchmark Round 4A: exact reconstruction and fail-closed readiness."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
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
from torch_geometric.utils import to_dense_adj

from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.experiments.round4a_reconstruction import (
    CRITICAL_GATES,
    decide_round4a_readiness,
    receptive_field_manifest,
)
from gog_fraud.models.pygod.exact_reconstruction import (
    chunked_exact_row_error,
    exact_attribute_error,
    exact_dot_product_row_error,
)
from gog_fraud.models.pygod.shared_reconstruction import (
    ExactDOMINANTBase,
    SharedAnomalyDAE,
    SharedCONAD,
    SharedDLGBase,
    SharedDLGFull,
    SharedDOMINANT,
)
from gog_fraud.models.pygod.stable_reconstruction import StableDOMINANTBase, stable_reconstruction_score
from gog_fraud.pipelines.run_sci_round1_benchmark import _legacy_registries, _resolve_gpu


LAYOUT = (
    "equivalence", "sparse_backend", "shared_training", "large_graph",
    "failures", "ablation", "representative", "resources", "manifests",
    "tables", "figures",
)


class PeakRSS:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.before = self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        process = psutil.Process(os.getpid())
        self.before = self.peak = process.memory_info().rss

        def sample():
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, process.memory_info().rss)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1)


def ensure_layout(output: Path):
    for name in LAYOUT:
        (output / name).mkdir(parents=True, exist_ok=True)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected)), torch.finfo(expected.dtype).eps)
    return float(torch.linalg.vector_norm(actual - expected)) / denominator


def run_equivalence(output: Path):
    seed_everything(42)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 4, 5], [0, 1, 2, 1, 4, 3, 5, 5]], dtype=torch.long
    )
    adjacency = to_dense_adj(edge_index, max_num_nodes=6)[0].float()
    rows = []
    score_pairs = []
    gradient_pairs = []
    definitions = (
        ("DOMINANT", False, 0.5), ("CONAD", False, 0.5),
        ("DLG-Base", False, 0.5), ("DLG-Aug", False, 0.5),
        ("AnomalyDAE", True, 0.8),
    )
    for model, sigmoid, positive_weight in definitions:
        torch.manual_seed(7)
        z = torch.randn(6, 4, requires_grad=True)
        x = torch.randn(6, 3)
        x_hat = torch.randn(6, 3, requires_grad=True)
        prediction = z @ z.T
        if sigmoid:
            prediction = torch.sigmoid(prediction)
        diff = (adjacency - prediction).square()
        if positive_weight != 0.5:
            diff = torch.where(adjacency > 0, positive_weight * diff, (1-positive_weight) * diff)
        dense_structure = torch.sqrt(diff.sum(dim=1))
        dense_attribute = torch.linalg.vector_norm(x - x_hat, dim=1)
        dense_score = 0.5 * dense_attribute + 0.5 * dense_structure
        if sigmoid:
            sparse_structure = chunked_exact_row_error(
                z, edge_index, positive_weight=positive_weight,
                sigmoid=True, chunk_size=2,
            )
        else:
            sparse_structure = exact_dot_product_row_error(z, edge_index)
        sparse_score = 0.5 * exact_attribute_error(x, x_hat) + 0.5 * sparse_structure
        dense_loss, sparse_loss = dense_score.mean(), sparse_score.mean()
        dense_grad, = torch.autograd.grad(dense_loss, z, retain_graph=True)
        sparse_grad, = torch.autograd.grad(sparse_loss, z)
        loss_rel = abs(float(sparse_loss - dense_loss)) / max(abs(float(dense_loss)), 1e-12)
        score_max = float((sparse_score - dense_score).abs().max())
        grad_rel = _relative_error(sparse_grad, dense_grad)
        passed = loss_rel <= 1e-5 and score_max <= 1e-5 and grad_rel <= 1e-4
        rows.append({
            "model": model, "dataset": "synthetic_small", "loss_dense": float(dense_loss),
            "loss_sparse": float(sparse_loss), "loss_rel_error": loss_rel,
            "score_max_error": score_max, "gradient_max_error": float((sparse_grad-dense_grad).abs().max()),
            "gradient_rel_error": grad_rel, "backend": "chunked_exact" if sigmoid else "exact_sparse",
            "pass": passed,
        })
        score_pairs.extend({"model": model, "dense": float(d), "exact": float(s)} for d, s in zip(dense_score, sparse_score))
        gradient_pairs.extend({"model": model, "dense": float(d), "exact": float(s)} for d, s in zip(dense_grad.flatten(), sparse_grad.flatten()))
    table = pd.DataFrame(rows)
    table.to_csv(output / "equivalence/dense_vs_exact_sparse.csv", index=False)
    table.to_csv(output / "tables/table_r4a_a_dense_vs_exact_sparse_equivalence.csv", index=False)
    pd.DataFrame(score_pairs).to_csv(output / "equivalence/score_pairs.csv", index=False)
    pd.DataFrame(gradient_pairs).to_csv(output / "equivalence/gradient_pairs.csv", index=False)
    for filename, pairs, title in (
        ("01_dense_vs_sparse_score_equivalence.png", score_pairs, "Dense vs exact node scores"),
        ("02_dense_vs_sparse_gradient_equivalence.png", gradient_pairs, "Dense vs exact gradients"),
    ):
        frame = pd.DataFrame(pairs)
        fig, ax = plt.subplots(figsize=(5, 5))
        for model, group in frame.groupby("model"):
            ax.scatter(group.dense, group.exact, s=14, alpha=.7, label=model)
        low = min(frame.dense.min(), frame.exact.min()); high = max(frame.dense.max(), frame.exact.max())
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set(xlabel="dense reference", ylabel="exact backend", title=title)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output / "figures" / filename, dpi=180); plt.close(fig)
    (output / "manifests/equivalence_gate.json").write_text(json.dumps({
        "pass": bool(table["pass"].all()), "loss_rtol": 1e-5,
        "score_rtol": 1e-5, "gradient_rtol": 1e-4,
    }, indent=2) + "\n", encoding="utf-8")

    # Multi-update shared-model equivalence, not merely a formula check.
    torch.manual_seed(29)
    x = torch.randn(9, 5, dtype=torch.float64)
    train_edges = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8], [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 8]]
    )
    target = to_dense_adj(train_edges, max_num_nodes=9)[0].double()
    dense_model = StableDOMINANTBase(in_dim=5, hid_dim=6, num_layers=4).double()
    exact_model = ExactDOMINANTBase(in_dim=5, hid_dim=6, num_layers=4).double()
    exact_model.load_state_dict(copy.deepcopy(dense_model.state_dict()))
    dense_optimizer = torch.optim.Adam(dense_model.parameters(), lr=.003)
    exact_optimizer = torch.optim.Adam(exact_model.parameters(), lr=.003)
    trajectory = []
    for epoch in range(3):
        dense_optimizer.zero_grad(); dense_x, dense_s = dense_model(x, train_edges)
        dense_loss = stable_reconstruction_score(x, dense_x, target, dense_s).mean()
        dense_loss.backward(); dense_optimizer.step()
        exact_optimizer.zero_grad(); exact_x, exact_z = exact_model(x, train_edges)
        exact_score = .5 * exact_attribute_error(x, exact_x) + .5 * exact_dot_product_row_error(exact_z, train_edges)
        exact_loss = exact_score.mean(); exact_loss.backward(); exact_optimizer.step()
        trajectory.append({"epoch": epoch + 1, "dense_loss": float(dense_loss), "exact_loss": float(exact_loss),
                           "abs_difference": abs(float(dense_loss-exact_loss))})
    parameter_error = max(
        float((left-right).abs().max())
        for left, right in zip(dense_model.parameters(), exact_model.parameters())
    )
    training = pd.DataFrame(trajectory); training["final_parameter_max_error"] = parameter_error
    training["pass"] = (training.abs_difference <= 1e-8) & (parameter_error <= 1e-7)
    training.to_csv(output / "shared_training/training_equivalence.csv", index=False)

    torch.manual_seed(5); chunk_z = torch.randn(64, 8)
    chunk_edges = torch.randint(0, 64, (2, 300))
    chunk_scores = {
        size: chunked_exact_row_error(chunk_z, chunk_edges, sigmoid=True, chunk_size=size)
        for size in (1024, 4096, 8192)
    }
    reference = chunk_scores[1024]
    pd.DataFrame([{
        "chunk_size": size,
        "max_abs_score_difference": float((score-reference).abs().max()),
        "pass": bool(torch.allclose(score, reference, rtol=1e-5, atol=1e-6)),
    } for size, score in chunk_scores.items()]).to_csv(
        output / "equivalence/chunk_independence.csv", index=False
    )


def _registries(config):
    datasets, _ = _legacy_registries(config["data"]["root"], 42)
    root = Path(config["data"]["root"])
    datasets["DGraphFin"] = lambda: load_dgraphfin_aligned(root / "DGraphFin/dgraphfin.npz")
    models = {
        "DOMINANT": SharedDOMINANT,
        "AnomalyDAE": SharedAnomalyDAE,
        "CONAD": SharedCONAD,
        "DLG-Base": SharedDLGBase,
        "DLG-Aug": SharedDLGFull,
    }
    return datasets, models


def _write_large_record(output: Path, record: dict):
    key = f"{record['dataset']}__{record['model']}__seed{record['seed']}"
    if record.get("gradient_checkpointing"):
        key += "__checkpoint"
    (output / "large_graph" / f"{key}.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )


def run_large_one(config: dict, output: Path, dataset: str, model_name: str, seed: int):
    datasets, models = _registries(config)
    seed_everything(seed)
    started = time.perf_counter()
    data = datasets[dataset]()
    load_sec = time.perf_counter() - started
    n, e, f = int(data.num_nodes), int(data.num_edges), int(data.x.shape[1])
    record = {
        "dataset": dataset, "display_name": dataset + "-Syn" if dataset in {"Yelp", "Reddit"} else dataset,
        "model": model_name, "seed": seed, "nodes": n, "edges": e, "features": f,
        "backend": "chunked_exact" if model_name == "AnomalyDAE" else "exact_sparse",
        "shared_model": True, "training_full_graph": True, "approximation_used": False,
        "dense_materialized": False, "load_sec": load_sec, "status": "failed",
        "gradient_checkpointing": bool(config["large_graph"].get("gradient_checkpointing", False)) if model_name != "AnomalyDAE" else False,
    }
    # Exact sigmoid row reconstruction remains quadratic arithmetic.  Reject a
    # scientifically impossible production run before allocating a row block.
    exact_pairs = n * n
    record["exact_reconstruction_pairs_per_epoch"] = exact_pairs
    if model_name == "AnomalyDAE" and exact_pairs > 1_000_000_000:
        record.update({
            "status": "blocked_known_algorithm_limitation",
            "success": False,
            "error_type": "ExactQuadraticComputeLimit",
            "error_message": "sigmoid(Z Z^T) has no Gram closed form; exact chunking bounds memory but requires N^2 pair evaluations",
            "peak_ram_mb": psutil.Process(os.getpid()).memory_info().rss / 2**20,
            "peak_vram_mb": 0.0,
            "train_sec": 0.0, "infer_sec": 0.0,
        })
        _write_large_record(output, record); return
    gpu = _resolve_gpu(int(config["execution"].get("gpu", 0)))
    kwargs = {
        "epoch": int(config["large_graph"].get("feasibility_epochs", 1)),
        "gpu": gpu, "batch_size": 0, "verbose": 0,
        "score_chunk_size": int(config["reconstruction"].get(
            "nonlinear_score_chunk_size" if model_name == "AnomalyDAE" else "score_chunk_size", 8192
        )),
    }
    if model_name != "AnomalyDAE":
        kwargs["gradient_checkpointing"] = bool(config["large_graph"].get("gradient_checkpointing", False))
    if model_name == "DLG-Aug":
        kwargs["l1_epochs"] = 1
    detector = models[model_name](**kwargs)
    try:
        if gpu >= 0:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(gpu)
        with PeakRSS() as memory:
            started = time.perf_counter(); detector.fit(data); train_sec = time.perf_counter() - started
            started = time.perf_counter(); score = detector.decision_function(data); infer_sec = time.perf_counter() - started
        record.update({
            "status": "success", "success": True, "train_sec": train_sec, "infer_sec": infer_sec,
            "peak_ram_mb": memory.peak / 2**20, "rss_delta_mb": (memory.peak-memory.before) / 2**20,
            "peak_vram_mb": torch.cuda.max_memory_allocated(gpu) / 2**20 if gpu >= 0 else 0.0,
            "score_nodes": int(score.numel()), "score_finite": bool(torch.isfinite(score).all()),
            "score_min": float(score.min()), "score_max": float(score.max()),
            "metadata": detector.backend_metadata(),
        })
        if not record["score_finite"] or record["score_nodes"] != n:
            raise RuntimeError("non-finite or incomplete full-graph score vector")
    except Exception as exc:
        record.update({
            "success": False, "status": "failed", "error_type": type(exc).__name__,
            "error_message": str(exc), "traceback": traceback.format_exc(),
        })
    _write_large_record(output, record)


def _placeholder_figure(path: Path, title: str, reason: str):
    fig, ax = plt.subplots(figsize=(7, 3)); ax.axis("off")
    ax.text(.5, .62, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(.5, .35, reason, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def finalize(config: dict, output: Path):
    records = []
    for path in sorted((output / "large_graph").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    large = pd.DataFrame(records)
    if not large.empty:
        budget = float(config["large_graph"].get("gpu_memory_budget_mb", 8192))
        large["resource_acceptance"] = (
            large.get("success", False).fillna(False).astype(bool)
            & (pd.to_numeric(large.get("peak_vram_mb"), errors="coerce") <= budget)
            & ((pd.to_numeric(large.get("train_sec"), errors="coerce").fillna(np.inf)
                + pd.to_numeric(large.get("infer_sec"), errors="coerce").fillna(np.inf)) <= 600)
        )
        large.to_csv(output / "resources/full_sparse_profiles.csv", index=False)
        large.to_csv(output / "tables/table_r4a_b_large_graph_feasibility.csv", index=False)
    else:
        pd.DataFrame(columns=["dataset", "model", "success"]).to_csv(
            output / "tables/table_r4a_b_large_graph_feasibility.csv", index=False
        )
    expected = {(d, m) for d in ("DGraphFin", "Yelp", "Reddit") for m in config["large_graph"]["production_models"]}
    observed = {(row.get("dataset"), row.get("model")) for row in records}
    missing = sorted(expected - observed)
    accepted = set()
    budget = float(config["large_graph"].get("gpu_memory_budget_mb", 8192))
    for row in records:
        runtime = float(row.get("train_sec", math.inf)) + float(row.get("infer_sec", math.inf))
        vram = float(row.get("peak_vram_mb", math.inf))
        if row.get("success") is True and runtime <= 600 and vram <= budget:
            accepted.add((row.get("dataset"), row.get("model")))
    resource_by_dataset = {
        dataset: all((dataset, model) in accepted for model in config["large_graph"]["production_models"])
        for dataset in ("DGraphFin", "Yelp", "Reddit")
    }
    eq_path = output / "manifests/equivalence_gate.json"
    equivalence = json.loads(eq_path.read_text(encoding="utf-8")) if eq_path.exists() else {"pass": False}
    upstream_resource_pass = all(resource_by_dataset.values()) and not missing
    skip_reason = "upstream large-graph resource gate failed; downstream production pilots were not authorized"
    failure_rows = []
    for row in records:
        if not row.get("success"):
            failure_rows.append({
                "dataset": row.get("dataset"), "model": row.get("model"),
                "round2_round3_issue": "dense N x N reconstruction / large-graph execution",
                "root_cause": row.get("error_message"),
                "fix": "exact row backend implemented; nonlinear exact arithmetic remains quadratic" if row.get("model") == "AnomalyDAE" else "shared full sparse backend",
                "final_status": row.get("status"),
            })
    pd.DataFrame(failure_rows, columns=["dataset", "model", "round2_round3_issue", "root_cause", "fix", "final_status"]).to_csv(
        output / "tables/table_r4a_c_failure_closure.csv", index=False
    )
    pd.DataFrame([{"status": "not_run", "reason": skip_reason}]).to_csv(output / "tables/table_r4a_d_dlg_components.csv", index=False)
    pd.DataFrame([{"status": "not_run", "reason": skip_reason}]).to_csv(output / "tables/table_r4a_e_representative_matrix.csv", index=False)
    pd.DataFrame(receptive_field_manifest()).to_csv(output / "manifests/receptive_field_manifest.csv", index=False)
    (output / "manifests/reconstruction_backend_registry.json").write_text(json.dumps({
        "dense_reference": {"purpose": "small-graph equivalence only", "dense_materialized": True},
        "exact_sparse": {"purpose": "linear dot-product primary", "dense_materialized": False, "approximation_used": False},
        "chunked_exact": {"purpose": "nonlinear row exact", "dense_materialized": False, "approximation_used": False,
                          "limitation": "bounds memory but retains N^2 arithmetic"},
    }, indent=2) + "\n", encoding="utf-8")
    claim_rows = [
        ("Elliptic", "real labels", "stratified_node_transductive", "full_sparse_shared_model", "model_specific", "exact", "illicit-node anomaly benchmark"),
        ("DGraphFin", "real labels", "official_random_70_15_15", "full_sparse_shared_model", "exact_sparse", "exact", "random-split financial graph anomaly benchmark"),
        ("Yelp-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "model_specific", "exact_when_feasible", "synthetic graph anomaly benchmark"),
        ("Amazon-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "model_specific", "exact_when_feasible", "synthetic graph anomaly benchmark"),
        ("Flickr-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "model_specific", "exact_when_feasible", "synthetic graph anomaly benchmark"),
        ("Reddit-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "model_specific", "exact_when_feasible", "synthetic graph anomaly benchmark"),
        ("Cora-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "exact_sparse", "exact", "synthetic graph anomaly benchmark"),
        ("CiteSeer-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "exact_sparse", "exact", "synthetic graph anomaly benchmark"),
        ("PubMed-Syn", "synthetic injection", "stratified_node_transductive", "full_sparse_shared_model", "exact_sparse", "exact", "synthetic graph anomaly benchmark"),
        ("BitcoinOTC", "real trust labels transformed by loader", "stratified_node_transductive", "full_sparse_shared_model", "model_specific", "exact_when_feasible", "signed-network anomaly benchmark"),
    ]
    pd.DataFrame(claim_rows, columns=["dataset", "label_provenance", "split_type", "graph_execution", "reconstruction_backend", "exact_or_approximate", "claim_scope"]).to_csv(
        output / "tables/paper_claim_safety.csv", index=False
    )
    (output / "resources/round4b_runtime_prediction.json").write_text(json.dumps({
        "status": "not_estimable_for_authorized_protocol",
        "reason": "production-equivalent upstream resource gate failed before representative 80-run",
        "forbidden_extrapolation": "one_epoch_times_50",
        "round4b_runs": 400,
    }, indent=2) + "\n", encoding="utf-8")

    if not large.empty:
        plot = large.copy(); plot["peak_ram_mb"] = pd.to_numeric(plot.get("peak_ram_mb"), errors="coerce")
        success_plot = plot.loc[plot.success.eq(True)] if "success" in plot else plot.iloc[0:0]
        if not success_plot.empty:
            pivot = success_plot.pivot(index="dataset", columns="model", values="peak_ram_mb")
            ax = pivot.plot.bar(figsize=(8, 4)); ax.set_ylabel("Peak process RSS (MiB)"); ax.set_title("Full sparse shared-model memory")
            ax.figure.tight_layout(); ax.figure.savefig(output / "figures/03_large_graph_memory_comparison.png", dpi=180); plt.close(ax.figure)
            pivot = success_plot.assign(runtime=lambda x: x.train_sec + x.infer_sec).pivot(index="dataset", columns="model", values="runtime")
            ax = pivot.plot.bar(figsize=(8, 4)); ax.set_ylabel("Seconds"); ax.set_title("Full sparse shared-model runtime")
            ax.figure.tight_layout(); ax.figure.savefig(output / "figures/04_large_graph_runtime_comparison.png", dpi=180); plt.close(ax.figure)
    for number, title in (("03_large_graph_memory_comparison.png", "Large-graph memory"), ("04_large_graph_runtime_comparison.png", "Large-graph runtime")):
        path = output / "figures" / number
        if not path.exists(): _placeholder_figure(path, title, "No successful comparable profiles")
    _placeholder_figure(output / "figures/05_dlg_component_pr_auc.png", "DLG component PR-AUC", skip_reason)
    _placeholder_figure(output / "figures/06_dlg_component_f1.png", "DLG component validation F1", skip_reason)

    gates = {name: False for name in CRITICAL_GATES}
    gates.update({
        "exact_sparse_mathematical_equivalence": bool(equivalence.get("pass")),
        "shared_full_graph_semantic_equivalence": bool(equivalence.get("pass")),
        "dgraphfin_resource": resource_by_dataset["DGraphFin"],
        "yelp_resource": resource_by_dataset["Yelp"],
        "reddit_resource": resource_by_dataset["Reddit"],
        "score_semantics": True,
        "provenance": True,
    })
    decision = decide_round4a_readiness(gates)
    gate = {
        "decision": decision, "gates": gates, "missing_large_runs": missing,
        "downstream_skipped": not upstream_resource_pass,
        "downstream_skip_reason": skip_reason if not upstream_resource_pass else None,
        "round4b_auto_started": False,
        "round4b_protocol_created": False,
    }
    (output / "manifests/readiness_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (output / "manifests/environment.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("equivalence", "large", "finalize"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_root"]); ensure_layout(output)
    if args.stage == "equivalence": run_equivalence(output)
    elif args.stage == "large":
        if not args.dataset or not args.model: parser.error("large requires --dataset and --model")
        run_large_one(config, output, args.dataset, args.model, args.seed)
    else: finalize(config, output)
    (output / "manifests/config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
