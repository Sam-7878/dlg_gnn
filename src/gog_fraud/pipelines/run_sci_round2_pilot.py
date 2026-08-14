"""Round-2 scientific-validity gate and representative pilot orchestrator."""
from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from analysis.utils import dataset_metadata
from gog_fraud.evaluation.graph_conventions import topology_convention
from gog_fraud.evaluation.partition_fidelity import audit_contiguous_partition
from gog_fraud.evaluation.score_semantics import SCORE_SEMANTICS
from gog_fraud.experiments.round2_validity import VARIANT_IDENTITIES, graph_fingerprints
from gog_fraud.pipelines.run_sci_round1_ablation import run as run_ablation
from gog_fraud.pipelines.run_sci_round1_benchmark import _legacy_registries, run as run_benchmark

log = logging.getLogger(__name__)
LAYOUT = ("manifests", "multiseed", "score_semantics", "topology", "partition", "ablation", "fusion", "resources", "tables", "figures", "logs", "injection")


def _layout(root: Path) -> None:
    for name in LAYOUT: (root / name).mkdir(parents=True, exist_ok=True)


def _gpu_readiness(requested: int = 0) -> dict[str, Any]:
    record = {
        "torch_cuda_available": bool(torch.cuda.is_available()), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "requested_gpu": requested, "cuda_usable": False,
        "gpu_model": None, "cpu_fallback_reason": None,
    }
    if not torch.cuda.is_available():
        record["cpu_fallback_reason"] = "torch.cuda.is_available() is false"; return record
    try:
        record["gpu_model"] = torch.cuda.get_device_name(requested)
        probe = torch.ones(1024, device=f"cuda:{requested}"); probe.square_(); torch.cuda.synchronize(requested)
        del probe; record["cuda_usable"] = True
    except Exception as exc:
        record["cpu_fallback_reason"] = f"{type(exc).__name__}: {exc}"
    return record


def run_preflight(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    evaluation = config.get("evaluation", {}); dataset_seed = int(evaluation.get("dataset_seed", 42))
    registry, _ = _legacy_registries(config["data"]["root"], dataset_seed)
    datasets = config.get("datasets", list(registry)); partition_sizes = evaluation.get("partition_sizes", {})
    integrity_rows, convention_rows, partition_rows, failures = [], [], [], []
    for name in datasets:
        log.info("[PREFLIGHT] loading %s", name)
        try:
            data = registry[name]()
            if data is None: raise RuntimeError("loader returned None")
            y = data.y.detach().cpu().numpy().reshape(-1)
            eligible = np.isin(y, (0, 1)); positive = int((y[eligible] == 1).sum())
            fingerprints = graph_fingerprints(data, injection_config={"dataset_seed": dataset_seed})
            integrity_rows.append({
                "dataset": name, **dataset_metadata(name), "nodes": int(data.num_nodes), "edges": int(data.num_edges),
                "positive": positive, "negative": int(eligible.sum() - positive),
                "positive_ratio": float(positive / eligible.sum()) if eligible.sum() else np.nan,
                "dataset_seed": dataset_seed, **fingerprints,
            })
            convention_rows.append(topology_convention(name, data.edge_index, int(data.num_nodes)))
            size = int(partition_sizes.get(name, 0))
            if size and data.num_nodes > size:
                fidelity = audit_contiguous_partition(data.edge_index, data.y, size, directed=True)
                partition_rows.append({"dataset": name, **fidelity.to_dict()})
        except Exception as exc:
            failures.append({"dataset": name, "type": type(exc).__name__, "message": str(exc)})
            log.exception("preflight failed: %s", name)
    pd.DataFrame(integrity_rows).to_csv(output_root / "tables/table_r2_a_dataset_integrity_preflight.csv", index=False)
    pd.DataFrame(convention_rows).to_csv(output_root / "topology/directed_undirected_conventions.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(output_root / "partition/partition_fidelity.csv", index=False)
    gpu = _gpu_readiness(int(evaluation.get("gpu", 0)))
    (output_root / "resources/gpu_readiness.json").write_text(json.dumps(gpu, indent=2) + "\n", encoding="utf-8")
    identities = [identity.to_dict() for identity in VARIANT_IDENTITIES.values()]
    (output_root / "manifests/dlg_variant_identities.json").write_text(json.dumps(identities, indent=2) + "\n", encoding="utf-8")
    semantics = [contract.to_dict() for contract in SCORE_SEMANTICS.values()]
    pd.DataFrame(semantics).to_csv(output_root / "score_semantics/detector_contracts.csv", index=False)
    result = {"datasets_ok": len(integrity_rows), "datasets_failed": failures, "gpu": gpu, "partition_audits": len(partition_rows)}
    (output_root / "manifests/preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_injection_sensitivity(config: dict[str, Any], output_root: Path) -> None:
    dataset = config.get("injection_sensitivity", {}).get("dataset", "Cora")
    dataset_seeds = config.get("injection_sensitivity", {}).get("dataset_seeds", [41, 42, 43])
    model_seeds = config.get("injection_sensitivity", {}).get("model_seeds", [42, 43])
    rows = []
    for dataset_seed in dataset_seeds:
        nested = json.loads(json.dumps(config)); nested["datasets"] = [dataset]
        nested["models"] = [config.get("injection_sensitivity", {}).get("model", "DLG")]
        nested["evaluation"]["dataset_seed"] = int(dataset_seed)
        nested["evaluation"]["seeds"] = model_seeds
        target = output_root / f"injection/dataset_seed_{dataset_seed}"
        run_benchmark(nested, output_root=target, resume=True, force=False)
        raw = pd.read_csv(target / "multiseed/raw_results.csv"); raw["dataset_seed"] = dataset_seed; rows.append(raw)
    combined = pd.concat(rows, ignore_index=True); combined.to_csv(output_root / "injection/injection_seed_sensitivity.csv", index=False)
    summary = combined.loc[combined.status.eq("success")].groupby("dataset_seed", as_index=False).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"), model_seed_variation=("pr_auc", "std"),
        dataset_hash=("dataset_hash", "first"), label_hash=("label_hash", "first"), edge_hash=("edge_hash", "first"))
    summary.to_csv(output_root / "injection/injection_seed_summary.csv", index=False)


def run_partition_sensitivity(config: dict[str, Any], output_root: Path) -> None:
    sensitivity = config.get("partition_sensitivity", {})
    dataset, sizes = sensitivity.get("dataset", "Reddit"), sensitivity.get("sizes", [2048, 4096, 8192])
    rows = []
    for size in sizes:
        nested = json.loads(json.dumps(config)); nested["datasets"] = [dataset]
        nested["models"] = sensitivity.get("models", ["DOMINANT", "DLG"])
        nested["evaluation"]["seeds"] = sensitivity.get("seeds", [42])
        nested["evaluation"]["partition_sizes"] = {dataset: int(size)}
        target = output_root / f"partition/sensitivity_{size}"
        run_benchmark(nested, output_root=target, resume=True, force=False)
        raw = pd.read_csv(target / "multiseed/raw_results.csv"); raw["partition_size"] = size; rows.append(raw)
    pd.concat(rows, ignore_index=True).to_csv(output_root / "partition/partition_sensitivity_performance.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-root")
    parser.add_argument("--stages", nargs="+", choices=("preflight", "benchmark", "ablation", "injection", "partition_sensitivity"), default=["preflight", "benchmark", "ablation", "injection", "partition_sensitivity"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    output = Path(args.output_root or config.get("experiment", {}).get("output_root", "outputs/sci_round2_pilot")); _layout(output)
    if "preflight" in args.stages: run_preflight(config, output)
    if "benchmark" in args.stages: run_benchmark(config, output_root=output, resume=args.resume, force=False)
    if "ablation" in args.stages:
        ablation = config.get("component_pilot", {}); run_ablation(config, output, datasets=ablation.get("datasets"), seeds=ablation.get("seeds"))
    if "injection" in args.stages: run_injection_sensitivity(config, output)
    if "partition_sensitivity" in args.stages: run_partition_sensitivity(config, output)
    return 0


if __name__ == "__main__": raise SystemExit(main())

