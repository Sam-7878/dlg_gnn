"""Finalize Round 4C accounting and freeze restricted Round 5 policy."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gog_fraud.experiments.round4c_completion import (
    account_cells, complete_case_views, completion_decision,
    frozen_support_matrix, load_classification_ledger,
)
from gog_fraud.experiments.round4c_policy import UNSUPPORTED_STATUSES, canonical_hash
from gog_fraud.pipelines.analyze_sci_round4c import (
    ROUND5_DATASETS, build_data_freeze, build_environment_freeze, collect_results,
)
from gog_fraud.pipelines.run_sci_round4c import _datasets, ensure_layout


FRAUD_REPRESENTATIVE = ["Elliptic", "DGraphFin", "Yelp", "Reddit"]
REMAINING_DATASETS = ["Amazon", "BitcoinOTC", "Flickr", "CiteSeer", "PubMed"]
DISPLAY_NAMES = {
    "Amazon": "Amazon-Syn", "BitcoinOTC": "BitcoinOTC", "Flickr": "Flickr-Syn",
    "CiteSeer": "CiteSeer-Syn", "PubMed": "PubMed-Syn",
}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    selected = selected.fillna("")
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
            for row in selected.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_completion_report(output: Path, gate: dict, accounting: pd.DataFrame,
                            support: pd.DataFrame, views: pd.DataFrame,
                            anomaly: pd.DataFrame, oom: pd.DataFrame,
                            preflight: pd.DataFrame, forecast: dict) -> Path:
    report = Path("docs/work_reports/206_benchmark_round_4c_completion/02_round4c_completion_and_round5_readiness.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    reddit = accounting.loc[accounting.dataset.eq("Reddit")]
    unsupported = support.loc[~support.primary_supported]
    consistency_path = output / "tables/dlg_component_consistency.csv"
    consistency = pd.read_csv(consistency_path) if consistency_path.exists() else pd.DataFrame()
    consistent_count = int(consistency.metric_semantics_consistent.sum()) if "metric_semantics_consistent" in consistency else 0
    consistency_total = len(consistency)
    protocol_created = bool(gate["round5_protocol_created"])
    text = f"""# DLG-GNN SCI Benchmark Round 4C Completion and Round 5 Readiness

## 1. Final decision

**{gate['decision']}**. Nominal {gate['nominal_cells']} cells 중 {gate['accounted_cells']} cells가 success 또는 scientifically classified unsupported로 accounted되었다. Success는 {gate['success_cells']}, classified unsupported는 {gate['classified_unsupported_cells']}, unknown/unaccounted는 {gate['failed_unknown']}이다. Round 5는 자동 실행하지 않았다.

가속화는 기존 success cell을 보존하는 hash-checked resume, fast-to-slow scheduling, 그리고 관측된 최단 epoch로도 guard 내 완주가 불가능함을 보이는 conservative lower-bound decision만 사용했다. AMP, sampling, partition, hidden-dimension 축소 또는 objective 변경은 사용하지 않았다.

## 2. 80-cell accounting

Raw execution observation은 보존하고 `support_reclassification.json` ledger가 final support status와 measured/by-policy 근거를 별도로 고정한다. Unsupported cell에는 score 또는 worst rank를 대입하지 않는다.

{_markdown_table(support, ['dataset','model','seed42_status','seed43_status','primary_supported','restriction_class'])}

## 3. Reddit production results

{_markdown_table(reddit, ['model','seed','observed_status','final_status','evidence_mode','accounted'])}

## 4. AnomalyDAE operational classification

Decoder-only microbenchmark보다 end-to-end production evidence를 우선한다. 즉시 OOM runtime은 completed runtime으로 해석하지 않는다. DGraphFin은 4 epoch/63,064초의 실측에 더해 관측된 최단 epoch 135분을 남은 46 epoch에 적용한 낙관적 하한만 103.5시간이므로, 누적 시간이 24-hour guard 안에서 50 epoch에 도달할 수 없다는 decision-complete evidence로 분류했다.

{_markdown_table(anomaly, ['dataset','estimated_gpu_hours','observed_runtime_sec','observed_epochs','production_projection_hours','actual_to_estimated_ratio','final_support_status'])}

## 5. GADNR support classification

Historical architecture, hidden dimension, neighbor policy를 바꾸지 않았다. 동일 exact production path의 반복 materialization OOM은 `unsupported_resource_exact_implementation`으로 분류한다.

아래 allocated/reserved 수치는 PyTorch가 WSL unified/virtual allocator 문구로 보고한 값이며 물리 GPU 용량으로 해석하지 않는다. 물리 장치는 8 GB-class이고, scientific classification은 requested allocation, traceback stage와 seed 재현성에 근거한다.

{_markdown_table(oom, ['dataset','seed','oom_stage','requested_allocation_gib','current_allocated_gib','current_reserved_unallocated_gib','N','E'])}

## 6. Final support matrix

`primary_supported=true`는 두 production seed가 모두 성공한 경우뿐이다. Restricted cells는 performance comparison에서 missing이며 scalability table에서 restriction으로 보고한다. Restricted model-dataset pairs: {len(unsupported)}.

Reddit CONAD처럼 1-epoch gate는 통과했지만 50-epoch exact production에서 두 seed 모두 동일 allocation OOM이 재현된 경우도 hidden dimension·sampling·objective를 바꾸지 않고 current exact implementation resource restriction으로 기록한다.

## 7. Complete-case statistical views

{_markdown_table(views, ['view_name','models','datasets','n_models','n_datasets','reason'])}

All-8-model block가 너무 작으면 main Friedman test로 사용하지 않는다. Seed를 dataset/model 내에서 먼저 집계하고 complete blocks에서만 Friedman, Wilcoxon signed-rank, Holm correction을 적용한다.

## 8. Runtime forecast

지원된 measured cells와 nearest N/E conservative extrapolation을 분리했다. Round 5 5-seed scheduling은 pessimistic estimate를 사용한다.

- optimistic: {forecast['optimistic_sec'] / 3600:.2f} GPU-hours
- median: {forecast['median_sec'] / 3600:.2f} GPU-hours
- pessimistic: {forecast['pessimistic_sec'] / 3600:.2f} GPU-hours
- measured/extrapolated model-dataset estimates: {forecast['measured_cells']}/{forecast['extrapolated_cells']}
- excluded restricted representative cells: {forecast['excluded_operational_cells']}

## 9. DLG consistency

Round 4B component rows와 Round 4C DLG-Base/DLG-Aug rows는 absolute PR-AUC 및 validation-F1 difference ≤ 0.02 criterion으로 {consistent_count}/{consistency_total} rows가 consistent하다. DLG-Fusion은 extended component analysis이며 historical main `DLG` alias는 DLG-Aug이다. Dataset-dependent component pattern은 descriptive association이지 causal mechanism claim이 아니다.

## 10. Remaining five-dataset preflight

{_markdown_table(preflight, ['dataset','N','E','F','label_ratio','label_provenance','expected_sparse_adjacency_bytes'])}

Long-limit model status는 N/E evidence에 따라 `likely_supported`, `requires_preflight`, `likely_unsupported`만 표시하며 실제 evidence 없이 unsupported로 확정하지 않는다.

## 11. Data/environment freeze

10개 registry dataset의 feature/edge/label/injection hash와 code/config/backend provenance를 다시 freeze했다. 환경에는 Python, PyTorch, PyG, PyGOD, torch-sparse, CUDA, GPU, WSL 정보를 기록했다.

## 12. Exact Round 5 protocol

Round 5 protocol created: **{str(protocol_created).lower()}**. Protocol은 frozen support matrix, seeds 42–46, fixed synthetic seed 42, fresh subprocess, full shared graph, sparse fused message passing, exact sparse/chunked-exact reconstruction, validation-selected threshold, complete-case statistics와 no worst-rank imputation을 고정한다. Round 5 benchmark 자체는 이 task에서 시작하지 않았다.

## Evidence paths

- `outputs/sci_round4c/manifests/readiness_gate.json`
- `outputs/sci_round4c/manifests/support_reclassification.json`
- `outputs/sci_round4c/tables/production_cell_accounting_final.csv`
- `outputs/sci_round4c/tables/model_dataset_support_matrix_final.csv`
- `outputs/sci_round4c/tables/statistical_complete_case_views.csv`
- `outputs/sci_round4c/tables/anomalydae_estimated_vs_actual_final.csv`
- `outputs/sci_round4c/tables/gadnr_oom_root_cause.csv`
- `outputs/sci_round4c/resources/round5_runtime_forecast_final.json`
- `outputs/sci_round4c/freeze/round5_data_and_code_freeze.json`
"""
    report.write_text(text, encoding="utf-8")
    return report


def _memory_gib(pattern: str, value: str) -> float | None:
    match = re.search(pattern + r"\s*([0-9.]+)\s*(GiB|MiB)", value or "", flags=re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    return amount if match.group(2).lower() == "gib" else amount / 1024.0


def _oom_stage(model: str, message: str) -> str:
    lowered = (message or "").lower()
    if model == "AnomalyDAE":
        return "encoder_gat_coo_aggregation"
    if "linalg_inv" in lowered or "linalg_solve" in lowered:
        return "neighborhood_covariance_inverse"
    return "native_neighborhood_or_sage_materialization"


def gadnr_oom_audit(raw: pd.DataFrame, freeze: dict) -> pd.DataFrame:
    graph = {row["dataset"]: row for row in freeze["datasets"]}
    rows = []
    selected = raw.loc[raw.model.eq("GADNR") & raw.status.eq("failed_oom")]
    dataset_stage = {}
    for dataset, group in selected.groupby("dataset"):
        stages = {_oom_stage("GADNR", str(row.failure_message)) for row in group.itertuples()}
        dataset_stage[dataset] = ("neighborhood_covariance_inverse"
                                  if "neighborhood_covariance_inverse" in stages
                                  else "native_neighborhood_or_sage_materialization")
    for row in selected.itertuples():
        message = str(row.failure_message)
        meta = graph.get(row.dataset, {})
        rows.append({
            "dataset": row.dataset, "seed": int(row.seed), "oom_stage": dataset_stage[row.dataset],
            "requested_allocation_gib": _memory_gib(r"Tried to allocate", message),
            "current_allocated_gib": _memory_gib(r"allocated memory", message),
            "current_reserved_unallocated_gib": _memory_gib(r"and", message),
            "tensor_shape": "not exposed by allocator traceback",
            "N": meta.get("nodes"), "E": meta.get("edges"),
            "observed_status": row.status,
        })
    return pd.DataFrame(rows)


def production_oom_audit(raw: pd.DataFrame, freeze: dict) -> pd.DataFrame:
    graph = {row["dataset"]: row for row in freeze["datasets"]}
    rows = []
    for row in raw.loc[raw.status.eq("failed_oom")].itertuples():
        message = str(row.failure_message)
        meta = graph.get(row.dataset, {})
        if row.model == "CONAD":
            stage = "native_contrastive_augmentation_or_reconstruction_materialization"
        else:
            stage = _oom_stage(row.model, message)
        rows.append({
            "dataset": row.dataset, "model": row.model, "seed": int(row.seed),
            "oom_stage": stage, "requested_allocation_gib": _memory_gib(r"Tried to allocate", message),
            "current_allocated_gib": _memory_gib(r"allocated memory", message),
            "current_reserved_unallocated_gib": _memory_gib(r"and", message),
            "tensor_shape": "not exposed by allocator traceback",
            "N": meta.get("nodes"), "E": meta.get("edges"),
        })
    return pd.DataFrame(rows)


def resource_preflight(config: dict, freeze: dict) -> pd.DataFrame:
    registry = _datasets(config)
    frozen = {row["dataset"]: row for row in freeze["datasets"]}
    rows = []
    for name in REMAINING_DATASETS:
        data = registry[name]()
        labels = data.y.detach().cpu().numpy()
        eligible = labels[np.isin(labels, [0, 1])]
        anomaly_ratio = float((eligible == 1).mean()) if len(eligible) else np.nan
        n, e, f = int(data.num_nodes), int(data.num_edges), int(data.x.shape[1])
        rows.append({
            "dataset": name, "display_name": DISPLAY_NAMES[name], "N": n, "E": e, "F": f,
            "label_ratio": anomaly_ratio,
            "label_provenance": "real labels" if name == "BitcoinOTC" else "synthetic injection",
            "synthetic": name != "BitcoinOTC",
            "expected_sparse_adjacency_bytes": int((n + 1) * 8 + e * (8 + 4)),
            "feature_hash": frozen.get(name, {}).get("feature_hash"),
            "edge_hash": frozen.get(name, {}).get("edge_hash"),
        })
        del data
    return pd.DataFrame(rows)


def long_limit_preclassification(preflight: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in preflight.itertuples():
        if row.N <= 46564 and row.E <= 83188:
            anomaly = "likely_supported"
        else:
            anomaly = "requires_preflight"
        if row.N <= 2708 and row.E <= 11276:
            gadnr = "likely_supported"
        elif row.N >= 46564 or row.E >= 83188:
            gadnr = "likely_unsupported"
        else:
            gadnr = "requires_preflight"
        rows.extend([
            {"dataset": row.dataset, "model": "AnomalyDAE", "preclassification": anomaly,
             "basis": "N/E compared with successful Elliptic and production-limited representative graphs"},
            {"dataset": row.dataset, "model": "GADNR", "preclassification": gadnr,
             "basis": "N/E compared with successful Cora and smallest reproduced production OOM"},
        ])
    return pd.DataFrame(rows)


def runtime_forecast_cells(
    success: pd.DataFrame,
    support: pd.DataFrame,
    freeze: dict,
    preflight: pd.DataFrame,
    models: list[str],
) -> tuple[pd.DataFrame, dict]:
    metadata = {row["dataset"]: {"N": row["nodes"], "E": row["edges"],
                                  "F": row.get("features", np.nan)} for row in freeze["datasets"]}
    metadata.update({row.dataset: {"N": row.N, "E": row.E, "F": row.F}
                     for row in preflight.itertuples()})
    supported_pairs = {(row.dataset, row.model) for row in support.itertuples() if row.primary_supported}
    measured = success.groupby(["dataset", "model"]).total_wall_sec.agg(["min", "median", "max"]).reset_index()
    rows = []
    for row in measured.itertuples():
        if (row.dataset, row.model) in supported_pairs:
            rows.append({"dataset": row.dataset, "model": row.model,
                         "optimistic_sec": row.min, "median_sec": row.median,
                         "pessimistic_sec": row.max, "confidence": "measured",
                         "source_dataset": row.dataset, "planned": True})
    measured_by_model = {model: group for model, group in measured.groupby("model")}
    for dataset in REMAINING_DATASETS:
        target = metadata[dataset]
        for model in models:
            candidates = measured_by_model.get(model)
            if candidates is None or candidates.empty:
                continue
            distances = []
            for candidate in candidates.itertuples():
                source = metadata[candidate.dataset]
                distance = abs(math.log1p(target["N"]) - math.log1p(source["N"]))
                distance += abs(math.log1p(target["E"]) - math.log1p(source["E"]))
                distances.append(distance)
            source_row = candidates.iloc[int(np.argmin(distances))]
            source_meta = metadata[source_row.dataset]
            scale = max(1.0, (target["N"] + target["E"]) / (source_meta["N"] + source_meta["E"]))
            rows.append({"dataset": dataset, "model": model,
                         "optimistic_sec": float(source_row["min"] * scale),
                         "median_sec": float(source_row["median"] * scale),
                         "pessimistic_sec": float(source_row["max"] * scale * 1.5),
                         "confidence": "extrapolated", "source_dataset": source_row.dataset,
                         "planned": True})
    cells = pd.DataFrame(rows)
    planned = cells.loc[cells.planned]
    forecast = {
        "method": "measured two-seed ranges plus conservative nearest-N/E extrapolation",
        "round5_seeds": 5,
        "optimistic_sec": float(planned.optimistic_sec.sum() * 5),
        "median_sec": float(planned.median_sec.sum() * 5),
        "pessimistic_sec": float(planned.pessimistic_sec.sum() * 5),
        "measured_cells": int(cells.confidence.eq("measured").sum()),
        "extrapolated_cells": int(cells.confidence.eq("extrapolated").sum()),
        "scheduling_basis": "pessimistic",
    }
    return cells, forecast


def anomaly_actual_table(raw: pd.DataFrame, ledger: pd.DataFrame, round4b_root: Path) -> pd.DataFrame:
    estimate_path = round4b_root / "anomalydae/feasibility.csv"
    estimates = pd.read_csv(estimate_path)[["dataset", "estimated_gpu_hours"]]
    rows = []
    for dataset in ["DGraphFin", "Yelp-Syn", "Reddit-Syn"]:
        raw_name = {"Yelp-Syn": "Yelp", "Reddit-Syn": "Reddit"}.get(dataset, dataset)
        observed = raw.loc[(raw.dataset.eq(raw_name)) & raw.model.eq("AnomalyDAE")]
        classified = ledger.loc[(ledger.dataset.eq(raw_name)) & ledger.model.eq("AnomalyDAE")]
        measured = classified.loc[classified.evidence_mode.eq("measured")]
        observed_runtime = float(observed.total_wall_sec.max()) if not observed.empty else np.nan
        observed_epochs = np.nan
        projection = np.nan
        if not measured.empty:
            observed_runtime = float(measured.iloc[0].get("observed_runtime_sec", observed_runtime))
            observed_epochs = measured.iloc[0].get("observed_epochs", np.nan)
            projection = measured.iloc[0].get("production_projection_hours", np.nan)
        status = classified.final_status.iloc[0] if not classified.empty else (
            observed.status.iloc[0] if not observed.empty else "not_attempted")
        reason = classified.restriction_reason.iloc[0] if not classified.empty else (
            observed.failure_message.iloc[0] if not observed.empty else "")
        estimated = float(estimates.loc[estimates.dataset.eq(dataset), "estimated_gpu_hours"].iloc[0])
        actual_hours = observed_runtime / 3600 if np.isfinite(observed_runtime) else np.nan
        ratio = (projection / estimated) if np.isfinite(projection) else (
            actual_hours / estimated if status == "success" else np.nan)
        rows.append({"dataset": dataset, "estimated_gpu_hours": estimated,
                     "observed_runtime_sec": observed_runtime, "observed_epochs": observed_epochs,
                     "production_projection_hours": projection,
                     "actual_to_estimated_ratio": ratio, "final_support_status": status,
                     "support_reason": reason})
    return pd.DataFrame(rows)


def finalize(config: dict, output: Path, round4b_root: Path) -> dict:
    ensure_layout(output)
    ledger_path = output / "manifests/support_reclassification.json"
    raw = collect_results(output)
    raw_paths = {}
    for path in sorted((output / "raw").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        raw_paths[(record["dataset"], record["model"], int(record["seed"]))] = str(path.relative_to(output))
    if not raw.empty:
        raw["evidence_path"] = [raw_paths[(row.dataset, row.model, int(row.seed))]
                                for row in raw.itertuples()]
    ledger = load_classification_ledger(ledger_path)
    expected = [(d, m, int(s)) for d in config["datasets"] for m in config["models"] for s in config["seeds"]]
    accounting = account_cells(raw, ledger, expected)
    support = frozen_support_matrix(accounting)
    decision, reasons = completion_decision(accounting)
    accounting.to_csv(output / "tables/production_cell_accounting_final.csv", index=False)
    support.to_csv(output / "tables/model_dataset_support_matrix_final.csv", index=False)
    views = complete_case_views(support, list(config["models"]), list(config["datasets"]), FRAUD_REPRESENTATIVE)
    views.to_csv(output / "tables/statistical_complete_case_views.csv", index=False)

    freeze = build_data_freeze(config, datasets=ROUND5_DATASETS)
    (output / "freeze/round5_data_and_code_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    environment = build_environment_freeze()
    (output / "freeze/environment_freeze.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    preflight = resource_preflight(config, freeze)
    preflight.to_csv(output / "tables/remaining_five_dataset_resource_preflight.csv", index=False)
    long_limit = long_limit_preclassification(preflight)
    long_limit.to_csv(output / "tables/remaining_five_long_limit_preclassification.csv", index=False)

    oom = gadnr_oom_audit(raw, freeze)
    oom.to_csv(output / "tables/gadnr_oom_root_cause.csv", index=False)
    all_oom = production_oom_audit(raw, freeze)
    all_oom.to_csv(output / "tables/production_oom_root_cause.csv", index=False)
    anomaly = anomaly_actual_table(raw, ledger, round4b_root)
    anomaly.to_csv(output / "tables/anomalydae_estimated_vs_actual_final.csv", index=False)

    success = raw.loc[raw.status.eq("success")].copy()
    runtime_cells, forecast = runtime_forecast_cells(success, support, freeze, preflight, list(config["models"]))
    runtime_cells.to_csv(output / "tables/round5_runtime_forecast_cells.csv", index=False)
    unsupported = support.loc[~support.primary_supported & support.accounted].copy()
    unsupported.to_csv(output / "tables/round5_excluded_operational_cells.csv", index=False)
    forecast["excluded_operational_cells"] = int(len(unsupported))
    forecast["excluded_cost_note"] = "unsupported complexity is reported separately and never treated as zero performance"
    (output / "resources/round5_runtime_forecast_final.json").write_text(json.dumps(forecast, indent=2) + "\n", encoding="utf-8")

    protocol_path = output / "manifests/round5_full_benchmark_protocol.yaml"
    protocol = None
    if decision in {"READY_FOR_FULL_RUN", "READY_WITH_RESTRICTIONS"}:
        unsupported_records = support.loc[~support.primary_supported, [
            "dataset", "model", "restriction_class", "restriction_reason"
        ]].to_dict("records")
        protocol = {
            "benchmark": {"datasets": ROUND5_DATASETS, "models": list(config["models"]),
                          "model_seeds": list(config["round5"]["model_seeds"]),
                          "historical_dlg_alias": {"DLG": "DLG-Aug"}},
            "support_policy": {"use_frozen_support_matrix": True,
                               "support_matrix": "tables/model_dataset_support_matrix_final.csv",
                               "support_matrix_hash": canonical_hash(support.to_dict("records")),
                               "unsupported_representative_cells": unsupported_records,
                               "remaining_long_limit_preclassification": "tables/remaining_five_long_limit_preclassification.csv",
                               "unsupported_performance_value": "missing", "worst_rank_imputation": False},
            "synthetic": {"fixed_dataset_seed": 42, "robustness_appendix": True},
            "training": {"epochs": 50, "early_stopping": False, "dlg_l1_epochs": 20},
            "execution": {"fresh_subprocess_per_run": True, "full_shared_graph": True,
                          "message_backend": "sparse_fused", "partition_fallback": False,
                          "cpu_fallback": False, "approximation": False},
            "reconstruction": {"linear": "exact_sparse", "anomalydae": "chunked_exact"},
            "threshold": {"primary": "validation_selected", "fixed_05": False},
            "statistics": {"aggregate_seed_first": True, "complete_case_blocks": True,
                           "friedman": True, "wilcoxon_holm": True},
            "runtime": {"forecast": "resources/round5_runtime_forecast_final.json",
                        "scheduling_basis": "pessimistic"},
            "auto_start": False,
        }
        protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    elif protocol_path.exists():
        protocol_path.unlink()

    gate = {
        "decision": decision, "reasons": reasons, "nominal_cells": len(expected),
        "observed_attempted_cells": int(len(raw)), "accounted_cells": int(accounting.accounted.sum()),
        "success_cells": int(accounting.final_status.eq("success").sum()),
        "classified_unsupported_cells": int(accounting.final_status.isin(UNSUPPORTED_STATUSES).sum()),
        "failed_unknown": int((~accounting.accounted).sum()),
        "support_matrix_frozen": bool(accounting.accounted.all()),
        "round5_protocol_created": protocol is not None, "round5_auto_started": False,
    }
    (output / "manifests/readiness_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    write_completion_report(output, gate, accounting, support, views, anomaly, oom, preflight, forecast)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--round4b-root", default="outputs/sci_round4b")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_root"])
    print(json.dumps(finalize(config, output, Path(args.round4b_root)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
