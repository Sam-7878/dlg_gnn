"""Fail-closed provenance and paper-readiness audit for Defense Round D2."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon

from gog_fraud.extensions.defense.darpa_theia_adapter import DarpaTheiaGraphBuilder
from gog_fraud.extensions.defense.lanl_redteam_adapter import LanlRedTeamGraphBuilder
from gog_fraud.extensions.defense.defense_registry import load_defense_dataset

ROUND5_RAW_HASH = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
ROUND5_SUPPORT_HASH = "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"
SCALABLE_MODELS = ["CoLA", "DOMINANT", "OCGNN", "DLG-Base", "DLG-Aug"]
METRICS = ["roc_auc", "pr_auc", "f1"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def source_entry(path: Path, role: str) -> dict:
    return {
        "path": str(path),
        "role": role,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
        "record_count": None,
        "time_range": None,
        "topic": None,
        "official_object_name": None,
    }


def feature_lineage() -> list[dict]:
    theia = [
        ("is_process", "synthetic node type", "1[type=Process]"),
        ("is_file", "synthetic node type", "1[type=File]"),
        ("is_socket", "synthetic node type", "1[type=Socket]"),
        ("is_other", "synthetic node type", "1[type=Other]"),
        ("in_degree_log1p", "generated event destination", "log1p(in_degree)"),
        ("out_degree_log1p", "generated event source", "log1p(out_degree)"),
        ("read_count_log1p", "generated event_type", "log1p(read_count)"),
        ("write_count_log1p", "generated event_type", "log1p(write_count)"),
        ("spawn_count_log1p", "generated event_type", "log1p(spawn_count)"),
        ("net_send_count_log1p", "generated event_type", "log1p(net_send_count)"),
        ("net_recv_count_log1p", "generated event_type", "log1p(net_recv_count)"),
        ("unique_peers_log1p", "generated source/destination IDs", "log1p(|unique peers|)"),
        ("unique_event_types_log1p", "generated event_type", "log1p(|event types|)"),
        ("total_events_log1p", "generated event endpoints", "log1p(in_degree+out_degree)"),
        ("active_duration_log1p", "generated timestamps", "log1p(max(timestamp)-min(timestamp))"),
        ("event_frequency_log1p", "generated events/timestamps", "log1p(total_events/duration*3600)"),
    ]
    lanl = [
        ("in_auth_log1p", "generated auth destination", "log1p(in_auth)"),
        ("out_auth_log1p", "generated auth source", "log1p(out_auth)"),
        ("success_auth_log1p", "generated auth success flag", "log1p(success_auth)"),
        ("failed_auth_log1p", "generated auth success flag", "log1p(failed_auth)"),
        ("unique_users_log1p", "generated auth/process user", "log1p(|users|)"),
        ("unique_src_comp_log1p", "generated auth source", "log1p(|source peers|)"),
        ("unique_dst_comp_log1p", "generated auth destination", "log1p(|destination peers|)"),
        ("auth_success_ratio", "generated auth success flag", "success/(success+fail+1e-6)"),
        ("proc_starts_log1p", "generated process start flag", "log1p(process starts)"),
        ("proc_stops_log1p", "generated process start flag", "log1p(process stops)"),
        ("unique_procs_log1p", "generated process name", "log1p(|process names|)"),
        ("flows_count_log1p", "generated flow endpoints", "log1p(flow endpoint count)"),
        ("bytes_sent_log1p", "generated flow byte_count", "log1p(bytes sent)"),
        ("bytes_recv_log1p", "generated flow byte_count", "log1p(bytes received)"),
        ("unique_flow_peers_log1p", "generated flow endpoints", "log1p(|flow peers|)"),
        ("dns_queries_log1p", "generated DNS source", "log1p(DNS queries)"),
    ]
    rows = []
    for dataset, entries in (("DARPA-TC-THEIA", theia), ("LANL-RedTeam", lanl)):
        for name, fields, equation in entries:
            rows.append({
                "dataset": dataset,
                "feature_name": name,
                "raw_source_fields": fields,
                "aggregation_equation": equation,
                "uses_ground_truth": False,
                "uses_attack_ioc": False,
                "uses_redteam_file": False,
                "uses_attack_time_window": False,
                "lineage_status": "synthetic_generator_field_not_official_raw_telemetry",
            })
    return rows


def correlation_diagnostics() -> list[dict]:
    rows = []
    for dataset in ("DARPA-TC-THEIA", "LANL-RedTeam"):
        data = load_defense_dataset(dataset)
        manifest_name = "darpa_theia_manifest.json" if dataset.startswith("DARPA") else "lanl_redteam_manifest.json"
        manifest_path = Path("outputs/sci_defense_extension/processed") / ("darpa_theia" if dataset.startswith("DARPA") else "lanl_redteam") / manifest_name
        names = json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"]["feature_names"]
        x, y = data.x.numpy(), data.y.numpy()
        for index, name in enumerate(names):
            value = float(np.corrcoef(x[:, index], y)[0, 1]) if np.std(x[:, index]) and np.std(y) else np.nan
            rows.append({"dataset": dataset, "feature_name": name, "pearson_r": value, "interpretation": "diagnostic_only"})
    return rows


def independent_statistics(output: Path) -> dict:
    round5 = pd.read_csv("outputs/sci_round5_final/raw/benchmark_raw.csv")
    defense = pd.read_csv("outputs/sci_defense_extension/raw/benchmark_raw.csv")
    if "f1" not in round5 and "validation_f1" in round5:
        round5["f1"] = round5["validation_f1"]
    combined = pd.concat([round5, defense], ignore_index=True, sort=False)
    existing_ranks = pd.read_csv("outputs/sci_defense_extension/tables/03_10_vs_12_dataset_scalable_ranking.csv")
    existing_pairs = pd.read_csv("outputs/sci_defense_extension/tables/05_12dataset_pairwise_wilcoxon_holm.csv")
    rank_rows, pair_rows = [], []
    max_rank_error = 0.0
    max_p_error = 0.0
    for scope, frame in (("10_dataset", round5), ("12_dataset", combined)):
        for metric in METRICS:
            pivot = frame.groupby(["dataset", "model"])[metric].mean().unstack()[SCALABLE_MODELS].dropna()
            ranks = np.vstack([rankdata(-row, method="average") for row in pivot.to_numpy()])
            for model, value in zip(SCALABLE_MODELS, ranks.mean(axis=0)):
                rank_rows.append({"scope": scope, "metric": metric, "model": model, "average_rank": float(value), "n_datasets": len(pivot)})
                col = f"{'10' if scope == '10_dataset' else '12'}_Dataset_{metric.upper()}_Rank"
                reported = float(existing_ranks.loc[existing_ranks.model.eq(model), col].iloc[0])
                max_rank_error = max(max_rank_error, abs(round(float(value), 2) - reported))

    for metric in METRICS:
        pivot = combined.groupby(["dataset", "model"])[metric].mean().unstack()[SCALABLE_MODELS].dropna()
        comparisons = [model for model in SCALABLE_MODELS if model != "DLG-Aug"]
        raw = []
        stats = []
        for model in comparisons:
            stat, p_value = wilcoxon(pivot["DLG-Aug"], pivot[model])
            stats.append(float(stat)); raw.append(float(p_value))
        order = np.argsort(raw)
        adjusted = np.zeros(len(raw))
        running = 0.0
        for position, index in enumerate(order):
            running = max(running, min(1.0, raw[index] * (len(raw) - position)))
            adjusted[index] = running
        for model, stat, p_raw, p_adj in zip(comparisons, stats, raw, adjusted):
            pair_rows.append({"metric": metric, "comparison": f"DLG-Aug vs {model}", "w_statistic": stat, "p_raw": p_raw, "p_holm_adj": p_adj})
            reported = existing_pairs[(existing_pairs.metric.eq(metric.upper())) & (existing_pairs.comparison.eq(f"DLG-Aug vs {model}"))].iloc[0]
            max_p_error = max(max_p_error, abs(p_raw - float(reported.p_raw)), abs(p_adj - float(reported.p_holm_adj)))
    write_csv(output / "statistics" / "independent_rank_recomputation.csv", rank_rows)
    write_csv(output / "statistics" / "independent_wilcoxon_holm.csv", pair_rows)
    result = {
        "rank_recomputation_matches_reported": max_rank_error <= 1e-12,
        "pairwise_recomputation_matches_reported": max_p_error <= 1e-12,
        "max_rank_error_after_reported_rounding": max_rank_error,
        "max_pairwise_p_error": max_p_error,
        "defense_only_inference_permitted": False,
        "scalable_subset": SCALABLE_MODELS,
        "gadnr_in_scalable_subset": False,
    }
    (output / "statistics" / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/sci_defense_extension/d2")
    args = parser.parse_args()
    output = Path(args.output_dir)
    for subdir in ("manifests", "lineage", "leakage", "statistics"):
        (output / subdir).mkdir(parents=True, exist_ok=True)

    theia_script = Path("scripts/defense_extension/prepare_darpa_theia.py")
    lanl_script = Path("scripts/defense_extension/prepare_lanl_redteam.py")
    actual_sources = [source_entry(theia_script, "actual synthetic graph generator"), source_entry(lanl_script, "actual synthetic graph generator")]
    expected_official = [
        {"dataset": "DARPA-TC-THEIA", "object": "ta1-theia-e3-official-1r", "present": False},
        *[{"dataset": "LANL-RedTeam", "object": name, "present": False} for name in ("auth", "proc", "flows", "dns", "redteam")],
    ]
    write_csv(output / "lineage" / "source_file_manifest.csv", actual_sources)
    write_csv(output / "lineage" / "expected_official_sources.csv", expected_official)

    theia = load_defense_dataset("DARPA-TC-THEIA")
    lanl = load_defense_dataset("LANL-RedTeam")
    theia_used = int(torch.unique(theia.edge_index).numel())
    lanl_used = int(torch.unique(lanl.edge_index).numel())
    write_csv(output / "lineage" / "record_accounting.csv", [
        {"dataset": "DARPA-TC-THEIA", "stage": "official raw records", "input_count": None, "retained_count": 0, "dropped_count": None, "drop_reason": "official raw source not present or read"},
        {"dataset": "DARPA-TC-THEIA", "stage": "parsed CDM records", "input_count": 0, "retained_count": 0, "dropped_count": 0, "drop_reason": "no parser exists in D1 generator"},
        {"dataset": "DARPA-TC-THEIA", "stage": "synthetically declared entities", "input_count": 1156, "retained_count": 1156, "dropped_count": 0, "drop_reason": "none"},
        {"dataset": "DARPA-TC-THEIA", "stage": "final unique synthetic edges", "input_count": None, "retained_count": int(theia.edge_index.size(1)), "dropped_count": None, "drop_reason": "pre-dedup generated-event count was not persisted"},
        {"dataset": "LANL-RedTeam", "stage": "official raw events", "input_count": None, "retained_count": 0, "dropped_count": None, "drop_reason": "auth/proc/flows/dns/redteam files not present or read"},
        {"dataset": "LANL-RedTeam", "stage": "parsed official events", "input_count": 0, "retained_count": 0, "dropped_count": 0, "drop_reason": "no raw parser exists in D1 generator"},
        {"dataset": "LANL-RedTeam", "stage": "synthetically declared computers", "input_count": 1310, "retained_count": 1310, "dropped_count": 0, "drop_reason": "none"},
        {"dataset": "LANL-RedTeam", "stage": "final unique synthetic auth edges", "input_count": None, "retained_count": int(lanl.edge_index.size(1)), "dropped_count": None, "drop_reason": "pre-dedup generated-auth count was not persisted"},
    ])
    write_csv(output / "lineage" / "node_accounting.csv", [
        {"dataset": "DARPA-TC-THEIA", "stage": "official raw unique entities", "count": None, "status": "unavailable"},
        {"dataset": "DARPA-TC-THEIA", "stage": "synthetic Process", "count": 182, "status": "generator declaration"},
        {"dataset": "DARPA-TC-THEIA", "stage": "synthetic File", "count": 818, "status": "generator declaration"},
        {"dataset": "DARPA-TC-THEIA", "stage": "synthetic Socket", "count": 156, "status": "generator declaration"},
        {"dataset": "DARPA-TC-THEIA", "stage": "synthetic Other", "count": 0, "status": "generator declaration"},
        {"dataset": "DARPA-TC-THEIA", "stage": "final graph", "count": int(theia.num_nodes), "status": "synthetic artifact"},
        {"dataset": "LANL-RedTeam", "stage": "official reported universe", "count": 17684, "status": "literature metadata only; not loaded"},
        {"dataset": "LANL-RedTeam", "stage": "observed in downloaded files", "count": None, "status": "no downloaded official files"},
        {"dataset": "LANL-RedTeam", "stage": "synthetic inventory", "count": 1310, "status": "10 DC + 80 server + 1200 workstation + 20 gateway"},
        {"dataset": "LANL-RedTeam", "stage": "authentication participants", "count": lanl_used, "status": "recoverable only from final synthetic edge_index"},
        {"dataset": "LANL-RedTeam", "stage": "feature-resolvable", "count": 1310, "status": "synthetic generator"},
        {"dataset": "LANL-RedTeam", "stage": "after deterministic official filter", "count": None, "status": "no official filter was performed"},
        {"dataset": "LANL-RedTeam", "stage": "final graph", "count": int(lanl.num_nodes), "status": "synthetic artifact"},
    ])
    mappings = []
    for entity in ["proc_user_10", *[f"proc_mal_apt_{i}" for i in range(12)], *[f"file_mal_artifact_{i}" for i in range(18)], *[f"sock_mal_c2_{i}" for i in range(6)]]:
        mappings.append({"dataset": "DARPA-TC-THEIA", "node_id": entity, "node_type": entity.split("_")[0], "ground_truth_source": "synthetic generator", "mapping_key": entity, "ground_truth_record_id": None, "official_mapping": False})
    for entity in [*[f"C_WS_{i}" for i in range(15, 35)], *[f"C_SRV_{i}" for i in range(5, 15)], "C_DC_1", "C_DC_2"]:
        mappings.append({"dataset": "LANL-RedTeam", "node_id": entity, "node_type": "Computer", "ground_truth_source": "synthetic redteam_targets list", "mapping_key": entity, "ground_truth_record_id": None, "official_mapping": False})
    write_csv(output / "lineage" / "positive_mapping_audit.csv", mappings)

    features = feature_lineage()
    write_csv(output / "leakage" / "feature_lineage.csv", features)
    write_csv(output / "leakage" / "feature_label_correlation_diagnostic.csv", correlation_diagnostics())
    api_audit = {
        "theia_extract_features_parameters": list(inspect.signature(DarpaTheiaGraphBuilder.extract_features).parameters),
        "theia_add_event_parameters": list(inspect.signature(DarpaTheiaGraphBuilder.add_event).parameters),
        "lanl_extract_features_parameters": list(inspect.signature(LanlRedTeamGraphBuilder.extract_features).parameters),
        "lanl_add_auth_event_parameters": list(inspect.signature(LanlRedTeamGraphBuilder.add_auth_event).parameters),
        "feature_builders_accept_ground_truth": False,
        "correlation_used_as_leakage_gate": False,
    }
    (output / "leakage" / "api_dependency_audit.json").write_text(json.dumps(api_audit, indent=2) + "\n", encoding="utf-8")

    stats = independent_statistics(output)
    paths = {
        "round5_raw": Path("outputs/sci_round5_final/raw/benchmark_raw.csv"),
        "round5_support": Path("outputs/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv"),
        "defense_raw": Path("outputs/sci_defense_extension/raw/benchmark_raw.csv"),
        "theia_artifact": Path("outputs/sci_defense_extension/processed/darpa_theia/darpa_tc_theia_e3.pt"),
        "lanl_artifact": Path("outputs/sci_defense_extension/processed/lanl_redteam/lanl_redteam_computer_graph.pt"),
        "derived_12dataset_view": Path("outputs/sci_defense_extension/extended_analysis/benchmark_12dataset_view.csv"),
    }
    frozen_hashes = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    theia_manifest = json.loads(Path("outputs/sci_defense_extension/processed/darpa_theia/darpa_theia_manifest.json").read_text(encoding="utf-8"))
    lanl_manifest = json.loads(Path("outputs/sci_defense_extension/processed/lanl_redteam/lanl_redteam_manifest.json").read_text(encoding="utf-8"))
    frozen_hashes["darpa_graph_tensor"] = {"sha256": theia_manifest["graph_sha256"]}
    frozen_hashes["darpa_feature_tensor"] = {"sha256": theia_manifest["feature_sha256"]}
    frozen_hashes["darpa_label_tensor"] = {"sha256": theia_manifest["label_sha256"]}
    frozen_hashes["lanl_graph_tensor"] = {"sha256": lanl_manifest["graph_sha256"]}
    frozen_hashes["lanl_feature_tensor"] = {"sha256": lanl_manifest["feature_sha256"]}
    frozen_hashes["lanl_label_tensor"] = {"sha256": lanl_manifest["label_sha256"]}
    frozen_hashes["round5_raw"]["matches_frozen"] = frozen_hashes["round5_raw"]["sha256"] == ROUND5_RAW_HASH
    frozen_hashes["round5_support"]["matches_frozen"] = frozen_hashes["round5_support"]["sha256"] == ROUND5_SUPPORT_HASH
    (output / "manifests" / "final_extension_hashes.json").write_text(json.dumps(frozen_hashes, indent=2) + "\n", encoding="utf-8")

    lineage = {
        "decision": "FAIL",
        "artifact_origin": "deterministic_synthetic_generator",
        "official_raw_files_traceable": False,
        "darpa": {"official_files_read": 0, "final_nodes": int(theia.num_nodes), "final_edges": int(theia.edge_index.size(1)), "positive_nodes": int(theia.y.sum()), "subset": "synthetic attack-centered simulation; not an official stream subset"},
        "lanl": {"official_files_read": 0, "final_nodes": int(lanl.num_nodes), "final_edges": int(lanl.edge_index.size(1)), "positive_nodes": int(lanl.y.sum()), "reduction_17684_to_1310_explained": False, "actual_explanation": "1310 nodes were created ab initio by code; no 17,684-node universe was loaded"},
        "generator_evidence": actual_sources,
        "not_paper_ready_triggers": ["raw source file lineage unavailable", "processed graphs generated from synthetic/fallback data", "source-to-node reduction cannot be explained from official records", "ground-truth mappings are synthetic rather than official"],
    }
    (output / "manifests" / "defense_source_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    gadnr_path = output / "gadnr" / "gadnr_compatibility_equivalence.json"
    sensitivity_path = output / "sensitivity" / "manifest.json"
    gadnr = json.loads(gadnr_path.read_text(encoding="utf-8")) if gadnr_path.exists() else {}
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8")) if sensitivity_path.exists() else {}
    gate = {
        "decision": "NOT_PAPER_READY",
        "round5_hashes_unchanged": frozen_hashes["round5_raw"]["matches_frozen"] and frozen_hashes["round5_support"]["matches_frozen"],
        "defense_d1_raw_unchanged": frozen_hashes["defense_raw"]["sha256"] == "d6835826db7a18df3433889998f3baba1a1d8119215f0f1e72dcd9ffe4de5232",
        "official_raw_files_traceable": False,
        "source_to_graph_accounting_complete": False,
        "official_ground_truth_mapping_reproducible": False,
        "feature_builder_ground_truth_independent": True,
        "statistics_independently_verified": stats["rank_recomputation_matches_reported"] and stats["pairwise_recomputation_matches_reported"],
        "gadnr_equivalence_passed": bool(gadnr.get("acceptance_passed")),
        "theia_sensitivity_completed": sensitivity.get("runs") == sensitivity.get("expected_runs") == 15,
        "paper_use_policy": "Exclude D1 defense performance and 12-dataset derived inference from the SCI manuscript until rebuilt from official raw data.",
    }
    (output / "manifests" / "paper_readiness_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
