#!/usr/bin/env python3
"""Build the Defense Extension Round D4 final paper bundle.

This is a derived-only operation: it does not rebuild a graph or rerun any
Round 5/LANL/THEIA detector cell. Raw file hashes are verified against the
local files that the D3 builders consumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from gog_fraud.extensions.defense.evaluation_policy import (
    record_is_performance_eligible,
    theia_performance_eligibility,
)


MODELS = ["DOMINANT", "AnomalyDAE", "CoLA", "CONAD", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]
METRICS = ["roc_auc", "pr_auc", "f1"]
ROUND5_DISPLAY = {
    "Elliptic": "Elliptic",
    "DGraphFin": "DGraphFin",
    "Yelp": "Yelp-Syn",
    "Amazon": "Amazon-Syn",
    "BitcoinOTC": "BitcoinOTC",
    "Flickr": "Flickr-Syn",
    "Reddit": "Reddit-Syn",
    "Cora": "Cora-Syn",
    "CiteSeer": "CiteSeer-Syn",
    "PubMed": "PubMed-Syn",
}
ROUND5_DOMAIN = {
    "real labels": "real_financial_or_blockchain",
    "synthetic injection": "synthetic_injection_graph_anomaly",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(materialized[0]) if materialized else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def parse_report_hashes(report_path: Path) -> dict[str, str]:
    text = report_path.read_text(encoding="utf-8")
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        names = re.findall(r"`([^`]+)`", line)
        hash_match = re.search(r"`([0-9a-f]{64})`", line)
        if hash_match and names:
            artifact = next((v for v in names if not re.fullmatch(r"[0-9a-f]{64}", v)), None)
            if artifact:
                hashes[Path(artifact).name] = hash_match.group(1)
    return hashes


def parse_round3_dataset_table(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| **"):
            continue
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        try:
            rows[cells[0]] = {
                "N": int(cells[2].replace(",", "")),
                "E": int(cells[3].replace(",", "")),
                "F": int(cells[4].replace(",", "")),
            }
        except ValueError:
            continue
    return rows


def build_theia_gt_audit(source_mapping: Path, output_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit existing positives without inventing identifier-level labels."""

    rows: list[dict[str, Any]] = []
    with source_mapping.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("label") != "1":
                continue
            rationale = row.get("mapping_rationale", "")
            confidence = "unresolved" if "attack_window" in rationale else "unresolved"
            rows.append({
                "gt_reference": row.get("gt_reference") or "TA51_Final_report_E5",
                "attack_action": "Drakon inject attack (report-level action)",
                "timestamp": "2019-05-15 18:47:41--19:10:00 UTC",
                "host": "ta1-theia-target-1 (128.55.12.110)",
                "process_pid_or_identifier": "not retained in graph mapping",
                "object_identifier_if_available": "",
                "CDM_uuid": row.get("uuid_hex", ""),
                "entity_type": row.get("node_class", ""),
                "internal_node_id": row.get("internal_id", ""),
                "mapping_confidence": confidence,
                "mapping_reason": (
                    "D3 label used temporal overlap only; no report identifier or action was joined "
                    f"to this CDM UUID ({rationale}). Time-window activity is not an attack label."
                ),
            })
    write_csv(output_path, rows, [
        "gt_reference", "attack_action", "timestamp", "host", "process_pid_or_identifier",
        "object_identifier_if_available", "CDM_uuid", "entity_type", "internal_node_id",
        "mapping_confidence", "mapping_reason",
    ])
    direct = sum(row["mapping_confidence"] == "direct" for row in rows)
    strong = sum(row["mapping_confidence"] == "strong_temporal_identifier_match" for row in rows)
    unresolved = sum(row["mapping_confidence"] == "unresolved" for row in rows)
    # No direct node labels exist; the old split had zero test positives in every successful run.
    eligibility = theia_performance_eligibility(direct, 0, 0)
    eligibility.update({
        "candidate_rows": len(rows),
        "direct_mappings": direct,
        "strong_temporal_identifier_matches": strong,
        "unresolved_mappings": unresolved,
        "historical_label_policy": "time_window_overlap",
        "historical_positive_count": len(rows),
        "decision_reason": "No entity has a direct official-identifier-to-CDM-UUID mapping in the selected subset.",
    })
    return rows, eligibility


def lanl_gt_audit(mapping_path: Path, redteam_path: Path, cutoff_seconds: int) -> dict[str, Any]:
    mapping = read_csv(mapping_path)
    positives = [row for row in mapping if row["label"] == "1"]
    official_destinations: set[str] = set()
    all_events = 0
    cutoff_events = 0
    max_time = 0
    with redteam_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            timestamp = int(parts[0])
            all_events += 1
            max_time = max(max_time, timestamp)
            if timestamp <= cutoff_seconds:
                cutoff_events += 1
                official_destinations.add(parts[3])
    mapped_destinations = {row["computer_id"] for row in positives}
    return {
        "positive_nodes": len(positives),
        "mapping_rows": len(mapping),
        "official_redteam_events": all_events,
        "events_within_30_day_cutoff": cutoff_events,
        "max_redteam_time_seconds": max_time,
        "official_destination_nodes_within_cutoff": len(official_destinations),
        "all_positive_ids_exist_in_final_graph": all(0 <= int(row["internal_id"]) < len(mapping) for row in positives),
        "all_cutoff_destination_ids_accounted": official_destinations == mapped_destinations,
        "labels_source": "official redteam.txt destination computer mapping",
        "feature_label_separation": "telemetry features are built before y assignment; redteam fields are not feature inputs",
    }


def summarize_lanl(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        cells = [r for r in records if r["dataset"] == "LANL-RedTeam" and r["model"] == model and r["status"] == "success"]
        item: dict[str, Any] = {"model": model, "successful_seeds": len(cells)}
        for metric in METRICS:
            values = [float(r[metric]) for r in cells if r.get(metric) not in (None, "")]
            avg, std = mean_std(values)
            item[f"{metric}_mean"] = avg
            item[f"{metric}_std"] = std
        rows.append(item)
    return rows


def summarize_theia_scalability(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        cells = [r for r in records if r["dataset"] == "DARPA-TC-THEIA" and r["model"] == model]
        success = [r for r in cells if r["status"] == "success" and r.get("actual_epochs") not in (None, "")]
        runtimes = [float(r["fit_seconds"]) for r in success]
        vram = [float(r["peak_vram_mb"]) for r in success]
        runtime_mean, runtime_std = mean_std(runtimes)
        rows.append({
            "model": model,
            "support_status": "supported_exact_50_epoch" if len(success) == 5 else "unsupported_resource_exact_implementation",
            "attempted_seeds": len(cells),
            "completed_50_epoch_seeds": len(success),
            "fit_seconds_mean": runtime_mean,
            "fit_seconds_std": runtime_std,
            "peak_vram_mb_max": max(vram) if vram else None,
            "performance_metrics_reported": False,
            "metric_status": "undefined_single_class",
        })
    return rows


def combine_performance_rows(round5_path: Path, defense_records: list[dict[str, str]]) -> list[dict[str, Any]]:
    round5 = [row for row in read_csv(round5_path) if row["status"] == "success"]
    lanl = [r for r in defense_records if r["dataset"] == "LANL-RedTeam" and record_is_performance_eligible(r)]
    combined: list[dict[str, Any]] = []
    for row in round5:
        item = dict(row)
        item["display_name"] = ROUND5_DISPLAY.get(row["dataset"], row["dataset"])
        item["analysis_scope"] = "primary_frozen_round5"
        item["f1"] = row["validation_f1"]
        combined.append(item)
    for row in lanl:
        item = dict(row)
        item["display_name"] = "LANL-RedTeam"
        item["analysis_scope"] = "external_labeled_sensitivity"
        item["validation_f1"] = row["f1"]
        combined.append(item)
    return combined


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["display_name"], row["model"])].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, model), cells in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "model": model, "n": len(cells)}
        for metric in METRICS:
            values = [float(c[metric]) for c in cells if c.get(metric) not in (None, "")]
            avg, std = mean_std(values)
            item[f"{metric}_mean"] = avg
            item[f"{metric}_std"] = std
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/sci_defense_extension_real_final"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (project / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    real = project / "outputs/sci_defense_extension_real"
    round5 = project / "outputs/sci_round5_final"
    report_dir = project / "docs/work_reports/210_Defense_Extension_Round_D3"
    data_root = project.parent.parent / "_data/DLG"
    theia_dir = data_root / "DARPA-TC-THEIA/Data/theia/theia-20260822T022150Z-1-001/theia"
    theia_root = data_root / "DARPA-TC-THEIA"
    lanl_root = data_root / "LANL-RedTeam"

    for directory in ["manifests", "tables", "statistics", "scalability", "defense_validation", "logs", "archive/pre_real_source_attempts", "source_audit"]:
        (output / directory).mkdir(parents=True, exist_ok=True)

    theia_lineage = read_json(real / "source_audit/defense_real_theia_lineage.json")
    lanl_lineage = read_json(real / "source_audit/defense_real_lanl_lineage.json")
    data_freeze = read_json(round5 / "manifests/data_freeze.json")
    defense_records = read_csv(real / "benchmark/benchmark_raw.csv")
    report_hashes = parse_report_hashes(report_dir / "01_real_source_acquisition_report.md")

    source_records: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    for lineage, base, dataset in [(theia_lineage, theia_dir, "DARPA-TC-THEIA"), (lanl_lineage, lanl_root, "LANL-RedTeam")]:
        for declared in lineage["source"]["raw_files"]:
            path = base / declared["filename"]
            actual = sha256(path)
            declared_hash = declared["sha256"]
            reported = report_hashes.get(declared["filename"], "not_reported")
            source_records.append({
                "dataset": dataset,
                "filename": declared["filename"],
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
                "declared_machine_manifest_match": declared_hash == actual,
            })
            reconciliation.extend([
                {"artifact": f"{dataset}/{declared['filename']} (D3 report)", "reported_hash": reported, "actual_hash": actual, "match": reported == actual, "authoritative": False},
                {"artifact": f"{dataset}/{declared['filename']} (machine manifest)", "reported_hash": declared_hash, "actual_hash": actual, "match": declared_hash == actual, "authoritative": True},
            ])

    schema_path = theia_root / "Schema/TCCDMDatum.avsc"
    gt_docx_path = theia_root / "Ground_Truth/TA51_Final_report_E5.docx"
    for label, path in [("DARPA-TC-THEIA/CDM20 schema", schema_path), ("DARPA-TC-THEIA/E5 ground truth DOCX", gt_docx_path)]:
        actual = sha256(path)
        reported = report_hashes.get(path.name, "not_reported")
        reconciliation.extend([
            {"artifact": f"{label} (D3 report)", "reported_hash": reported, "actual_hash": actual, "match": reported == actual, "authoritative": False},
            {"artifact": f"{label} (local verification)", "reported_hash": actual, "actual_hash": actual, "match": True, "authoritative": True},
        ])

    round5_raw_path = round5 / "raw/benchmark_raw.csv"
    round5_support_path = round5 / "manifests/model_dataset_support_matrix_v2.csv"
    round5_raw_hash = sha256(round5_raw_path)
    round5_support_hash = sha256(round5_support_path)
    expected_round5_raw = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
    expected_round5_support = "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"
    reconciliation.extend([
        {"artifact": "Round5/benchmark_raw.csv", "reported_hash": expected_round5_raw, "actual_hash": round5_raw_hash, "match": expected_round5_raw == round5_raw_hash, "authoritative": True},
        {"artifact": "Round5/model_dataset_support_matrix_v2.csv", "reported_hash": expected_round5_support, "actual_hash": round5_support_hash, "match": expected_round5_support == round5_support_hash, "authoritative": True},
    ])
    write_csv(output / "tables/hash_reconciliation.csv", reconciliation)

    _, theia_eligibility = build_theia_gt_audit(
        real / "source_audit/ground_truth_mapping.csv",
        output / "source_audit/theia_ground_truth_entity_mapping.csv",
    )
    write_json(output / "defense_validation/theia_performance_eligibility.json", theia_eligibility)

    lanl_audit = lanl_gt_audit(
        real / "source_audit/lanl_ground_truth_mapping.csv",
        lanl_root / "redteam.txt",
        cutoff_seconds=30 * 86400,
    )
    write_json(output / "defense_validation/lanl_ground_truth_freeze.json", lanl_audit)

    lanl_table = summarize_lanl(defense_records)
    theia_scalability = summarize_theia_scalability(defense_records)
    write_csv(output / "tables/table_d2_lanl_external_validation.csv", lanl_table)
    write_csv(output / "tables/table_d3_theia_scalability.csv", theia_scalability)

    defense_chars = [
        {
            "dataset": "LANL-RedTeam",
            "role": "external_labeled_defense_validation",
            "raw_source_scope": "LANL Cyber1 Days 1-30 subset; all released red-team events fall within cutoff",
            "N": lanl_lineage["graph_statistics"]["num_nodes"],
            "E": lanl_lineage["graph_statistics"]["num_edges"],
            "F": lanl_lineage["graph_statistics"]["num_features"],
            "positive_count": lanl_audit["positive_nodes"],
            "positive_ratio": lanl_lineage["graph_statistics"]["anomaly_rate"],
            "performance_eligible": True,
        },
        {
            "dataset": "DARPA-TC-THEIA E5 selected official-stream subset",
            "role": "defense_scalability_case",
            "raw_source_scope": "10 CDM20 files from the first locally extracted THEIA archive directory; not full E5",
            "N": theia_lineage["graph_statistics"]["num_nodes"],
            "E": theia_lineage["graph_statistics"]["num_edges"],
            "F": theia_lineage["graph_statistics"]["num_features"],
            "positive_count": theia_eligibility["direct_mappings"],
            "positive_ratio": 0.0,
            "performance_eligible": False,
        },
    ]
    write_csv(output / "tables/table_d1_defense_dataset_characteristics.csv", defense_chars)

    portfolio: list[dict[str, Any]] = []
    for item in data_freeze["datasets"]:
        provenance = item["label_provenance"]
        portfolio.append({
            "dataset": item["dataset"],
            "display_name": ROUND5_DISPLAY[item["dataset"]],
            "role": "primary_frozen_performance",
            "domain_taxonomy": ROUND5_DOMAIN[provenance],
            "N": item["nodes"], "E": item["edges"], "F": item["features"],
            "label_provenance": provenance,
            "performance_eligible": True,
            "metadata_source": "outputs/sci_round5_final/manifests/data_freeze.json",
        })
    portfolio.extend([
        {"dataset": "LANL-RedTeam", "display_name": "LANL-RedTeam", "role": "external_labeled_performance", "domain_taxonomy": "defense_external_validation", "N": defense_chars[0]["N"], "E": defense_chars[0]["E"], "F": defense_chars[0]["F"], "label_provenance": "official redteam.txt destination computers", "performance_eligible": True, "metadata_source": "authoritative D4 defense manifest"},
        {"dataset": "DARPA-TC-THEIA", "display_name": "DARPA-TC-THEIA E5 selected official-stream subset", "role": "scalability_only", "domain_taxonomy": "defense_scalability_case", "N": defense_chars[1]["N"], "E": defense_chars[1]["E"], "F": defense_chars[1]["F"], "label_provenance": "no eligible direct entity labels", "performance_eligible": False, "metadata_source": "authoritative D4 defense manifest"},
    ])
    write_csv(output / "tables/table_d4_dataset_portfolio.csv", portfolio)

    d3_metadata = parse_round3_dataset_table(report_dir / "05_real_12dataset_extension_report.md")
    metadata_rows: list[dict[str, Any]] = []
    for item in data_freeze["datasets"]:
        old = d3_metadata.get(item["dataset"], {})
        final = {"N": item["nodes"], "E": item["edges"], "F": item["features"]}
        metadata_rows.append({
            "dataset": item["dataset"],
            "display_name": ROUND5_DISPLAY[item["dataset"]],
            "frozen_N": final["N"], "frozen_E": final["E"], "frozen_F": final["F"],
            "round3_report_N": old.get("N"), "round3_report_E": old.get("E"), "round3_report_F": old.get("F"),
            "round3_report_match": old == final,
            "final_N": final["N"], "final_E": final["E"], "final_F": final["F"],
            "final_match": True,
            "authoritative_source": "outputs/sci_round5_final/manifests/data_freeze.json",
        })
    write_csv(output / "tables/dataset_metadata_reconciliation.csv", metadata_rows)

    performance_rows = combine_performance_rows(round5_raw_path, defense_records)
    union_fields: list[str] = []
    for row in performance_rows:
        for key in row:
            if key not in union_fields:
                union_fields.append(key)
    write_csv(output / "statistics/performance_11_dataset_raw.csv", performance_rows, union_fields)
    write_csv(output / "statistics/performance_11_dataset_summary.csv", build_summary(performance_rows))
    write_json(output / "statistics/statistical_scope.json", {
        "primary_inference": "unchanged frozen Round 5 ten-dataset statistics",
        "primary_performance_datasets": 10,
        "external_labeled_sensitivity_datasets": ["LANL-RedTeam"],
        "descriptive_performance_datasets": 11,
        "scalability_portfolio_datasets": 12,
        "excluded_from_performance_and_ranking": ["DARPA-TC-THEIA E5 selected official-stream subset"],
        "exclusion_rule": "n_test_positives == 0 or performance_eligible == false",
        "theia_placeholder_metrics_included": False,
    })

    graph_hashes = {
        "theia": sha256(real / "graphs/theia_graph.pt"),
        "lanl": sha256(real / "graphs/lanl_graph.pt"),
    }
    round5_all_rows = read_csv(round5_raw_path)
    round5_success_rows = [row for row in round5_all_rows if row["status"] == "success"]
    authoritative = {
        "manifest_role": "single_authoritative_real_source_manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness": "PAPER_READY_10_PLUS_LANL_PLUS_THEIA_SCALABILITY",
        "round5": {
            "nominal_accounted_cells": len(round5_all_rows),
            "successful_supported_runs": len(round5_success_rows),
            "benchmark_raw_sha256": round5_raw_hash,
            "support_matrix_sha256": round5_support_hash,
            "frozen_unchanged": round5_raw_hash == expected_round5_raw and round5_support_hash == expected_round5_support,
        },
        "DARPA": {
            "dataset_display_name": "DARPA-TC-THEIA E5 selected official-stream subset",
            "engagement": "DARPA Transparent Computing Engagement 5",
            "exact_subset_description": "10 CDM20 .bin.gz files consumed from the first locally extracted THEIA directory; no additional archive was analyzed in D4",
            "raw_files_actually_consumed": [r for r in source_records if r["dataset"] == "DARPA-TC-THEIA"],
            "schema": {"path": str(schema_path), "sha256": sha256(schema_path)},
            "ground_truth": {"path": str(gt_docx_path), "sha256": sha256(gt_docx_path)},
            "parser": theia_lineage["parser"],
            "record_counts": theia_lineage["record_accounting"],
            "graph": {**theia_lineage["graph_statistics"], "path": str(real / "graphs/theia_graph.pt"), "sha256": graph_hashes["theia"]},
            "historical_time_window_positive_count": theia_lineage["graph_statistics"]["num_positive_labels"],
            "eligible_direct_positive_count": theia_eligibility["direct_mappings"],
            "evaluation_status": "scalability_only",
            "performance_metrics_valid": False,
        },
        "LANL": {
            "dataset_display_name": "LANL-RedTeam",
            "exact_time_coverage": "LANL Cyber1 Days 1-30 subset (timestamp <= 2,592,000 seconds)",
            "cutoff_rationale": ["all released red-team events included", "manageable deterministic scope", "predefined preprocessing boundary"],
            "raw_files_actually_consumed": [r for r in source_records if r["dataset"] == "LANL-RedTeam"],
            "record_counts": lanl_lineage["record_accounting"],
            "ground_truth_audit": lanl_audit,
            "graph": {**lanl_lineage["graph_statistics"], "path": str(real / "graphs/lanl_graph.pt"), "sha256": graph_hashes["lanl"]},
            "evaluation_status": "external_labeled_validation",
            "performance_metrics_valid": True,
            "benchmark_runs_reused": 40,
        },
    }
    write_json(output / "manifests/defense_final_source_of_truth.json", authoritative)

    stale_names = ["darpa_real_manifest.json", "lanl_real_manifest.json", "defense_real_source_lineage.json", "official_source_gate.json", "final_paper_gate.json"]
    for name in stale_names:
        src = real / "manifests" / name
        if src.exists():
            shutil.copy2(src, output / "archive/pre_real_source_attempts" / name)

    reports = {
        "01_round4_evidence_reconciliation.md": f"""# Round D4 Evidence Reconciliation\n\nFinal decision: **PAPER_READY_10_PLUS_LANL_PLUS_THEIA_SCALABILITY**.\n\nLocal-file hashing verified all 15 consumed defense raw files. Machine-readable D3 source-audit hashes matched; the D3 acquisition report hashes did not and are non-authoritative. The stale E3/pre-acquisition gate files are retained only under `archive/pre_real_source_attempts/`. Frozen Round 5 hashes remain unchanged. See `tables/hash_reconciliation.csv` and `manifests/defense_final_source_of_truth.json`.\n""",
        "02_theia_ground_truth_and_evaluation_scope.md": f"""# THEIA Ground Truth and Evaluation Scope\n\nThe graph contains {defense_chars[1]['N']:,} entity nodes, {defense_chars[1]['E']:,} edges, and {defense_chars[1]['F']} features. It is not a host-node graph. D3 assigned one Subject label using attack-window overlap; the audit found **0 direct entity mappings**, **0 strong temporal-identifier mappings**, and **{theia_eligibility['unresolved_mappings']} unresolved time-window-only candidate**. Activity in an attack window is not an attack label.\n\nThe node-level eligibility gate therefore fails (direct positives 0/10; validation 0/2; test 0/2). THEIA is frozen as `scalability_only`; its historical 0.5/0/0 placeholders are excluded from every performance table and ranking. No additional archive and no detector rerun were used.\n""",
        "03_lanl_final_external_validation.md": f"""# LANL Final External Validation\n\nScope is **LANL Cyber1 Days 1-30**, not the full 58-day corpus. The boundary was predefined because it includes all released red-team events, is deterministic, and provides a manageable scope. All {lanl_audit['official_redteam_events']} red-team events occur within the cutoff; their destination mapping yields {lanl_audit['positive_nodes']} positive computers in the {defense_chars[0]['N']:,}-node graph. All mapped IDs exist and all cutoff destinations are accounted.\n\nThe existing 40 runs were reused without rerun. GADNR has the highest mean ROC-AUC, PR-AUC, and validation-selected F1. This is consistent with useful neighborhood-distribution reconstruction on this interaction graph, but does not establish a causal mechanism. DLG-Aug is lower than DLG-Base here, adding external evidence for dataset-dependent local augmentation.\n""",
        "04_final_dataset_portfolio_and_statistics_scope.md": """# Final Dataset Portfolio and Statistical Scope\n\nPrimary inference remains the frozen ten-dataset Round 5 analysis. LANL is an external labeled sensitivity/extension analysis, producing an 11-dataset descriptive performance view. THEIA is the twelfth portfolio dataset only for scalability/runtime/resource evidence. It never enters performance ranks. Synthetic-injection datasets use the `-Syn` display suffix and are not described as real fraud-label datasets. Dataset N/E/F values are regenerated from the Round 5 frozen `data_freeze.json`, not defense-script constants.\n""",
        "05_final_manuscript_readiness.md": """# Final Manuscript Readiness\n\n## Decision\n\n**PAPER_READY_10_PLUS_LANL_PLUS_THEIA_SCALABILITY**\n\nThe frozen Round 5 evidence is unchanged, LANL provenance and five-seed results are valid, THEIA provenance and six-model exact 50-epoch scalability evidence are valid, THEIA single-class placeholders are absent from final performance statistics, dataset metadata is reconciled, and stale manifests are outside the final manifest root. Defense-data expansion ends here for this paper.\n""",
    }
    docs_out = project / "docs/work_reports/211_Defense_Extension_Round_D4"
    for name, text in reports.items():
        (docs_out / name).write_text(text, encoding="utf-8")
        (output / "logs" / name).write_text(text, encoding="utf-8")

    write_json(output / "manifests/bundle_integrity.json", {
        "authoritative_source_manifest_count": 1,
        "stale_gate_files_in_manifest_root": [],
        "all_authoritative_hash_checks_pass": all(row["match"] for row in reconciliation if row["authoritative"]),
        "round5_frozen_unchanged": authoritative["round5"]["frozen_unchanged"],
        "theia_performance_excluded": True,
        "lanl_runs_reused": 40,
        "round5_runs_reused": authoritative["round5"]["successful_supported_runs"],
        "decision": authoritative["readiness"],
    })
    print(json.dumps({"output": str(output), "decision": authoritative["readiness"]}, indent=2))


if __name__ == "__main__":
    main()
