"""Official-source hard gate for Defense Extension Round D3.

No graph parser, builder, smoke run, or benchmark may execute unless both
official datasets pass this gate. There is deliberately no synthetic fallback.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROUND5_RAW_HASH = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
ROUND5_SUPPORT_HASH = "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"
D1_RAW_HASH = "d6835826db7a18df3433889998f3baba1a1d8119215f0f1e72dcd9ffe4de5232"
D1_VIEW_HASH = "196d6566f1f2f831c3b61490ebcd1558e7678452452f3b2eda176596e51680e9"

ROOT = Path("outputs/sci_defense_extension_real")
DARPA = ROOT / "source/darpa"
LANL = ROOT / "source/lanl"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, official_name: str, role: str) -> dict:
    return {
        "official_name": official_name,
        "role": role,
        "local_path": str(path),
        "available": path.is_file() and path.stat().st_size > 0,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_hash(path) if path.is_file() else None,
    }


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main() -> None:
    manifests = ROOT / "manifests"
    source_audit = ROOT / "source_audit"
    manifests.mkdir(parents=True, exist_ok=True)
    source_audit.mkdir(parents=True, exist_ok=True)

    darpa_records = [
        file_record(DARPA / "README-E3.md", "README-E3.md", "official release manifest"),
        file_record(DARPA / "TC_Ground_Truth_Report_E3_Update.pdf", "TC_Ground_Truth_Report_E3_Update.pdf", "official ground truth"),
        file_record(DARPA / "TCCDMDatum.avsc", "TCCDMDatum.avsc", "CDM18 Avro schema"),
        file_record(DARPA / "CDM18.avdl", "CDM18.avdl", "human-readable CDM18 schema"),
        file_record(DARPA / "operational_event_log.md", "operational_event_log.md", "official operational log"),
        file_record(DARPA / "ta1-theia-e3-official-6r.json.tar.gz", "ta1-theia-e3-official-6r.json.tar.gz", "candidate raw stream"),
        file_record(DARPA / "ta1-theia-e3-official-6r.bin.tar.gz", "ta1-theia-e3-official-6r.bin.tar.gz", "candidate raw stream"),
        file_record(DARPA / "candidates/ta1-theia-e3-official-5m.json.tar.gz", "ta1-theia-e3-official-5m.json.tar.gz", "candidate raw stream"),
        file_record(DARPA / "candidates/ta1-theia-e3-official-3.json.tar.gz", "ta1-theia-e3-official-3.json.tar.gz", "candidate raw stream"),
    ]
    write_csv(source_audit / "darpa_raw_manifest.csv", darpa_records)
    raw_available = any(record["available"] for record in darpa_records if record["role"] == "candidate raw stream")
    gt_available = next(record["available"] for record in darpa_records if record["role"] == "official ground truth")
    schema_available = any(record["available"] for record in darpa_records if "schema" in record["role"])
    darpa_pass = raw_available and gt_available and schema_available

    lanl_records = [file_record(LANL / name, name, "official telemetry" if name != "redteam.txt.gz" else "official ground truth") for name in ("auth.txt.gz", "proc.txt.gz", "flows.txt.gz", "dns.txt.gz", "redteam.txt.gz")]
    write_csv(source_audit / "lanl_raw_manifest.csv", lanl_records)
    lanl_pass = darpa_pass and all(record["available"] for record in lanl_records)

    attempts = {
        "darpa": [
            {"object": "ta1-theia-e3-official-6r.json.tar.gz", "drive_id": "1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0", "status": "blocked_google_drive_download_quota"},
            {"object": "ta1-theia-e3-official-6r.bin.tar.gz", "drive_id": "13rgPgHunDV1dSNX9U8bSOem56StlUmqF", "status": "blocked_google_drive_download_quota"},
            {"object": "ta1-theia-e3-official-5m.json.tar.gz", "drive_id": "1zbgWJgF7F0fI6JhViqQZoo6AWdoV5YFK", "status": "blocked_google_drive_download_quota"},
            {"object": "ta1-theia-e3-official-3.json.tar.gz", "drive_id": "1dWJecuLXZMksKAPo8348Q6L5DiccsS1u", "status": "blocked_google_drive_download_quota"},
        ],
        "lanl": {"status": "not_attempted_due_sequential_darpa_gate_failure", "official_page": "https://csr.lanl.gov/data/cyber1/", "access_note": "download form requests an email address and intended usage"},
    }
    (source_audit / "acquisition_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
    darpa_manifest = {
        "dataset": "DARPA-TC-THEIA",
        "gate": "PASS" if darpa_pass else "FAIL",
        "files": darpa_records,
        "raw_record_accounting": None,
        "reason": None if darpa_pass else "No official THEIA raw archive was acquired; preprocessing forbidden.",
    }
    lanl_manifest = {
        "dataset": "LANL-RedTeam",
        "gate": "PASS" if lanl_pass else ("NOT_RUN" if not darpa_pass else "FAIL"),
        "files": lanl_records,
        "raw_record_accounting": None,
        "reason": "Sequential DARPA-first hard stop." if not darpa_pass else None,
    }
    (manifests / "darpa_real_manifest.json").write_text(json.dumps(darpa_manifest, indent=2) + "\n", encoding="utf-8")
    (manifests / "lanl_real_manifest.json").write_text(json.dumps(lanl_manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(source_audit / "raw_record_accounting.csv", [{"dataset": "DARPA-TC-THEIA", "stage": "official raw", "count": None, "status": "NOT_AVAILABLE_SOURCE_GATE_FAILED"}, {"dataset": "LANL-RedTeam", "stage": "official raw", "count": None, "status": "NOT_RUN_SEQUENTIAL_HARD_STOP"}])
    write_csv(source_audit / "feature_lineage.csv", [{"dataset": "DARPA-TC-THEIA", "feature_name": None, "raw_field": None, "status": "NOT_GENERATED_SOURCE_GATE_FAILED"}, {"dataset": "LANL-RedTeam", "feature_name": None, "raw_field": None, "status": "NOT_RUN_SEQUENTIAL_HARD_STOP"}])
    write_csv(source_audit / "ground_truth_mapping.csv", [{"dataset": "DARPA-TC-THEIA", "node_id": None, "official_reference": None, "status": "NOT_MAPPABLE_WITHOUT_RAW_CDM"}, {"dataset": "LANL-RedTeam", "node_id": None, "official_reference": None, "status": "NOT_RUN_SEQUENTIAL_HARD_STOP"}])
    write_csv(source_audit / "darpa_event_to_edge_mapping.csv", [{"cdm_event_type": None, "edge_semantics": None, "status": "NOT_GENERATED_SOURCE_GATE_FAILED"}])

    frozen = {
        "round5_raw": {"path": "outputs/sci_round5_final/raw/benchmark_raw.csv", "expected": ROUND5_RAW_HASH},
        "round5_support": {"path": "outputs/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv", "expected": ROUND5_SUPPORT_HASH},
        "d1_synthetic_raw": {"path": "outputs/sci_defense_extension/raw/benchmark_raw.csv", "expected": D1_RAW_HASH},
        "d1_synthetic_12dataset_view": {"path": "outputs/sci_defense_extension/extended_analysis/benchmark_12dataset_view.csv", "expected": D1_VIEW_HASH},
    }
    for value in frozen.values():
        value["actual"] = file_hash(Path(value["path"])); value["matches"] = value["actual"] == value["expected"]
    (manifests / "frozen_integrity.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    archive_root = ROOT / "archive/defense_d1_synthetic"
    archive_files = []
    for path in sorted(archive_root.rglob("*")):
        if path.is_file():
            archive_files.append({"path": str(path.relative_to(archive_root)), "size_bytes": path.stat().st_size, "sha256": file_hash(path), "paper_input": False})
    archive = {"archive_name": "defense_d1_synthetic", "files": archive_files, "paper_input_excluded": True}
    (manifests / "d1_synthetic_archive.json").write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")

    gate = {
        "decision": "PAPER_READY_10_DATASET_ONLY",
        "official_raw_available": darpa_pass and lanl_pass,
        "official_ground_truth_available": gt_available and lanl_pass,
        "source_sha256_recorded": all(record["sha256"] for record in darpa_records if record["available"]),
        "darpa": {"gate": "PASS" if darpa_pass else "FAIL", "raw_available": raw_available, "ground_truth_available": gt_available, "schema_available": schema_available, "coverage_warning": "6r begins after 2018-04-10 22:30 per operational log and cannot alone cover the two earlier THEIA attacks"},
        "lanl": {"gate": "NOT_RUN" if not darpa_pass else ("PASS" if lanl_pass else "FAIL"), "reason": "sequential DARPA-first hard stop" if not darpa_pass else None},
        "preprocessing_executed": False,
        "graph_artifacts_generated": False,
        "smoke_runs_executed": 0,
        "qualification_runs_executed": 0,
        "production_runs_executed": 0,
        "new_12dataset_view_generated": False,
        "synthetic_fallback_used": False,
        "round5_frozen_unchanged": all(frozen[key]["matches"] for key in ("round5_raw", "round5_support")),
        "d1_synthetic_archived_and_excluded": archive["paper_input_excluded"] and bool(archive_files),
        "next_action": "Retry official DARPA raw acquisition after Drive quota clears; only then proceed to LANL acquisition.",
    }
    (manifests / "official_source_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (manifests / "final_paper_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    lineage = {
        "decision": gate["decision"],
        "chain_complete": False,
        "darpa": darpa_manifest,
        "lanl": lanl_manifest,
        "parser_executed": False,
        "synthetic_fallback_used": False,
        "final_graphs": [],
    }
    (manifests / "defense_real_source_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
