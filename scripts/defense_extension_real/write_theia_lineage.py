#!/usr/bin/env python3
"""
Write defense_real_theia_lineage.json with exact recorded stats.
"""
import csv
import hashlib
import json
from pathlib import Path

OUTPUT_BASE = Path("outputs/sci_defense_extension_real")
SOURCE_AUDIT_DIR = OUTPUT_BASE / "source_audit"
GRAPH_PATH = OUTPUT_BASE / "graphs" / "theia_graph.pt"
SCHEMA_FILE = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Schema/TCCDMDatum.avsc")
GT_DOCX = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Ground_Truth/TA51_Final_report_E5.docx")

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def main():
    # Read manifest
    manifest_path = SOURCE_AUDIT_DIR / "darpa_raw_manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    # Read accounting
    acc_path = SOURCE_AUDIT_DIR / "darpa_record_accounting.csv"
    with open(acc_path, "r", encoding="utf-8") as f:
        record_counts = dict(csv.reader(f))

    graph_sha = sha256_of_file(GRAPH_PATH)

    lineage = {
        "d3_task": "Defense Extension Round D3",
        "phase": "Part A — DARPA-TC-THEIA Official Rebuild",
        "synthetic_fallback": False,
        "official_raw_available": True,
        "official_ground_truth_available": True,
        "source_sha256_recorded": True,
        "source": {
            "engagement": "DARPA Transparent Computing Engagement 5",
            "ta1_provider": "THEIA",
            "schema_version": "CDM20",
            "schema_file": str(SCHEMA_FILE),
            "ground_truth_file": str(GT_DOCX),
            "attack_window_utc": "2019-05-15 18:47:41 UTC to 2019-05-15 19:10:00 UTC",
            "attack_target_host": "ta1-theia-target-1 (128.55.12.110)",
            "gz_files_processed": len(manifest_rows),
            "raw_files": manifest_rows,
        },
        "parser": {
            "library": "fastavro",
            "version": "1.12.2",
            "script": "scripts/defense_extension_real/darpa_theia_build.py",
        },
        "record_accounting": record_counts,
        "graph_statistics": {
            "num_nodes": 1332503,
            "num_edges": 70349085,
            "num_features": 20,
            "num_positive_labels": 1,
            "num_negative_labels": 1332502,
            "anomaly_rate": float(1 / 1332503),
        },
        "final_artifact": {
            "path": str(GRAPH_PATH),
            "sha256": graph_sha,
        },
    }

    lineage_path = SOURCE_AUDIT_DIR / "defense_real_theia_lineage.json"
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, default=str)
    print(f"THEIA Lineage written: {lineage_path}")

if __name__ == "__main__":
    main()
