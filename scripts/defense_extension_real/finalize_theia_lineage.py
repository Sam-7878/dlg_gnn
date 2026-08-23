#!/usr/bin/env python3
"""
Finalize DARPA THEIA E5 manifest and lineage JSON from built graph and accounting CSVs.
"""
import csv
import hashlib
import json
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

THEIA_DATA_DIR = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Data/theia/theia-20260822T022150Z-1-001/theia")
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
    files = sorted(THEIA_DATA_DIR.glob("*.bin.*.gz"))
    log.info(f"Computing manifest for {len(files)} files...")
    manifest_rows = []
    for fpath in files:
        log.info(f"  SHA-256: {fpath.name} ({fpath.stat().st_size/1024**2:.1f} MB)...")
        sha = sha256_of_file(fpath)
        manifest_rows.append({
            "filename": fpath.name,
            "official_topic": fpath.name.split(".bin.")[0] if ".bin." in fpath.name else fpath.name,
            "compressed_size_bytes": fpath.stat().st_size,
            "sha256": sha,
            "cdm_version": "CDM20",
            "ta1_provider": "THEIA",
            "engagement": "Engagement 5",
        })

    # Save darpa_raw_manifest.csv (and darpa_e5_raw_manifest.csv)
    for mname in ["darpa_raw_manifest.csv", "darpa_e5_raw_manifest.csv"]:
        mpath = SOURCE_AUDIT_DIR / mname
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
        log.info(f"Manifest written: {mpath}")

    # Load graph stats
    data = torch.load(GRAPH_PATH, weights_only=False)
    graph_sha = sha256_of_file(GRAPH_PATH)

    # Load accounting
    acc_path = SOURCE_AUDIT_DIR / "darpa_record_accounting.csv"
    record_counts = {}
    if acc_path.exists():
        with open(acc_path, "r", encoding="utf-8") as f:
            record_counts = dict(csv.reader(f))

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
            "gz_files_processed": len(files),
            "raw_files": manifest_rows,
        },
        "parser": {
            "library": "fastavro",
            "version": "1.12.2",
            "script": "scripts/defense_extension_real/darpa_theia_build.py",
        },
        "record_accounting": record_counts,
        "graph_statistics": {
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.shape[1],
            "num_features": data.x.shape[1],
            "num_positive_labels": int(data.y.sum().item()),
            "num_negative_labels": int((data.y == 0).sum().item()),
            "anomaly_rate": float(data.y.sum().item() / max(1, data.num_nodes)),
        },
        "final_artifact": {
            "path": str(GRAPH_PATH),
            "sha256": graph_sha,
        },
    }

    lineage_path = SOURCE_AUDIT_DIR / "defense_real_theia_lineage.json"
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, default=str)
    log.info(f"Lineage written: {lineage_path}")
    print("THEIA Lineage Finalized Successfully!")

if __name__ == "__main__":
    main()
