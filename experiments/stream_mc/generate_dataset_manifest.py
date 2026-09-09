"""
experiments/generate_dataset_manifest.py

Generates results/dataset_manifest.json detailing:
  - Dataset fingerprint (SHA256 of synthetic context config / records)
  - Number of events, fraud ratio
  - Seed list and generator parameters
  - Output format schema
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="results/dataset_manifest.json")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    cfg_path = root / args.config
    cfg_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
    cfg_hash = hashlib.sha256(cfg_text.encode()).hexdigest()

    # Calculate raw predictions hash if available
    raw_dir = root / "results" / "raw_predictions"
    pred_hashes = {}
    if raw_dir.exists():
        for fp in sorted(raw_dir.glob("*.csv")):
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            pred_hashes[fp.name] = h

    manifest = {
        "dataset_name": "dlg_gnn_synthetic_multimodal_graph",
        "n_events": 1000,
        "fraud_ratio": 0.10,
        "seeds": [7, 17, 27, 37, 47],
        "generator_config_hash": cfg_hash,
        "config_file": str(args.config),
        "split_policy": "Chronological (pre-transaction window 300s, no future leakage)",
        "prediction_artifacts": pred_hashes,
        "provenance": {
            "graph_type": "Heterogeneous dynamic financial graph (accounts, transactions)",
            "context_type": "Synthetic communication & situational text",
            "kb_nodes": 28,
            "kb_edges": 59,
        }
    }

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Dataset manifest generated at {out_path}")


if __name__ == "__main__":
    main()
