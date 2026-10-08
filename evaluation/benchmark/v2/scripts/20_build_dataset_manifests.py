#!/usr/bin/env python3
"""20_build_dataset_manifests.py: Build dataset freeze manifests and integrity hashes for 13+1 datasets (Phase H).

Indexes and hashes the canonical 13 primary datasets + 1 external (LANL) dataset in:
  /mnt/d/_Work/_data/DLG/

Outputs:
  - evaluation/benchmark/v2/manifests/datasets/dataset_manifests.json
  - evaluation/benchmark/v2/manifests/datasets/dataset_manifests.csv
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path("/mnt/d/_Work/_data/DLG")
OUTPUT_DIR = REPO_ROOT / "evaluation" / "benchmark" / "v2" / "manifests" / "datasets"


def compute_sha256(filepath: Path, max_bytes: int = 100 * 1024 * 1024) -> str:
    """Compute sha256 for a file (capped at max_bytes for multi-gigabyte archives)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        bytes_read = 0
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            bytes_read += len(chunk)
            if bytes_read >= max_bytes:
                break
    return h.hexdigest()


DATASETS_META = [
    # Real-label financial / fraud
    {"name": "Elliptic", "domain": "Blockchain / AML", "label_type": "real", "rel_path": "Elliptic/raw/elliptic_txs_classes.csv", "estimated_nodes": 203769, "estimated_edges": 234355},
    {"name": "DGraphFin", "domain": "Financial Loan Fraud", "label_type": "real", "rel_path": "DGraphFin/dgraphfin.npz", "estimated_nodes": 3700550, "estimated_edges": 4300999},
    {"name": "Yelp", "domain": "Review Spam", "label_type": "real", "rel_path": "Yelp/yelp_reviews.csv", "estimated_nodes": 45954, "estimated_edges": 3846979},
    {"name": "Amazon", "domain": "E-Commerce Fraud", "label_type": "real", "rel_path": "Amazon/amazon_reviews.csv", "estimated_nodes": 11944, "estimated_edges": 4398392},
    {"name": "BitcoinOTC", "domain": "Cryptocurrency Trust", "label_type": "real", "rel_path": "BitcoinOTC/raw/soc-sign-bitcoinotc.csv", "estimated_nodes": 5881, "estimated_edges": 35592},
    {"name": "Cora-Syn", "domain": "Citation / Benchmark", "label_type": "synthetic", "rel_path": "Cora/Cora/raw/ind.cora.x", "estimated_nodes": 2708, "estimated_edges": 5429},
    {"name": "CiteSeer-Syn", "domain": "Citation / Benchmark", "label_type": "synthetic", "rel_path": "CiteSeer/CiteSeer/raw/ind.citeseer.x", "estimated_nodes": 3327, "estimated_edges": 4732},
    {"name": "PubMed-Syn", "domain": "Medical Citation", "label_type": "synthetic", "rel_path": "PubMed/PubMed/raw/ind.pubmed.x", "estimated_nodes": 19717, "estimated_edges": 44338},
    {"name": "Flickr-Syn", "domain": "Social Network", "label_type": "synthetic", "rel_path": "Flickr/raw", "estimated_nodes": 89250, "estimated_edges": 899756},
    {"name": "Reddit-Syn", "domain": "Social Forum", "label_type": "synthetic", "rel_path": "Reddit/reddit_posts.csv", "estimated_nodes": 232965, "estimated_edges": 11606919},
    {"name": "Twitch-Syn", "domain": "Streaming Social", "label_type": "synthetic", "rel_path": "Twitch/EN", "estimated_nodes": 7126, "estimated_edges": 35324},
    {"name": "CryptoScamDB", "domain": "Web3 Scam / Phishing", "label_type": "real", "rel_path": "CryptoScamDB/urls.csv", "estimated_nodes": 5600, "estimated_edges": 14200},
    {"name": "CryptoScamTracker", "domain": "Web3 Fraud Addresses", "label_type": "real", "rel_path": "CryptoScamTracker/dan_dataset.csv", "estimated_nodes": 8400, "estimated_edges": 21000},
    # External dataset
    {"name": "LANL-RedTeam", "domain": "Enterprise Cybersecurity", "label_type": "external_real", "rel_path": "LANL-RedTeam/redteam.txt", "estimated_nodes": 156117, "estimated_edges": 140000000},
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = []

    print(f"Scanning and hashing 13+1 datasets from {DATA_ROOT}...")

    for meta in DATASETS_META:
        target_path = DATA_ROOT / meta["rel_path"]
        exists = target_path.exists()
        
        file_size = target_path.stat().st_size if exists and target_path.is_file() else (
            sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file()) if exists else 0
        )
        
        sha256 = compute_sha256(target_path) if exists and target_path.is_file() else "DIRECTORY_OR_MULTI_PART"

        entry = {
            "dataset_name": meta["name"],
            "domain": meta["domain"],
            "label_type": meta["label_type"],
            "path": str(target_path),
            "exists": exists,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "sha256_head": sha256[:16] if sha256 != "DIRECTORY_OR_MULTI_PART" else sha256,
            "estimated_nodes": meta["estimated_nodes"],
            "estimated_edges": meta["estimated_edges"],
            "status": "VERIFIED_PRESENT" if exists else "MISSING",
        }
        manifests.append(entry)
        status_str = f"PRESENT ({entry['size_mb']} MB)" if exists else "MISSING"
        print(f"  [{meta['name']:18s}] -> {status_str}")

    df = pd.DataFrame(manifests)
    json_path = OUTPUT_DIR / "dataset_manifests.json"
    csv_path = OUTPUT_DIR / "dataset_manifests.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)
    df.to_csv(csv_path, index=False)

    print(f"\nManifests successfully written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
