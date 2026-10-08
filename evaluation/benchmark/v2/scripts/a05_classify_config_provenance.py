#!/usr/bin/env python3
"""Reconstruct A03 crypto and LANL configuration sources for Tier P evidence."""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "evaluation/benchmark/v2"
REGISTRY = V2 / "paper_ready_final/publication_evidence_a04/approved_run_registry.csv"
OUTPUT = V2 / "paper_ready_final/a05_config_provenance_crypto_lanl.csv"
CRYPTO = V2 / "scripts/a03_run_crypto_production.py"
CRYPTO_REPAIR = V2 / "scripts/a03_rerun_crypto_dlg_aug.py"
LANL_REVISION = "f0d63791da5dc031d34dd4ed0c9f9deba1020d74"
LANL_RUNNER = "scripts/defense_extension_real/run_defense_multiseed_real.py"
LANL_CONFIG = "configs/benchmark/sci_defense_extension_real.yaml"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def historic(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{LANL_REVISION}:{path}"], cwd=ROOT)


def main() -> None:
    rows = []
    for row in csv.DictReader(REGISTRY.open()):
        if row["status"] != "success" or row["source_kind"] not in {"a03_json", "lanl_real_json"}:
            continue
        model, dataset = row["model"], row["dataset"]
        if row["source_kind"] == "a03_json":
            repaired = model == "DLG-Aug"
            source = CRYPTO_REPAIR if repaired else CRYPTO
            cfg = {
                "epochs": 30 if dataset == "Ethereum" else 40,
                "batch_size": (64 if int(row["nodes"]) > 5000 else 32) if model in {"CoLA", "OCGNN"} else 0,
                "model_constructor": "SharedDLGFull(l1_epochs=20)" if repaired else f"a03_run_crypto_production.train_and_eval_detector/{model}",
                "seed": int(row["seed"]),
                "split": "numpy RandomState(seed) permutation, 60/20/20",
                "scores": "detector.decision_function(data)",
                "threshold": "validation F1 grid of 40 quantiles, 80..99.5" if repaired else "validation prevalence percentile",
                "hyperparameter_defaults": "detector constructor defaults from A03 source environment",
            }
            source_ref = str(source.relative_to(ROOT))
            source_hash = sha(source.read_bytes())
            caveat = ("Repaired DLG-Aug metric JSON has no run-bound source hash; "
                      "deterministic UUID and output schema identify the repair runner."
                      if repaired else "Original metric JSON has no run-bound source hash; runner source and recorded epoch/seed identify settings.")
        else:
            cfg = {
                "epochs": 50, "dlg_l1_epochs": 20 if model == "DLG-Aug" else None,
                "batch_size": 0, "seed": int(row["seed"]),
                "split": "stratified_split_indices(y,seed,0.2,0.2)",
                "feature_scaling": "train-set mean/std after log1p on columns max>10",
                "scores": "detector.decision_score_",
                "threshold": "validation best F1 among up to 201 score quantiles",
                "model_constructor": f"run_defense_multiseed_real.instantiate_detector/{model}",
                "hyperparameter_defaults": "model constructor defaults in historical source snapshot",
            }
            source_ref = f"{LANL_REVISION}:{LANL_RUNNER} + {LANL_CONFIG}"
            source_hash = sha(historic(LANL_RUNNER) + historic(LANL_CONFIG))
            caveat = ("Historical Git snapshot postdates raw metric file timestamps; the unique "
                      "runner's record schema, 50 epochs, model/seed, test counts, and split "
                      "support match the archived JSON. No run-bound source hash exists.")
        rows.append({
            "dataset": dataset, "model": model, "seed": row["seed"], "run_id": row["run_id"],
            "source_kind": row["source_kind"], "source_path": row["source_path"],
            "classification": "CONFIG_UNAMBIGUOUS_RECONSTRUCTION",
            "configuration_source": source_ref, "configuration_source_sha256": source_hash,
            "reconstructed_config_json": json.dumps(cfg, sort_keys=True),
            "reconstructed_config_sha256": sha(json.dumps(cfg, sort_keys=True).encode()),
            "run_bound_source_hash": "false", "caveat": caveat,
        })
    assert len(rows) == 120, f"Expected 120 affected successful runs, found {len(rows)}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "classifications": {"CONFIG_UNAMBIGUOUS_RECONSTRUCTION": len(rows)},
                      "output": str(OUTPUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
