"""Truthful SCI evaluation orchestrator.

This entrypoint creates provenance and dataset-audit artifacts first. Expensive model
stages run only when explicit commands are present in the config; it never synthesizes
or estimates paper metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gog_fraud.data.io.dataset_manifest import build_dataset_manifest
from gog_fraud.experiments.manifest import RunManifest


log = logging.getLogger(__name__)
RESULT_DIRS = ("manifests", "tuning", "main", "ablation", "streaming", "calibration", "temporal", "cross_chain", "memory", "statistics", "tables", "figures", "logs")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("SCI config must contain a mapping")
    return config


def _create_layout(root: Path) -> None:
    for name in RESULT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def _run_explicit_commands(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stage in config.get("pipeline_commands", []):
        name, command = str(stage["name"]), list(stage["command"])
        started = datetime.now(timezone.utc).isoformat()
        log_path = root / "logs" / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, text=True, stdout=handle, stderr=subprocess.STDOUT, check=False)
        records.append({"name": name, "command": command, "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(), "returncode": completed.returncode, "log": str(log_path)})
        if completed.returncode:
            break
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", "--output", dest="output_root")
    parser.add_argument("--skip-dataset-scan", action="store_true")
    parser.add_argument("--run-configured-stages", action="store_true", help="run only pipeline_commands explicitly listed in the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    experiment = config.get("experiment", {})
    output_root = Path(args.output_root or experiment.get("output_root", "results_sci")).resolve()
    _create_layout(output_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}_{experiment.get('name', 'dlg_streammc')}_audit"
    labels_path = config.get("dataset", {}).get("labels_path")
    dataset_root = config.get("dataset", {}).get("root")
    dataset_files = [config_path] + ([Path(labels_path)] if labels_path else [])
    manifest = RunManifest.capture(experiment_id=experiment_id, config=config, seed=int((experiment.get("seeds") or [42])[0]), dataset_files=dataset_files, repo_root=Path.cwd())
    manifest_path = output_root / "manifests" / f"{experiment_id}.json"

    try:
        dataset_artifacts: list[str] = []
        if not args.skip_dataset_scan:
            if not dataset_root:
                raise ValueError("dataset.root is required for the audit")
            for chain in config.get("dataset", {}).get("chains", []):
                log.info("auditing dataset chain=%s", chain)
                dataset_manifest = build_dataset_manifest(dataset_root, chain=str(chain), labels_path=labels_path)
                json_path = output_root / "manifests" / f"dataset_{chain}.json"
                csv_path = output_root / "manifests" / f"dataset_{chain}.csv"
                dataset_manifest.write(json_path, csv_path)
                dataset_artifacts.extend((str(json_path), str(csv_path)))
        stage_records = _run_explicit_commands(config, output_root) if args.run_configured_stages else []
        failed = next((record for record in stage_records if record["returncode"]), None)
        audit = {
            "experiment_id": experiment_id,
            "status": "failed" if failed else "audit_complete",
            "dataset_manifests": dataset_artifacts,
            "configured_stage_records": stage_records,
            "paper_metrics_generated": False,
            "note": "No paper metric is generated unless an explicit real experiment command runs successfully.",
        }
        audit_path = output_root / "manifests" / f"{experiment_id}_audit.json"
        audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest.finalize(status="failed" if failed else "success", output_files=[*dataset_artifacts, audit_path], failure=failed)
        manifest.write(manifest_path)
        return 1 if failed else 0
    except Exception as exc:
        manifest.finalize(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        manifest.write(manifest_path)
        log.exception("SCI audit failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
