"""Fail-closed SCI evaluation orchestrator with immutable provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gog_fraud.data.io.dataset_manifest import build_dataset_manifest
from gog_fraud.experiments.manifest import RunManifest

log = logging.getLogger(__name__)
RESULT_DIRS = ("manifests", "tuning", "main", "ablation", "streaming", "calibration", "temporal", "cross_chain", "memory", "statistics", "tables", "figures", "logs", "splits")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("SCI config must contain a mapping")
    return config


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_status(repo_root: Path) -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-c", f"safe.directory={repo_root.resolve()}", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _create_layout(root: Path) -> None:
    for name in RESULT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def _validate_prerequisites(
    config: dict[str, Any], output_root: Path, *, require_clean_git: bool,
    repo_root: Path, initial_git_dirty: bool | None = None,
) -> tuple[list[str], list[Path]]:
    errors: list[str] = []
    evidence: list[Path] = []
    dirty = initial_git_dirty
    if dirty is None:
        _, dirty = _git_status(repo_root)
    if require_clean_git and dirty:
        errors.append("working tree is dirty")
    dataset = config.get("dataset", {})
    for required in ("version", "canonical_root", "physical_transaction_root", "labels_path"):
        if not dataset.get(required):
            errors.append(f"dataset.{required} is required")
    manifest_dir = output_root / "manifests"
    split_dir = output_root / "splits"
    for chain in dataset.get("chains", []):
        manifest_path = manifest_dir / f"{chain}.json"
        split_path = split_dir / f"{chain}_holdout_v1.json"
        rolling_path = split_dir / f"{chain}_rolling5_v1.json"
        audit_path = split_dir / f"{chain}_leakage_audit_v1.json"
        for path in (manifest_path, split_path, rolling_path, audit_path):
            if not path.is_file():
                errors.append(f"missing prerequisite: {path}")
            else:
                evidence.append(path)
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not payload.get("manifest_complete") or payload.get("files_failed") != 0:
                errors.append(f"dataset manifest is incomplete: {chain}")
            if payload.get("canonical_root") != dataset.get("canonical_root"):
                errors.append(f"canonical root mismatch: {chain}")
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("status") != "PASS" or audit.get("violations") != 0:
                errors.append(f"sample-level leakage audit not PASS: {chain}")
    return errors, evidence


def _run_explicit_commands(config: dict[str, Any], root: Path, experiment_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stage in config.get("pipeline_commands", []):
        name, command = str(stage["name"]), [str(item) for item in stage["command"]]
        started = datetime.now(timezone.utc).isoformat()
        log_path = root / "logs" / f"{experiment_dir.name}_{name}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=handle, stderr=subprocess.STDOUT, check=False)
        records.append({"name": name, "command": command, "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(), "returncode": completed.returncode, "log": str(log_path), "log_sha256": _sha256(log_path)})
        if completed.returncode:
            break
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", "--output", dest="output_root")
    parser.add_argument("--skip-dataset-scan", action="store_true")
    parser.add_argument("--run-configured-stages", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path.cwd().resolve()
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    experiment = config.get("experiment", {})
    output_root = Path(args.output_root or experiment.get("output_root", "results_sci")).resolve()
    _, initial_git_dirty = _git_status(repo_root)
    _create_layout(output_root)
    config_digest = hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}_{experiment.get('name', 'dlg_streammc')}_{config_digest}"
    experiment_dir = output_root / "manifests" / experiment_id
    if args.resume:
        candidates = sorted((output_root / "manifests").glob(f"*_{experiment.get('name', 'dlg_streammc')}_{config_digest}"))
        if candidates:
            experiment_dir = candidates[-1]; experiment_id = experiment_dir.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = experiment_dir / "resolved_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    dataset = config.get("dataset", {})
    dataset_files: list[Path] = [config_path, resolved_config]
    labels_path = dataset.get("labels_path")
    if labels_path:
        dataset_files.append(Path(labels_path))
    manifest = RunManifest.capture(experiment_id=experiment_id, config=config, seed=int((experiment.get("seeds") or [42])[0]), dataset_files=dataset_files, repo_root=repo_root)
    manifest_path = experiment_dir / "run_manifest.json"
    audit_path = experiment_dir / "audit.json"
    errors: list[str] = []
    try:
        if not args.skip_dataset_scan:
            physical_root = dataset.get("physical_transaction_root") or dataset.get("root")
            for chain in dataset.get("chains", []):
                built = build_dataset_manifest(
                    physical_root, chain=str(chain), labels_path=labels_path, max_files=args.max_files,
                    canonical_root=dataset.get("canonical_root"), preprocessing_source=physical_root,
                    source_version=dataset.get("version"), hash_index_path=output_root / "manifests/hash_index.json",
                )
                built.write(output_root / f"manifests/{chain}.json", output_root / f"manifests/{chain}.csv")
        errors, prerequisite_files = _validate_prerequisites(
            config, output_root, require_clean_git=args.require_clean_git,
            repo_root=repo_root, initial_git_dirty=initial_git_dirty,
        )
        if args.strict and not config.get("pipeline_commands"):
            errors.append("strict evaluation requires explicit real pipeline_commands; no paper stage is configured")
        if errors and args.strict:
            raise RuntimeError("; ".join(errors))
        stage_records = _run_explicit_commands(config, output_root, experiment_dir) if (args.run_configured_stages or args.strict) and not errors else []
        failed = next((record for record in stage_records if record["returncode"]), None)
        status = "failed" if failed else ("blocked" if errors else "success")
        audit = {
            "experiment_id": experiment_id, "status": status, "prerequisite_errors": errors,
            "configured_stage_records": stage_records, "paper_metrics_generated": bool(stage_records) and not failed,
            "resolved_config": str(resolved_config), "resolved_config_hash": _sha256(resolved_config),
        }
        audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        outputs = [resolved_config, audit_path, *prerequisite_files]
        manifest.finalize(status=status, output_files=outputs, failure=failed or ({"type": "PrerequisiteError", "messages": errors} if errors else None))
        manifest.write(manifest_path)
        return 0 if status == "success" else 2
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        audit = {
            "experiment_id": experiment_id,
            "status": "failed",
            "prerequisite_errors": errors,
            "configured_stage_records": [],
            "paper_metrics_generated": False,
            "resolved_config": str(resolved_config),
            "resolved_config_hash": _sha256(resolved_config),
            "failure": failure,
        }
        audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest.finalize(status="failed", output_files=[resolved_config, audit_path], failure=failure)
        manifest.write(manifest_path)
        log.exception("SCI evaluation failed closed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
