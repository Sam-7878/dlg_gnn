"""Round 4 fail-closed prerequisite and paper-eligibility policy."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DATASET_VERSION = "gog-sci-v2.0"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GateResult:
    authorized: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_main_prerequisites(
    dataset_root: str | Path,
    *,
    git_clean_at_start: bool,
    dependency_lock: str | Path | None,
    demo_metrics_configured: bool = False,
    chains: tuple[str, ...] = ("ethereum", "bsc", "polygon"),
) -> GateResult:
    """Validate SCI-v2 truth sources; legacy compatibility is informational."""
    root = Path(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {"dataset_root": str(root.resolve())}

    if not git_clean_at_start:
        errors.append("git_clean_at_experiment_start is false")
    lock = Path(dependency_lock) if dependency_lock else None
    if lock is None or not lock.is_file():
        errors.append("dependency lock is missing")
    else:
        evidence["dependency_lock"] = str(lock.resolve())
        evidence["dependency_lock_sha256"] = _hash_file(lock)
    if demo_metrics_configured:
        errors.append("demo or synthetic metric path is configured")

    summary_path = root / "manifests/dataset_summary.json"
    if not summary_path.is_file():
        errors.append("dataset summary is missing")
        summary = {}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evidence["dataset_summary_sha256"] = _hash_file(summary_path)
        if summary.get("dataset_version") != DATASET_VERSION:
            errors.append("dataset version is not gog-sci-v2.0")
        if summary.get("semantic_status") != "RESOLVED":
            errors.append("label semantics are not resolved")

    leakage_path = root / "audit/leakage_audit_all.json"
    if not leakage_path.is_file():
        errors.append("sample-level leakage audit is missing")
    else:
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        evidence["leakage_audit_sha256"] = _hash_file(leakage_path)
        evidence["leakage_status"] = leakage.get("status")
        if leakage.get("status") != "PASS" or leakage.get("violations") != 0:
            errors.append("sample-level leakage audit is not PASS")

    semantics_path = root / "labels/label_semantics.json"
    if not semantics_path.is_file():
        errors.append("label semantics artifact is missing")
    else:
        semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
        evidence["label_semantics_sha256"] = _hash_file(semantics_path)
        if semantics.get("semantic_status") != "RESOLVED":
            errors.append("label semantics artifact is unresolved")

    total_samples = 0
    legacy_states: dict[str, str] = {}
    split_hashes: dict[str, str] = {}
    for chain in chains:
        manifest_path = root / f"manifests/{chain}.json"
        mapping_path = root / f"mappings/{chain}_raw_to_graph.json"
        split_path = root / f"splits/{chain}_holdout_v2.json"
        normalizer_path = root / f"normalizers/{chain}/holdout/normalizer.json"
        relation_path = root / f"relations/{chain}/holdout/relation_state.json"
        required = (manifest_path, mapping_path, split_path, normalizer_path, relation_path)
        for path in required:
            if not path.is_file():
                errors.append(f"missing prerequisite: {path}")
        if not all(path.is_file() for path in required):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
        relation = json.loads(relation_path.read_text(encoding="utf-8"))
        count = int(manifest.get("files_succeeded", -1))
        total_samples += max(count, 0)
        if not manifest.get("manifest_complete") or manifest.get("files_failed") != 0:
            errors.append(f"v2 manifest is incomplete: {chain}")
        if len(manifest.get("records", [])) != count or len(mapping) != count:
            errors.append(f"v2 internal mapping count mismatch: {chain}")
        if not split.get("split_hash"):
            errors.append(f"fixed split hash is missing: {chain}")
        else:
            split_hashes[chain] = str(split["split_hash"])
        if normalizer.get("fit_scope") != "train_only" or not normalizer.get("fit_hash"):
            errors.append(f"train-only normalizer is invalid: {chain}")
        if relation.get("future_nodes_included") != 0 or relation.get("future_relations_included") != 0:
            errors.append(f"historical relation pool contains future data: {chain}")
        legacy = str(manifest.get("legacy_mapping_status", "NOT_AVAILABLE"))
        legacy_states[chain] = "PASS" if legacy == "PASS" else ("PARTIAL" if legacy == "INCOMPLETE" else "NOT_AVAILABLE")
        if legacy_states[chain] != "PASS":
            warnings.append(f"legacy compatibility is {legacy_states[chain]}: {chain}")

    evidence.update({
        "dataset_version": summary.get("dataset_version"),
        "sample_count": total_samples,
        "split_hashes": split_hashes,
        "legacy_compatibility": legacy_states,
    })
    return GateResult(not errors, tuple(errors), tuple(warnings), evidence)


ELIGIBILITY_FIELDS = (
    "dataset_version", "leakage_audit_status", "split_hash", "run_manifest",
    "resolved_config", "git_clean_at_start", "real_model_inference",
    "demo_or_synthetic_metric", "sample_count_consistent", "status",
)


def assess_paper_eligibility(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return eligibility and exact fail-closed reasons for one experiment."""
    missing = [field for field in ELIGIBILITY_FIELDS if field not in record]
    reasons = [f"missing {field}" for field in missing]
    checks = (
        (record.get("dataset_version") == DATASET_VERSION, "wrong dataset_version"),
        (record.get("leakage_audit_status") == "PASS", "leakage audit is not PASS"),
        (bool(record.get("split_hash")), "split_hash is empty"),
        (bool(record.get("run_manifest")), "run_manifest is empty"),
        (bool(record.get("resolved_config")), "resolved_config is empty"),
        (record.get("git_clean_at_start") is True, "git was dirty at experiment start"),
        (record.get("real_model_inference") is True, "real model inference not evidenced"),
        (record.get("demo_or_synthetic_metric") is False, "demo/synthetic metric detected"),
        (record.get("sample_count_consistent") is True, "sample count mismatch"),
        (record.get("status") == "SUCCESS", "experiment status is not SUCCESS"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return not reasons, reasons
