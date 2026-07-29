from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import VerificationIssue


def detect_issues(*, repo_root: str | Path, datasets: dict[str, Any], experiments: list[dict[str, Any]], git: dict[str, Any], configs: list[dict[str, Any]]) -> list[VerificationIssue]:
    root = Path(repo_root).resolve()
    issues: list[VerificationIssue] = []

    def add(severity: str, category: str, message: str, evidence: tuple[str, ...] = ()) -> None:
        issues.append(VerificationIssue(f"ISS-{len(issues)+1:03d}", severity, category, message, evidence))

    if not experiments:
        add("CRITICAL", "experiment", "No immutable scientific experiment result rows exist; detection, routing, calibration, resource, and statistical claims are not verified.")
    missing_chains = sorted({"ethereum", "bsc", "polygon"} - set(datasets))
    if missing_chains:
        add("HIGH", "dataset", f"Missing dataset manifests: {', '.join(missing_chains)}")
    for chain, manifest in datasets.items():
        evidence = (str(manifest.get("_source", "")),)
        if not manifest.get("manifest_complete", False):
            add("HIGH", "dataset", f"{chain} dataset manifest is incomplete or partial.", evidence)
        if int(manifest.get("files_failed", 0)):
            add("HIGH", "dataset", f"{chain} dataset manifest contains failed files.", evidence)
        if manifest.get("hash_verification") != "full":
            add("HIGH", "data_hash", f"{chain} does not have full raw-file hash coverage.", evidence)
        ratio = manifest.get("positive_ratio")
        if ratio is not None and float(ratio) > 0.5:
            add("MEDIUM", "label", f"{chain} fraud positive ratio is {float(ratio):.3f}; label semantics and fraud-oriented corpus sampling must be disclosed.", evidence)
    if git.get("dirty"):
        add("HIGH", "provenance", "The report working tree is dirty, so the commit SHA alone cannot reproduce the evaluated code state.")
    manifest_source = root / "src/gog_fraud/data/io/dataset_manifest.py"
    if manifest_source.is_file():
        source = manifest_source.read_text(encoding="utf-8", errors="replace")
        if "st_mtime_ns" not in source or 'cached.get("size")' not in source:
            add("HIGH", "data_hash", "The resumable hash cache lacks metadata-guarded invalidation.", ("src/gog_fraud/data/io/dataset_manifest.py",))
        if "truncated" not in source or "manifest_complete" not in source:
            add("HIGH", "dataset", "Partial manifests are not explicitly marked incomplete.", ("src/gog_fraud/data/io/dataset_manifest.py",))
    if not (root / "results_sci/manifests").exists():
        add("HIGH", "provenance", "The standard results_sci/manifests registry is missing.")
    if not list((root / "configs/sci").rglob("*.yaml")) if (root / "configs/sci").exists() else True:
        add("HIGH", "config", "SCI configs are missing.")
    elif len(configs) < 2:
        add("MEDIUM", "config", "SCI configs are not separated into immutable data/model/routing/hardware snapshots.")
    if not any((root / name).is_file() for name in ("requirements-sci-lock.txt", "requirements-lock.txt", "poetry.lock")):
        add("MEDIUM", "environment", "A reproducible Python dependency lock is missing.")
    split_dir = root / "results_sci/splits"
    if not split_dir.exists() or not list(split_dir.glob("*_holdout_v1.json")):
        add("HIGH", "temporal", "Fixed temporal split/hash artifacts are missing.")
    audit_paths = list(split_dir.glob("*_leakage_audit_v1.json")) if split_dir.exists() else []
    if audit_paths:
        import json
        incomplete = [path.name for path in audit_paths if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS"]
        if incomplete:
            add("HIGH", "temporal", f"Sample-level leakage audit is not PASS: {', '.join(incomplete)}")
    return issues
