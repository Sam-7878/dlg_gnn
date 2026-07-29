from __future__ import annotations

import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .evidence_index import sha256_file
from .schema import EvidenceRecord


SCAN_ROOTS = ("results_sci", "configs/sci", "tests", "docs/work_reports/100_stream_mc_update", "docs/work_reports/101_stream_mc_check_result", "docs/work_reports/102_stream_mc_update2")
ALLOWED_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".csv", ".parquet", ".md", ".txt", ".log", ".xml", ".png", ".svg", ".pdf", ".py"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", "archive"}
DEFAULT_INCLUDE = (
    "tests/data/**", "tests/streaming/**", "tests/selection/**",
    "tests/experiments/**", "tests/profiling/**", "tests/reporting/**",
    "tests/unit/test_mc_dropout.py", "tests/unit/test_phase1_level1.py",
    "tests/unit/test_phase2_level1_data_and_eval.py",
    "tests/unit/test_phase3_level2.py", "tests/unit/test_phase4_fusion.py",
)
DEFAULT_EXCLUDE = ("tests/llama/**", "tests/micro_rag/**", "tests/mock/**")


def _test_in_scope(relative: str) -> bool:
    if not relative.startswith("tests/"):
        return True
    if any(fnmatch.fnmatch(relative, pattern) for pattern in DEFAULT_EXCLUDE):
        return False
    return any(fnmatch.fnmatch(relative, pattern) for pattern in DEFAULT_INCLUDE)


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", *args],
            cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_metadata(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    return {
        "branch": _git(repo_root, "branch", "--show-current"),
        "git_sha": _git(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
        "dirty_paths": status.splitlines() if status and status != "unknown" else [],
    }


def environment_metadata() -> dict[str, Any]:
    def version(name: str) -> str:
        try: return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: return "NOT_INSTALLED"
    hardware: dict[str, Any] = {"machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        hardware["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        hardware["ram_bytes"] = "UNKNOWN"
    cuda = "UNAVAILABLE"
    try:
        import torch
        cuda = torch.version.cuda or "CPU_ONLY"
        hardware["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    except Exception:
        hardware["gpu"] = "UNKNOWN"
    return {"os": platform.platform(), "python": platform.python_version(), "torch": version("torch"), "torch_geometric": version("torch-geometric"), "pygod": version("pygod"), "cuda": cuda, "hardware": hardware}


def _category(path: Path) -> str:
    parts = set(path.parts)
    if "tests" in parts: return "test"
    if "configs" in parts: return "config"
    if path.suffix.lower() in {".png", ".svg", ".pdf"}: return "figure"
    if path.suffix.lower() == ".log": return "log"
    if "results_sci" in parts: return "result"
    if "dataset_manifests" in parts: return "data"
    return "code" if path.suffix.lower() == ".py" else "report"


def collect_evidence(repo_root: str | Path, *, include_archive: bool = False) -> list[EvidenceRecord]:
    root = Path(repo_root).resolve()
    git_sha = git_metadata(root)["git_sha"]
    paths: list[Path] = []
    for relative in SCAN_ROOTS:
        scan_root = root / relative
        if not scan_root.exists(): continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES: continue
            if any(part in SKIP_PARTS for part in path.parts) and not include_archive: continue
            if path.name.startswith("DLG_StreamMC_SCI_Integrated_Verification_Report") or path.name in {"DLG_StreamMC_SCI_Evidence_Index.csv", "DLG_StreamMC_SCI_Report_Validation.json", "REPORT_MANIFEST.json"}: continue
            if not _test_in_scope(path.relative_to(root).as_posix()): continue
            paths.append(path)
    # The reporting allowlist defines the whole gog_fraud package and profiler as in-scope code.
    for relative in ("src/gog_fraud", "src/profiling"):
        source_dir = root / relative
        for path in source_dir.rglob("*.py") if source_dir.exists() else ():
            if "__pycache__" not in path.parts:
                paths.append(path)
    records: list[EvidenceRecord] = []
    for index, path in enumerate(sorted(set(paths))):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        category = _category(Path(relative))
        sections = {
            "data": "4,5", "test": "3,5,12,22", "config": "2,6,22",
            "result": "6-19", "figure": "9-17", "log": "19,Appendix E",
        }.get(category, "2,3,Appendix D")
        records.append(EvidenceRecord(
            evidence_id=f"EV-{index + 1:05d}", category=category,
            experiment_id="", path=relative, sha256=sha256_file(path),
            generated_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            producer="collector.py", config_hash="", git_sha=git_sha,
            status="valid", used_in_report=category in {"data", "test", "config", "result", "code", "report"},
            report_sections=sections,
        ))
    return records


def load_dataset_manifests(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifests: dict[str, Any] = {}
    candidates = list((root / "results_sci" / "manifests").glob("*.json")) if (root / "results_sci" / "manifests").exists() else []
    candidates += list((root / "docs/work_reports/100_stream_mc_update/artifacts/dataset_manifests").glob("*.json"))
    for path in candidates:
        if path.name in {"build_summary.json", ".hash_index.json"}: continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("chain"):
                data["_source"] = path.relative_to(root).as_posix()
                data["_manifest_sha256"] = sha256_file(path)
                data["file_hash_count"] = len(data.get("file_hashes", {}))
                data.pop("file_hashes", None)
                manifests[str(data["chain"])] = data
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def load_experiment_registry(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve() / "results_sci"
    if not root.exists(): return []
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), 2):
                    if row.get("experiment_id"):
                        rows.append({**row, "result_file": path.as_posix(), "row_id": row_number})
        except (OSError, csv.Error):
            continue
    return rows


def load_configs(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    result = []
    for path in sorted((root / "configs/sci").rglob("*.yaml")) if (root / "configs/sci").exists() else []:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            result.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "config": payload})
        except (OSError, yaml.YAMLError):
            result.append({"path": path.relative_to(root).as_posix(), "status": "corrupt"})
    return result
