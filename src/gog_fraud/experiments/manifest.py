from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


@dataclass
class RunManifest:
    experiment_id: str
    config: dict[str, Any]
    cli_args: list[str]
    seed: int
    dataset_hashes: dict[str, str]
    git_sha: str
    git_dirty: bool
    diff_hash: str
    environment: dict[str, Any]
    started_at: str
    ended_at: str | None = None
    status: str = "running"
    outputs: dict[str, str] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def capture(cls, *, experiment_id: str, config: dict[str, Any], seed: int, dataset_files: list[str | Path] | None = None, repo_root: str | Path = ".") -> "RunManifest":
        root = Path(repo_root)
        def git(*args: str) -> str:
            return subprocess.check_output(["git", "-c", f"safe.directory={root.resolve()}", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        try:
            sha, status, diff = git("rev-parse", "HEAD"), git("status", "--porcelain"), git("diff", "--binary")
        except (OSError, subprocess.CalledProcessError):
            sha, status, diff = "unknown", "unavailable", ""
        files = [Path(item) for item in (dataset_files or [])]
        return cls(
            experiment_id=experiment_id, config=config, cli_args=sys.argv[1:], seed=seed,
            dataset_hashes={str(path.resolve()): _sha256(path) for path in files if path.is_file()},
            git_sha=sha, git_dirty=bool(status), diff_hash=hashlib.sha256(diff.encode()).hexdigest(),
            environment={"os": platform.platform(), "python": platform.python_version(), "torch": _version("torch"), "torch_geometric": _version("torch-geometric"), "pygod": _version("pygod")},
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def finalize(self, *, status: str, output_files: list[str | Path] = (), failure: dict[str, Any] | None = None) -> None:
        self.status = status; self.ended_at = datetime.now(timezone.utc).isoformat()
        self.outputs = {str(Path(path).resolve()): _sha256(Path(path)) for path in output_files if Path(path).is_file()}
        if failure:
            self.failures.append(failure)

    def write(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), sort_keys=True, indent=2) + "\n", encoding="utf-8")
