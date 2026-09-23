#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"
INCLUDE_FILES = ["README.md", "INSTALL.md", "CITATION.cff", "LICENSE", "pyproject.toml", "environment.yml", "requirements-core.txt", "scripts/reproduce_frozen_artifacts.py", "scripts/verify_environment.py", "scripts/publication/build_p5_release.py"]
INCLUDE_TREES = ["artifacts", "provenance", "configs/benchmark", "docs/math", "src/gog_fraud", "experiments/benchmark"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = json.loads((ROOT / "publication/benchmark/publication_metadata.json").read_text())
    metadata.update({"commit_sha": commit, "is_public_release": False, "release_state": "prepared_for_public_release"})
    release_meta = OUT.parent / "release_metadata.json"
    release_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    files = [ROOT / p for p in INCLUDE_FILES]
    for tree in INCLUDE_TREES:
        files.extend(p for p in (ROOT / tree).rglob("*") if p.is_file())
    files = sorted({p for p in files if not (set(p.parts) & EXCLUDE_PARTS) and p.suffix not in EXCLUDE_SUFFIXES})
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in files:
            zf.write(path, "dlg_gnn/" + path.relative_to(ROOT).as_posix())
        zf.writestr("dlg_gnn/release_metadata.json", json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "sha256": sha256(OUT), "files": len(files) + 1, "commit": commit}, indent=2))
if __name__ == "__main__":
    main()
