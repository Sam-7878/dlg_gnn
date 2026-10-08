#!/usr/bin/env python3
"""
scripts/manuscript/build_public_release_p4.py

Round P4 Public Release Builder:
Outputs:
outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip

Key Enhancements in Round P4:
- Canonical Release Filename: DLG_GNN_Benchmark_v1.0.0_preprint.zip
- Zero stale metadata: Uses root README.md (Benchmark Landing Page), updated CITATION.cff,
  INSTALL.md, and release_metadata.json (prepared_for_public_release).
- Zero 'Under Review' or 'goat-bank' strings in packaged artifacts.
- Exact Author ORCIDs: SeongSu Park (0009-0008-4056-3875) and Ki-Hyung Kim (0000-0002-2321-4475).
- Environment versioning: frozen_execution_environment.json & current_reproduction_environment.json.
- Full automated clean unpack and reproduction verification test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_public_release_p4")

REPO_ROOT = Path(__file__).resolve().parents[2]
M5_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5"
RELEASE_DIR = M5_DIR / "release"
PROVENANCE_DIR = M5_DIR / "provenance"
ART_DIR = M5_DIR / "artifacts"
ZIP_PATH = RELEASE_DIR / "DLG_GNN_Benchmark_v1.0.0_preprint.zip"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_release():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Building Round P4 Public Release Bundle at {ZIP_PATH}...")

    # File mapping: (zip_relative_path, local_source_path)
    file_map: dict[str, Path] = {}

    # 1. Models: src/gog_fraud/models/
    models_dir = REPO_ROOT / "src" / "gog_fraud" / "models"
    for py_file in models_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 2. Data modules: src/gog_fraud/data/
    data_dir = REPO_ROOT / "src" / "gog_fraud" / "data"
    for py_file in data_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 3. Evaluation modules: src/gog_fraud/evaluation/
    eval_dir = REPO_ROOT / "src" / "gog_fraud" / "evaluation"
    for py_file in eval_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 4. Experiments modules: src/gog_fraud/experiments/
    exp_dir = REPO_ROOT / "src" / "gog_fraud" / "experiments"
    for py_file in exp_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 5. Core pipelines: src/gog_fraud/pipelines/
    pipe_dir = REPO_ROOT / "src" / "gog_fraud" / "pipelines"
    for py_file in pipe_dir.glob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # __init__.py and analysis helper files
    file_map["dlg_gnn/src/gog_fraud/__init__.py"] = REPO_ROOT / "src" / "gog_fraud" / "__init__.py"
    file_map["dlg_gnn/src/analysis/__init__.py"] = REPO_ROOT / "src" / "analysis" / "__init__.py"
    file_map["dlg_gnn/src/analysis/utils.py"] = REPO_ROOT / "src" / "analysis" / "utils.py"

    # 6. Runners & configs
    file_map["dlg_gnn/experiments/benchmark/run_sci_round5_final.py"] = REPO_ROOT / "experiments" / "benchmark" / "run_sci_round5_final.py"
    file_map["dlg_gnn/experiments/benchmark/run_capacity_controls_m3.py"] = REPO_ROOT / "experiments" / "benchmark" / "run_capacity_controls_m3.py"
    file_map["dlg_gnn/configs/benchmark/sci_round5_final.yaml"] = REPO_ROOT / "configs" / "benchmark" / "sci_round5_final.yaml"

    # 7. Scripts
    file_map["dlg_gnn/scripts/reproduce_frozen_artifacts.py"] = REPO_ROOT / "scripts" / "reproduce_frozen_artifacts.py"
    file_map["dlg_gnn/scripts/verify_environment.py"] = REPO_ROOT / "scripts" / "verify_environment.py"
    manuscript_scripts = [
        "generate_capacity_controls_m4.py",
        "generate_m3_architecture_manifest.py",
        "calculate_lanl_diagnostics_m3.py",
        "generate_appendix_tables.py",
    ]
    for s in manuscript_scripts:
        p = REPO_ROOT / "scripts" / "manuscript" / s
        if p.exists():
            file_map[f"dlg_gnn/scripts/manuscript/{s}"] = p

    # 8. Artifacts folder (all small derived CSVs, JSONs, manifests)
    for art_file in ART_DIR.rglob("*"):
        if art_file.is_file():
            rel = art_file.relative_to(ART_DIR)
            file_map[f"dlg_gnn/artifacts/{rel.as_posix()}"] = art_file

    # 9. Math documentation
    file_map["dlg_gnn/docs/math/exact_sparse_reconstruction.md"] = REPO_ROOT / "docs" / "math" / "exact_sparse_reconstruction.md"

    # 10. Metadata, packaging, installation (from live root & provenance)
    file_map["dlg_gnn/README.md"] = REPO_ROOT / "README.md"
    file_map["dlg_gnn/LICENSE"] = REPO_ROOT / "LICENSE"
    file_map["dlg_gnn/pyproject.toml"] = REPO_ROOT / "pyproject.toml"
    file_map["dlg_gnn/release_metadata.json"] = RELEASE_DIR / "release_metadata.json"
    file_map["dlg_gnn/CITATION.cff"] = REPO_ROOT / "CITATION.cff"
    file_map["dlg_gnn/INSTALL.md"] = REPO_ROOT / "INSTALL.md"
    file_map["dlg_gnn/environment.yml"] = REPO_ROOT / "environment.yml"
    file_map["dlg_gnn/requirements-core.txt"] = REPO_ROOT / "requirements-core.txt"
    file_map["dlg_gnn/requirements-cuda121.txt"] = REPO_ROOT / "requirements-cuda121.txt"

    file_map["dlg_gnn/provenance/environment_manifest.json"] = PROVENANCE_DIR / "environment_manifest.json"
    file_map["dlg_gnn/provenance/frozen_execution_environment.json"] = PROVENANCE_DIR / "frozen_execution_environment.json"
    file_map["dlg_gnn/provenance/current_reproduction_environment.json"] = PROVENANCE_DIR / "current_reproduction_environment.json"

    # 11. Tests: tests/benchmark/manuscript_m5/
    m5_tests = REPO_ROOT / "tests" / "benchmark" / "manuscript_m5"
    for tp in m5_tests.glob("*.py"):
        file_map[f"dlg_gnn/tests/benchmark/manuscript_m5/{tp.name}"] = tp

    # Sanitize check: assure zero banned patterns in packaged files
    banned_patterns = [
        "/mnt/d/_work/",
        "d:\\_work\\",
        "file:///d:",
        "darpa",
        "theia",
        "goat-bank",
        "under review",
    ]
    for zip_rel, src_path in file_map.items():
        if src_path.is_file() and src_path.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv", ".yml", ".cff"):
            if zip_rel.endswith("test_no_private_paths.py") or zip_rel.endswith("test_no_unrelated_defense_scope.py"):
                continue
            txt = src_path.read_text(encoding="utf-8", errors="ignore").lower()
            for b in banned_patterns:
                if b in txt:
                    raise AssertionError(f"Banned pattern '{b}' detected in {src_path} (mapped to {zip_rel})")

    # Generate source tree manifest
    manifest_rows = []
    for zip_rel, src_path in sorted(file_map.items()):
        file_hash = sha256_file(src_path)
        manifest_rows.append({"relative_path": zip_rel, "sha256": file_hash, "size_bytes": src_path.stat().st_size})

    manifest_csv_path = PROVENANCE_DIR / "source_tree_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha = sha256_file(manifest_csv_path)
    (PROVENANCE_DIR / "source_tree_manifest_sha256.txt").write_text(manifest_sha + "\n", encoding="utf-8")
    log.info(f"Generated source tree manifest: {len(manifest_rows)} files (SHA: {manifest_sha[:16]}...)")

    file_map["dlg_gnn/provenance/source_tree_manifest.csv"] = manifest_csv_path
    file_map["dlg_gnn/provenance/source_tree_manifest_sha256.txt"] = PROVENANCE_DIR / "source_tree_manifest_sha256.txt"

    # Create Zip
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for zip_rel, src_path in file_map.items():
            zf.write(src_path, zip_rel)

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Successfully packaged P4 Public Release Bundle: {ZIP_PATH} ({size_mb:.2f} MB, {len(file_map)} files)")


def verify_release():
    log.info("Verifying release bundle from clean temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)

        root = tmp_path / "dlg_gnn"
        assert root.exists(), "Root folder dlg_gnn missing in zip"
        assert (root / "README.md").exists(), "README.md missing"
        assert (root / "release_metadata.json").exists(), "release_metadata.json missing"
        assert (root / "CITATION.cff").exists(), "CITATION.cff missing"
        assert (root / "INSTALL.md").exists(), "INSTALL.md missing"
        assert (root / "artifacts" / "primary" / "benchmark_raw.csv").exists(), "artifacts missing"
        assert (root / "scripts" / "reproduce_frozen_artifacts.py").exists(), "reproduce script missing"
        assert (root / "experiments" / "benchmark" / "run_sci_round5_final.py").exists(), "primary runner missing"
        assert (root / "experiments" / "benchmark" / "run_capacity_controls_m3.py").exists(), "controls runner missing"

        # Check banned strings
        banned = ["/mnt/d/_work/", "d:\\_work\\", "file:///d:", "theia", "darpa", "goat-bank", "under review"]
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv", ".yml", ".cff"):
                if p.name in ("test_no_private_paths.py", "test_no_unrelated_defense_scope.py"):
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                for b in banned:
                    assert b not in txt, f"Unpacked file {p.name} contains banned '{b}'"

        # Execute Mode 1 Master Reproduction from clean unpack!
        python_cmd = sys.executable
        repro_cmd = [
            python_cmd,
            str(root / "scripts" / "reproduce_frozen_artifacts.py"),
            "--artifact-root", str(root / "artifacts"),
            "--output-dir", str(root / "reproduced_tables")
        ]
        res = subprocess.run(repro_cmd, cwd=root, capture_output=True, text=True)
        if res.returncode != 0:
            log.error(f"Reproduction failed in clean unpack:\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}")
            raise RuntimeError(f"Clean unpack reproduction failed with code {res.returncode}")

        log.info("Clean unpack reproduction executed with 0 errors!")
        assert (root / "reproduced_tables" / "table_appendix_all_detailed.tex").exists()
        assert (root / "reproduced_tables" / "table_capacity_controls_paired_deltas.tex").exists()
        assert (root / "reproduced_tables" / "table_lanl_diagnostics.tex").exists()

        log.info("Clean unpack public release bundle verification PASSED 100%!")


if __name__ == "__main__":
    build_release()
    verify_release()
