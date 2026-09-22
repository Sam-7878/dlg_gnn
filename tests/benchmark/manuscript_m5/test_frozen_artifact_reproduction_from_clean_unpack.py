#!/usr/bin/env python3
"""
test_frozen_artifact_reproduction_from_clean_unpack.py

Round M5 Freeze Gate: Tests complete Mode 1 reproduction from a completely clean
unpack of the public release zip file with zero reliance on the surrounding repo.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_M5_Release.zip"


def test_clean_unpack_reproduction():
    assert RELEASE_ZIP.exists(), f"Release zip missing: {RELEASE_ZIP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
            zf.extractall(tmp_path)

        unpacked_root = tmp_path / "dlg_gnn"
        assert unpacked_root.exists(), "Root dlg_gnn folder missing in unpack"

        # Execute reproduction script
        cmd = [
            sys.executable,
            str(unpacked_root / "scripts/reproduce_frozen_artifacts.py"),
            "--artifact-root", str(unpacked_root / "artifacts"),
            "--output-dir", str(unpacked_root / "reproduced_tables")
        ]
        res = subprocess.run(cmd, cwd=unpacked_root, capture_output=True, text=True)
        assert res.returncode == 0, f"Reproduction failed:\nStdout: {res.stdout}\nStderr: {res.stderr}"

        # Verify generated tables
        out_dir = unpacked_root / "reproduced_tables"
        expected_tables = [
            "table_appendix_all_detailed.tex",
            "table_statistical_rankings.tex",
            "table_capacity_controls_summary.tex",
            "table_capacity_controls_paired_deltas.tex",
            "table_dlg_architecture_budget.tex",
            "table_lanl_diagnostics.tex"
        ]
        for tbl in expected_tables:
            tbl_path = out_dir / tbl
            assert tbl_path.exists(), f"Expected table {tbl} was not generated"
            content = tbl_path.read_text(encoding="utf-8")
            assert len(content) > 50, f"Table {tbl} is empty or truncated"
