#!/usr/bin/env python3
"""
test_no_unrelated_defense_scope.py

Round M5 Freeze Gate: Verifies that the public release package contains zero
unrelated DARPA/THEIA historical defense traces or unassociated modules.
"""

from pathlib import Path
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"

BANNED_STRINGS = ["th" + "eia", "da" + "rpa"]


def test_release_contains_no_unrelated_defense_scope():
    assert RELEASE_ZIP.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
            zf.extractall(tmp_path)

        for p in tmp_path.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv"):
                if p.name == "test_no_unrelated_defense_scope.py":
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                for b in BANNED_STRINGS:
                    assert b not in txt, f"Unrelated string '{b}' found in release file {p.relative_to(tmp_path)}"
