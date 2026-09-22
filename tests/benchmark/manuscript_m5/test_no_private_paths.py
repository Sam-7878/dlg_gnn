#!/usr/bin/env python3
"""
test_no_private_paths.py

Round M5 Freeze Gate: Scans all files inside the public release archive to guarantee
that zero private absolute paths (/mnt/d/_work/, d:\\_work\\, file:///d:) exist.
"""

from pathlib import Path
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_M5_Release.zip"

BANNED_PATTERNS = [
    "/mnt/d" + "/_work/",
    "d:" + "\\_work\\",
    "file:" + "///d:",
]


def test_release_contains_no_private_paths():
    assert RELEASE_ZIP.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
            zf.extractall(tmp_path)

        for p in tmp_path.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv", ".yml", ".cff"):
                if p.name == "test_no_private_paths.py":
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                for b in BANNED_PATTERNS:
                    assert b not in txt, f"Private path '{b}' found in release file {p.relative_to(tmp_path)}"
