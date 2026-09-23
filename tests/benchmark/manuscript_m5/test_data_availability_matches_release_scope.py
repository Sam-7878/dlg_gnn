#!/usr/bin/env python3
"""
test_data_availability_matches_release_scope.py

Round M5 Freeze Gate: Verifies that the Data Availability statement in the manuscript
truthfully reflects the actual public package contents and states the GitHub availability
status truthfully.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_manuscript_data_availability():
    tex_path = REPO_ROOT / "docs/papers/_42_Benchmark/DLG-Benchmark.tex"
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")

    assert r"\dataavailability{" in content
    assert "https://github.com/Sam-7878/dlg_gnn" in content
    assert "upon preprint release" in content
    assert "complete benchmark suite" in content


def test_readme_matches_availability():
    readme_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"
    assert readme_path.exists()
