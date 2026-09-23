#!/usr/bin/env python3
"""
test_root_readme_is_benchmark_landing_page.py

Round P4 Publication Gate:
Verifies that the repository root README.md serves as a primary benchmark
reproduction landing page rather than a historical application README.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_readme_landing_page_structure():
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")

    # Required sections
    assert "# A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs" in content or "DLG-GNN Benchmark" in content
    assert "SeongSu Park" in content
    assert "Ki-Hyung Kim" in content
    assert "0009-0008-4056-3875" in content
    assert "0000-0002-2321-4475" in content
    assert "Mode 1: Frozen-Artifact Instant Verification" in content or "Mode 1" in content
    assert "Mode 2: Full Benchmark Re-Execution" in content or "Mode 2" in content
    assert "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c" in content
    assert "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914" in content
    assert "10.20944/preprints202609.0848.v1" in content
