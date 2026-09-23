#!/usr/bin/env python3
"""
test_graphical_abstract_primary_counts.py

Round P2 Gate: Verifies that the Graphical Abstract generation script and assets
declare the exact, frozen benchmark counts:
  - 10 primary heterogeneous graphs (3 real financial + 7 synthetic injected)
  - External validation on LANL-RedTeam
  - 8 detector configurations (6 baselines + 2 DLG variants)
  - 355 primary runs and 45 sensitivity control runs
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"
PNG_PATH = REPO_ROOT / "publication" / "benchmark" / "preprints" / "graphical_abstract.png"


def test_graphical_abstract_png_exists():
    assert PNG_PATH.exists(), f"Missing graphical abstract image: {PNG_PATH}"
    assert PNG_PATH.stat().st_size > 50 * 1024, "Graphical abstract PNG is suspiciously small"


def test_graphical_abstract_script_counts():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # 10 primary graphs
    assert "10 Heterogeneous Primary Graphs" in content or "10 Primary Graphs" in content
    assert "Elliptic, DGraphFin, BitcoinOTC" in content
    assert "Yelp, Amazon, Flickr" in content
    assert "Reddit, Cora, CiteSeer, PubMed" in content

    # External validation
    assert "External Validation: LANL-RedTeam" in content

    # 8 detector configurations
    assert "8 Detector Configurations" in content
    assert "6 Established Graph Baselines" in content
    assert "2 DLG Variants: DLG-Base & DLG-Aug" in content

    # Runs
    assert "355 successful primary runs" in content
    assert "45 Targeted Sensitivity Controls" in content or "45 sensitivity" in content
