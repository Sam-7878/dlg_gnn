#!/usr/bin/env python3
"""
test_graphical_abstract_syn_suffixes.py

Round P3 Gate: Verifies that all seven synthetic-injection datasets in the
Graphical Abstract script carry the explicit '-Syn' provenance suffix:
  - Yelp-Syn, Amazon-Syn, Flickr-Syn, Reddit-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"

REQUIRED_SYN_DATASETS = [
    "Yelp-Syn",
    "Amazon-Syn",
    "Flickr-Syn",
    "Reddit-Syn",
    "Cora-Syn",
    "CiteSeer-Syn",
    "PubMed-Syn",
]


def test_graphical_abstract_has_syn_suffixes():
    assert SCRIPT_PATH.exists(), f"Missing file: {SCRIPT_PATH}"
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    for syn_name in REQUIRED_SYN_DATASETS:
        assert syn_name in content, f"Missing '-Syn' dataset name '{syn_name}' in graphical abstract script"
