#!/usr/bin/env python3
"""
test_cover_letter_dataset_count.py

Round P2 Gate: Verifies that the MDPI cover letter accurately declares
ten primary graphs, composed of 3 real-label financial graphs and 7 controlled
synthetic-injection graphs (eliminating the erroneous 'six PyGOD' claim).
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVER_LETTER = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "cover_letter.md"


def test_cover_letter_dataset_counts():
    assert COVER_LETTER.exists(), f"Missing {COVER_LETTER}"
    content = COVER_LETTER.read_text(encoding="utf-8")

    # Erroneous count must be absent
    assert "six PyGOD" not in content
    assert "6 PyGOD" not in content
    assert "six synthetic" not in content

    # Accurate counts must be present
    assert "ten primary graphs" in content or "10 primary graphs" in content
    assert "three real-label financial/blockchain datasets" in content or "3 real" in content
    assert "seven controlled synthetic-injection datasets" in content or "7 synthetic" in content
    assert "Elliptic, DGraphFin, and BitcoinOTC" in content
    assert "Yelp, Amazon, Flickr, Reddit, Cora, CiteSeer, and PubMed" in content
