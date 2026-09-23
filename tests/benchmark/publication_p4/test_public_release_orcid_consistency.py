#!/usr/bin/env python3
"""
test_public_release_orcid_consistency.py

Round P4 Publication Gate:
Verifies that author ORCIDs across root README.md, CITATION.cff,
and release metadata strictly match canonical author identities:
- SeongSu Park: 0009-0008-4056-3875
- Ki-Hyung Kim: 0000-0002-2321-4475
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

PARK_ORCID = "0009-0008-4056-3875"
KIM_ORCID = "0000-0002-2321-4475"
OLD_PARK_ORCID = "0009-0005-0424-7809"


def test_citation_cff_orcids():
    cff = REPO_ROOT / "CITATION.cff"
    assert cff.exists()
    content = cff.read_text(encoding="utf-8")
    assert PARK_ORCID in content
    assert KIM_ORCID in content
    assert OLD_PARK_ORCID not in content


def test_root_readme_orcids():
    readme = REPO_ROOT / "README.md"
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    assert PARK_ORCID in content
    assert KIM_ORCID in content
    assert OLD_PARK_ORCID not in content


def test_manuscript_orcids():
    tex_path = REPO_ROOT / "docs/papers/_42_Benchmark/DLG-Benchmark.tex"
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")
    assert PARK_ORCID in content
    assert KIM_ORCID in content
    assert OLD_PARK_ORCID not in content
