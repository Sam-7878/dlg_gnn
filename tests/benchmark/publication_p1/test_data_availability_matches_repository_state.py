#!/usr/bin/env python3
"""
test_data_availability_matches_repository_state.py

Round P1 Freeze Gate: Verifies that the Data Availability statement in both
Preprints.org manuscript and MDPI manuscript accurately states that code,
sparse backends, and replication scripts are made openly available upon
preprint release at the repository URL declared in release metadata.
"""

from pathlib import Path
import json
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"


def extract_data_availability(text: str) -> str:
    m = re.search(r"\\dataavailability\{([\s\S]*?)\}", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def test_data_availability_statements_exist_and_match():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")
    mast_text = MASTER_TEX.read_text(encoding="utf-8")

    prep_da = extract_data_availability(prep_text)
    mdpi_da = extract_data_availability(mdpi_text)
    mast_da = extract_data_availability(mast_text)

    assert len(prep_da) > 100, "Preprint data availability statement missing"
    assert len(mdpi_da) > 100, "MDPI data availability statement missing"
    assert prep_da == mdpi_da, "Data availability statement differs between Preprint and MDPI"
    assert prep_da == mast_da, "Data availability statement differs between Preprint and Master"


def test_data_availability_references_repository_url():
    meta = json.loads(RELEASE_META.read_text(encoding="utf-8"))
    repo_url = meta.get("repository_url")
    assert repo_url, "Missing repository_url in release metadata"

    prep_da = extract_data_availability(PREPRINT_TEX.read_text(encoding="utf-8"))
    assert repo_url in prep_da, f"Repository URL {repo_url} not in Data Availability statement"
    assert "openly available upon preprint release" in prep_da
    assert "Elliptic" in prep_da
    assert "DGraphFin" in prep_da
    assert "LANL-RedTeam" in prep_da
