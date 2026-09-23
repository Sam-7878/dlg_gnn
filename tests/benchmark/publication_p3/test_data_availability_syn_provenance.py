#!/usr/bin/env python3
"""
test_data_availability_syn_provenance.py

Round P3 Gate: Verifies that the Data Availability statement across master,
Preprint, and MDPI manuscripts explicitly describes the provenance of the seven
'-Syn' benchmarks (constructed from public base graphs with anomaly injection)
and distinguishes them from real-label graphs (BitcoinOTC from SNAP, Elliptic, DGraphFin).
"""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"


def extract_data_availability(content: str) -> str:
    m = re.search(r"\\dataavailability\{([\s\S]*?)\}", content)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_data_availability_syn_provenance(tex_path):
    assert tex_path.exists(), f"Missing file: {tex_path}"
    content = tex_path.read_text(encoding="utf-8")
    da = extract_data_availability(content)

    # Injected dataset distinction
    assert "seven ``-Syn'' benchmarks" in da or "seven \"-Syn\" benchmarks" in da or "seven `-Syn` benchmarks" in da
    assert "constructed from public base graphs" in da
    assert "anomaly-injection protocol" in da

    # Real-label distinctions
    assert "BitcoinOTC" in da
    assert "SNAP" in da
    assert "Elliptic" in da
    assert "DGraphFin" in da
    assert "LANL-RedTeam" in da

    # Official repository
    assert "https://github.com/Sam-7878/dlg_gnn" in da
