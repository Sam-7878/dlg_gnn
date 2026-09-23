#!/usr/bin/env python3
"""
test_capacity_control_intent_is_sensitivity_not_causal.py

Round P3 Gate: Verifies that the capacity-control section frames controls
as sensitivity probing rather than causal identification across master,
Preprint, and MDPI manuscripts:
  - DLG-Aug-Permuted: "To probe sensitivity to node alignment while preserving the marginal distribution..."
  - DLG-Base-70: "To probe sensitivity to an additional global-epoch budget..."
  - Causal phrasing ("To determine whether...", "To verify whether...") eliminated.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"

FORBIDDEN_CAUSAL = [
    "To determine whether performance gains depend on",
    "To verify whether DLG-Aug's multi-stage budget explains",
]

REQUIRED_SENSITIVITY = [
    "To probe sensitivity to node alignment while preserving the marginal distribution",
    "To probe sensitivity to an additional global-epoch budget",
]


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_capacity_control_phrasing_is_sensitivity_not_causal(tex_path):
    assert tex_path.exists(), f"Missing file: {tex_path}"
    content = tex_path.read_text(encoding="utf-8")

    for f_phrase in FORBIDDEN_CAUSAL:
        assert f_phrase not in content, (
            f"Forbidden causal phrase '{f_phrase}' found in {tex_path.name}"
        )

    for r_phrase in REQUIRED_SENSITIVITY:
        assert r_phrase in content, (
            f"Required sensitivity phrase '{r_phrase}' missing in {tex_path.name}"
        )
