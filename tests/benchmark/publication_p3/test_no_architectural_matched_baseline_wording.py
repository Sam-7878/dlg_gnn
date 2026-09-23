#!/usr/bin/env python3
"""
test_no_architectural_matched_baseline_wording.py

Round P3 Gate: Verifies that inaccurate architectural "matched baseline" wording
is eliminated from Related Work across master, Preprint, and MDPI manuscripts,
accurately describing DLG-Base as the historical non-augmentation implementation.
"""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"

FORBIDDEN_PHRASE = "matched non-augmentation implementation"


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_no_architectural_matched_baseline_wording(tex_path):
    assert tex_path.exists(), f"Missing file: {tex_path}"
    content = tex_path.read_text(encoding="utf-8")

    assert FORBIDDEN_PHRASE not in content, (
        f"Forbidden phrase '{FORBIDDEN_PHRASE}' found in {tex_path.name}"
    )
    # Check that historical non-augmentation description is present (allowing optional \emph{})
    assert re.search(r"historical non-augmentation implementation is reported as (\\emph\{)?DLG-Base", content), (
        f"Historical non-augmentation wording missing in {tex_path.name}"
    )
