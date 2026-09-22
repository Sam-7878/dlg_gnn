#!/usr/bin/env python3
"""
test_ai_disclosure_matches_author_attestation.py

Round P2 Gate: Verifies that the manuscript AI disclosure exactly mirrors
the author attestation matrix (declaring software-development and code-review
assistance alongside drafting/editing tools, with full author responsibility).
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ATTESTATION_MD = REPO_ROOT / "publication" / "benchmark" / "ai_use_author_attestation.md"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"

REQUIRED_TOOLS = ["ChatGPT", "Claude", "Gemini"]
REQUIRED_PHRASE = "software-development and code-review assistance"
REQUIRED_RESPONSIBILITY = "The authors take full responsibility for the scientific claims, analyses, software, results, and final manuscript."


def test_attestation_matrix_exists_and_complete():
    assert ATTESTATION_MD.exists(), f"Missing {ATTESTATION_MD}"
    content = ATTESTATION_MD.read_text(encoding="utf-8")

    for tool in REQUIRED_TOOLS:
        assert tool in content, f"Tool '{tool}' missing from attestation matrix"

    assert "Strictly forbidden" in content or "No" in content
    assert REQUIRED_RESPONSIBILITY in content


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_manuscript_disclosure_matches_attestation(tex_path):
    assert tex_path.exists(), f"Missing {tex_path}"
    content = tex_path.read_text(encoding="utf-8")

    for tool in REQUIRED_TOOLS:
        assert tool in content, f"Tool '{tool}' not mentioned in {tex_path.name}"

    assert REQUIRED_PHRASE in content, f"Phrase '{REQUIRED_PHRASE}' missing in {tex_path.name}"
    assert REQUIRED_RESPONSIBILITY in content, f"Responsibility statement missing in {tex_path.name}"
