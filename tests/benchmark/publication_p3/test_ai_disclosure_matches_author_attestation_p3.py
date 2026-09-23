#!/usr/bin/env python3
"""
test_ai_disclosure_matches_author_attestation_p3.py

Round P3 Gate: Verifies that the manuscript AI disclosure in master,
Preprint, and MDPI LaTeX files matches the scope declared in the Author
Attestation Matrix, specifically including scientific-review feedback
alongside drafting, coding, and consistency checking.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"
ATTESTATION_MD = REPO_ROOT / "publication" / "benchmark" / "ai_use_author_attestation.md"

REQUIRED_TOOLS = ["OpenAI ChatGPT", "Anthropic Claude", "Google Gemini"]
REQUIRED_PHRASES = [
    "software-development and code-review assistance",
    "scientific-review feedback",
    "cross-document consistency checking",
    "draft organization",
    "The authors take full responsibility for the scientific claims, analyses, software, results, and final manuscript.",
]


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_manuscript_ai_disclosure_scope_p3(tex_path):
    assert tex_path.exists(), f"Missing file: {tex_path}"
    content = tex_path.read_text(encoding="utf-8")

    for tool in REQUIRED_TOOLS:
        assert tool in content, f"Missing tool '{tool}' in {tex_path.name}"

    for phrase in REQUIRED_PHRASES:
        assert phrase in content, f"Missing phrase '{phrase}' in {tex_path.name}"


def test_attestation_matrix_scope_p3():
    assert ATTESTATION_MD.exists(), f"Missing file: {ATTESTATION_MD}"
    content = ATTESTATION_MD.read_text(encoding="utf-8")

    for tool in REQUIRED_TOOLS:
        assert tool in content, f"Missing tool '{tool}' in {ATTESTATION_MD.name}"

    for phrase in REQUIRED_PHRASES:
        assert phrase in content, f"Missing phrase '{phrase}' in {ATTESTATION_MD.name}"
