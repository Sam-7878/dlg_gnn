#!/usr/bin/env python3
"""
test_ai_assistance_disclosure_present.py

Round P1 Freeze Gate: Verifies that the AI-assisted tools disclosure is
consistently present in master LaTeX, Preprints.org manuscript, and MDPI manuscript,
declaring tool usage (ChatGPT, Claude, Gemini) and author full responsibility statement.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "mdpi" / "DLG-Benchmark.tex"

REQUIRED_SUBSECTION = "Use of AI-Assisted Tools in Manuscript and Software Preparation"
REQUIRED_TOOLS = ["ChatGPT", "Claude", "Gemini"]
REQUIRED_RESPONSIBILITY_SENTENCE = (
    "The authors take full responsibility for the scientific claims, analyses, software, results, and final manuscript."
)


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_ai_disclosure_in_file(tex_path):
    assert tex_path.exists(), f"File missing: {tex_path}"
    content = tex_path.read_text(encoding="utf-8")

    assert REQUIRED_SUBSECTION in content, f"Missing subsection '{REQUIRED_SUBSECTION}' in {tex_path.name}"

    for tool in REQUIRED_TOOLS:
        assert tool in content, f"Missing mention of '{tool}' in {tex_path.name}"

    assert REQUIRED_RESPONSIBILITY_SENTENCE in content, (
        f"Missing author responsibility statement in {tex_path.name}"
    )
