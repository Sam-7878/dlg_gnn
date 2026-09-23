#!/usr/bin/env python3
"""
test_abstract_word_count_200_max.py

Round P3 Gate: Verifies that the abstract across master, Preprint, and MDPI
manuscripts strictly complies with the Applied Sciences author guideline:
  - Maximum 200 words (and >= 150 words)
  - Single paragraph structure (no blank line breaks / no \\par)
"""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"


def extract_abstract_text(tex_path: Path) -> str:
    content = tex_path.read_text(encoding="utf-8")
    # In master / MDPI: \abstract{...}
    # In Preprint: \begin{abstract}...\end{abstract}
    m = re.search(r"\\abstract\{([\s\S]*?)\}\s*\\keyword\{", content)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", content)
    if m2:
        # Strip \vspace and \noindent\textbf{Keywords:} if present inside
        text = m2.group(1)
        text = re.sub(r"\\vspace\{[^}]+\}", "", text)
        text = re.sub(r"\\noindent\\textbf\{Keywords:\}.*", "", text)
        return text.strip()
    raise ValueError(f"Could not extract abstract from {tex_path.name}")


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_abstract_length_within_200_words(tex_path):
    assert tex_path.exists(), f"Missing file: {tex_path}"
    abstract = extract_abstract_text(tex_path)

    # Clean citations and special characters for word count
    words = abstract.split()
    word_count = len(words)

    assert word_count <= 200, (
        f"Abstract in {tex_path.name} exceeds 200 words: {word_count} words"
    )
    assert word_count >= 150, (
        f"Abstract in {tex_path.name} is suspiciously short: {word_count} words"
    )


@pytest.mark.parametrize("tex_path", [MASTER_TEX, PREPRINT_TEX, MDPI_TEX])
def test_abstract_is_single_paragraph(tex_path):
    abstract = extract_abstract_text(tex_path)
    assert "\n\n" not in abstract, f"Abstract in {tex_path.name} contains multiple paragraphs"
    assert r"\par" not in abstract, f"Abstract in {tex_path.name} contains \\par command"
