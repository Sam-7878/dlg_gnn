#!/usr/bin/env python3
"""
test_preprint_mdpi_scientific_content_parity.py

Round P1 Freeze Gate: Verifies 100% scientific content parity between
Preprints.org manuscript (publication/benchmark/preprints/DLG-Benchmark-Preprint.tex) and
MDPI manuscript (publication/benchmark/mdpi/DLG-Benchmark.tex).
"""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"


def extract_abstract(text: str) -> str:
    # Preprint style: \begin{abstract} ... \end{abstract}
    m_prep = re.search(r"\\begin\{abstract\}(.*?)(?:\\vspace|\\end\{abstract\})", text, re.DOTALL)
    if m_prep:
        return re.sub(r"\s+", " ", m_prep.group(1)).strip()

    # MDPI style: \abstract{ ... }
    m_mdpi = re.search(r"\\abstract\{(.*?)\}\s*\n", text, re.DOTALL)
    if m_mdpi:
        return re.sub(r"\s+", " ", m_mdpi.group(1)).strip()

    return ""


def test_manuscript_files_exist():
    assert PREPRINT_TEX.exists(), f"Missing {PREPRINT_TEX}"
    assert MDPI_TEX.exists(), f"Missing {MDPI_TEX}"


def test_abstract_scientific_parity():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")

    prep_abs = extract_abstract(prep_text)
    mdpi_abs = extract_abstract(mdpi_text)

    assert len(prep_abs) > 200, "Preprint abstract not found or too short"
    assert len(mdpi_abs) > 200, "MDPI abstract not found or too short"

    # Preprints abstract and MDPI abstract must have 100% identical wording
    assert prep_abs == mdpi_abs, "Preprint abstract does not exactly match MDPI abstract"


def test_section_titles_parity():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")

    # Extract all section levels
    prep_sections = re.findall(r"\\(?:section|subsection|subsubsection)\{([^}]+)\}", prep_text)
    mdpi_sections = re.findall(r"\\(?:section|subsection|subsubsection)\{([^}]+)\}", mdpi_text)

    # Filter out MDPI specific back-matter sections
    ignore = {
        "Supplementary Materials", "Author Contributions", "Funding",
        "Institutional Review Board Statement", "Informed Consent Statement",
        "Data Availability Statement", "Conflicts of Interest"
    }
    prep_main = [s for s in prep_sections if s not in ignore]
    mdpi_main = [s for s in mdpi_sections if s not in ignore]

    assert prep_main == mdpi_main, f"Section structure mismatch:\nPreprint={prep_main}\nMDPI={mdpi_main}"


def test_table_inputs_parity():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")

    prep_inputs = re.findall(r"\\input\{([^}]+)\}", prep_text)
    mdpi_inputs = re.findall(r"\\input\{([^}]+)\}", mdpi_text)

    assert prep_inputs == mdpi_inputs, f"Input mismatch: Preprint={prep_inputs} vs MDPI={mdpi_inputs}"


def test_table_and_figure_labels_parity():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")

    prep_tabs = sorted(re.findall(r"\\label\{(tab:[^}]+)\}", prep_text))
    mdpi_tabs = sorted(re.findall(r"\\label\{(tab:[^}]+)\}", mdpi_text))
    assert prep_tabs == mdpi_tabs, f"Table labels mismatch: {set(prep_tabs) ^ set(mdpi_tabs)}"

    prep_figs = sorted(re.findall(r"\\label\{(fig:[^}]+)\}", prep_text))
    mdpi_figs = sorted(re.findall(r"\\label\{(fig:[^}]+)\}", mdpi_text))
    assert prep_figs == mdpi_figs, f"Figure labels mismatch: {set(prep_figs) ^ set(mdpi_figs)}"


def test_equations_parity():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")

    prep_eqs = sorted(re.findall(r"\\label\{(eq:[^}]+)\}", prep_text))
    mdpi_eqs = sorted(re.findall(r"\\label\{(eq:[^}]+)\}", mdpi_text))
    assert prep_eqs == mdpi_eqs, f"Equation labels mismatch: {set(prep_eqs) ^ set(mdpi_eqs)}"


def test_citations_parity():
    prep_text = PREPRINT_TEX.read_text(encoding="utf-8")
    mdpi_text = MDPI_TEX.read_text(encoding="utf-8")

    # Extract all citation keys
    def extract_cites(txt):
        keys = set()
        for m in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", txt):
            for k in m.split(","):
                keys.add(k.strip())
        return keys

    prep_cites = extract_cites(prep_text)
    mdpi_cites = extract_cites(mdpi_text)
    assert prep_cites == mdpi_cites, f"Citation mismatch: {prep_cites ^ mdpi_cites}"
