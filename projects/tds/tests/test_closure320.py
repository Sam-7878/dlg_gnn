"""Pytest verification for TDS R320 publication closure."""
import json
import hashlib
from pathlib import Path
import pytest

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
TDS = ROOT / 'projects/tds'
SUB320 = TDS / 'paper/submission_320'
PRE_OUT = TDS / 'submission/preprints/320'
MDPI_OUT = TDS / 'submission/applsci/320'
AUTH_OUT = TDS / 'submission/author_records/320'
REPORTS_OUT = TDS / 'reports/320'

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def test_shared_science_byte_parity():
    """Verify 100% byte-level parity between Preprints and MDPI science sources."""
    for fn in ['manuscript_body.tex', 'references.bib', 'declarations.tex']:
        p_pre = SUB320 / 'preprints' / fn
        p_mdpi = SUB320 / 'mdpi_applsci_anomaly' / fn
        assert p_pre.exists(), f"Missing {p_pre}"
        assert p_mdpi.exists(), f"Missing {p_mdpi}"
        assert p_pre.read_bytes() == p_mdpi.read_bytes(), f"Mismatch in {fn}!"

def test_architecture_figure_present():
    """Verify Section 3 architecture overview figure exists in both venues."""
    for venue in ['preprints', 'mdpi_applsci_anomaly']:
        fig_pdf = SUB320 / venue / 'figures/fig_tds_architecture.pdf'
        fig_png = SUB320 / venue / 'figures/fig_tds_architecture.png'
        assert fig_pdf.exists() and fig_pdf.stat().st_size > 0
        assert fig_png.exists() and fig_png.stat().st_size > 0

    body = (SUB320 / 'preprints/manuscript_body.tex').read_text()
    assert 'fig:tds-architecture' in body
    assert 'figures/fig_tds_architecture.pdf' in body

def test_enriched_related_work_and_references():
    """Verify Related Work section expansion and references integration."""
    body = (SUB320 / 'preprints/manuscript_body.tex').read_text()
    bib = (SUB320 / 'preprints/references.bib').read_text()

    # Check subsections
    assert r'\subsection{Unsupervised and Supervised Graph Anomaly Detection}' in body
    assert r'\subsection{Blockchain Transaction Networks, Smart Contracts, and Label Shift}' in body
    assert r'\subsection{Graph Neural Networks versus Tabular Baselines}' in body
    assert r'\subsection{Dynamic and Temporal Graph Learning Limitations}' in body

    # Check key citations
    for key in ['park2026dlg', 'dilini2026comprehensive', 'heidrich2026systematic', 'jiang2025mvcg', 'ma2023survey']:
        assert f'\\cite{{{key}}}' in body or f'{key}' in body
        assert f'{key},' in bib

def test_submission_packages_exist_and_match_manifest():
    """Verify compiled submission packages match candidate manifest."""
    manifest = json.loads((REPORTS_OUT / 'frozen_candidate_manifest_320.json').read_text())
    
    # Preprints
    pre_pdf = PRE_OUT / 'preprint_article.pdf'
    pre_supp = PRE_OUT / 'supplement.pdf'
    pre_zip = PRE_OUT / 'preprint_source.zip'
    assert pre_pdf.exists() and sha256_file(pre_pdf) == manifest['artifacts']['preprints']['preprint_article.pdf']['sha256']
    assert pre_supp.exists() and sha256_file(pre_supp) == manifest['artifacts']['preprints']['supplement.pdf']['sha256']
    assert pre_zip.exists() and sha256_file(pre_zip) == manifest['artifacts']['preprints']['preprint_source.zip']['sha256']

    # MDPI
    mdpi_pdf = MDPI_OUT / 'applsci_article.pdf'
    mdpi_supp = MDPI_OUT / 'supplement.pdf'
    mdpi_zip = MDPI_OUT / 'mdpi_source.zip'
    assert mdpi_pdf.exists() and sha256_file(mdpi_pdf) == manifest['artifacts']['applsci']['applsci_article.pdf']['sha256']
    assert mdpi_supp.exists() and sha256_file(mdpi_supp) == manifest['artifacts']['applsci']['supplement.pdf']['sha256']
    assert mdpi_zip.exists() and sha256_file(mdpi_zip) == manifest['artifacts']['applsci']['mdpi_source.zip']['sha256']

def test_final_publication_gate_status():
    """Verify final publication gate status is strictly HOLD_AUTHOR_ACTION."""
    gate = json.loads((REPORTS_OUT / 'final_publication_gate_320.json').read_text())
    assert gate['scientific_evidence_pass'] is True
    assert gate['dual_build_pass'] is True
    assert gate['rights_verified'] is True
    assert gate['funding_official_proof_verified'] is True
    assert gate['exact_file_author_signoff_complete'] is False
    assert gate['status'] == 'HOLD_AUTHOR_ACTION'

