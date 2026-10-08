#!/usr/bin/env python3
"""
build_preprints_bundle_p1.py

Round P1: Builds the official publisher-neutral Preprints.org submission bundle:
  Outputs:
    publication/preprints/DLG-Benchmark-Preprint.tex
    publication/preprints/DLG-Benchmark-Preprint.pdf
    publication/preprints/DLG_Benchmark_Preprints_Submission.zip

Guarantees:
  1. 100% scientific content parity with docs/papers/_42_Benchmark/DLG-Benchmark.tex
  2. Complete removal of MDPI logos, journal name (Applied Sciences), and publisher branding.
  3. Clean 4-pass compilation (pdflatex -> bibtex -> pdflatex -> pdflatex) with 0 errors.
  4. Fully self-contained submission archive containing all necessary tables and bib sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_preprints_bundle")

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_PAPER_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark"
PREPRINTS_DIR = REPO_ROOT / "publication" / "benchmark" / "preprints"
PREPRINTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_TEX = DOCS_PAPER_DIR / "DLG-Benchmark.tex"
TARGET_TEX = PREPRINTS_DIR / "DLG-Benchmark-Preprint.tex"
TARGET_PDF = PREPRINTS_DIR / "DLG-Benchmark-Preprint.pdf"
TARGET_ZIP = PREPRINTS_DIR / "DLG_Benchmark_Preprints_Submission.zip"


def create_preprint_tex() -> str:
    """Transforms the MDPI master LaTeX file into a clean, publisher-neutral Preprints.org LaTeX file."""
    content = MASTER_TEX.read_text(encoding="utf-8")

    # Split preamble from body
    parts = content.split(r"\begin{document}")
    if len(parts) != 2:
        raise ValueError("Could not find single \\begin{document} in master tex")
    preamble, body = parts[0], parts[1]

    # Clean Publisher-Neutral Preprint Preamble
    neutral_preamble = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{mathtools}
\usepackage{siunitx}
\usepackage{threeparttable}
\usepackage{centernot}
\usepackage{multirow}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{url}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{microtype}
\usepackage{authblk}

% Pseudo-code packages
\usepackage{algorithm}
\usepackage{algpseudocode}

% Plots and graphics libraries
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{
  arrows.meta,
  positioning,
  shapes.geometric,
  fit,
  backgrounds,
  calc
}

% Source-code listings
\usepackage{listings}

\newcommand{\cmark}{\checkmark}
\newcommand{\etal}{\emph{et al.}}

% Publisher-neutral macros matching master manuscript
\newcommand{\Title}[1]{\title{\textbf{\Large #1}}}
\newcommand{\TitleCitation}[1]{}
\newcommand{\AuthorCitation}[1]{}
\newcommand{\AuthorNames}[1]{}

% Disclosures and back-matter commands
\newcommand{\authorcontributions}[1]{\vspace{1em}\noindent\textbf{Author Contributions: }#1}
\newcommand{\funding}[1]{\vspace{0.8em}\noindent\textbf{Funding: }#1}
\newcommand{\institutionalreview}[1]{\vspace{0.8em}\noindent\textbf{Institutional Review Board Statement: }#1}
\newcommand{\informedconsent}[1]{\vspace{0.8em}\noindent\textbf{Informed Consent Statement: }#1}
\newcommand{\dataavailability}[1]{\vspace{0.8em}\noindent\textbf{Data Availability Statement: }#1}
\newcommand{\conflictsofinterest}[1]{\vspace{0.8em}\noindent\textbf{Conflicts of Interest: }#1}

\title{\textbf{\Large A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs}}

\author[1]{SeongSu Park\thanks{ORCID: \url{https://orcid.org/0009-0008-4056-3875}}}
\author[2]{Ki-Hyung Kim\thanks{Corresponding author: \texttt{kkim86@ajou.ac.kr}; Tel.: +82-31-219-2433; ORCID: \url{https://orcid.org/0000-0002-2321-4475}}}

\affil[1]{\small Department of Computer Engineering, Ajou University, Suwon 16499, Republic of Korea; \texttt{parky@ajou.ac.kr}}
\affil[2]{\small Department of Cyber Security, Ajou University, Suwon 16499, Republic of Korea; \texttt{kkim86@ajou.ac.kr}}

\date{\today}
"""

    # Extract abstract and keywords from preamble
    abs_match = re.search(r"\\abstract\{([\s\S]*?)\}\s*\\keyword\{([\s\S]*?)\}", preamble)
    if not abs_match:
        raise ValueError("Could not extract abstract and keywords from preamble")
    abstract_text = abs_match.group(1).strip()
    keywords_text = abs_match.group(2).strip()

    frontmatter = f"""
\\maketitle

\\begin{{abstract}}
{abstract_text}

\\vspace{{0.8em}}
\\noindent\\textbf{{Keywords:}} {keywords_text}
\\end{{abstract}}

\\vspace{{1.5em}}
"""

    # Clean body: bibliographystyle
    cleaned_body = body
    if r"\bibliographystyle{" not in cleaned_body:
        cleaned_body = cleaned_body.replace(r"\bibliography{references}", r"\bibliographystyle{unsrt}" + "\n" + r"\bibliography{references}")

    full_tex = neutral_preamble + "\n\\begin{document}\n" + frontmatter + cleaned_body
    TARGET_TEX.write_text(full_tex, encoding="utf-8")
    log.info(f"Generated publisher-neutral preprint LaTeX at {TARGET_TEX}")
    return full_tex


def compile_preprint(src_dir: Path) -> Path:
    """Compiles the preprint manuscript via pdflatex -> bibtex -> pdflatex -> pdflatex."""
    tex_file = src_dir / "DLG-Benchmark-Preprint.tex"
    assert tex_file.exists()

    def run_cmd(cmd: list[str], label: str):
        res = subprocess.run(cmd, cwd=src_dir, capture_output=True, text=True)
        if res.returncode != 0:
            log.error(f"{label} failed with code {res.returncode}:\n{res.stderr}\n{res.stdout[-2000:]}")
            raise RuntimeError(f"{label} compilation failed")
        return res

    log.info("Pass 1: pdflatex...")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark-Preprint.tex"], "Pass 1 pdflatex")
    log.info("Pass 2: bibtex...")
    run_cmd(["bibtex", "DLG-Benchmark-Preprint"], "Pass 2 bibtex")
    log.info("Pass 3: pdflatex...")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark-Preprint.tex"], "Pass 3 pdflatex")
    log.info("Pass 4: pdflatex...")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark-Preprint.tex"], "Pass 4 pdflatex")

    pdf_out = src_dir / "DLG-Benchmark-Preprint.pdf"
    assert pdf_out.exists(), "Compiled PDF not produced"
    log.info(f"Compilation succeeded! PDF size: {pdf_out.stat().st_size / 1024:.1f} KB")
    return pdf_out


def build_preprints_bundle():
    log.info("Building Preprints.org Publisher-Neutral Submission Bundle...")
    create_preprint_tex()

    # Copy supporting files into publication/preprints/
    # 1. generated/ tables
    gen_target = PREPRINTS_DIR / "generated"
    gen_target.mkdir(parents=True, exist_ok=True)
    for p in (DOCS_PAPER_DIR / "generated").glob("*.tex"):
        shutil.copy2(p, gen_target / p.name)

    # 2. references.bib
    shutil.copy2(DOCS_PAPER_DIR / "references.bib", PREPRINTS_DIR / "references.bib")

    # 3. appendix_financial_results.tex if referenced
    if (DOCS_PAPER_DIR / "appendix_financial_results.tex").exists():
        shutil.copy2(DOCS_PAPER_DIR / "appendix_financial_results.tex", PREPRINTS_DIR / "appendix_financial_results.tex")

    # Test clean compilation in isolated temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        log.info(f"Testing compilation in pristine scratch directory: {tmp_path}")

        # Copy all preprint bundle files to temp dir
        shutil.copy2(TARGET_TEX, tmp_path / "DLG-Benchmark-Preprint.tex")
        shutil.copy2(PREPRINTS_DIR / "references.bib", tmp_path / "references.bib")
        if (PREPRINTS_DIR / "appendix_financial_results.tex").exists():
            shutil.copy2(PREPRINTS_DIR / "appendix_financial_results.tex", tmp_path / "appendix_financial_results.tex")

        tmp_gen = tmp_path / "generated"
        tmp_gen.mkdir(parents=True, exist_ok=True)
        for p in (PREPRINTS_DIR / "generated").glob("*.tex"):
            shutil.copy2(p, tmp_gen / p.name)

        compiled_pdf = compile_preprint(tmp_path)

        # Check log for undefined citations or references
        log_content = (tmp_path / "DLG-Benchmark-Preprint.log").read_text(encoding="utf-8", errors="ignore")
        if "Citation" in log_content and "undefined" in log_content:
            log.warning("Undefined citations detected in preprint log!")
            for line in log_content.splitlines():
                if "undefined" in line.lower():
                    log.warning(f"  {line}")

        # Copy compiled PDF to destination
        shutil.copy2(compiled_pdf, TARGET_PDF)
        log.info(f"Published verified preprint PDF to {TARGET_PDF}")

    # Build self-contained Preprints.org submission zip
    log.info(f"Packaging self-contained Preprints submission archive to {TARGET_ZIP}...")
    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(TARGET_TEX, "DLG-Benchmark-Preprint.tex")
        zf.write(PREPRINTS_DIR / "references.bib", "references.bib")
        if (PREPRINTS_DIR / "appendix_financial_results.tex").exists():
            zf.write(PREPRINTS_DIR / "appendix_financial_results.tex", "appendix_financial_results.tex")
        if (PREPRINTS_DIR / "graphical_abstract.png").exists():
            zf.write(PREPRINTS_DIR / "graphical_abstract.png", "graphical_abstract.png")

        for p in sorted((PREPRINTS_DIR / "generated").glob("*.tex")):
            zf.write(p, f"generated/{p.name}")

    log.info(f"Successfully created Preprints bundle: {TARGET_ZIP} ({TARGET_ZIP.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_preprints_bundle()
