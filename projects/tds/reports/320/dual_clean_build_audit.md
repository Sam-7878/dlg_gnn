# Dual Clean Build Audit — R320

Comprehensive build verification of both venue source packages.

## 1. Preprints.org Source Build
- **Source Directory**: `projects/tds/paper/submission_320/preprints`
- **Compiler**: `pdfTeX 3.141592653-2.6-1.40.28 (TeX Live 2025/Debian)` via `latexmk 4.87`
- **Main Output**: `preprint_main.pdf` -> `preprint_article.pdf`
- **Page Count**: **21 pages** (includes expanded Section 2 Related Work, Section 3 Figure 1, and complete references)
- **Exit Code**: `0 (SUCCESS)`
- **SHA-256**: `dd4a532b917262fa59636d37914cd6ceb9c847982fc920aa53f417975a67c17d`
- **Undefined References**: `0`
- **Undefined Citations**: `0`

## 2. MDPI Applied Sciences Source Build
- **Source Directory**: `projects/tds/paper/submission_320/mdpi_applsci_anomaly`
- **Class / Style**: Official `mdpi.cls` (v18/06/2025), `palatino` font engine
- **Main Output**: `applsci_article.pdf`
- **Page Count**: **19 pages** (compact 2-column MDPI layout)
- **Exit Code**: `0 (SUCCESS)`
- **SHA-256**: `ad0f9b8d62353dcb69355a5013a38329b5fd3e38062e760cba5fe48cedbfde6c`
- **Undefined References**: `0`
- **Undefined Citations**: `0`

## 3. Supplement Build
- **Main Output**: `supplement.pdf`
- **Page Count**: **5 pages**
- **Exit Code**: `0 (SUCCESS)`
- **SHA-256**: `c574576168a8c4eab07573c936114ad601a518463104bbc06a24c5dd09ae2153`

## 4. Shared Science Byte-Level Parity
- `manuscript_body.tex`: **100% IDENTICAL** (`3cf4e89faf14de66eed7f536ec2cf01334feae56f5f83bb72795cc05adf7b339`)
- `references.bib`: **100% IDENTICAL** (`d177ad15f028fccf9b81a18dd11fd2ddf1e1fff6aa8b09024c26f2014219c371`)
- `declarations.tex`: **100% IDENTICAL** (`17d5133d5505d3debd1b7145021584f13103e171f615f466e8120894f7b42347`)
- **Verdict**: **PASS (PERFECT PARITY)**.
