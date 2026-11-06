# A25 final PDF and source audit

The A26 revision has its own review and current submission hashes. This document records the A25 revision. [Current record](../a26/A26_PDF_QA.md)

Checked on 2026-11-06 KST. Rendered and reviewed every page of both PDFs at 95 DPI. Compared the original and rebuilt page renders. All pages passed native word bounds checks. Both builds and clean ZIP rebuilds returned exit code 0 with a populated bibliography and identical PDF hashes under the same local TeX environment. The logs contain no overfull hboxes or unresolved references or citations.

Both deliberate Section 8 mutations compiled and failed the PDF citation check. The section-aware A24 validator checked 19 source tables per variant and S1 n=8, Q=29.0357, W=0.6049. The exact source diff covers the remaining tables and primary AP, ROC-AUC and F1. Native coordinate checks matched all seven Table A5 ROC-AUC mean and sample SD cells.

## preprints

28 pages. Corrected citation on page 15. Figures 1–3, Tables 1–9 and S3–S8, author metadata, declarations and reference list checked in the page renders. The PDF text confirms separate historical and matched A24 control captions. The LANL comparison retains the explicit preprocessing qualification.

| Item | PDF page(s) |
|---|---|
| Table 1 | 6, 7 |
| Table 2 | 10 |
| Table 3 | 11 |
| Table 4 | 11 |
| Table 5 | 11 |
| Table 6 | 12 |
| Table 7 | 13 |
| Table 8 | 14 |
| Table 9 | 16 |
| Figure 1 | 4, 5 |
| Figure 2 | 9 |
| Figure 3 | 12 |
| Table A5 | 23 |
| Table S3 | 25 |
| Table S4 | 25 |
| Table S5 | 25 |
| Table S6 | 26 |
| Table S7 | 27 |
| Table S8 | 28 |

Pixel-identical A24 pages at the same page numbers are 1, 2, 3, 4, 5, 7, 11, 21, 22, 23, 24, 25, 26, 27, 28. All other pages were reviewed in the rebuilt render.

## applsci

25 pages. Corrected citation on page 14. Figures 1–3, Tables 1–9 and S3–S8, author metadata, declarations and reference list checked in the page renders. The PDF text confirms separate historical and matched A24 control captions. The LANL comparison retains the explicit preprocessing qualification.

| Item | PDF page(s) |
|---|---|
| Table 1 | 6, 7 |
| Table 2 | 10 |
| Table 3 | 10 |
| Table 4 | 11 |
| Table 5 | 11 |
| Table 6 | 12 |
| Table 7 | 13 |
| Table 8 | 14 |
| Table 9 | 15 |
| Figure 1 | 4, 5 |
| Figure 2 | 9 |
| Figure 3 | 11 |
| Table A5 | 19 |
| Table S3 | 22 |
| Table S4 | 22 |
| Table S5 | 22 |
| Table S6 | 23 |
| Table S7 | 24 |
| Table S8 | 25 |

Pixel-identical A24 pages at the same page numbers are 1, 2, 3, 4, 5, 7, 20, 22, 23, 24, 25. All other pages were reviewed in the rebuilt render.
## Build environment and repeat verification

LaTeX ran through Latexmk 4.87 with pdfTeX 1.40.28 and BibTeX in the local TeX Live 2025/Debian environment. The builds used the A24 fixed source date epoch 1793318400. Compare PDF bytes only under the same environment. The private archive at `local_storage/benchmark/a25/verification` contains the build and mutation logs, scripts and native coordinate results. The retained build script reads the original files from `rollback_a24` and writes to a temporary directory. For submission, use the A25 SHA-256 manifest to identify the corrected copies. The author-local A24 writer generates the earlier editorial revision.
