# A27 PDF Quality Assurance

Two fresh compilations per source bundle (`clean1` and `clean2`) were byte-identical with fixed `SOURCE_DATE_EPOCH=1793923200`. Both final LaTeX logs were clear of overfull boxes, undefined references, multiply defined labels, and oversized floats. In both variants, Table 4 appears on or before the page of Table 5 (`A27-09`), Appendix Table A3 includes all 12 dataset–estimator rows (`A27-01`), Table 5 reports exact DP tail probabilities $p_{\mathrm{exact}}$ and $p_{\chi^2}/p_{\mathrm{exact}}$ (`A27-03`), Table 6 and Table 9 report the directly measured RTX 3090 peak VRAM for `Reddit-Syn / DLG-Aug` (`13.848/17.461 GiB`, `A27-10`), and the *Applied Sciences* front matter includes the `Featured Application` block and continuous page range `1–25` rather than `16, 0` (`A27-08`).

Visual and programmatic QA inspected all 56 rendered pages (29 Preprints and 27 Applied Sciences).

| Artifact | Preprints page | Applied Sciences page |
|---|---:|---:|
| Table 1 | 7 | 7 |
| Table 2 | 10 | 10 |
| Table 3 | 11 | 11 |
| Table 4 | 12 | 12 |
| Table 5 | 12 | 12 |
| Table 6 | 13 | 13 |
| Table 7 | 14 | 14 |
| Table 8 | 16 | 16 |
| Table 9 | 16 | 16 |
| Table A1 | 22 | 20 |
| Table A2 | 22 | 20 |
| Table A3 | 23 | 21 |
| Table A4 | 23 | 21 |
| Table A5 | 24 | 22 |
| Table S1 | 25 | 23 |
| Table S2 | 25 | 23 |
| Table S3 | 26 | 24 |
| Table S4 | 26 | 24 |
| Table S5 | 26 | 24 |
| Table S6 | 27 | 25 |
| Table S7 | 28 | 26 |
| Table S8 | 29 | 27 |
| Figure 1 | 5 | 5 |
| Figure 2 | 9 | 9 |
| Figure 3 | 11 | 11 |

Preprints PDF SHA-256: `786833a3a92e4bede9ba22ed21cd3bf5030c49cfc68b7955aab417d90522f8dc`

Applied Sciences PDF SHA-256: `29ff916d9b88a47c509665d4008bc99ba187ba07d4a21f1489696d15371c1b43`
