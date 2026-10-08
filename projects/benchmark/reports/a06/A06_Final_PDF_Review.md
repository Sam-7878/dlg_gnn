# A06 final PDF review

Date: 2026-10-05 (Asia/Seoul). Scope: the publisher-neutral preprint candidate and the MDPI-class journal candidate generated from the frozen A05 evidence. No new detector training was performed.

| Artifact | Pages | SHA-256 |
|---|---:|---|
| `submission_a06/DLG-Benchmark_A06.pdf` | 13 | `2972ecb1e95541fc540949b9d328087b2971333830faa4e30a6a3e69fa49a722` |
| `submission_a06/mdpi/DLG-Benchmark_A06_MDPI.pdf` | 13 | `015f52291c3d6cb399552c2b6dd60404dc021da818b98e94f26b3d193ea2339a` |

Both PDFs were compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error`. The final logs contain no unresolved citation/reference or overfull-box messages. `pdftoppm` rendered every page; all 26 page images were examined visually. The independent extraction test of `DLG-Benchmark_A06_MDPI_source.zip` also compiled to a 13-page PDF after the source archive was corrected to include the MDPI EPS-logo conversion files.

| Page | Neutral preprint review | MDPI review |
|---:|---|---|
| 1 | Title, authors/affiliations, abstract, keywords, introduction and ORCID notes readable. | Title, author/ORCID icons, abstract, keywords, journal side panel and opening introduction readable; no clipping. |
| 2 | Three contributions, Related Work, exact-execution equation and Methods readable. | Related Work continuation, equation and Methods readable; references render correctly. |
| 3 | Table 1 legible; dataset provenance, protocol and GenAI disclosure present. | Table 1 legible; Methods, environment/provenance and GenAI disclosure present. |
| 4 | Tables 2 and 3 legible; unsupported cells shown as dashes; S3 `p=0.082284` visible. | Tables 2 and 3 legible; S3 mixed-label wording and non-significant interpretation visible. |
| 5 | Table 4 and measured Figure 1 fit; OOM is marked with an X, without a fabricated bar. | Table 4 and Figure 1 fit; selected-case scope and OOM symbol visible. |
| 6 | Tables 5 and 6 fit; projected hours distinguished from completed five-seed runs; alert-budget caption readable. | Same table/caption checks pass; DGraphFin clock note and recovery qualification readable. |
| 7 | Limitations, usage guide, conclusion, declarations and availability text readable. | Usage guide, conclusion, declarations, availability text and appendix opening readable. |
| 8 | DLG-Base and DLG-Aug appendix equations fit. | Appendix J.1 equations and explanatory text fit. |
| 9 | DLG score and Gram derivation fit with numbered equations. | Appendix J.2 derivation and numbered equations fit. |
| 10 | Sparse GCN and support policy readable. | Appendix J.3/J.4 readable; no formula clipping. |
| 11 | ROC-AUC and F1 appendix tables fit within margins; references begin on next page. | Appendix K ROC-AUC/F1 tables fit; reference heading begins on page 12. |
| 12 | References 1–14 readable. | References 1–20 readable, including linked identifiers. |
| 13 | References 15–23 readable; remaining whitespace is a normal consequence of a separate reference page. | References 21–23 readable; remaining whitespace is a normal consequence of a separate reference page. |

**Visual verdict:** PASS for the two PDFs identified by the hashes above. Tables are not split or clipped; Figure 1 labels are visible; author names, ORCIDs, captions, appendix order, and reference entries are legible. The PDF's `Data Availability Statement` explicitly says the public evidence release location is pending. This is a publication gate, not a visual defect.

**Template limitation:** the bundled `Definitions/mdpi.cls` reports `12/09/2024 MDPI paper class`. The candidate compiles and renders, but the authors must compare it with the current official MDPI template immediately before journal upload. Neither PDF is claimed to have been submitted.
