# A07 candidate PDF visual review — updated 2026-10-06

## Exact files reviewed

- MDPI candidate: `projects/benchmark/paper/current/mdpi/DLG-Benchmark_A07_MDPI.pdf`, 14 A4 pages, SHA-256 `f8118a4f18012aa8e480790624b1e36af2ba8220a3eb3fc95353a14e439f109e`.
- Neutral preprint candidate: `projects/benchmark/paper/current/DLG-Benchmark_A07.pdf`, SHA-256 `5544a7f7d57570f4def0892d91a88c7f9047e95218b6f94ff92cb593f34064a5`.
- F1 supplement: `projects/benchmark/paper/current/Supplementary_F1_A07.pdf`, one page, SHA-256 `1150813524d64b924c2a336cc683ce1e7b1405fbbdd726492da11a4732b9ba51`.

After relocation, the neutral, MDPI and supplement PDFs were recompiled. The extracted text matches the 2026-10-05 reviewed package byte for byte after normalizing only the automatically printed October 5/6 date. All 14 new MDPI pages and the supplement page were rendered to PNG and inspected. `latexmk` completed without undefined citations/references or overfull box warnings in the final MDPI log. This is a **candidate visual pass**, not a submission approval: Data Availability still states that the public immutable release identity will be inserted after publication.

| Page | Manual review |
|---|---|
| 1 | Title, both author ORCID symbols, affiliations, corresponding email and telephone visible; abstract/introduction readable. |
| 2 | Four Related Work axes and citations visible; exact-execution equation fits. |
| 3 | Dataset portfolio table and provenance text fit; no clipped cells. |
| 4 | Main PR-AUC table and Friedman view table fit; five-seed and mixed-label statements visible. |
| 5 | Support table and BSC negative result/variance discussion visible; no caption overlap. |
| 6 | Measured memory Figure 1 has one caption; targeted AnomalyDAE and alert-budget tables fit. |
| 7 | Discussion, author/funding/availability sections readable; constructed crypto graph access limitation stated. |
| 8 | Appendix model definitions and equations fit. |
| 9 | Local/global implementation equations fit. |
| 10 | Exact sparse/fused execution derivations fit. |
| 11 | Support policy, four-row Holm table and ROC-AUC appendix table fit; no truncated text. |
| 12 | References 1–18 readable; links and line numbers remain within page. |
| 13 | References 19–39 readable. |
| 14 | References 40–44 readable; remaining whitespace is due to bibliography end, not overflow. |

The separate F1 supplement preserves all 14 dataset rows (13 primary plus LANL), seven model columns and unsupported dashes. The earlier isolated F1 table page and two Related Work line-overflow warnings were repaired by the generator; this review applies only to the PDF hashes above. The MDPI class template prints “submitted to Appl. Sci.” in its running header; that is template boilerplate and is **not evidence of an actual submission**.
