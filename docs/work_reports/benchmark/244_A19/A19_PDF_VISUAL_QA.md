# Gate A19-G5: Visual Quality Assurance Report

**Date:** 2026-10-28 / 2026-10-29  
**Status:** PASS (All Key Visual Artifacts Inspected and Verified)  
**Evaluator:** Automated & Rendered High-Resolution Inspection (Poppler 26.01.0, 150 DPI)  

---

## 1. Inspection Targets & Artifact Verification

| Target Artifact | PDF Document | Compiled Pages | Key Inspected Elements | QA Status |
|---|---|---|---|---|
| **Figure 1** (DLG Architecture) | Preprints (p. 5) & Applied Sciences (p. 5) | Preprints: 20 pages<br>Appl. Sci.: 17 pages | Level 1 local pretraining diagram, finite 2-hop message-passing annotation, detached $\mathbf{Z}_\mathrm{local}$, Level 2 global concatenation $[\mathbf{X} \parallel \mathbf{Z}_\mathrm{local}]$, closed-form Gram decoder $\hat{S} = \mathbf{Z}\mathbf{Z}^\top$. | **PASS** (Zero clipping, vector crispness, exact terminology matching methodology) |
| **Table 3** (Alert Budgets) | Preprints (p. 10) & Applied Sciences (p. 10) | Preprints: 20 pages<br>Appl. Sci.: 17 pages | Multi-row structure for Ethereum, BSC, Polygon across $P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$; exact bolding of optimal detectors (DOMINANT on Ethereum, AnomalyDAE on BSC/Polygon). | **PASS** (100% arithmetic & tabular alignment with `a19_alert_budget_raw_recheck.csv`) |
| **Table A2** (F1 Scores) | Preprints (p. 20) & Applied Sciences (p. 17) | Preprints: 20 pages<br>Appl. Sci.: 17 pages | Full 14-dataset × 7-model matrix; exact mean $\pm$ SD across 5 seeds; validation-thresholded test F1 caption with supplementary provenance reference. | **PASS** (Zero text overflow, perfect column spacing, exact match with `f1_per_seed_verified.csv`) |
| **Final / Availability Pages** | Preprints (p. 14--20) & Applied Sciences (p. 13--17) | Preprints: 20 pages<br>Appl. Sci.: 17 pages | Data availability statement citing immutable release tag `benchmark-a19-submission-2026-10-28`; full references with zero undefined citations or references (`undef refs: []`, `undef cits: []`). | **PASS** (Clean pagination, zero broken cross-references) |

---

## 2. High-Resolution Visual Render Manifest

Rendered visual assets stored in `docs/work_reports/benchmark/244_A19/visual_qa/`:
1. `prep_p05-05.png` (346 KB) — Preprints Architecture Diagram (Figure 1)
2. `prep_p10-10.png` (399 KB) — Preprints Operational Alert Budget Table (Table 3)
3. `prep_p20-20.png` (315 KB) — Preprints Supplementary F1 & ROC Tables (Table A1, Table A2)
4. `appl_p05-05.png` (362 KB) — Applied Sciences Architecture Diagram (Figure 1)
5. `appl_p10-10.png` (439 KB) — Applied Sciences Operational Alert Budget Table (Table 3)
6. `appl_p17-17.png` (320 KB) — Applied Sciences Supplementary F1 & ROC Tables (Table A1, Table A2)

---

## 3. Editorial & Typographical Verification
1. **Terminology Harmonization:** Residual phrase *"local ego-net augmentation"* was replaced in the Results section with *"pretrained local-feature augmentation"*.
2. **Methodological Truthfulness:** The accurate clarification *"Crucially, this pre-training operates in full-graph forward passes without explicit ego-net subgraph extraction"* is retained intact in Section 2.
3. **Duplication Eradication:** A duplicated introductory paragraph for the DLG architecture was removed from both manuscripts.
4. **Tag Consistency:** Both manuscripts cite the predeclared release tag `benchmark-a19-submission-2026-10-28` in their Data Availability sections.

