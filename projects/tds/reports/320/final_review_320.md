# TDS R320 Final Review & Publication Readiness Report

**Project**: `dlg_gnn/projects/tds`  
**Milestone**: R320 — Scientific Closure Reconciled & Publication Readiness Gate  
**Status**: `SCIENTIFIC_LOCKED / TECHNICAL_PUBLICATION_READY`  
**Submission Gate**: `HOLD_AUTHOR_ACTION` (Awaiting Final Meeting Signoff)  
**Date**: 2026-11-02  

---

## 1. Executive Summary

As instructed, scientific development for the TDS project officially concluded with R319. R320 successfully completes all evidence reconciliation, manuscript refinements, and publication gate verifications. 

Key milestones achieved in R320:
1. **Evidence & Numerical Replay (G0/G1)**: Full independent verification of all 227 files in `projects/tds/results/319` with **0 hash mismatches**. Recomputed 7-day cohort support (3,627 samples, 2,037 positives, 56.16%), matched model means, 4 exploratory statistics, calibration shifts, and alert budgets with **100% exact numerical agreement**.
2. **Section 3 Architecture Overview Figure Added (G2)**: Generated a publication-quality 4-phase framework overview diagram (`fig_tds_architecture.pdf`) and integrated it into Section 3 of both manuscripts as Figure 1.
3. **Related Work Major Expansion (G2)**: Enriched Section 2 into 4 comprehensive subsections, importing 27 high-impact references from the accompanying benchmark preprint (`dlg_gnn/projects/benchmark/submission/preprints`). Directly integrated key *Applied Sciences* literature (Dilini et al. 2026, Heidrich et al. 2026, Jiang & Tsai 2025) and the predecessor DLG-GNN preprint (Park & Kim 2026).
4. **Dual Clean Builds (G6)**: Successfully compiled both Preprints.org (21 pages) and MDPI Applied Sciences (19 pages) manuscripts with **0 errors, 0 undefined citations, and 100% byte-level parity** across shared science sources.
5. **Publication Sequence & Gate Control (G7)**: Prepared exact submission bundles (`preprint_source.zip`, `mdpi_source.zip`) and author signoff records. The gate status is strictly maintained at `HOLD_AUTHOR_ACTION` pending the scheduled final meeting with the advising professor.

---

## 2. Gate Verification Matrix

| Gate | Target / Requirement | Verification Outcome | Gate Status |
|---|---|---|---|
| **G0** Evidence Freeze | SHA-256 verification of 227 result files & candidate freeze | All 227 files match exactly; `candidate_identity_320.json` generated | **PASS** |
| **G1** Numerical Audit | Independent recomputation of all metrics, statistics, and tables | Exact agreement on all 12 key claims; 0 discrepancies | **PASS** |
| **G2** Manuscript Polish | Overview figure in Sec 3, enriched Related Work, scientific bounds | Figure 1 added; 27 references imported; GNN vs size clarified | **PASS** |
| **G3** Rights & Privacy | CC BY-NC-SA 4.0 review, secret leak check, PII exclusion | 0 secrets/PII leaked; deterministic rebuild payload defined | **PASS** |
| **G4** Author & Funding | Exact file approval sheet, IITP grant proof, CRediT, COI | All declarations compiled; signoff staged for meeting | **HOLD_AUTHOR_ACTION** |
| **G5** Public Release | TDS-specific immutable release on GitHub & anonymous replay | Staged (`tds-r320-v1.0.0`); held until author meeting | **HOLD_PENDING_SIGNOFF** |
| **G6** Dual Clean Build | Independent clean compilation of Preprints and MDPI sources | Both compiled cleanly (Preprints: 21p, MDPI: 19p, Supp: 5p) | **PASS** |
| **G7** Platform Dispatch | Preprints.org posting -> DOI -> MDPI Applied Sciences submission | Two-stage submission pipeline ready for author execution | **HOLD_AUTHOR_ACTION** |

---

## 3. Publication Roadmap & Next Steps

1. **Professor Final Meeting**:
   - Review the final compiled PDFs (`projects/tds/submission/preprints/320/preprint_article.pdf` and `projects/tds/submission/applsci/320/applsci_article.pdf`).
   - Confirm author CRediT roles and sign `exact_file_approval_320.md`.
2. **Preprints.org First Posting**:
   - Dispatch `preprint_source.zip` and `supplement.pdf` to Preprints.org.
   - Obtain formal preprint receipt and allocated DOI (e.g., `10.20944/preprints...`).
3. **MDPI Applied Sciences Special Issue Submission**:
   - Update the assigned preprint DOI into `declarations.tex` and cover letter.
   - Submit `mdpi_source.zip` and `supplement.pdf` to the Special Issue *"Machine Learning and Its Application for Anomaly Detection"* (URL: `https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt`).
4. **Public GitHub Release**:
   - Tag and publish release `tds-r320-v1.0.0` on `https://github.com/Sam-7878/dlg_gnn` including the L1 numerical replay package.
