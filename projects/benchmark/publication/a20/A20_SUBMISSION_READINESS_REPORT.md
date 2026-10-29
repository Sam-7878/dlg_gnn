# DLG-GNN Benchmark A20 Submission Readiness Report

**Report ID:** `DLG-BENCH-REP-A20-READINESS`  
**Date:** 2026-10-29  
**Work Order:** `DLG-BENCH-WO-A20-EDITORIAL-ONLY`  
**Status:** **READY FOR SUBMISSION (ALL P0 / P1 GATES PASS)**  
**Target Repository:** https://github.com/Sam-7878/dlg_gnn  
**Release Tag Target:** `benchmark-a20-preprint-2026-10-29` (superseding `benchmark-a19-submission-2026-10-28`)  

---

## 1. Executive Summary & Gate Verification Matrix

The A20 editorial and evidence submission cycle has completed all work order requirements without opening a new benchmark run cycle, without modifying model architectures or weights, and without altering the frozen empirical score tables.

| Gate | Category | Scope | Verification Status | Artifacts / Evidence |
|---|---|---|---|---|
| **G0** | P0 HARD | Snapshot & Input Freeze | **PASS** | `a20_input_freeze.json` |
| **G1** | P0 HARD | BSC Source-Map Contradiction | **PASS** | `a20_source_map_consistency.json`, `a20_authoritative_source_map.md` ($M_\mathrm{test} = [215, 240, 232, 215, 203]$) |
| **G2** | P0 HARD | Score Audit & Bug Exclusion Removal | **PASS** | `a20_claims_to_evidence.csv`, Data Availability disclosed 105 local vs 350 summary-only |
| **G3** | P0 HARD | Practitioner Decision Matrix (Table 7) | **PASS** | Table 7 redesigned (Feasibility vs Accuracy separated, coverage noted) |
| **G4** | P0 HARD | Effect-Size & Uncertainty Phrasing | **PASS** | Absolute $\Delta\mathrm{AP}$ prioritized, prevalence-anchored, Figure 3 re-rendered |
| **G5** | P0 HARD | Memory Evidence Hierarchy | **PASS** | `a20_memory_evidence_classes.csv`, Table 6 and Sec 6 theoretical bounds vs VRAM separated |
| **G6** | P0 HARD | Split Protocol Audit | **PASS** | `a20_split_protocol_audit.md`, unstratified cryptocurrency footnotes, Polygon $K=5$ sensitivity |
| **G7** | P1 REQ | Manuscript Editorial Cleanup | **PASS** | Duplicate paragraphs deleted, citations fixed, `a20_lanl_f1_threshold_audit.md` |
| **G8** | P1 REQ | Statistical Phrasing Clarification | **PASS** | $b=0$ MC tail bound ($p_\mathrm{MC} \approx 1.0 \times 10^{-6}$, 95% UCB $\approx 3.0 \times 10^{-6}$), equivalence claim softened |
| **G9** | P0 HARD | Public Release, PDF/ZIP & Gates | **PASS** | Clean TeX Live 2025 compilation (20 & 19 pp), 0 undef refs/cits, SHA-256 manifests synchronized |

---

## 2. Deliverables and Checksum Ledger

All 6 primary publication deliverables have been compiled, verified for byte-for-byte consistency with their source bundles, and recorded in the canonical hash ledger:

| Track | Artifact Filename | Size (Bytes) | SHA-256 Digest | Clean Check |
|---|---|---|---|---|
| **Preprints.org** | `DLG-Benchmark_A16_Preprints.pdf` | 602,671 | `a4bc3dd65dba431dd23f7d0cbd74bfd744506c192e79e7d64cbd0bcb1805c96f` | 20 pages, 0 undef |
| **Preprints.org** | `DLG-Benchmark_A16_Preprints.tex` | 66,043 | `f202bdf160fea13e9abfddf769890aaa3e0d6fc7e65afe17533b0c53809ba6d3` | 1:1 match in zip |
| **Preprints.org** | `source_bundle.zip` | 816,138 | `1b685586ec2e34fad47615db952cc7977cb0280b07874bce6666418c8293ffc7` | Verified clean unpack |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.pdf` | 352,898 | `c44aa4c6cda155346b7e7eb5cfd13154952b198a634ae3afe841c77b80957af9` | 19 pages, 0 undef |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.tex` | 65,671 | `257bc901b097709e891880a7d3cbde6bc77426fc7436125c48a7bff0f6d8a03a` | 1:1 match in zip |
| **Applied Sciences** | `source_bundle.zip` | 1,852,125 | `2cf89c7e4376cded7daab1d3290542bcb78b2350cf931badeb5f5a28a56955fb` | Verified clean unpack |

---

## 3. Evidence Consistency and Scientific Integrity Audit

1. **BSC Source Map Alignment:**
   - Evaluated test positive count vector $[215, 240, 232, 215, 203]$ is strictly verified across `a19_alert_budget_raw_recheck.csv`, `a20_source_map_consistency.json`, `a20_authoritative_source_map.md`, and the manuscript narrative. The erroneous narrative vector from A19 (`[215, 237, 217, 206, 230]`) is completely eliminated.
2. **Score-Level Audit & Bug Exclusion Claims:**
   - Overreaching assertions that historical baselines were audited at the per-node continuous score level and "ruled out implementation defects or sign bugs" have been excised.
   - The manuscript truthfully discloses that high-resolution per-node anomaly score vectors ($y_\mathrm{score}$) and labels ($y_\mathrm{true}$) are preserved in compressed `.npz` format for the 105 cryptocurrency evaluations, while the 350 historical baseline runs archived summary metric scalars at runtime.
3. **Practitioner Decision Matrix (Table 7):**
   - Strictly separates computational feasibility (memory and runtime support) from empirical predictive accuracy.
   - DGraphFin entries explicitly warn of performance near no-skill prevalence ($\text{AP} \approx 0.014$, $\text{ROC-AUC} \approx 0.54$); Elliptic entries note performance below prevalence for CoLA ($\text{AP} \approx 0.046 < \pi = 0.098$); Reddit entries caveat AnomalyDAE hardware limitations.
4. **Memory Evidence Hierarchy (Table 6 & Sec 6):**
   - Explicitly classifies memory values into theoretical lower bounds ($4N^2 \approx 217$ GB on Reddit, 6.0 TB on DGraphFin), illustrative profile estimates ($\approx 5.5$ GB), measured VRAM allocations, and observed OOMs.
5. **Split Protocol & Small-Budget Sensitivity:**
   - Audited split generator code (`a04_build_dataset_manifest.py`); confirmed cryptocurrency graphs use seeded unstratified permutations, correctly reflecting the seed-dependent test positive count variation ($M_\mathrm{test} \in [7, 18]$ on Polygon).
   - Quantified discrete step sensitivity on Polygon ($K=5, \Delta P = 0.20$ per true positive).
6. **LANL F1 Evaluation Nuance:**
   - Audited LANL run logs; qualified Table A2 caption regarding single-threshold recording.

---

## 4. Submission Routing and Next Steps

1. **Deposit to Preprints.org:**
   - Upload `submission/preprints/source_bundle.zip` (or standalone TeX + figures) and `DLG-Benchmark_A16_Preprints.pdf`.
   - Complete Preprints.org submission to obtain a dedicated benchmark preprint DOI (distinct from the prior DLG-GNN architectural preprint `10.20944/preprints202610.1878.v1`).
2. **Deposit to MDPI Applied Sciences:**
   - Target Special Issue: "Machine Learning and Its Application for Anomaly Detection" (SI ID: `C67H3U9QGT`, deadline 2026-12-31).
   - Upload `submission/applsci/source_bundle.zip` and `DLG-Benchmark_A16_AppliedSciences.pdf`.
   - Submit `COVER_LETTER.md` incorporating the newly minted Preprints DOI.
3. **Repository Finalization:**
   - Commit all editorial changes.
   - Create annotated tag `benchmark-a20-preprint-2026-10-29` pointing to the release commit.
   - Push commit and tag to remote GitHub repository.

