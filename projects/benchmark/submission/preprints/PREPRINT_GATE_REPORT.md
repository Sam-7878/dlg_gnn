# Preprints.org Gate Verification Report (A12)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A12  
**Date:** 2026-10-24  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Status:** **ALL GATES PASSED (GREEN) — A12 PREPRINT READY**

---

## 1. A12 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A12-G1** | Dataset Identity Reconciliation | **PASS** | 14 canonical model-input graph tensors ($N, E, d$) reconciled against approved hashes. Upstream raw source totals cleanly separated. Table 1 synchronized. |
| **A12-G2** | Global Numeric Claims Gate | **PASS** | Abstract prose strictly aligned with frozen registry (`Reddit-Syn` AP 0.4044/0.3388, $\Delta -0.0656$; Ethereum OCGNN 0.4996 > Aug 0.4359; BSC AnomalyDAE 0.3609 > Aug 0.3411; Polygon AnomalyDAE 0.1878 > Aug 0.0731). |
| **A12-G3** | Empirical Low ROC-AUC Diagnostic | **PASS** | Audited raw score file availability (`MISSING_NOT_STORED_BY_SOURCE_RUN`). Claims downgraded to informed topological hypotheses; authentic empirical below-0.50 ROC-AUC retained with zero sign inversion. |
| **A12-G4** | Monte Carlo Statistical Audit | **PASS** | Seed 20261024 within-block 1,000,000 permutations: S1 ($p=4.00\times 10^{-6}$), S2 ($p=2.10\times 10^{-5}$), S3 ($p=0.1921$, non-sig), S4 ($p=1.00\times 10^{-6}$). Table 5 and JSON synchronized. |
| **A12-G5** | Hardware & Memory Visuals | **PASS** | Figure 2 distinguishes exact support (5/5), LANL external, OOM, Err, and prohibitive scaling (N/A). Figure 3 incorporates seed-paired difference uncertainty ($n=5$). |
| **A12-G6** | Literature Integration | **PASS** | 3 verified *Applied Sciences* citations integrated into Related Work (Dilini et al. 2026, Heidrich et al. 2026, Jiang & Tsai 2025). |
| **A12-G7** | Publisher-Neutral LaTeX Build | **PASS** | Clean build with `article` class under TeX Live (`pdflatex` + `bibtex`). 17 pages, zero undefined references, zero unresolved citations. |
| **A12-G8** | Submission Package Artifacts | **PASS** | Complete `source_bundle.zip` created with verifiable SHA-256 hashes. |
| **A12-G9** | Proportionate Scientific Claims | **PASS** | Title, abstract, discussion, and conclusion accurately state empirical findings without unsupported causal overclaims. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A12_Preprints.tex`  
  - SHA-256: `215964401034d1619c5cd1d093a91a75bde38c957d912aefaae998d242fcb4b4`
- **Rendered PDF (17 pages):** `DLG-Benchmark_A12_Preprints.pdf`  
  - SHA-256: `d595962eb43e3cf55c6eba193c72bf05b5eb20dceb43fafaf6cf23f10f7e6a97`  
  - Size: `511,963` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `61fbdcd680586d19f5d289a476a8514982492cefb8f4358681867eb4e9adcaf4`  
  - Size: `101,231` bytes
  - Contents: `DLG-Benchmark_A12_Preprints.tex`, `references.bib`, `figures/*.pdf`
- **Preprint DOI Policy:** To be registered as a distinct, standalone benchmark preprint upon upload.
