# Preprints.org Gate Verification Report (A11)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A11  
**Date:** 2026-10-24  
**Target Platform:** Preprints.org  
**Status:** **ALL GATES PASSED (GREEN) — READY FOR PREPRINT REGISTRATION**

---

## 1. Gate Execution Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A11-P0-01** | Low ROC-AUC Diagnostic | **PASS** | Audited Elliptic and DGraphFin. Score, label mapping ($y=1$ illicit), and masks verified. Confirmed phenomenon of normal hub reconstruction residuals. No test-driven sign inversion. |
| **A11-P0-02** | Friedman / Permutation Audit | **PASS** | Permutation engine independently verified with unit tests. Independent 1M simulation reproduced exact magnitude ($p \sim 4.0\times 10^{-6}$). Finite-MC resolution caveat documented. |
| **A11-P0-03** | Visual Navigation & Cross-Refs | **PASS** | Added 3 high-resolution publication figures (Fig A: Conditional Delta, Fig B: Support Matrix, Fig C: DLG Pipeline). All tables and figures explicitly cited via `Table~\ref{...}` and `Figure~\ref{...}`. |
| **A11-P0-04** | Application Results in Main Text | **PASS** | Average Precision paired with evaluation prevalence baselines; Alert-budget precision/recall ($P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$) moved to main Results. |
| **A11-P0-05** | Claim vs Registry Alignment | **PASS** | Abstract, text claims, delta numbers, and support counts verified against frozen `number_registry.json`. Corrected blockchain claim: established comparators exceed DLG-Aug on all three chains. |
| **A11-P0-06** | Bibliography & LaTeX Clean Build | **PASS** | Source compiled cleanly with TeX Live 2025/2026 (`pdflatex` + `bibtex`). All 48 references cited (including `kumar2016edge`). Zero float errors. |
| **A11-P0-07** | Data & AI Disclosure | **PASS** | Immutable release tag `benchmark-a09-2026-10-10` and commit `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d` bound. Transparent Generative AI disclosure included. |

---

## 2. Submission Deliverables

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A11_Preprints.tex`
- **Rendered PDF (17 pages):** `DLG-Benchmark_A11_Preprints.pdf` (SHA-256 verified)
- **Source Bundle Archive:** `source_bundle.zip` (Contains `.tex`, `.bib`, and vector figures)
- **Distinct Benchmark Preprint DOI Policy:** To be registered as a distinct, standalone benchmark preprint (separate from the preceding DLG-GNN architecture preprint).
