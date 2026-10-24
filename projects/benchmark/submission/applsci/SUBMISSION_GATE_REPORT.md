# Applied Sciences Special Issue Submission Gate Report (A11)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A11  
**Date:** 2026-10-24  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — READY FOR FORMAL SUBMISSION**

---

## 1. Comprehensive Gate Checklist (Work Order Section 12)

| Criteria Check | Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **1. AUC Direction Audit** | Score/label/mask contract signed off; below-random ROC-AUC explained; no post-hoc sign flipping | `diagnostics/a11_score_direction_audit.md` completed. Hub reconstruction residual verified. | **PASS** |
| **2. Friedman Implementation** | Permutation logic independently checked; within-block rank permutation verified | `analysis/a11_permutation_unit_tests.md` completed. Code passed toy null and invariance tests. | **PASS** |
| **3. $\chi^2$ vs MC $p$-value** | Explanation of sample size $n$, discrete permutation space, and finite-MC resolution added | Rigorous statistical explanation incorporated in Methods and Results. 1M independent simulation matches scale. | **PASS** |
| **4. 13+LANL AP Delta** | Aug minus Base signed difference covers Elliptic ($+0.0350$), Reddit-Syn ($-0.0656$), and LANL ($-0.0253$) | Figure~\ref{fig:conditional_delta} and Table~\ref{tab:conditional_delta} fully plot and tabulate all 14 datasets. | **PASS** |
| **5. Substantive Visuals** | At least 2 evidence-based figures integrated with explicit cross-references | 3 publication figures integrated: Fig 1 (DLG pipeline), Fig 2 (Support matrix), Fig 3 (AP delta). | **PASS** |
| **6. Application-Facing Results** | AP + prevalence reference + alert budgets ($P@1\%, R@1\%$) visible in main text | Table~\ref{tab:pr_auc_main} (AP vs Prevalence) and Table~\ref{tab:alert_budget} (Alert budget) prominent in Results. | **PASS** |
| **7. Claim vs Registry Match** | All prose statements, deltas, support counts, and seed stats match frozen registry | Verified against `number_registry.json`. Blockchain comparator claim strictly corrected. | **PASS** |
| **8. Bibliography & Clean Build** | All 48 BibTeX entries cited legitimately; clean PDF build; zero unresolved refs | Clean compile with TeX Live 2025/2026 (`pdflatex` + `bibtex`). `kumar2016edge` cited in Section 4. Zero float errors. | **PASS** |
| **9. Preprints Publisher-Neutral** | Independent `article` class source compiles without MDPI logos/branding; author metadata complete | `submission/preprints/DLG-Benchmark_A11_Preprints.pdf` compiled cleanly (17 pages, SHA-256 verified). | **PASS** |
| **10. Preprint Posting Protocol** | Distinct Benchmark Preprint DOI protocol established (separate from preceding DLG preprint) | Staged in `submission/preprints/` for sequential Preprints.org upload prior to journal submission. | **PASS** |
| **11. Official MDPI Template** | Official `Definitions/mdpi.cls` template applied with current back-matter sections | `submission/applsci/DLG-Benchmark_A11_AppliedSciences.pdf` compiled cleanly (15 pages, exit code 0). | **PASS** |
| **12. Data Availability Truthfulness** | Data Availability accurately distinguishes public GitHub assets from private author-local score files | `submission/applsci/DATA_AVAILABILITY.md` completed with explicit commit, tag, and request policy. | **PASS** |
| **13. AI Disclosure Transparency** | Material uses of Claude Opus 5.5 and ChatGPT/Codex transparently disclosed under author oversight | Explicit AI assistance statement placed in Back Matter per current MDPI ethical guidelines. | **PASS** |
| **14. SI-Specific Cover Letter** | Cover letter specifically maps to "Machine Learning and Its Application for Anomaly Detection" SI | `submission/applsci/COVER_LETTER.md` addressed to Guest Editors Dr. Ayoub and Dr. Mouakher. | **PASS** |
| **15. Preceding Work Disclosure** | Relationship and distinct contributions relative to preceding DLG-GNN preprint clearly disclosed | Disclosed in Cover Letter, Introduction, and Abstract; clear differentiation established. | **PASS** |

---

## 2. Source Bundle & Artifact Digests

### 2.1 Preprints.org Submission Package (`submission/preprints/`)
- `DLG-Benchmark_A11_Preprints.tex`
- `DLG-Benchmark_A11_Preprints.pdf` (17 pages, 484,133 bytes)
- `source_bundle.zip` (720,919 bytes)
- `PREPRINT_GATE_REPORT.md`

### 2.2 MDPI Applied Sciences Submission Package (`submission/applsci/`)
- `DLG-Benchmark_A11_AppliedSciences.tex`
- `DLG-Benchmark_A11_AppliedSciences.pdf` (15 pages, 324,233 bytes)
- `source_bundle.zip` (1,757,119 bytes)
- `COVER_LETTER.md`
- `DATA_AVAILABILITY.md`
- `SUBMISSION_GATE_REPORT.md`

### 2.3 Supporting Diagnostic & Analytical Evidence (`projects/benchmark/`)
- `publication/a11/evidence_inventory.csv` (98 rows of detailed model-dataset cell metrics)
- `publication/a11/claims_to_evidence.csv` (10 core manuscript claims bound to evidence files)
- `publication/a11/benchmark_campaign_map.md` (Multi-campaign provenance documentation)
- `diagnostics/a11_score_direction_audit.md` (P0-B Low ROC-AUC audit report)
- `diagnostics/a11_score_label_alignment.json` (P0-B diagnostic machine-readable outcome)
- `analysis/a11_permutation_unit_tests.md` (P0-C Friedman audit report)
- `analysis/a11_friedman_repro_check.json` (P0-C 1M simulation results and unit tests)
- `analysis/a11_asymptotic_mc_comparison.csv` (P0-C S1--S4 comparison table)

---

## 3. Final Recommendation

All non-negotiable P0 gates and journal-specific criteria have been satisfied without requiring fresh neural network training. The manuscript presents an honest, rigorous, and visually compelling empirical benchmark that directly serves the objectives of the Applied Sciences Special Issue on Machine Learning for Anomaly Detection. 

The author may now proceed with the declared sequential submission:
1. Deposit `submission/preprints/source_bundle.zip` on **Preprints.org** and record the newly issued Preprint DOI.
2. Submit `submission/applsci/source_bundle.zip` and `COVER_LETTER.md` to **Applied Sciences** under Special Issue `C67H3U9QGT`.
