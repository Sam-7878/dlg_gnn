# Preprints.org Gate Verification Report (A16 Closure & A17 Gate Check)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A16  
**Date:** 2026-10-28  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Benchmark Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Status:** **ALL GATES PASSED (GREEN) — PREPRINT READY**

---

## 1. A16 / A17 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **Gate R1** | Remote Git Tag Independent Verification | **PASS** | Annotated tag `benchmark-a16-submission-2026-10-28` verified live on GitHub remote via `git ls-remote` and GitHub API HTTP 200, dereferencing to immutable target commit `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f`. |
| **Gate R2** | Private Review Language Removal | **PASS** | Removed private review reference; replaced with neutral mathematical text explaining permutation vs asymptotic $\chi^2$ differences. |
| **Gate R3** | Internal Source ZIP Filename Normalization | **PASS** | Rebuilt `source_bundle.zip` with internal main file `DLG-Benchmark_A16_Preprints.tex`. Clean unpack build verifies 18 pages, 0 undefined references/citations. |
| **Gate R4** | Memory Claim Verification & Reframing | **PASS** | Table 6 reframed from measured VRAM to illustrative peak memory materialization profiles on RTX 3090, accurately noting AnomalyDAE preflight OOM on Reddit-Syn. |
| **Gate R5** | Decision Matrix Bounded to Observations | **PASS** | Table 7 replaces unverified causal assertions with concrete empirical AP observations (+50.97% on Elliptic, -0.0656 on Reddit-Syn, -0.0252 on LANL). |
| **Gate R6** | Final Submission Gate & Negative Controls | **PASS** | 15/15 negative control mutation tests passed; true disk SHA-256 hashes verified; zero cross-artifact discrepancies. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A16_Preprints.tex`  
  - SHA-256: `a226b2dca3bf874f81f9c95d4e8790ced85020f9abd201d7d121653eeeba0422`
- **Rendered PDF (18 pages):** `DLG-Benchmark_A16_Preprints.pdf`  
  - SHA-256: `35343549afcda1d75768eca24ef8b146d6cdb6a765e7cfc12d4fbb8a41212a39`  
  - Size: `590,838` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `ab13612ffeaf4188dd97b3e827b1cd526c8f117a1a8e23b9bddbb04e1276d042`  
  - Size: `107,005` bytes
  - Contents: `DLG-Benchmark_A16_Preprints.tex`, `references.bib`, `figures/*.pdf`
