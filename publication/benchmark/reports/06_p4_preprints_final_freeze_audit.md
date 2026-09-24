# Round P4 Audit Report 06: Preprints.org Deposit Package Final Freeze Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite (Round P4)  
**Target Package:** `publication/benchmark/preprints/DLG_Benchmark_Preprints_Submission.zip`  
**Target PDF:** `publication/benchmark/preprints/DLG-Benchmark-Preprint.pdf`  
**PDF SHA-256:** `26B289207E598676C40A342E0D4E6D2DB6FA2A28E0DAE4829EC5A123FF7B62D3`  
**Zip SHA-256:** `ABFC66B5925FB17BF680B6FE1F3CCD01050AE84A4F5B20AA91F5C220B431F0FF`

---

## 1. Executive Summary

Round P4 finalized the editorial review of the manuscript abstract and recompiled the publisher-neutral Preprints.org submission bundle. All mathematical symbols, author metadata, institutional emails, ORCIDs, and data availability statements are frozen and verified against automated compilation tests.

---

## 2. Manuscript Editorial Pass Audit

- **Previous Wording:** *"local augmentation proved strongly dataset-dependent"*
- **Refined Wording (Section 11):** *"local augmentation showed strong dataset dependence"*
- **Parity Check:** Synchronized simultaneously across:
  - `publication/benchmark/preprints/DLG-Benchmark-Preprint.tex`
  - `docs/papers/_42_Benchmark/DLG-Benchmark.tex` (MDPI source)
- **Scientific Impact:** Zero change to experimental findings or conclusions; enhances natural academic flow while strictly maintaining $\le 200$ words length (188 substantive words).

---

## 3. Submission Bundle Contents Audit

The submission archive `DLG_Benchmark_Preprints_Submission.zip` contains all self-contained assets for direct deposit on Preprints.org:

| Item | Included in Package | Verification Status |
|---|:---:|:---:|
| **LaTeX Source** | `DLG-Benchmark-Preprint.tex` | **PASS** (Publisher-neutral, zero MDPI branding) |
| **Compiled PDF** | `DLG-Benchmark-Preprint.pdf` | **PASS** (32 pages, 501.6 KB, 0 errors) |
| **Bibliography** | `references.bib` | **PASS** (Foundational DLG paper included with DOI) |
| **Graphical Abstract** | `Figure_GA_Revised.png` & `.pdf` | **PASS** (High-res, 300 DPI, `-Syn` notation verified) |
| **All Manuscript Figures** | Figures 1–8 (PNG / PDF) | **PASS** (Zero external dependencies) |
| **Generated Tables** | Tables 1–8 | **PASS** (Generated from frozen benchmark data) |

---

## 4. Metadata & Formatting Conformance

- **Author 1:** SeongSu Park (ORCID: `0009-0008-4056-3875`, email: `sspark@ajou.ac.kr`)
- **Author 2:** Ki-Hyung Kim (ORCID: `0000-0002-2321-4475`, email: `kkim8473@ajou.ac.kr`, Corresponding Author)
- **Data Availability URL:** `https://github.com/Sam-7878/dlg_gnn`
- **Publisher Branding:** 0 occurrences of MDPI logos, watermarks, or journal headers.

---

## 5. Automated Test Verification

- `tests/benchmark/publication_p4/test_preprint_abstract_editorial_wording.py` (PASS)
- `tests/benchmark/publication_p4/test_clean_preprints_submission_zip_contents.py` (PASS)
- `tests/benchmark/publication_p3/test_preprints_bundle_compiles.py` (PASS)

---

## 6. Audit Verdict

**STATUS: PASS**  
The Preprints.org submission package is frozen, mathematically clean, and prepared for immediate deposit upon release publication.
