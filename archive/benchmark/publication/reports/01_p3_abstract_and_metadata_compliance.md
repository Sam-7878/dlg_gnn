# Round P3 Audit Report 01: Abstract & Metadata Compliance

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite  
**Scope:** Abstract word count compression, paragraph structure, Preprints/MDPI parity, and author ORCID metadata.

---

## 1. Abstract Length & Formatting Verification

Applied Sciences and general MDPI submission guidance recommend an abstract maximum of approximately 200 words. The master manuscript and both derivative submission bundles have been compressed and synchronized:

- **Target Word Count:** $\le 200$ words
- **Master Manuscript (`DLG-Benchmark.tex`):** 188 words (single paragraph, 0 line breaks)
- **Preprints Manuscript (`DLG-Benchmark-Preprint.tex`):** 188 words (single paragraph, clean `abstract` environment)
- **MDPI Manuscript (`publication/benchmark/mdpi/DLG-Benchmark.tex`):** 188 words (single paragraph, `\abstract{...}` macro)
- **Character Count:** 1,549 characters
- **Automated Test:** `tests/benchmark/publication_p3/test_abstract_word_count_200_max.py` (PASS)

---

## 2. Scientific Content & Cross-Source Parity

Both Preprints.org and MDPI Applied Sciences sources were audited for exact abstract text identity after whitespace normalization. All mandatory empirical elements from the frozen benchmark are fully preserved:

| Empirical Element | Master LaTeX | Preprints LaTeX | MDPI LaTeX | Verification Status |
|---|:---:|:---:|:---:|:---:|
| **Detector Count** | 8 configurations (6 baselines + 2 DLG) | 8 configurations | 8 configurations | **IDENTICAL** |
| **Primary Datasets** | 10 frozen primary datasets | 10 datasets | 10 datasets | **IDENTICAL** |
| **External Validation** | LANL-RedTeam | LANL-RedTeam | LANL-RedTeam | **IDENTICAL** |
| **Seeds & Runs** | 5 seeds, 355 successful runs | 5 seeds, 355 runs | 5 seeds, 355 runs | **IDENTICAL** |
| **Exact Support Accounting** | 71 of 80 pairs supported | 71 of 80 supported | 71 of 80 supported | **IDENTICAL** |
| **Fraud-Oriented Rankings** | DLG-Aug 1st (ROC-AUC 1.71, PR-AUC 1.71, F1 1.86) | 1.71 / 1.71 / 1.86 | 1.71 / 1.71 / 1.86 | **IDENTICAL** |
| **Positive Delta** | Elliptic PR-AUC $+0.0350$ | $+0.0350$ | $+0.0350$ | **IDENTICAL** |
| **Negative Delta** | Reddit-Syn PR-AUC $-0.0656$ | $-0.0656$ | $-0.0656$ | **IDENTICAL** |
| **LANL Delta** | DLG-Base exceeds DLG-Aug | DLG-Base exceeds DLG-Aug | DLG-Base exceeds DLG-Aug | **IDENTICAL** |
| **Final Scientific Framing** | Conditional rather than universal | Conditional rather than universal | Conditional rather than universal | **IDENTICAL** |

**Automated Test:** `tests/benchmark/publication_p3/test_preprint_mdpi_abstract_identity.py` (PASS)

---

## 3. Author Identity & ORCID Verification

Author metadata has been unified across all LaTeX headers, templates, and citation files:
- **First Author:** SeongSu Park (`parky@ajou.ac.kr`)
  - Department of Computer Engineering, Ajou University, Suwon 16499, Republic of Korea
  - ORCID: `0009-0008-4056-3875`
- **Corresponding Author:** Prof. Ki-Hyung Kim (`kkim86@ajou.ac.kr`)
  - Department of Cyber Security, Ajou University, Suwon 16499, Republic of Korea
  - ORCID: `0000-0002-2321-4475`
- **Citation Metadata:** `CITATION.cff` generated and synchronized with official ORCIDs.

---

## 4. Compliance Verdict
**STATUS: PASS (`READY_FOR_PREPRINTS_ORG_DEPOSIT`)**  
Abstract length, structure, parity, and metadata satisfy all Preprints.org and MDPI Applied Sciences author instructions without deviation.
