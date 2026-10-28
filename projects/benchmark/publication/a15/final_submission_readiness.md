# Final Submission Readiness Report (A16 Closure)

**Document ID:** DLG-BENCH-READINESS-A16  
**Date:** 2026-10-28  
**Benchmark Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Dual Submission Sequence:**
1. **Preprints.org FIRST:** Deposit publisher-neutral package (`18` pages) to secure distinct benchmark preprint DOI.
2. **MDPI Applied Sciences SECOND:** Submit Special Issue package (`16` pages) to Special Issue "Machine Learning and Its Application for Anomaly Detection" (Special Issue ID: `C67H3U9QGT`, deadline 2026-12-31).

---

## 1. Compliance Checklist (Work Order Gates A16-G1 through A16-G6)

| Gate | Description | Outcome | Sign-off |
|---|---|---|---|
| **A16-G1** | Programmatic Denominator Audit | `evidence_denominator_audit.md` and `denominator_crosscheck_a16.csv` auto-generated from canonical CSVs; 14/14 datasets match Table 1 & Table 2 identically. | **APPROVED** |
| **A16-G2** | Resource Failure Provenance | 84/91 primary support; 7 unsupported pairs mapped to 3 exact categories in `support_14x7.csv`, manuscript Section 5.1, Figure 2 caption, and `release_manifest.json`. | **APPROVED** |
| **A16-G3** | DGraphFin Split Hash Identity | Verified `SAME_MASK_DIFFERENT_METADATA`: all 6 supported models on DGraphFin evaluate the identical 183,840 test nodes and 2,326 positives. | **APPROVED** |
| **A16-G4** | Concise 172-Word Abstract | Single-paragraph 172-word abstract verified across both manuscripts; Author Contributions and Acknowledgments updated. | **APPROVED** |
| **A16-G5** | SHA-256 Byte Verification & Clean Unpack | True disk byte hashes computed after final build; clean unpack build succeeds (18p Preprints, 16p MDPI). | **APPROVED** |
| **A16-G6** | Extended Validator & Mutation Testing | 12/12 negative control injection tests rejected; zero cross-artifact discrepancies. | **APPROVED** |

---

## 2. Definitive Deliverable Checksums

See `publication/a16/final_artifact_hashes.sha256` and submission `SHA256SUMS.txt` for cryptographic byte-exact records.
