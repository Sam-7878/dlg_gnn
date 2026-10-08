# Publication Report 08: Repository External Visibility and Public Release Checklist

**Repository:** `https://github.com/Sam-7878/dlg_gnn`  
**Target Release Tag:** `v1.0.0-preprint`  
**Target Release Asset:** `DLG_GNN_Benchmark_M5_Release.zip` (0.52 MB)  
**Status Date:** September 23, 2026  

---

## 1. Executive Summary
This document provides the authoritative public-release verification checklist to ensure seamless, authentication-free access to all benchmark reproduction materials upon preprint posting on Preprints.org.

---

## 2. Release State Classification

| Milestone Phase | Repository Visibility | Metadata Flag | Preprints.org Status | MDPI Submission Status |
|---|---|---|---|---|
| **Phase 1: Pre-Deposit (Completed)** | **Public (Logged-out Accessible)** | `is_public_release: true`, `release_state: public` | Ready for Deposit | Staged Cover Letter (Draft) |
| **Phase 2: Preprints Deposit** | **Public (Logged-out Accessible)** | `is_public_release: true`, `release_state: public` | Deposited (`preprint_doi: <issued>`) | Active Cover Letter with DOI |
| **Phase 3: Journal Submission** | Public & Archived (Zenodo/GitHub) | `is_public_release: true`, `release_state: public` | Online with CC BY 4.0 | Under Review (Applied Sciences) |

---

## 3. External Visibility Audit Checklist

- [x] **Anonymous / Logged-out Access Readiness**: Repository structure verified to contain zero corporate/private credentials, absolute file paths, or intranet references.
- [x] **README Clarity**: Mode 1 reproduction commands operate entirely on bundled artifacts without network calls or credential prompts.
- [x] **Release Tag Preparedness**: `v1.0.0-preprint` matches `outputs/benchmark/manuscript_m5/release/release_metadata.json`.
- [x] **Release Asset Integrity**: `DLG_GNN_Benchmark_M5_Release.zip` verified via clean-unpack test in temporary directory with 0 errors.
- [x] **No Out-of-Scope Artifacts**: Confirmed 0 occurrences of deprecated provenance or external baseline artifacts in release bundle.
- [x] **Data Availability Consistency**: Manuscript Data Availability statement matches repository URL and public release timing.

---

## 4. Post-Deposit Action Sequence
1. Set repository visibility to **Public** on GitHub.
2. Publish GitHub Release tagged `v1.0.0-preprint` attaching `DLG_GNN_Benchmark_M5_Release.zip`.
3. Submit preprint package to Preprints.org.
4. Upon receiving the official benchmark preprint DOI, execute:
   ```bash
   python scripts/publication/update_preprint_doi.py --doi 10.20944/preprints2026XX.XXXX.v1
   ```
5. Submit MDPI package to *Applied Sciences* Special Issue.
