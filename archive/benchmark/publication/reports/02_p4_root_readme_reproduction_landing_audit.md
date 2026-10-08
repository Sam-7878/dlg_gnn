# Round P4 Audit Report 02: Root README Reproduction Landing Page Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite (Round P4)  
**Target File:** `README.md` (root repository landing page)

---

## 1. Executive Summary

In previous rounds, the root `README.md` served primarily as a historical AML/financial fraud system document for the internal GoatBank codebase. In Round P4, root `README.md` was redesigned into an authoritative, publisher-neutral Benchmark Reproduction Landing Page that immediately greets visitors and reviewers following the Preprints.org Data Availability URL (`https://github.com/Sam-7878/dlg_gnn`).

---

## 2. Key Sections Audit

The transformed `README.md` includes all structural components required by the Round P4 Work Order:

| Section | Implementation Status | Notes |
|---|:---:|---|
| **Header & Title** | **PASS** | *"DLG-GNN Benchmark: Empirical Evaluation of Directed Local-Global Graph Neural Networks for Fraud and Anomaly Detection"* |
| **Authors & ORCID** | **PASS** | SeongSu Park ([0009-0008-4056-3875](https://orcid.org/0009-0008-4056-3875))<br>Ki-Hyung Kim ([0000-0002-2321-4475](https://orcid.org/0000-0002-2321-4475)) |
| **Publication State** | **PASS** | Declares *"Preprint preparation / deposit pending at Preprints.org"* |
| **Benchmark Summary** | **PASS** | 10 primary datasets, 8 detector configurations, 71/80 supported pairs, 355 primary runs, 45 sensitivity controls |
| **Mode 1 Quick Reproduction** | **PASS** | Exact command: `python scripts/reproduce_frozen_artifacts.py` (< 1 sec execution) |
| **Mode 2 Full Pipeline** | **PASS** | Full rerun runner: `experiments/benchmark/run_sci_round5_final.py` |
| **Release Claim Gate** | **PASS** | Honestly displays *"Release prepared; publication pending"* prior to GitHub Release |
| **Frozen Hashes Table** | **PASS** | Displays SHA-256 hashes for `benchmark_raw.csv` and support matrix |
| **Data Provenance** | **PASS** | Distinguishes public base graphs from synthetic anomaly injection protocol |
| **Preceding Work DOI** | **PASS** | Clearly identifies DLG-GNN foundational paper: `10.20944/preprints202609.0848.v1` |
| **Historical Context** | **PASS** | Preserves historical GoatBank context under a distinct bottom section |

---

## 3. Claim Gate Compliance

In accordance with Section 6 of the Round P4 Work Order:
- **No premature release claims:** Does not claim the release is live or download links are active before the user publishes the release on GitHub.
- **Accurate phrasing used:** *"Release prepared; publication pending on GitHub"*
- **Preprints DOI:** Declared as *Pending issuance upon Preprints.org screening*.

---

## 4. Automated Test Verification

- `tests/benchmark/publication_p4/test_root_readme_is_benchmark_landing_page.py` (PASS)
- `tests/benchmark/publication_p4/test_public_release_orcid_consistency.py` (PASS)
- `tests/benchmark/publication_p4/test_public_release_preceding_doi_identity.py` (PASS)

---

## 5. Audit Verdict

**STATUS: PASS**  
The root `README.md` is fully compliant with the reproduction landing page specification.
