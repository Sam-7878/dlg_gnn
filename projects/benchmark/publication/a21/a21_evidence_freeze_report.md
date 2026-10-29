# Gate L3: Evidence Freeze Metadata Reconciliation Report

**Document ID:** `DLG-BENCH-GATE-L3-FREEZE-REPORT`  
**Date:** 2026-10-29  
**Parent Work Order:** `DLG-BENCH-WO-A21-LEDGER-ONLY`  
**Status:** **PASS** (Hash Keys Strictly Separated into Valid 64-Hex Digests and Path Strings)  

---

## 1. Context and Objective

During independent submission review of A20 (`DLG_Benchmark_A20_Independent_Submission_Review_2026-10-29.md`, Finding F3), it was observed that in `projects/benchmark/publication/a20/a20_input_freeze.json`:
- The fields `canonical_run_registry_hash` and `support_matrix_hash` erroneously held relative file paths rather than 64-character SHA-256 hex digests.
- While the six submission package hashes and eight empirical evidence hashes were valid SHA-256 strings, archival rigor requires separating file path pointers from cryptographic digest fields.
- Objective: Update `a20_input_freeze.json` to store distinct `<field>_path` and `<field>_sha256` keys, produce `projects/benchmark/publication/a21/a21_evidence_freeze_patch.json`, and verify all hashes against disk bytes.

---

## 2. Cryptographic Verification of Registry & Support Matrix

Both files were read directly from disk and hashed using standard SHA-256:

| Referenced Entity | File Path | File Size (Bytes) | SHA-256 Digest | Verification Status |
|---|---|---|---|---|
| **Canonical Run Registry** | `projects/benchmark/publication/a16/canonical_selected_run_registry.csv` | 671,520 | `45d4ca7b4a9b010aa779e7cb60a1816e0f1c08e98039a119d008a1d1cd5db907` | **PASS (Exact Disk Match)** |
| **Hardware Support Matrix** | `projects/benchmark/publication/a16/support_14x7.csv` | 6,061 | `bbd89bf88f05047d2f8f29c2065c73791d480bdfcadb78293f7d03a543b8ecd8` | **PASS (Exact Disk Match)** |

---

## 3. Preservation of Truthful Provenance Tiers

In full alignment with the submission disclosures in Section 4.2 and `DATA_AVAILABILITY.md`:
1. **Tier 1 (105 Cryptocurrency Evaluations):** Per-node continuous prediction arrays ($y_\mathrm{score}$) and binary labels ($y_\mathrm{true}$) for Ethereum, BSC, and Polygon (7 models $\times$ 5 seeds) remain author-locally preserved in compressed NumPy `.npz` format and are available upon reasonable academic request.
2. **Tier 2 (350 Historical Baseline Evaluations):** 280 primary runs, 35 BitcoinOTC runs, and 35 LANL runs originally archived scalar summary metrics rather than continuous score arrays. As truthfully disclosed, these arrays are not retroactively synthesized.

---

## 4. Verification Verdict

- [x] All keys ending in `_sha256` are strictly 64-hexadecimal characters; no file paths in hash fields.
- [x] File paths preserved in corresponding `_path` fields.
- [x] Disk bytes match 100% across all 8 evidence files and 2 registry tables.
- **GATE L3 STATUS: PASS**

