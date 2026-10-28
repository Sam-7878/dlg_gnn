# DGraphFin Evaluation Split Identity Audit (A16-G3)

**Document ID:** DLG-BENCH-DGRAPHFIN-SPLIT-AUDIT-A16  
**Date:** 2026-10-28  
**Classification:** **`SAME_MASK_DIFFERENT_METADATA`**  
**Verification Status:** **PASS (100% Identical Test Node Indices and Positive Labels)**

---

## 1. Executive Summary

During the A15 independent audit, inspection of `canonical_selected_run_registry.csv` noted that on `DGraphFin`, five supported models (`DOMINANT`, `CoLA`, `OCGNN`, `DLG-Base`, `DLG-Aug`) recorded `split_hash = 4ebf6ded53b8054653f40e6157617b71c8230e0e2ed0cf32c3d5d83b1e76b50c`, whereas `AnomalyDAE` recorded `split_hash = 3fd8c88fa4b5e4919c78bc9ad5bcce33d2e76b78c7e0462b512e79ee441e63b6`, even though all six models reported `test_total = 183,840`, `test_positive = 2,326`, and `test_prevalence = 0.0126523064`.

Direct cryptographic reconstruction from `/mnt/d/_Work/_data/DLG/DGraphFin/dgraphfin.npz` and `dataset_manifest_canonical.json` proves that **all six models evaluated the exact same 183,840 test node indices and 2,326 positive fraud labels** on the identical 1,225,601-node induced DGraphFin graph. The two hash strings differ solely because two export scripts serialized different metadata representations of the same boolean masks:

1. **Boolean Mask Hash (`4ebf6ded53b8054653f40e6157617b71c8230e0e2ed0cf32c3d5d83b1e76b50c`):**
   - **Models:** `DOMINANT`, `CoLA`, `OCGNN`, `DLG-Base`, `DLG-Aug` (25 runs across seeds 42--46)
   - **Source Script:** `evaluation/benchmark/v2/scripts/a04_export_evidence.py`
   - **Serialization Formula:** SHA-256 of the sorted JSON dictionary `{'test_mask_hash', 'train_mask_hash', 'val_mask_hash'}` where each inner hash is computed over the length-`1,225,601` boolean mask tensor.
   - **Evaluated Test Set:** `183,840` nodes, `2,326` positives (`0.0126523064`).

2. **Flatnonzero Index & Label Slice Hash (`3fd8c88fa4b5e4919c78bc9ad5bcce33d2e76b78c7e0462b512e79ee441e63b6`):**
   - **Models:** `AnomalyDAE` (5 runs across seeds 42--46)
   - **Source Script:** `evaluation/benchmark/v2/scripts/a05_run_anomalydae_large.py`
   - **Serialization Formula:** SHA-256 of the sorted JSON dictionary `{'test', 'test_y', 'val', 'val_y'}` where `'test'` is `sha256(np.flatnonzero(test_mask).tobytes())` and `'test_y'` is `sha256(y[test_idx].tobytes())`.
   - **Evaluated Test Set:** `183,840` nodes, `2,326` positives (`0.0126523064`).

---

## 2. Cryptographic Verification Summary

| Property | Value |
|---|---|
| **Classification** | `SAME_MASK_DIFFERENT_METADATA` |
| **Induced Graph Nodes ($N_{\text{input}}$)** | `1,225,601` |
| **Official Train Mask Nodes** | `857,899` |
| **Official Valid Mask Nodes / Positives** | `183,862` / `2,326` |
| **Official Test Mask Nodes / Positives** | `183,840` / `2,326` |
| **Test Positive Rate ($\pi_{\text{test}}$)** | `0.0126523064` (`0.0127`) |
| **Boolean Mask Hash (`a04_export_evidence.py`)** | `4ebf6ded53b8054653f40e6157617b71c8230e0e2ed0cf32c3d5d83b1e76b50c` |
| **Index Array Hash (`a05_run_anomalydae_large.py`)** | `3fd8c88fa4b5e4919c78bc9ad5bcce33d2e76b78c7e0462b512e79ee441e63b6` |
| **`np.array_equal(flatnonzero(mask), test_idx)`** | `True` |
| **`np.array_equal(y[mask], y[test_idx])`** | `True` |
