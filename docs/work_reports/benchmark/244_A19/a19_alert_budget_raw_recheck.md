# Gate A19-G1: Raw-Score Alert Budget Verification Gate Report

**Date:** 2026-10-28 / 2026-10-29  
**Status:** PASS (Source-Level Raw Array Reconstruction Established)  
**Scope:** 105 Cryptocurrency runs (Ethereum, BSC, Polygon × 7 models × 5 seeds) evaluated at q=1% and q=5% (210 evaluations)

## 1. Specification of Evaluation Mechanics

In accordance with Gate A19-G1 work order specifications:
1. **Target Population:** Restricted strictly to nodes where `test_mask == True`.
   - Ethereum: `N_test = 2893`, seed-specific `M_test = [1143, 1246, 1153, 1184, 1189]`, mean = 1183.4
   - BSC: `N_test = 1500`, seed-specific `M_test = [215, 237, 217, 206, 230]`, mean = 221.0
   - Polygon: `N_test = 471`, seed-specific `M_test = [16, 11, 7, 18, 9]`, mean = 12.2
2. **K Rounding Convention:** `K = int(round(q * N_test))`.
   - Ethereum: K@1% = 29, K@5% = 145
   - BSC: K@1% = 15, K@5% = 75
   - Polygon: K@1% = 5, K@5% = 24
3. **Score Orientation:** Higher score denotes higher predicted anomalousness (`sort_order = np.argsort(-test_scores, kind='mergesort')`).
4. **Tie-breaking Convention:** Stable merge sort (`kind='mergesort'`) preserving native array index sequence in the presence of tied scores.
5. **No Manipulation Clause:** No relabeling, score-inversion, or threshold-tuning was performed.

## 2. Verification Results Summary

- **Total Evaluations Checked:** 210 / 210
- **Exact Matches against `metrics.json`:** 210 / 210 (100.0%)
- **Exact Matches against A18 `alert_budget_per_seed_verified.csv`:** 210 / 210 (100.0%)
- **Exact Set Matches on `selected_node_ids`:** 210 / 210 (100.0%)
- **Arithmetic Invariants Checked:**
  - $0 \le \mathrm{TP} \le \min(K, M_\mathrm{test})$: 0 violations
  - $P@K = \mathrm{TP} / K$: 0 violations
  - $R@K = \mathrm{TP} / M_\mathrm{test}$: 0 violations

## 3. Comparison with A18 Finding
A18 independently verified the algebraic consistency of `metrics.json`.
A19 has advanced this verification from **metric-JSON-derived** to **genuine source-level raw array reconstruction**:
Every top-K node set, TP count, Precision, and Recall was recomputed from the actual floating-point prediction vectors in `scores.npz` and ground-truth binary labels in `labels.npy` on the `test_mask`.

## 4. Artifact Outputs
- `projects/benchmark/publication/a19/a19_alert_budget_raw_recheck.csv`
- `projects/benchmark/publication/a19/a19_crypto_scores_manifest.csv`
- `docs/work_reports/benchmark/244_A19/a19_alert_budget_raw_recheck.md`
