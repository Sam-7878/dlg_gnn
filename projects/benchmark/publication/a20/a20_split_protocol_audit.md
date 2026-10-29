# DLG-GNN Benchmark — A20 Split Generator Audit & Small-Budget Uncertainty Report

**Document ID:** `DLG-BENCH-A20-SPLIT-AUDIT-20261029`  
**Date:** 2026-10-29  
**Target:** Evaluation split provenance across cryptocurrency and synthetic benchmark datasets.

---

## 1. Executive Summary

A critical discrepancy was raised during peer review (Claude R5) regarding the manuscript's claim that all datasets utilized "random stratified cross-seed splits":
- In the frozen Tier-1 cryptocurrency benchmarks (`Ethereum`, `BSC`, `Polygon`), the positive test count $M_\mathrm{test}$ varies across seeds 42–46 (e.g., Polygon has 16, 11, 7, 18, and 9 positives across seeds 42–46), which contradicts stratified sampling where positive counts are held fixed proportional to class prevalence.
- An independent code audit of the split generators in `evaluation/benchmark/v2/scripts/a04_build_dataset_manifest.py` and upstream pipeline scripts was conducted to establish exact implementation provenance.

---

## 2. Code-Level Inspection Findings

### 2.1. Cryptocurrency Splits (`Ethereum`, `BSC`, `Polygon`)
In `evaluation/benchmark/v2/scripts/a04_build_dataset_manifest.py` (lines 94–105):
```python
if name in {'Ethereum', 'BSC', 'Polygon'}:
    # ...
    observe(row, data, source, prediction_unit='smart_contract_node', split_policy='seeded_random_60_20_20')
    n = data.num_nodes
    for seed in range(42, 47):
        perm = np.random.RandomState(seed).permutation(n)
        tr = np.zeros(n, dtype=np.bool_); va = tr.copy(); te = tr.copy()
        a = int(0.6 * n); b = int(0.8 * n)
        tr[perm[:a]] = True
        va[perm[a:b]] = True
        te[perm[b:]] = True
```
- **Audit Verdict:** The splits were generated using pure random permutations of all $N$ node indices (`np.random.RandomState(seed).permutation(n)`) into fixed 60% train, 20% validation, and 20% test partitions.
- **Stratification Status:** **Non-stratified.** The split generator does NOT condition on labels (`data.y`), nor does it invoke scikit-learn's `stratify` parameter.
- **Consequence:** The number of anomalies $M_\mathrm{test}$ in the test set is a binomial random variable governed by the test slice permutation, leading directly to the observed seed-to-seed positive variation:
  - Ethereum ($N=14,385, N_\mathrm{test}=2,893$): $M_\mathrm{test} \in [1143, 1246, 1153, 1184, 1189]$ (mean 1183.0, $\pi \in [0.3951, 0.4307]$)
  - BSC ($N=7,481, N_\mathrm{test}=1,500$): $M_\mathrm{test} \in [215, 240, 232, 215, 203]$ (mean 221.0, $\pi \in [0.1353, 0.1600]$)
  - Polygon ($N=2,303, N_\mathrm{test}=471$): $M_\mathrm{test} \in [16, 11, 7, 18, 9]$ (mean 12.2, $\pi \in [0.0149, 0.0382]$)

### 2.2. Synthetic Benchmark Splits
In contrast, synthetic graphs (lines 58–62 of `a04_build_dataset_manifest.py`) used:
```python
val_local, test_local = train_test_split(np.arange(len(eligible)), test_size=0.5, random_state=seed, stratify=labels)
```
These were indeed stratified on the eligible evaluation subset.

---

## 3. Small-Budget Statistical Uncertainty on Polygon

On Polygon ($N=2,303$):
- Test set size is $N_\mathrm{test} = 471$.
- At budget $\tau = 1\%$, the alert capacity is $K = \lfloor 0.01 \times 471 \rfloor = 4$ or $\lceil 0.01 \times 471 \rceil = 5$ (frozen convention: $K=5$).
- Ground-truth positives across seeds 42–46 are: $[16, 11, 7, 18, 9]$ (mean $12.2$).
- **Granularity of Precision@1%:** With $K=5$, each correctly detected true positive increments Precision@1% by:
  $$\Delta P = \frac{1}{5} = 0.200 \quad (20.0\%)$$
- **Granularity of Recall@1%:** With $M_\mathrm{test}$ as small as 7 (seed 44), one TP corresponds to $1/7 \approx 14.29\%$ recall.
- **Editorial Action:** The manuscript must explicitly describe Polygon alert-budget metrics as **descriptive and exploratory evidence subject to small-sample discrete step variance**, rather than operational certainty.

---

## 4. Manuscript Text Revisions Required

1. **Table 1 and Table 2 Footnotes:**
   - Change: *"random stratified cross-seed splits"*
   - To: *"seed-dependent evaluation splits (stratified for synthetic graphs; seeded random non-stratified 60/20/20 splits for cryptocurrency graphs)"*
2. **Table 3 Caption and Discussion:**
   - Explicitly cite the non-stratified nature of the cryptocurrency evaluation sets.
   - Note the low-count sensitivity of Polygon ($N_\mathrm{test}=471, K@1\%=5, M_\mathrm{test} \in [7, 18]$) where $\Delta P = 0.20$ per true positive.

