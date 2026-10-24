# A11 Friedman & Permutation Statistics Audit Report

**Date:** 2026-10-24  
**Audit Target:** Phase P0-C Friedman/Permutation Statistics Audit  
**Status:** **PASS — METHODOLOGICALLY SOUND & INDEPENDENTLY REPRODUCED**

---

## 1. Executive Summary

Claude Review questioned why asymptotic $\chi^2$ p-values and Monte Carlo permutation p-values differ by an order of magnitude in views S1, S2, and S4. 

This audit demonstrates conclusively:
1. **The difference is mathematically expected, not a software bug:** The asymptotic Friedman test assumes large $N$ (number of datasets) so that the rank totals converge in distribution to multivariate normal, yielding a $\chi^2_{k-1}$ reference distribution. In small-$N$ benchmark matrices ($N=7$ or $N=8$), the discrete permutation distribution of within-dataset rank vectors has substantially lighter tails than the continuous $\chi^2$ distribution. Consequently, an extreme rank configuration yields a much smaller tail probability under the exact discrete within-block permutation reference than under the $\chi^2$ approximation.
2. **Independent 1,000,000 Permutations Reproduce the Exact Scale:** Running 1,000,000 within-block permutations confirms the reported magnitude of Monte Carlo p-values (S1 $\approx 4.0 \times 10^{-6}$, S2 $\approx 2.1 \times 10^{-5}$, S4 $\approx 8.0 \times 10^{-6}$).
3. **Finite-Monte-Carlo Resolution Caveat:** When the number of empirical exceedances is very small ($0 \le b \le 3$), the Monte Carlo estimate has finite resolution ($p = (b+1)/(B+1)$) and should be reported with its simulation envelope, rather than as a continuous six-decimal truth. No p-value should ever be printed as exactly 0.0.

---

## 2. Unit Test Suite Results

The following unit tests verified the correctness of the permutation engine:

| Test Name | Description | Verification Criterion | Status |
|---|---|---|---|
| `test_identical_ranks_maximum` | Equal ranking across all blocks | Computed $Q = n(k-1)$ exactly ($9.0 == 9.0$) | **PASS** |
| `test_within_block_permutation` | Column permutation within each dataset block | Preserves valid rank distributions under null | **PASS** |
| `test_zero_exceedance_formula` | Zero exceedances ($b=0$) floor | $p = (0+1)/(B+1) = 5.0 \times 10^{-6}$ ($B=200,000$) | **PASS** |

---

## 3. Comprehensive View Comparison: Asymptotic vs 200k MC vs 1M MC

| View | N Datasets | K Models | Friedman $Q$ | Asymptotic $\chi^2$ $p$ | Frozen 200k MC $p$ | Independent 1M MC $p$ | Exceedances in 1M ($b$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S1** (Complete primary) | 8 | 7 | 29.0357 | $5.99 \times 10^{-5}$ | $5.00 \times 10^{-6}$ (floor) | $\mathbf{4.00 \times 10^{-6}}$ | 3 |
| **S2** (Continuity across all) | 13 | 5 | 23.7538 | $8.95 \times 10^{-5}$ | $2.00 \times 10^{-5}$ | $\mathbf{2.10 \times 10^{-5}}$ | 20 |
| **S3** (Financial only) | 6 | 5 | 6.1333 | 0.1894 | 0.193584 | $\mathbf{0.193610}$ | 193,609 |
| **S4** (Injected only) | 7 | 5 | 21.8286 | $2.17 \times 10^{-4}$ | $5.00 \times 10^{-6}$ (floor) | $\mathbf{8.00 \times 10^{-6}}$ | 7 |

### Detailed View Membership:
- **S1 (8 datasets, 7 models):** Elliptic, BitcoinOTC, Ethereum, BSC, Polygon, Reddit-Syn, Flickr-Syn, Cora-Syn. (Excluded due to OOM on GADNR: DGraphFin, Yelp-Syn, Amazon-Syn, CiteSeer-Syn, PubMed-Syn).
- **S2 (13 datasets, 5 continuity models):** All 13 primary datasets evaluated across DOMINANT, CoLA, OCGNN, DLG-Base, DLG-Aug.
- **S3 (6 financial datasets, 5 continuity models):** Elliptic, DGraphFin, BitcoinOTC, Ethereum, BSC, Polygon. ($Q=6.1333, p \approx 0.19$, non-significant).
- **S4 (7 synthetic injected datasets, 5 continuity models):** Yelp-Syn, Amazon-Syn, Reddit-Syn, Flickr-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn. ($Q=21.8286, p < 10^{-4}$, highly significant).

---

## 4. Manuscript Recommendations for Methods & Results

1. **Retain Both $p$-values:** Present both asymptotic $\chi^2$ p-value and the finite-sample permutation p-value in Table and narrative.
2. **Explicitly Clarify Permutation Mechanism:**
   $$\text{Null hypothesis: Within each dataset block } i, \text{ all } k! \text{ model ranking permutations are equally likely.}$$
3. **State the Finite-MC Caveat:**
   Explain that at $B=200,000$, zero observed exceedances yields the estimator floor $p = 1 / (B+1) \approx 5.0 \times 10^{-6}$. The 1M independent resimulation demonstrates 3 to 7 exceedances, confirming the true tail probability is on the order of $4\times 10^{-6}$ to $8\times 10^{-6}$.
4. **No Model Retraining Required:** Code and statistics are verified to be mathematically sound. Gate A11-P0-02 is **CLEARED**.
