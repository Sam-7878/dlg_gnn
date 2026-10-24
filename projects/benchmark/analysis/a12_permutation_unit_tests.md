# A12 Friedman Statistics & Monte Carlo Permutation Verification (P0-04)

**Date:** 2026-10-24  
**Pinned Seed:** `20261024`  
**Monte Carlo Draws:** `1,000,000`  
**Status:** **A12-G4 PASSED — Single Auditable Monte Carlo Output Synchronized**

## 1. Unit Test Verification Matrix

| Test Name | Description | Computed | Expected | Status |
|---|---|---|---|---|
| `test_identical_ranks_theoretical_maximum` | Mathematical invariant check | PASS | PASS | **PASS** |
| `test_within_block_permutation_invariants` | Mathematical invariant check | PASS | PASS | **PASS** |
| `test_zero_exceedance_formula` | Mathematical invariant check | PASS | PASS | **PASS** |

## 2. Statistical Reproduction Table (1,000,000 Permutations)

| View | $N$ | $K$ | Friedman $Q$ | Asymptotic $\chi^2$ $p$ | Exceedances ($b$) | Exact MC $p$ | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **S1_SevenModel_CompleteCase** | 8 | 7 | 29.0357 | 5.989741e-05 | 3 | **3.999996e-06** | [8.000580e-08, 7.919986e-06] |
| **S2_ContinuityFive** | 13 | 5 | 23.7538 | 8.948082e-05 | 20 | **2.099998e-05** | [1.201823e-05, 2.998173e-05] |
| **S3_FinancialBlockchainSix_MixedLabels** | 6 | 5 | 6.1333 | 1.894097e-01 | 192107 | **1.921078e-01** | [1.913357e-01, 1.928800e-01] |
| **S4_SyntheticSeven** | 7 | 5 | 21.8286 | 2.167973e-04 | 0 | **9.999990e-07** | [0.000000e+00, 2.959997e-06] |

