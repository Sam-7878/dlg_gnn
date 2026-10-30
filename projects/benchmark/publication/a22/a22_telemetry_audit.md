# Gate A22-G7: Telemetry, Memory Classification, and GADNR Diagnostic

- **Date:** 2026-10-30
- **Work Order Reference:** `DLG-BENCH-WO-A22-VALIDITY-AND-REVIEWER-READINESS`
- **Status:** PASS

---

## 1. Hardware Memory and Execution Classification Taxomony

To prevent reviewer confusion between theoretical memory bounds and observed hardware execution, Table 6 entries are categorized under the following strict taxonomy:

- `MATH_LOWER_BOUND`: The absolute minimum decimal memory to materialize a dense float32 matrix ($4N^2$ bytes). For Reddit-Syn ($N=232,965$), this is $217\,\mathrm{GB}$; for DGraphFin ($N=1,225,601$), $6.0\,\mathrm{TB}$. This is a mathematical impossibility for single-GPU 24GB execution without profiling.
- `ILLUSTRATIVE_ESTIMATE`: Peak VRAM profiles derived under standard PyTorch memory allocation models (batch size $B=2{,}048$). E.g., $\approx 7.1\,\mathrm{GB}$ for row-block AnomalyDAE structural reconstruction.
- `OBSERVED_CUDA_OOM`: Actual hardware runtime failure on 24-GB RTX 3090. E.g., AnomalyDAE full-graph encoder forward pass on Reddit-Syn and Yelp-Syn.
- `HISTORICAL_LEGACY_8GB_OOM`: GADNR on Flickr-Syn, which was observed to exceed memory on an 8-GB legacy environment; it was not executed on 24-GB hardware.
- `NUMERICAL_FAILURE (NO_APPROVED_RAW_RUN)`: GADNR on Reddit-Syn, Yelp-Syn, Elliptic, and DGraphFin. Failed during Gaussian mixture covariance determinant / Cholesky computation due to feature collinearity; never reached memory limits.

## 2. GADNR Numerical Singularity Diagnostic Results

- Raw Covariance Determinant: `-0.0` (Cholesky Error: `linalg.cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).`)
- Stabilization with Ridge Jitter $\Sigma + \epsilon I$ ($\epsilon = 10^{-4}$): Determinant $> 0$, Cholesky decomposition succeeds.
- **Conclusion:** Future extensions of GADNR can resolve this failure mode by adding numerical jitter, but within our frozen benchmark protocol, these cells are correctly disclosed as unapproved numerical execution failures.

