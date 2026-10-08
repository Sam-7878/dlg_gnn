# Grok M2 Audit Report: Root Cause of CONAD ≈ DOMINANT Identity

## 1. Executive Summary
ROOT CAUSE CONFIRMED: PyGOD's implementation of CONAD contains two fatal issues: 1) Upstream CONAD calls margin_loss_func(h, h, h_aug), passing (input1=h, input2=h). Because (input1 - input2) == 0, the contrastive loss has an analytical gradient of EXACTLY ZERO. 2) During inference, CONAD returns score = eta * reconstruction_score, which is merely a positive constant scalar multiple of DOMINANT's reconstruction score. In rank-based anomaly detection metrics (ROC-AUC and PR-AUC), scalar multiplication by eta preserves rankings identically (Spearman rho = 1.0). Hence, CONAD behaves as DOMINANT with inert contrastive loss and identical anomaly rankings.

## 2. Mathematical Proof of Zero Contrastive Gradient
- MarginRankingLoss call: `margin_loss_func(h, h, h_aug)`
- Loss value: `0.22499999403953552` (constant margin)
- Gradient norm $\nabla_h L_{contrastive}$: `0.00e+00`
- Zero gradient confirmed: **True**

## 3. Seed-by-Seed Empirical Comparison

|   seed |   dom_loss |   conad_loss | dom_weight_checksum   | conad_weight_checksum   | weights_identical   |   score_spearman_corr |   score_pearson_corr |   score_conad_equals_dom_times_eta_max_diff | conad_ranking_identical_to_dominant   |
|-------:|-----------:|-------------:|:----------------------|:------------------------|:--------------------|----------------------:|---------------------:|--------------------------------------------:|:--------------------------------------|
|     42 |    3.25635 |      1.62818 | 22fbf2cbc71900ab      | 078f852074eb9851        | False               |                     1 |                    1 |                                 1.16825e-05 | True                                  |
|     43 |    3.30818 |      1.65409 | 3bde7c8ddce46161      | 9fd9bce5fcc8f3de        | False               |                     1 |                    1 |                                 3.93391e-06 | True                                  |
|     44 |    3.1795  |      1.58975 | 55252e81d693a2d7      | d3679aec3821ec84        | False               |                     1 |                    1 |                                 6.91414e-06 | True                                  |
|     45 |    3.02189 |      1.51095 | d4f58cf5c7b9e33f      | 42971f5ba6c73134        | False               |                     1 |                    1 |                                 4.88758e-06 | True                                  |
|     46 |    3.02651 |      1.51326 | 5d0a53b02aee94d1      | 90995a1e2ba0473a        | False               |                     1 |                    1 |                                 2.95639e-05 | True                                  |

## 4. Work Order A02 D-03 Checklist Status
- [x] **MarginRankingLoss argument order**: Confirmed bug `(h, h, h_aug)` -> `h - h = 0`.
- [x] **Contrastive loss gradient path**: Confirmed gradient is analytically zero.
- [x] **Score aggregation**: Confirmed CONAD score is `eta * reconstruction_score`, yielding identical rankings.
