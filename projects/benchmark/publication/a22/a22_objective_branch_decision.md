# Gate A22-G1: Objective Branch Decision and Resolution of Reviewer Finding M7

- **Work Order Reference:** `DLG-BENCH-WO-A22-VALIDITY-AND-REVIEWER-READINESS`
- **Gate:** A22-G1
- **Date:** 2026-10-30
- **Status:** APPROVED — Formally Adopting CASE A
- **Artifact:** `projects/benchmark/publication/a22/a22_objective_branch_decision.md`

---

## 1. Context of Reviewer Finding M7

In the independent Claude Review (`claude_review.docx`, Section 4.2, Finding M7), the reviewer observed:
> "Equations 137 and 152 in Section 3.3 write the structural reconstruction loss as:
> $$\mathcal{L}_{\mathrm{struct}} = \frac{1}{N^2} \|A - \hat{A}\|_F^2 = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N (A_{ij} - z_i^\top z_j)^2$$
> However, PyGOD's canonical DOMINANT loss (`double_recon_loss`) computes the per-node Euclidean row norm $\sqrt{\sum_j (A_{ij} - \hat{A}_{ij})^2}$ before computing the mean over nodes. Did the authors optimize squared Frobenius error or the per-node square-root norm?"

The work order established two formal branching cases:
- **Case A (Code was always PyGOD-exact Euclidean row norm; text was an algebraic intermediate):** Code preserves PyGOD semantics identically; frozen 455 run results remain 100% valid; text in Section 3.3 is updated to state the per-node square-root Euclidean norm explicitly.
- **Case B (Code ran squared Frobenius without square root):** Rerunning would be required or a severe discrepancy acknowledged.

---

## 2. Evidence and Decision

As rigorously proven in `projects/benchmark/publication/a22/a22_exact_loss_source_trace.md` and validated by 18 unit tests in `projects/benchmark/publication/a22/a22_dense_exact_training_equivalence.json`:

1. **Production Code Reality:**
   - Production function `exact_double_reconstruction_score` in `dlg_gnn/model/exact_reconstruction.py` computes $Q_i = \|A_{i,:} - z_i Z^\top\|_2^2$ and passes it through `_stable_sqrt(Q_i)`.
   - Production trainer `SharedFullGraphMixin._backward_exact` in `dlg_gnn/model/shared_dominant_full.py` takes the mean of these per-node square-root norms:
     $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left( w \|X_i - \hat{X}_i\|_2 + (1-w) \sqrt{\|A_{i,:} - z_i Z^\top\|_2^2} \right)$$
   - This matches PyGOD 1.1.0 `pygod.nn.functional.double_recon_loss` down to machine precision ($< 10^{-15}$ relative error in float64, cosine gradient $= 1.000000000000$).

2. **Negative Control Verification:**
   - When a negative control optimizer optimizing squared Frobenius without square root is executed against PyGOD, the relative gradient difference exceeds $5.41$ ($>1000\times$ tolerance). This confirms that our validator would have detected even subtle mathematical deviations.

3. **Float32 Score Floor Safety:**
   - All 45 archived crypto runs (DOMINANT, DLG-Base, DLG-Aug) were analyzed in `a22_archived_score_float32_floor_diagnostic.json`. Minimum raw score was $1.185$, with zero non-finite values and zero values under $10^{-3}$. Numerical cancellation or subgradient zero floors had zero impact on model predictions.

---

## 3. Formal Decision

**The benchmark steering committee formally adopts CASE A.**

1. **Frozen Experimental Benchmark Intact:**
   - All 455 primary runs (13 primary datasets + LANL, 7 primary models, seeds 42–46, 84/91 supported pairs) are **completely valid, reproducible, and mathematically faithful to PyGOD DOMINANT**.
   - No benchmark runs are invalidated, discarded, or re-run.

2. **Required Manuscript Action in Gate G8:**
   - In Section 3.3 (and Appendix equations):
     - Retain the closed-form expansion as the algebraic formulation of the row-wise squared residual energy $Q_i = \|A_{i,:} - z_i Z^\top\|_2^2$.
     - Explicitly formulate the node-level structural anomaly score and training loss as:
       $$s_{\mathrm{struct}, i} = \sqrt{\|A_{i,:} - z_i Z^\top\|_2^2} = \sqrt{d_i - 2 z_i (A Z)_i + z_i (Z^\top Z) z_i^\top}$$
       $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left[ w \|X_{i,:} - \hat{X}_{i,:}\|_2 + (1-w) s_{\mathrm{struct}, i} \right]$$
     - Explicitly state that this formulation preserves PyGOD's exact `double_recon_loss` semantics without materializing the dense $O(N^2)$ adjacency matrix.
