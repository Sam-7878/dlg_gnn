# Gate A22-G1: Exact Loss and Optimizer Semantics Source Trace Audit

- **Date:** 2026-10-30
- **Auditor:** DLG-GNN Benchmark Integrity Taskforce
- **Status:** PASS (Case A Confirmed — Exact Match to PyGOD double_recon_loss)
- **Artifact:** `projects/benchmark/publication/a22/a22_exact_loss_source_trace.md`

---

## 1. Executive Summary

Claude Reviewer M7 identified a potential discrepancy between the loss equation reported in the manuscript (Equations 137 and 152) and standard PyGOD implementation:
- **Manuscript presentation:** $\frac{1}{N^2} \|A - ZZ^\top\|_F^2 = \frac{1}{N^2} \sum_{i,j} (A_{ij} - z_i^\top z_j)^2$ (Squared Frobenius norm / mean squared error over all pairs).
- **PyGOD `double_recon_loss`:** $\ell_i = w \|X_i - \hat{X}_i\|_2 + (1-w) \|A_i - \hat{A}_i\|_2$, with training loss $\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \ell_i$ (Euclidean row-norm residual per node, with square-root per node before taking the sample mean).

A comprehensive mathematical and code-level audit was conducted across the PyGOD 1.1.0 codebase, the production model implementation (`SharedDLGBase`, `SharedDLGFull`, `SharedDOMINANT`), and the closed-form Gram reduction routines (`exact_reconstruction.py`).

### Key Findings
1. **PyGOD Reference Implementation:**
   - In PyGOD 1.1.0 (`pygod/nn/functional.py`), `double_recon_loss` computes:
     ```python
     diff_attribute = torch.pow(x - x_, 2)
     diff_structure = torch.pow(s - s_, 2)
     score_attribute = torch.sqrt(torch.sum(diff_attribute, 1))
     score_structure = torch.sqrt(torch.sum(diff_structure, 1))
     score = weight * score_attribute + (1.0 - weight) * score_structure
     loss = torch.mean(score)
     ```
   - Both structural and attribute residuals take the per-node Euclidean norm (square-root of row sum) prior to mean reduction.

2. **Benchmark Production Implementation:**
   - In `dlg_gnn/model/exact_reconstruction.py`, `exact_double_reconstruction_score` computes the exact structural row residual closed-form without materializing $N \times N$:
     $$Q_i = \|A_{i,:} - z_i Z^\top\|_2^2 = d_i - 2 z_i (Z^\top A_{:, i}) + z_i (Z^\top Z) z_i^\top$$
     It then applies `_stable_sqrt(Q_i)`:
     ```python
     structure_score = _stable_sqrt(exact_dot_product_row_squared_error(...))
     ```
   - In `dlg_gnn/model/shared_dominant_full.py`, `_backward_exact` takes this exact score and computes:
     ```python
     total_loss = float(self.weight) * attr_loss + float(1.0 - self.weight) * struct_loss
     ```
     where both `attr_loss` and `struct_loss` are `.mean()` of the per-node Euclidean residuals.

3. **Conclusion:**
   - **The benchmark code has ALWAYS executed the exact PyGOD `double_recon_loss` objective identically down to machine precision.**
   - The manuscript Equations (137) and (152) presented the structural residual energy / algebraic intermediate prior to Euclidean row-norm reduction, leading the external reviewer to question whether squared Frobenius or Euclidean row-norm was optimized.
   - Rigorous unit tests across seeds, asymmetric graphs, isolated nodes, and self-loops confirm loss relative error $\le 3.12 \times 10^{-16}$ (float64) and parameter gradient cosine $= 1.000000000000$.
   - **Decision: Case A is confirmed.** No retraining is required or justified. The manuscript equations and accompanying text in Section 3.3 will be clarified to present the per-node Euclidean row norm and mean reduction explicitly.

---

## 2. Source Code SHA-256 Hashes and Locations

| Component | Path | SHA-256 |
|---|---|---|
| PyGOD Functional Reference | `pygod/nn/functional.py` | `5294db5d415953e2aed7b52c545fd81a2b98868652649f24614e0fc7950fc1b9` |
| PyGOD DOMINANT Base | `pygod/models/dominant.py` | Installed PyGOD 1.1.0 |
| Production Exact Reconstruction | `dlg_gnn/model/exact_reconstruction.py` | `283ff6e96bbd9f2da05b765ec177b966ddaa1ce3f3ef8a7985aa0a911765ad8d` |
| Production Stable Reconstruction | `dlg_gnn/model/stable_reconstruction.py` | `d17743ea44da0df1c50e417122a2754942ec4cf41103c80bb01a4e15ba3c6981` |
| Production Shared DOMINANT Base | `dlg_gnn/model/shared_dominant_base.py` | `0b776a3ff88d2f592205562723c3b0dfb2fef40d42bc19472e35327ec23fbc04` |
| Production Shared DLG Base | `dlg_gnn/model/shared_dlg_base.py` | `07920dc7c449db77b94ce50f385c4bf49a0d8c0b561df0b87c7aa32c029b35bc` |
| Production Shared Full Graph Mixin | `dlg_gnn/model/shared_dominant_full.py` | `e985b8eb8ba1ae89a0785196dc4126d4001d234a62243d6dbfbf1386fc7430c7` |

---

## 3. Mathematical Equivalence Proof

Let $A \in \{0, 1\}^{N \times N}$ be the adjacency matrix and $Z \in \mathbb{R}^{N \times d}$ be the node embeddings.
The reconstructed adjacency is $\hat{A} = Z Z^\top$.

The row residual for node $i$ in dense PyGOD is:
$$R_{i,:} = A_{i,:} - \hat{A}_{i,:} = A_{i,:} - z_i Z^\top$$
The squared Euclidean norm of this row vector is:
$$\|R_{i,:}\|_2^2 = \sum_{j=1}^N (A_{ij} - z_i^\top z_j)^2 = \sum_{j=1}^N A_{ij}^2 - 2 \sum_{j=1}^N A_{ij} z_i^\top z_j + \sum_{j=1}^N (z_i^\top z_j)^2$$
Since $A_{ij} \in \{0, 1\}$, $A_{ij}^2 = A_{ij}$, and $\sum_j A_{ij} = d_i$ (the node degree).
Furthermore, $\sum_j A_{ij} z_j = (Z^\top A_{:, i}) = (A Z)_i$.
And $\sum_j (z_i^\top z_j)^2 = z_i \left( \sum_j z_j z_j^\top \right) z_i^\top = z_i (Z^\top Z) z_i^\top$.

Therefore:
$$Q_i = d_i - 2 z_i (A Z)_i + z_i (Z^\top Z) z_i^\top$$
This identity holds identically without approximation.

The Euclidean row norm per node is:
$$\mathcal{A}_{\mathrm{struct}, i} = \sqrt{\max(Q_i, 0)}$$
And the attribute reconstruction error per node is:
$$\mathcal{A}_{\mathrm{attr}, i} = \|X_{i,:} - \hat{X}_{i,:}\|_2 = \sqrt{\sum_{k=1}^F (X_{ik} - \hat{X}_{ik})^2}$$

The training loss is:
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left[ w \, \mathcal{A}_{\mathrm{attr}, i} + (1-w) \, \mathcal{A}_{\mathrm{struct}, i} \right]$$

This is mathematically identical to PyGOD's `double_recon_loss(x, x_, s, s_, weight)`.

---

## 4. Controlled Unit Test Results

The suite in `projects/benchmark/scripts/a22_g1_dense_exact_training_equivalence.py` executed 18 unit tests across:
- Float64 and Float32 precision
- Models: DOMINANT, DLG-Base, DLG-Aug
- Graph configurations: Symmetric, Asymmetric, Isolated nodes, Duplicate edges, Self-loops
- Optimizer trajectories: Single forward, Single backward, 10-step Adam optimizer trajectory

### Summary Metrics:
- **Float64 Forward Loss Relative Error:** $\le 3.12 \times 10^{-16}$ (Predeclared tolerance: $1.0 \times 10^{-12}$)
- **Float64 Backward Gradient Cosine:** $= 1.000000000000$ (Predeclared tolerance: $> 0.999999$)
- **Float64 Trajectory Parameter Step Diff:** $\le 1.11 \times 10^{-16}$
- **Float32 Forward Loss Relative Error:** $0.00$
- **Float32 Backward Gradient Cosine:** $= 1.000000000000$
- **Float32 Gradient Relative L2 Error:** $\le 1.87 \times 10^{-7}$
- **Float32 Trajectory Loss Diff:** $\le 2.05 \times 10^{-7}$
- **Negative Controls (Objective Mismatch Detection):**
  - Squared Per-Node Mean vs PyGOD: Gradient Relative L2 Error $= 0.7818$ (Tolerance violation $> 100\times$, reliably caught)
  - Global Squared Frobenius vs PyGOD: Gradient Relative L2 Error $= 5.4140$ (Tolerance violation $> 1000\times$, reliably caught)

Full structured JSON results are deposited in:
`projects/benchmark/publication/a22/a22_dense_exact_training_equivalence.json`.

