# CONAD Model Fairness Correction Specification (Gate A03-3)
**Document ID:** DLG-BENCH-V2-CONAD-SPEC-A03  
**Status:** Adopted per Work Order A03 §5  

## 1. Dual Identity Policy
To maintain scientific transparency and historical reproducibility, two distinct CONAD identities are established:

1. **`CONAD-PyGOD-1.1-reference` (Diagnostic Only)**:
   - Evaluates the upstream PyGOD 1.1.0 codebase verbatim.
   - Contrastive loss: `MarginRankingLoss(h, h, h_aug)`.
   - Analytical gradient: $\nabla_h L \equiv 0.0$.
   - Outcome: Acts as an uncalibrated scalar multiple of DOMINANT (Spearman $\rho = 1.0000$).
   - Usage: Retained exclusively in the Appendix/Diagnostic Audit; not used to claim DLG-GNN superiority over a functioning contrastive baseline.

2. **`CONAD-corrected` (Functional Contrastive Baseline)**:
   - Restores the intended node-wise contrastive objective from Xu et al. (IJCAI 2022).
   - Formulated as:
     $$L_{con} = \frac{1}{N} \sum_{i=1}^N \left[ (1 - y_i) \|h_i - \tilde{h}_i\|_2^2 + y_i \max(0, \text{margin} - \|h_i - \tilde{h}_i\|_2)^2 \right]$$
     where $y_i = \text{label\_aug}_i \in \{0, 1\}$.
   - Verified active gradient: $\|\nabla_h L_{con}\| > 0.08$ on representative batches.
   - Evaluated across supported benchmark graphs over seeds 42–46.
