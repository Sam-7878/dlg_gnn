# P2 Audit Report 03: Cover Letter Scope and Count Audit

## 1. Executive Summary
- **Objective**: Ensure that the MDPI Applied Sciences submission cover letter (`publication/benchmark/mdpi/cover_letter.md`) accurately aligns with the Special Issue scope, reports exact dataset counts, and uses safe prior-art-sensitive wording.
- **Status**: **100% PASSED** (All criteria verified).

---

## 2. Remediated Cover Letter Items

1. **Dataset Counts**:
   - Corrected from "six PyGOD" to:
     > *"ten primary graphs, including three real-label financial/blockchain datasets (Elliptic, DGraphFin, and BitcoinOTC) and seven controlled synthetic-injection datasets (Yelp, Amazon, Flickr, Reddit, Cora, CiteSeer, and PubMed)..."*
   - Total: 3 real + 7 synthetic = 10 primary graphs, plus LANL-RedTeam external validation.

2. **Special Issue Scope Bullets Alignment**:
   - Matches official *Applied Sciences* Special Issue: *"Graph Neural Networks: Theory, Methods and Applications"*:
     - *Scalable and efficient GNN architectures*
     - *Self-supervised and unsupervised graph learning*
     - *Graph representation learning and embeddings*
     - *Financial modeling and other applied GNN settings*
   - Appropriately frames anomaly detection as the application setting of our study.

3. **Exact Sparse Formulation Claim**:
   - Changed `"we develop and mathematically prove an exact sparse formulation"` to:
     > *"We derive, implement, and numerically verify a mathematically equivalent sparse reformulation of the linear dot-product reconstruction objective..."*
   - Avoids aggressive priority claims ("prove for the first time", "novel").

4. **Two-State Preprint Staging**:
   - Marked as draft pending Preprints.org deposit.
   - Companion tool `scripts/publication/update_preprint_doi.py` ready for one-command DOI injection once deposit confirmation is received.
