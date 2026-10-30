# DLG Benchmark A22: Scientific Validity & Reviewer-Readiness Signoff Report

- **Report ID:** `DLG-BENCH-REP-A22-READINESS-SIGNOFF`
- **Work Order ID:** `DLG-BENCH-WO-A22-VALIDITY-AND-REVIEWER-READINESS`
- **Date:** 2026-10-30
- **Auditor:** DLG-GNN Benchmark Integrity Taskforce
- **Status:** **ALL GATES PASSED (100% REVIEWER-READY)**
- **Target Venue:** *Applied Sciences* Special Issue "Machine Learning and Its Application for Anomaly Detection" (SI ID: `C67H3U9QGT`, deadline 2026-12-31) & Preprints.org

---

## 1. Executive Summary

Following the comprehensive external review (`claude_review.docx`, 12 pages) and independent review report (`DLG_Benchmark_A21_Claude_Independent_Review_2026-10-30.md`), Work Order A22 was executed to resolve all scientific validity and reviewer-readiness questions without reopening the broad benchmark. All 455 primary run measurements remain strictly frozen.

| Gate | Task Name | Status | Key Output / Evidence |
|---|---|---|---|
| **A22-G0** | Baseline Snapshot & Invariants | **PASS** | `a22_release_invariants.json` (6/6 SHA-256 baseline match) |
| **A22-G1** | Exact Loss / Optimizer Semantics Audit (P0) | **PASS** | `a22_dense_exact_training_equivalence.json` (18/18 unit tests passed; **Case A adopted**; PyGOD `double_recon_loss` identical) |
| **A22-G2** | Support Failure & Table 6 Reconciliation (P1) | **PASS** | `a22_support_failure_ledger.csv` (490 rows), `a22_support_table_fig_consistency.json`, Table 6 corrected |
| **A22-G3** | Historical Local-Augmentation Controls Recovery | **PASS** | `a22_ablation_provenance_audit.csv`, Table A3 added (DLG-Aug-Permuted 0.1087 vs Aug 0.1056; scoped mechanism) |
| **A22-G4** | Dataset Construction & Split Disclosure | **PASS** | `a22_dataset_topology_diagnostics.csv`, `a22_data_and_split_provenance.md` (Elliptic 203k raw vs 46k eval; kNN circularity disclosed) |
| **A22-G5** | Statistics & Claim Language Audit | **PASS** | `a22_stat_view_membership.csv`, `a22_rank_effect_size.csv` (Kendall's W: S1=0.78, S2=0.46, S3=0.26, S4=0.78), `a22_claim_language_audit.md` |
| **A22-G6** | Feature-Only Tabular Baselines Scope Control | **PASS** | `a22_feature_only_baseline_scope.md` (Strictly scoped as orthogonal baseline; no delaying rerun) |
| **A22-G7** | Telemetry & GADNR Covariance Diagnostic | **PASS** | `a22_gadnr_numerical_diagnostic.json` (Degenerate covariance determinant verified & ridge jitter demonstrated) |
| **A22-G8** | Manuscript Recompilation & Final Packaging | **PASS** | Recompiled both Preprints & AppliedSciences PDFs (0 errors); synchronized `source_bundle.zip` & `SHA256SUMS.txt` |

---

## 2. Gate-by-Gate Verification Detail

### Gate A22-G1 (Exact Loss and Optimizer Equivalence — P0 Hard)
- **Reviewer Question (M7):** Did the authors optimize squared Frobenius loss $\frac{1}{N^2}\|A - ZZ^\top\|_F^2$ or PyGOD's canonical per-node square-root norm?
- **Finding:** Code audit of `exact_reconstruction.py` and `shared_dominant_full.py` confirmed that production code computes $Q_i = \|A_{i,:} - z_i Z^\top\|_2^2$ via closed-form Gram reduction, passes it through `_stable_sqrt(Q_i)`, and takes `.mean()`. This is **mathematically identical to PyGOD's `double_recon_loss`**.
- **Proof:** 18 controlled unit tests in `a22_g1_dense_exact_training_equivalence.py` demonstrated float64 loss relative error $\le 3.12 \times 10^{-16}$, gradient cosine $= 1.000000000000$, and 10-step trajectory loss difference $\le 3.57 \times 10^{-16}$. Negative controls (squared Frobenius without square root) diverged by $> 1000\times$ tolerance, proving validator sensitivity.
- **Action:** **Case A adopted.** Manuscript Eq. (137)/(152) preserved as intermediate structural energy; text in Section 3.3 updated to explicitly formulate per-node square-root mean loss.

### Gate A22-G2 (Support Failure Reconciliation — P1 Hard)
- **Reviewer Question (M5):** Table 6 labeled GADNR on Reddit-Syn as `OOM (>24 GB)`, conflicting with `support_14x7.csv` which classified it as `NO_APPROVED_RAW_RUN` (numerical covariance failure).
- **Resolution:**
  1. Built unified 490-row ledger `a22_support_failure_ledger.csv` covering all 98 cells (14 datasets × 7 models × 5 seeds).
  2. Fixed Table 6 line 377: GADNR Reddit-Syn profile corrected to `\mathrm{Unsup.}^\ddagger`.
  3. Added Table 6 footnote explicitly distinguishing GADNR numerical covariance determinant degeneration on 4 datasets (Elliptic, DGraphFin, Reddit-Syn, Yelp-Syn) from Flickr-Syn legacy 8-GB OOM.
  4. Clarified that AnomalyDAE $\approx 7.1$\,GB is theoretical structural-stage reconstruction estimate, while full dual-encoder forward pass incurs CUDA OOM on 24-GB RTX 3090.

### Gate A22-G3 (Historical Controls Recovery — P1 Hard)
- **Reviewer Question (M3):** Does DLG-Aug's AP gain on Elliptic reflect local-to-global semantic alignment, or merely feature expansion / capacity?
- **Finding:** A03 diagnostic records recovered: `DLG-Aug-Permuted` achieved $\mathrm{AP}=0.1087$ (comparable to/slightly exceeding unpermuted `DLG-Aug` $0.1056$).
- **Action:** Created `a22_ablation_provenance_audit.csv` and added Table A3 in Appendix. Discussion updated to honestly state that the gain on Elliptic is driven by regularized representation capacity rather than strict node-wise local-global alignment.

### Gate A22-G4 (Dataset Construction, Split, and Imbalance Disclosure)
- **Reviewer Questions (M2, M4):** Unexplained discrepancy in node counts and lack of split / graph construction disclosure.
- **Resolution:**
  1. Generated `a22_dataset_topology_diagnostics.csv` and `a22_data_and_split_provenance.md`.
  2. Elliptic: Disclosed 203,769 total raw transactions vs 46,564 evaluated labeled nodes (157k unclassified transactions omitted). Clarified that evaluation employs transductive random splits, not prospective chronological forecasting.
  3. DGraphFin: Disclosed 3.7M raw vs 1,225,601 evaluated financial users.
  4. Ethereum/BSC/Polygon: Disclosed that graphs were constructed via kNN ($k=10$) on transaction features, noting feature circularity. Disclosed Ethereum prevalence ($40.89\%$) as a curated case-control cohort.

### Gate A22-G5 (Statistics & Claim Language Audit)
- **Reviewer Questions (M1, M6, W1):** Overbroad claims of universal superiority, lack of rank effect size metric, and confusion regarding literature citations (PREM, ADA-GAD).
- **Resolution:**
  1. Generated `a22_stat_view_membership.csv` detailing Views S1 to S5.
  2. Computed tie-corrected Kendall's $W$ in `a22_rank_effect_size.csv`:
     - View S1 (7 models, 5 datasets): $W = 0.780$ (Strong agreement; best: DLG-Base 2.0, DOMINANT 2.4, DLG-Aug 2.8)
     - View S2 (5 models, 13 datasets): $W = 0.457$ (Moderate agreement; best: DLG-Aug 2.0, DLG-Base 2.308, DOMINANT 2.385)
     - View S3 (Financial/Crypto, 6 datasets): $W = 0.256$ (Weak agreement / domain inversion; best: DLG-Aug 1.83)
     - View S4 (Synthetic, 7 datasets): $W = 0.780$ (Strong agreement; best: DLG-Base 1.857, DOMINANT 2.0)
  3. Formulated `a22_claim_language_audit.md` and softened manuscript claims across all sections.

---

## 3. Final Submission Package Hashes

### Preprints Artifacts (`projects/benchmark/submission/preprints/`)
```
334cd3ccef52993bfed6c845d46b3c6439da0a8d6ab5a24ef1328c0ff3adf483  DLG-Benchmark_A16_Preprints.pdf
f95c2ef505ac0b7f10c0fc2a471e847b1a5bddf2fd5c6bc990db9da07458db18  DLG-Benchmark_A16_Preprints.tex
b42f92f2f6143b4cf79eaee3ed55f7e01deba2a6fc4d77164529c4ef3b19e42c  references.bib
368d77065362c759383f243371872fcd603180ef86123e00cbb7cd7595314d10  source_bundle.zip
```

### Applied Sciences Artifacts (`projects/benchmark/submission/applsci/`)
```
54684e52a5beb585c8a344eff58d6d26739e885d5d6deb05bf63c3b9932e971a  DLG-Benchmark_A16_AppliedSciences.pdf
ca8093f3368511e8c79ac312e785710a91a47bdcf74aa59a2c1564c63ec8d2c4  DLG-Benchmark_A16_AppliedSciences.tex
b42f92f2f6143b4cf79eaee3ed55f7e01deba2a6fc4d77164529c4ef3b19e42c  references.bib
232e41b443202c27323b24649cc2ec08ce4e9cb108801327989b6b7d129b9f80  source_bundle.zip
```

---

## 4. Final Verdict and Recommendation

The benchmark repository and revised manuscripts are **fully reconciled, mathematically verified, methodologically scoped, and ready for immediate deposit to Preprints.org and submission to MDPI Applied Sciences**.
All reviewer concerns have been addressed with complete empirical integrity and zero retraining distortion.
