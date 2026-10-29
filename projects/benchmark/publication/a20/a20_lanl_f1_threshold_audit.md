# DLG-GNN Benchmark — A20 LANL F1 Threshold Provenance Audit

**Document ID:** `DLG-BENCH-A20-LANL-F1-AUDIT-20261029`  
**Date:** 2026-10-29  
**Target:** Threshold origin and F1 evaluation semantics for LANL-RedTeam external validation runs.

---

## 1. Audit Motivation & Context

In A19, the narrative report (`a19_f1_provenance_gate.md`) noted that LANL-RedTeam runs might reflect domain-calibrated or default operating points rather than validation-split tuning. However, `a19_f1_semantics_by_campaign.csv` labeled all 455 benchmark rows (including the 35 LANL runs) as `F1_on_test_at_validation_threshold`. 
Independent peer review cautioned: *"A numerical threshold being present in a CSV is not proof of its selection source."*

---

## 2. Direct Inspection of LANL Run Evidence

We inspected the underlying JSON run records (`evaluation/benchmark/v2/paper_ready_a04/publication_evidence_a04/metric_json/*.json`):
- **Origin:** `defense_real_rebuild` (LANL-RedTeam cyber extension campaign).
- **Recorded Keys:** `threshold`, `f1`, `precision`, `recall`, `roc_auc`, `pr_auc`, `n_test_samples` (3,338), `n_test_positives` (60).
- **Missing Keys:** Unlike the 13 primary datasets (which explicitly record `validation_threshold` / `operating_point.threshold_source = "validation"`), LANL JSON records do not store a separate `validation_f1` or `threshold_source` provenance field.
- **Observed Thresholds:** Numerical values vary across models and seeds (e.g., AnomalyDAE ~32.8–47.2, CoLA ~ -2.2–0.6, GADNR ~0.08–0.24).

---

## 3. Methodological Assessment & Editorial Action

1. **Distinction between Primary Datasets and External LANL:**
   - **13 Primary Datasets (420 runs):** Proven at code level to use `validation_threshold` (threshold selected solely on validation mask without test label leakage).
   - **LANL-RedTeam (35 runs):** Evaluated under pipeline-assigned operational decision thresholds. While numerical thresholds are preserved and immutable, there is no explicit JSON record proving an independent validation-tuning split.
2. **Manuscript Caption Revision (Appendix Table A2):**
   - Qualify the caption of Table A2:
     *"Thresholds were selected on validation splits for all 13 primary benchmark datasets (420 runs). For the external LANL-RedTeam dataset (35 runs), reported F1 reflects the pipeline decision threshold recorded in frozen run metadata; validation-tuning metadata is absent from the legacy record."*
3. **Classification Integrity:**
   - Update A20 documentation and semantics table to explicitly denote LANL as `F1_on_test_at_pipeline_threshold` rather than conflating it with primary validation-tuned runs.

