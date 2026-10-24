# A12 Score Direction Audit & Hub Bias Hypothesis Evaluation (P0-03)

**Date:** 2026-10-24  
**Audit ID:** A12-DIAG-P0-03  
**Status:** **A12-G3 PASSED — Score Direction Alignment Verified; Causal Claim Downgraded to Hypothesis**

---

## 1. Executive Summary

In A11, several sections claimed that below-random ROC-AUC values (<0.50) observed on Elliptic and DGraphFin were 'mathematically and empirically proven' to arise from normal hub node reconstruction residuals. An independent evidence review determined that because per-node degree and score arrays are not archived in public run records, this causal mechanism cannot be asserted as an established empirical proof.

In accordance with A12 Work Order P0-03, this audit:
1. **Verifies Score and Label Alignment:** Confirms $y=1$ positive anomaly semantics, higher-score anomaly orientation in PyGOD, and strict adherence to the non-negotiable rule forbidding post-hoc test-driven sign inversion.
2. **Refines Scientific Wording:** Downgrades 'proved / confirmed hub reconstruction bias' to 'empirically observed below-random performance under frozen evaluation; topological hub residual explanation remains an informed unverified hypothesis'.
3. **Documents Evidence Boundaries:** Explicitly distinguishes verified summary metric tables from unarchived per-node score arrays.

## 2. Empirical Metric Summary (Elliptic & DGraphFin)

| Dataset | Model | PR-AUC (Mean ± SD) | ROC-AUC (Mean ± SD) | Below Random ($<0.50$)? | Scientific Finding |
|---|---|---:|---:|:---:|---|
| Elliptic | DOMINANT | 0.0682 ± 0.0036 | 0.3432 ± 0.0367 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| Elliptic | AnomalyDAE | 0.0538 ± 0.0019 | 0.1296 ± 0.0246 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| Elliptic | CoLA | 0.0947 ± 0.0021 | 0.4878 ± 0.0088 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| Elliptic | GADNR | N/A | N/A | - | Unsupported (OOM) |
| Elliptic | OCGNN | 0.0667 ± 0.0046 | 0.3140 ± 0.0466 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| Elliptic | DLG-Base | 0.0687 ± 0.0022 | 0.3500 ± 0.0221 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| Elliptic | DLG-Aug | 0.1037 ± 0.0043 | 0.5705 ± 0.0228 | NO | ABOVE_RANDOM_OBSERVED |
| DGraphFin | DOMINANT | 0.0102 ± 0.0002 | 0.4088 ± 0.0049 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| DGraphFin | AnomalyDAE | 0.0119 ± 0.0031 | 0.4691 ± 0.0971 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| DGraphFin | CoLA | 0.0127 ± 0.0003 | 0.4994 ± 0.0088 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| DGraphFin | GADNR | N/A | N/A | - | Unsupported (OOM) |
| DGraphFin | OCGNN | 0.0126 ± 0.0015 | 0.4951 ± 0.0486 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| DGraphFin | DLG-Base | 0.0101 ± 0.0002 | 0.4092 ± 0.0086 | YES | EMPIRICAL_BELOW_RANDOM_OBSERVED |
| DGraphFin | DLG-Aug | 0.0134 ± 0.0017 | 0.5185 ± 0.0364 | NO | ABOVE_RANDOM_OBSERVED |

## 3. Mandatory Manuscript Text Corrections

- **Abstract:** Replace 'proving through exhaustive label, score, and mask verification that this stems from normal hub reconstruction residuals' with 'auditing the below-random ROC-AUC values (<0.50) observed on Elliptic across reconstruction baselines, confirming score and label alignment while leaving topological hub bias as an informed hypothesis'.
- **Section 1 (Introduction):** Replace 'demonstrating that below-0.50 ROC-AUC values reflect intrinsic structural hub biases' with 'verifying that below-0.50 ROC-AUC values reflect authentic empirical behavior under standard detector orientation rather than software defects, while discussing potential topological hypotheses'.
- **Section 5 (Discussion):** Explicitly state that direct per-node residual correlation against entity transaction degree requires fine-grained node telemetry not retained in historical summary archives.
