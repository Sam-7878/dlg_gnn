# P2 Audit Report 02: P1 Audit Report Reconciliation

## 1. Executive Summary
- **Objective**: Reconcile all metric tables and textual descriptions in P1 publication audit reports against the frozen source-of-truth benchmark artifacts, eliminating human data-entry errors.
- **Status**: **100% PASSED** (All reports programmatically regenerated from frozen CSV artifacts).

---

## 2. Reconciliation Matrix

| Metric / Description | Previous Value (Error) | Frozen Artifact Value (Reconciled) | Source Artifact |
|---|---|---|---|
| **Elliptic DLG-Aug PR-AUC** | 0.941 | **0.1037 ± 0.0048** (DLG-Base: 0.0687, delta +0.0350) | `seed_aggregated_performance.csv` |
| **DGraphFin DLG-Aug PR-AUC** | 0.781 | **0.0134 ± 0.0019** (DLG-Base: 0.0101, delta +0.0034) | `seed_aggregated_performance.csv` |
| **Reddit-Syn DLG-Aug PR-AUC** | 0.803 | **0.3388 ± 0.0102** (DLG-Base: 0.4044, delta -0.0656) | `seed_aggregated_performance.csv` |
| **LANL DLG-Base ROC-AUC** | 0.963 | **0.7923 ± 0.0452** | `table_d2_lanl_external_validation.csv` |
| **LANL DLG-Aug ROC-AUC** | 0.942 | **0.7188 ± 0.1010** | `table_d2_lanl_external_validation.csv` |
| **LANL GADNR ROC-AUC** | N/A | **0.8261 ± 0.0218** | `table_d2_lanl_external_validation.csv` |
| **Scope Description** | Deprecated baseline scope | **8 detectors on 10 primary graphs + LANL** | Master LaTeX & Canonical Manifest |

---

## 3. Automated Generation Verification
`scripts/manuscript/generate_p1_audit_reports.py` was refactored to read directly from frozen CSV files, completely eliminating hardcoded numerical assertions.
