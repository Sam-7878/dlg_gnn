# A19 Authoritative Source Map and Evidence Consistency Gate Report

**Document ID:** `DLG-BENCH-MAP-A19-SOURCES`  
**Date:** 2026-10-28 / 2026-10-29  
**Status:** PASS (Zero Cross-File Disagreements; Machine-Readable Origin Verified)  
**Release Tag:** `benchmark-a19-submission-2026-10-28`  

---

## 1. Frozen Benchmark Configuration & Scope

The benchmark evaluates an exact matrix of 14 datasets (13 primary graphs plus 1 external enterprise cyber network) across 7 anomaly detector architectures and 5 evaluation seeds:

- **Datasets (14 total):**
  - **Native Financial / Blockchain (5):** `Elliptic`, `DGraphFin`, `Ethereum`, `BSC`, `Polygon`
  - **Injected Synthetic / Real-Graph (8):** `BitcoinOTC`, `Yelp-Syn`, `Amazon-Syn`, `Reddit-Syn`, `Flickr-Syn`, `Cora-Syn`, `CiteSeer-Syn`, `PubMed-Syn`
  - **External Enterprise Cybersecurity (1):** `LANL-RedTeam`
- **Architectures (7):** `DOMINANT`, `AnomalyDAE`, `CoLA`, `GADNR`, `OCGNN`, `DLG-Base`, `DLG-Aug`
- **Evaluation Seeds (5):** `42`, `43`, `44`, `45`, `46`
- **Total Matrix Cells:** 14 × 7 = 98 dataset-model pairs
- **Supported Pairs:** 91 pairs (84 primary + 7 LANL)
- **Approved Runs:** 91 × 5 = 455 complete successful runs
- **Unsupported Pairs:** Exactly 7 pairs (all documented in `support_14x7.csv`)

### Verified 7 Unsupported Pairs (Per Frozen Contract `support_14x7.csv`)
1. `(DGraphFin, GADNR)`: `NO_APPROVED_RAW_RUN` (Numerical covariance determinant limitation)
2. `(Elliptic, GADNR)`: `NO_APPROVED_RAW_RUN` (Numerical covariance determinant limitation)
3. `(Flickr-Syn, GADNR)`: `OBSERVED_HISTORICAL_CUDA_OOM` (Historical CUDA OOM on 8GB legacy environment)
4. `(Reddit-Syn, AnomalyDAE)`: `OBSERVED_CUDA_OOM` (Preflight CUDA OOM in full encoder forward on 24GB RTX 3090)
5. `(Reddit-Syn, GADNR)`: `NO_APPROVED_RAW_RUN` (Numerical covariance determinant limitation)
6. `(Yelp-Syn, AnomalyDAE)`: `OBSERVED_CUDA_OOM` (Preflight CUDA OOM in full encoder forward on 24GB RTX 3090)
7. `(Yelp-Syn, GADNR)`: `NO_APPROVED_RAW_RUN` (Numerical covariance determinant limitation)

> [!NOTE]
> Historical erroneous references to `BHP` (a non-contract dataset) and `CONAD` (a diagnostic preliminary baseline excluded from the frozen contract) present in A18 draft reports are eradicated.

---

## 2. Truthful Evidence Tiers and Score Storage Hierarchy

A19 establishes an explicit, transparent, and non-retroactive evidence hierarchy:

| Evidence Tier | Datasets | Runs | Metric Source | Raw Prediction Scores (`scores.npz`) | Evaluation Verification Status |
|---|---|---|---|---|---|
| **Tier 1: Source-Level Raw Array Recomputable** | `Ethereum`, `BSC`, `Polygon` (3 datasets) | **105 runs** (3 × 7 × 5) | `metrics.json` in `local_storage/benchmark/a08_data_repair/runs/` | **Fully Archived & Verified** (`scores.npz` containing continuous prediction vector, node IDs, labels, and test masks) | **PASS:** 210/210 alert-budget evaluations ($P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$) verified with 100% exact top-$K$ node set match. |
| **Tier 2: Historical Preserved Metric Artifacts** | `Amazon-Syn`, `CiteSeer-Syn`, `Cora-Syn`, `PubMed-Syn`, `DGraphFin`, `Elliptic`, `Flickr-Syn`, `Reddit-Syn`, `Yelp-Syn`, `BitcoinOTC`, `LANL-RedTeam` (11 datasets) | **350 runs** (11 datasets, 70 supported pairs × 5 seeds) | Frozen `metrics.json` in `evaluation/benchmark/v2/` publication evidence | **NOT ARCHIVED** (Raw continuous scores were not preserved by source campaigns A04/A05/Round 5; metric JSONs preserved with SHA-256 checksums) | **PASS:** Frozen JSON hashes and numeric invariants verified. No unarchived historical raw scores are falsely claimed. |

---

## 3. Authoritative Cryptocurrency Evaluation Populations

Table 1, Table 3, and data availability descriptions strictly adhere to the empirical ground-truth test splits:

| Dataset | Total Nodes ($N$) | Total Edges ($E$) | Dim ($d$) | Graph Fraud Rate ($\pi_\mathrm{graph}$) | Test Split ($N_\mathrm{test}$) | Test Positives ($M_\mathrm{test}$ per seed [42--46]) | Mean Test Positives ($\bar{M}_\mathrm{test}$) | Mean Test Prevalence ($\pi_\mathrm{test}$) | K@1% | K@5% |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ethereum** | 14,385 | 115,065 | 8 | 0.4184 | **2,893** | [1143, 1246, 1153, 1184, 1189] | **1183.0** | **0.4089** (40.89%) | 29 | 145 |
| **BSC** | 7,481 | 59,832 | 8 | 0.1476 | **1,500** | [215, 237, 217, 206, 230] | **221.0** | **0.1473** (14.73%) | 15 | 75 |
| **Polygon** | 2,303 | 18,411 | 8 | 0.0261 | **471** | [16, 11, 7, 18, 9] | **12.2** | **0.0259** (2.59%) | 5 | 24 |

### Negative Control & Population Sanity Invariants
1. **Ethereum:** Test population is $N_\mathrm{test} = 2893$, **not** $1428$; test fraud count is $\bar{M}_\mathrm{test} = 1183.0$, **not** $584$. (The numbers 1428 / 584 belonged to an upstream validation slice and were erroneously cited in draft text).
2. **BSC:** Test population is $N_\mathrm{test} = 1500$, test fraud count is $\bar{M}_\mathrm{test} = 221.0$.
3. **Polygon:** Test population is $N_\mathrm{test} = 471$, **not** $4641$; test fraud count is $\bar{M}_\mathrm{test} = 12.2$, **not** $120$.
4. **Obsolete Prevalences Rejected:** Old candidate values (Ethereum 0.0195, BSC 0.0210, Polygon 0.0052) are definitively rejected as mismatched graph-label slices.

---

## 4. Machine-Readable Cross-Check Matrix

| Target Verification Item | Authoritative Machine Input | Manuscript Table | Status | Discrepancy Count |
|---|---|---|---|---|
| Total Runs | `canonical_selected_run_registry.csv` | Section 4 / Data Availability | 455 runs | 0 |
| Model Support Matrix | `support_14x7.csv` | Table 2 (PR-AUC) / Table A1 (ROC) | 91 supported, 7 unsupported | 0 |
| Table 3 Alert Budgets | `a19_alert_budget_raw_recheck.csv` | Table 3 (Ethereum, BSC, Polygon) | Exact match (210/210) | 0 |
| Table A2 F1 Semantics | `a19_f1_semantics_by_campaign.csv` | Table A2 (Validation-thresholded F1) | 455/455 verified thresholds | 0 |
| Crypto Test Populations | `a19_crypto_scores_manifest.csv` | Table 1 / Table Notes | Exact match (N=2893, 1500, 471) | 0 |

---

## 5. Artifact Provenance
- Generated by: `projects/benchmark/scripts/a19_gate_g3_source_map.py`
- Machine-readable consistency check: `projects/benchmark/publication/a19/a19_source_map_consistency.json`
- Historical draft archived: `docs/work_reports/benchmark/243_A18/a18_authoritative_source_map.md` (flagged obsolete)
