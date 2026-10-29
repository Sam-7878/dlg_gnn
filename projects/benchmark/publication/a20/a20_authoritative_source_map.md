# DLG-GNN Benchmark — Authoritative Source Map (A20 Revision)

**Release Scope:** A20 Editorial and Evidence Submission Gate  
**Date:** 2026-10-29  
**Parent:** A19 Final Evidence and Release Closure  
**Authoritative Tag:** `benchmark-a20-preprint-2026-10-29`  
**Core Policy:** FROZEN SCIENTIFIC EVIDENCE — EDITORIAL & INTEGRITY ALIGNMENT ONLY

---

## 1. Primary Benchmark Contract

- **Primary Datasets (13):**
  - **Injected Synthetic Benchmark Graphs (8):** `Amazon-Syn`, `CiteSeer-Syn`, `Cora-Syn`, `PubMed-Syn`, `Flickr-Syn`, `Reddit-Syn`, `Yelp-Syn`, `BitcoinOTC`
  - **Native Financial / Blockchain (5):** `Elliptic`, `DGraphFin`, `Ethereum`, `BSC`, `Polygon`
- **External Validation Graph (1):** `LANL-RedTeam` (cybersecurity authentication event graph)
- **Primary Functioning Detectors (7):** `DOMINANT`, `AnomalyDAE`, `CoLA`, `GADNR`, `OCGNN`, `DLG-Base`, `DLG-Aug`
- **Reference / Diagnostic-Only Detector (1):** `CONAD` (excluded from primary benchmark matrix)
- **Evaluation Seeds (5):** 42, 43, 44, 45, 46
- **Total Matrix Cells:** 14 datasets × 7 models = 98 cells (91 supported, 7 unsupported)
- **Total Qualified Runs:** 420 primary runs + 35 external LANL runs = **455 total successful runs**

---

## 2. Evidence Provenance Tiers

| Evidence Tier | Datasets | Runs | Source Artifacts | Verification Status |
|---|---|---|---|---|
| **Tier 1: Source-Level Raw Array Recomputable** | `Ethereum`, `BSC`, `Polygon` (3 datasets) | **105 runs** (3 × 7 × 5) | `scores.npz`, `metrics.json` in `local_storage/benchmark/a08_data_repair/runs/` | **Fully Archived & Verified**: 210/210 alert-budget evaluations ($P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$) verified with 100% exact top-$K$ node set match. |
| **Tier 2: Certified Run-Metric JSON Preserved** | `Amazon-Syn`, `CiteSeer-Syn`, `Cora-Syn`, `PubMed-Syn`, `DGraphFin`, `Elliptic`, `Flickr-Syn`, `Reddit-Syn`, `Yelp-Syn`, `BitcoinOTC`, `LANL-RedTeam` (11 datasets) | **350 runs** (10 × 7 × 5 + 1 × 7 × 5) | Certified publication JSONs in `publication_evidence` archives | **Metric Checksums Verified**: Summary metrics frozen; raw continuous score arrays were not archived by historical runs. |

---

## 3. Cryptocurrency Test Populations and Ground Truth (Authoritative Seed-Indexed Vectors)

| Dataset | Total Nodes $N$ | Total Edges $E$ | Node Features $D$ | Global Fraud Ratio | Test Population $N_\mathrm{test}$ | Test Positives $M_\mathrm{test}$ per seed (Seeds 42–46) | Test Mean Positives $\bar{M}_\mathrm{test}$ | Test Prevalence $\bar{\pi}_\mathrm{test}$ | Alert Budget $K@1\%$ | Alert Budget $K@5\%$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ethereum** | 14,385 | 115,065 | 8 | 0.4184 | **2,893** | [1143, 1246, 1153, 1184, 1189] | **1183.0** | **0.4089** (40.89%) | 29 | 145 |
| **BSC** | 7,481 | 59,832 | 8 | 0.1476 | **1,500** | **[215, 240, 232, 215, 203]** | **221.0** | **0.1473** (14.73%) | 15 | 75 |
| **Polygon** | 2,303 | 18,411 | 8 | 0.0261 | **471** | [16, 11, 7, 18, 9] | **12.2** | **0.0259** (2.59%) | 5 | 24 |

### A20 Resolution of the BSC Source-Map Contradiction
In the A19 narrative report, an obsolete draft vector `[215, 237, 217, 206, 230]` was inadvertently transcribed in markdown tables, even though both `a19_alert_budget_raw_recheck.csv` and `a19_source_map_consistency.json` contained the verified vector `[215, 240, 232, 215, 203]`. Both vectors possess the identical mean (221.0).
In A20, this discrepancy is resolved: all narrative documentation, machine-readable JSON manifests, and recheck ledgers strictly enforce the certified seed-indexed vector:
$$\text{BSC } M_\mathrm{test} \text{ (seeds 42--46)} = [215, 240, 232, 215, 203]$$

---

## 4. Evaluation Split Protocol Audit (Summary)

- **Cryptocurrency Graphs (Ethereum, BSC, Polygon):** Evaluated under **seed-dependent random 60/20/20 non-stratified node splits** generated via `np.random.RandomState(seed).permutation(n)`. Because splits are non-stratified, positive test counts vary across seeds.
- **Polygon Low-Budget Uncertainty:** With $N_\mathrm{test}=471$ and $K@1\%=5$, each single true positive changes $P@1\%$ by $\Delta P = 0.200$ (20.0 percentage points). Results at $K@1\%$ on Polygon are descriptive and exploratory, rather than operationally definitive.
- **Synthetic Graphs:** Transductive seeded 50/50 evaluation splits from labeled candidate pools.
- **DGraphFin:** Official transductive masks (70/15/15) with evaluation mask applied.
- **Elliptic:** Labeled transaction nodes with transductive evaluation splits.

---

## 5. Verification Invariant Summary

1. **Exact Alert-Budget Identities:** 210/210 entries satisfy $P@K = \mathrm{TP}@K / K$ and $R@K = \mathrm{TP}@K / M_\mathrm{test}$.
2. **Unsupported Cell Identity:** Exactly 7 matrix cells unsupported (AnomalyDAE on Reddit/Yelp; GADNR on Flickr/Reddit/Yelp/DGraphFin/Elliptic).
3. **No Retraining:** Zero model weights altered, zero metric tables overwritten.

