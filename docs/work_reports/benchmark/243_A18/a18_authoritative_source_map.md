> [!WARNING] OBSOLETE HISTORICAL RECORD
> This document contains superseded draft populations and references (e.g. BHP, N=1428, N=4641).
> For the verified authoritative source map, refer to `docs/work_reports/benchmark/244_A19/a19_authoritative_source_map.md`.

# A18 Authoritative Source Map
Generated Date: 2026-10-28
Document ID: `DLG-BENCH-MAP-A18-SOURCES`

## 1. Frozen Benchmark Configuration
- **Approved Tag:** `benchmark-a16-submission-2026-10-28`
- **Target Commit:** `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f`
- **Current HEAD Commit:** `91fb2a76369c09a0d9131fd03e70965082190871`
- **Supported Datasets:** 13 Primary + LANL (14 datasets total)
- **Supported Models:** DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug (7 models)
- **Evaluation Seeds:** 42, 43, 44, 45, 46 (5 seeds)
- **Approved Primary Runs:** 420 (84 pairs × 5 seeds) + 35 LANL (7 pairs × 5 seeds) = 455 complete runs.
- **Unsupported Pairs:** 7 pairs (CONAD diagnostic-only; OCGNN on DGraphFin/BHP; CoLA on DGraphFin/BHP; AnomalyDAE on large graphs where memory ceiling hit).

## 2. Evidence Tier & Storage Hierarchy
| Dataset Group | Campaign / Origin | Metric Source Path | Raw Scores Path | Population Masks |
|---|---|---|---|---|
| Cryptocurrency (Ethereum, BSC, Polygon) | A08 Data Repair (`local_storage/benchmark/a08_data_repair/runs/`) | `metrics.json` (per-run) | `scores.npz` (full array) | Exact test masks & labels stored |
| Primary Synthetic & Real (Amazon-Syn, Elliptic, Yelp, etc.) | Paper Ready A04 (`evaluation/benchmark/v2/paper_ready_a04/publication_evidence_a04/metric_json/`) | JSON per run | Stored in historical v2 outputs | Stored in split masks |
| BitcoinOTC | Paper Ready A05 (`evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/`) | JSON per run | Stored in historical v2 outputs | Stored in split masks |
| LANL Cyber | Round 5 Cyber Campaign | JSON per run | Stored in raw outputs | Full evaluation slice |

## 3. Metric Invariants & Population Standards
- **Test Prevalences:**
  - Ethereum: ~0.4089 (M=584, N=1428)
  - BSC: ~0.1473 (M=221, N=1500)
  - Polygon: ~0.0259 (M=120, N=4641)
- **Obsolete Prevalences (REJECTED by Negative Control):**
  - Ethereum: 0.0195
  - BSC: 0.0210
  - Polygon: 0.0052
