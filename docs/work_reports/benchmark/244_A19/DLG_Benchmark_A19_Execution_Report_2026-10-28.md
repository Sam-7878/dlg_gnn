# DLG Benchmark A19 Execution and Release Closure Report

**Document ID:** `DLG-BENCH-REP-A19-EXECUTION`  
**Date:** 2026-10-28 / 2026-10-29  
**Status:** ALL GATES PASSED (GREEN) — `PREPRINT READY`  
**Approved Release Tag:** `benchmark-a19-submission-2026-10-28`  
**Target Submission Platform:** Preprints.org (First) & *Applied Sciences* (MDPI) Special Issue  

---

## 1. Executive Summary & Verification Matrix

The A19 execution successfully resolves all findings raised in the Independent Review (`DLG_Benchmark_A18_Independent_Review_2026-10-28.md`) across six formal engineering and evidence closure gates (G1–G6). No GNN models were retrained, no historical raw scores were retroactively fabricated, and all operations were strictly bounded to CPU arithmetic verification, source code tracing, and packaging integrity.

| Gate ID | Verification Focus | Prior Status / Review Finding | A19 Resolution & Final Audit | Final Gate Status |
|---|---|---|---|---|
| **Gate A19-G2** | F1 Semantics & Operating Points | **H2:** 420/455 missing thresholds; ambiguous validation vs test semantics | Traced code paths in `evaluate_threshold_protocol()`; established that all 455 runs represent test F1 at validation-selected threshold (`F1_on_test_at_validation_threshold`); recovered all 105 crypto thresholds from `operating_point.threshold` in A08 runs. Table A2 caption reconciled with supplementary provenance table. | **PASS (100%)** |
| **Gate A19-G1** | Raw Alert Budget Recomputation | **H1:** Metric-JSON check only; lack of source array recomputation | Recomputed all 210 crypto evaluations ($P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$) from continuous floating-point arrays in `scores.npz` and binary test labels via stable mergesort. Exact matches on TP, precision, recall, and selected node IDs: **210 / 210 (100.0%)**. | **PASS (100%)** |
| **Gate A19-G3** | Authoritative Source Map & Populations | **H3, H4:** Erroneous crypto populations ($N=1428, 4641$); references to BHP/CONAD; false raw score claim for historical runs | Regenerated `a19_authoritative_source_map.md` and `a19_source_map_consistency.json` directly from machine-readable registry and raw data. Ground truth test populations established: Ethereum ($N=2893, \bar{M}=1183.0$), BSC ($N=1500, \bar{M}=221.0$), Polygon ($N=471, \bar{M}=12.2$). Removed BHP; matched 7 unsupported pairs in `support_14x7.csv`; truthfully separated 105 stored raw scores from 350 unarchived historical runs. | **PASS (100%)** |
| **Gate A19-G5** | Canonical Digest Ledger & Editorial Sweep | **H5, M1:** Hash discrepancy between execution report and zip; obsolete A13 reports in submission; residual "local ego-net" | Replaced residual "local ego-net augmentation" in Results with "pretrained local-feature augmentation"; removed duplicate architecture paragraphs; archived obsolete A13/A16 reports into `history/`; made reproduction scripts repo-relative via `Path(__file__)`; recompiled both PDFs in TeXLive 2025 (20 pages Preprints, 17 pages Appl. Sci.; 0 undef refs/cits); verified clean-unpack compilation of `source_bundle.zip`; rendered 150 DPI visual QA; generated single canonical ledger `A19_SHA256SUMS.txt`. | **PASS (100%)** |
| **Gate A19-G4** | Immutable Release Tag & Predeclaration | **Finding:** `benchmark-a16` tag did not include A18/A19 evidence | Predeclared tag `benchmark-a19-submission-2026-10-28` inserted in both manuscripts and `DATA_AVAILABILITY.md`. Staged and tagged final commit. | **PASS (100%)** |
| **Gate A19-G6** | Benchmark Engineering Stop & Hand-off | **Condition:** Strict stop condition; no A20 campaign | All G1–G5 gates passed. Benchmark contract (14 datasets, 7 models, 5 seeds, 91 supported pairs, 455 runs, exact AP/ROC/Alert/F1 metrics) frozen. | **PREPRINT READY** |

---

## 2. Canonical Artifact Hashes (Single Disk-Verified Ledger)

All cryptographic checksums below match the actual bytes on disk and in `projects/benchmark/publication/a19/A19_SHA256SUMS.txt`:

| Artifact | File Path | Size (Bytes) | SHA-256 Digest |
|---|---|---|---|
| **Preprints PDF** | `submission/preprints/DLG-Benchmark_A16_Preprints.pdf` | 609,252 | `2354ba74577ea629353ff94c920fe867275c56de7bd971510e42e0d342b1e8e9` |
| **Preprints TeX** | `submission/preprints/DLG-Benchmark_A16_Preprints.tex` | 42,912 | `7fbeeef46d0a4d702a5b9a0212f9a25d12ee9b7409a64c54393a1da6da0e64a4` |
| **Preprints Bundle** | `submission/preprints/source_bundle.zip` | 884,516 | `dd90abd80dd076ba46ad036b7e71c250264ad8589eb0838d29c9bc5f09f2a527` |
| **Applied Sciences PDF** | `submission/applsci/DLG-Benchmark_A16_AppliedSciences.pdf` | 368,793 | `0666f4500216e127501d215b2e97c035985164b5d2c760e1a081f736c7ddd942` |
| **Applied Sciences TeX** | `submission/applsci/DLG-Benchmark_A16_AppliedSciences.tex` | 42,284 | `bc0bcf0ce1a2a3f44f1800d51db54310f9cdb9a6956283f4673bd3cc3f2ade8b` |
| **Applied Sciences Bundle** | `submission/applsci/source_bundle.zip` | 1,920,500 | `745308b3fbe524b2a9080985821870a94bdd77d5a0408b1a5a5341595f773c80` |

---

## 3. Protocol Verification & Ground Truth Denominators

### Cryptocurrency Evaluation Populations (Table 1 & Table 3)
- **Ethereum:** $N_\mathrm{test} = 2893$, $M_\mathrm{test} \in [1143, 1246, 1153, 1184, 1189]$, mean $\bar{M}_\mathrm{test} = 1183.0$, prevalence $\pi_\mathrm{test} = 0.4089$ ($40.89\%$).
- **BSC:** $N_\mathrm{test} = 1500$, $M_\mathrm{test} \in [215, 237, 217, 206, 230]$, mean $\bar{M}_\mathrm{test} = 221.0$, prevalence $\pi_\mathrm{test} = 0.1473$ ($14.73\%$).
- **Polygon:** $N_\mathrm{test} = 471$, $M_\mathrm{test} \in [16, 11, 7, 18, 9]$, mean $\bar{M}_\mathrm{test} = 12.2$, prevalence $\pi_\mathrm{test} = 0.0259$ ($2.59\%$).

### Supported / Unsupported Statuses (Table 2 & Support Ledger)
- Total combinations: 14 datasets × 7 models = 98 pairs.
- Supported combinations: 91 pairs (84 primary pairs + 7 LANL cyber pairs = 455 runs across 5 seeds).
- Unsupported combinations: Exactly 7 pairs:
  - `(DGraphFin, GADNR)`, `(Elliptic, GADNR)`, `(Reddit-Syn, GADNR)`, `(Yelp-Syn, GADNR)`: `NO_APPROVED_RAW_RUN` (Covariance determinant limitation)
  - `(Flickr-Syn, GADNR)`: `OBSERVED_HISTORICAL_CUDA_OOM` (Historical 8-GB legacy environment)
  - `(Reddit-Syn, AnomalyDAE)`, `(Yelp-Syn, AnomalyDAE)`: `OBSERVED_CUDA_OOM` (24-GB RTX 3090 full encoder forward)

### Truthful Evidence Tiers
- **105 Cryptocurrency Runs:** Preserved in `scores.npz` continuous score arrays; recomputed and verified at the raw prediction level with 100% exact top-$K$ node IDs.
- **350 Historical Baseline Runs:** Summary metric scalars preserved in frozen JSON publication artifacts with SHA-256 verification. Raw continuous score vectors were not preserved at runtime by upstream campaigns and are truthfully disclosed as not archived.

---

## 4. Final Disposition & Handoff

In accordance with Work Order A19:
1. **Status:** `PREPRINT READY`.
2. **Next Steps:**
   - Upload `source_bundle.zip` and `DLG-Benchmark_A16_Preprints.pdf` to Preprints.org.
   - Obtain distinct Benchmark Preprint DOI.
   - Cross-disclose preprint DOI in *Applied Sciences* submission.
   - **No further experimental cycles or A20 retraining.** Benchmark engineering is definitively stopped and closed.

