# DLG Benchmark A23 — Final Provenance & Submission Coherence Readiness Report

- **Work Order ID:** `DLG-BENCH-WO-A23-FINAL-PROVENANCE-ONLY`
- **Date:** 2026-10-30
- **Release Git Tag:** `benchmark-a23-preprint-2026-10-30` (superseding historic tag `benchmark-a22-submission-2026-10-30` at `7e764da3a55d9d13ab50a2dbebab13670946a6cd`)
- **Primary Deposit Target:** Preprints.org (`DLG-Benchmark_A16_Preprints.pdf`)
- **Journal Submission Target:** MDPI *Applied Sciences*, Special Issue `"Machine Learning and Its Application for Anomaly Detection"` (SI ID: `C67H3U9QGT`)
- **Frozen Empirical Contract:** 455 valid benchmark runs (`420` primary across `84 / 91` supported pairs + `35` external `LANL-RedTeam` runs) in `canonical_selected_run_registry.csv` (`sha256 = 45d4ca7b4a9b010aa779e7cb60a1816e0f1c08e98039a119d008a1d1cd5db907`) — **100% untouched**.

---

## 1. Executive Summary of A23 Closure

Following the independent evidence review of A22 (`DLG_Benchmark_A22_Independent_Evidence_Review_2026-10-30.md`), which confirmed that Gate G1 (PyGOD exact training objective equivalence) and Gate G2 (490-row support failure ledger) passed, Work Order **A23** was executed to resolve six documentary and provenance coherence items (`A22-F1` through `A22-F6`) without running any new benchmark experiments or altering any empirical metric cell:

| Gate | Scope & Resolution Summary | Primary Deliverable | Status |
| :--- | :--- | :--- | :---: |
| **A23-G0** | Archived pre-A23 snapshot of `publication/a22/` and `submission/*/SHA256SUMS.txt`; preserved historic `A22_SUBMISSION_READINESS_REPORT.md` and tag `benchmark-a22-submission-2026-10-30` untouched; issued `a22_review_findings.csv` and `A23_CORRECTION_NOTE_ON_A22.md`. | `publication/a23/a22_archived_evidence_manifest.json` | **PASS** |
| **A23-G1** | Reconciled all 14 datasets across raw upstream counts, constructed/model-input $(N, E, F)$, labeled evaluation pools, and seed 42–46 test splits; fixed `DGraphFin` `model_input_N = 1,225,601` (vs. `367,702` val+test labeled pool and `183,840` test split/seed); removed misleading `node_retention_ratio` column. | `publication/a23/dataset_graph_identity_canonical.csv` | **PASS** |
| **A23-G2** | Replaced over-causal mechanistic wording in Table A3 caption, Limitations item 7, and ablation audit CSV/MD with non-causal language; verified `0` prohibited phrases across 11 active files. | `publication/a23/ablation_provenance_noncausal.csv` | **PASS** |
| **A23-G3** | Explicitly labeled View S3 as `FinancialBlockchain_MixedLabel_View` (`5` external fraud-label graphs + `BitcoinOTC` with canonical `180/6,005 = 3.00%` injected node labels); strictly separated GADNR historical `CUDA_OOM` (Flickr-Syn), historical `NO_APPROVED_RAW_RUN` (4 datasets), and post-hoc synthetic rank-deficient covariance illustration. | `publication/a23/s3_label_provenance_gate.csv`, `gadnr_failure_evidence_note.md` | **PASS** |
| **A23-G4** | Corrected directed row-residual cross term to $\sum_j A_{ij} \mathbf{z}_j = (A\mathbf{Z})_i$ (noting $\mathbf{Z}^\top A_{:,i} = (A^\top \mathbf{Z})_i$ holds only for symmetric $A$), verified on an asymmetric 12-node directed graph (`max_abs_diff = 0.00e+00`), listed exact `src/gog_fraud/models/pygod/*` paths/hashes, and stated the $N=40$ CPU fixture scope of the 18 unit tests. | `publication/a23/g1_exact_loss_canonical_trace.md` | **PASS** |
| **A23-G5** | Rebuilt both PDFs and `source_bundle.zip` archives, verified isolated clean-unpack compilation in `/tmp` (`23` and `20` pages, `0` errors/warnings), and synchronized all SHA-256 digests in a single post-build pass. | `publication/a23/A23_SHA256SUMS.txt` | **PASS** |

---

## 2. Authoritative Single-Source SHA-256 Checksum Table (`A23-G5`)

Every SHA-256 digest below is computed directly from the final frozen files on disk and is identical across `submission/preprints/SHA256SUMS.txt`, `submission/applsci/SHA256SUMS.txt`, `PREPRINT_GATE_REPORT.md`, `SUBMISSION_GATE_REPORT.md`, and `publication/a23/A23_SHA256SUMS.txt`:

| Package / Directory | File Path | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | ---: | :--- |
| **Preprints.org** | `projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.pdf` | 615,068 | `2ba2757a5372ccb05659ea664d4827956479a33acafa525eac397d94e341b719` |
| **Preprints.org** | `projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex` | 72,260 | `f7d88d32810dbc545068b48310b5ca1f438769e9fd1f8b6775a4105ffe358b86` |
| **Preprints.org** | `projects/benchmark/submission/preprints/references.bib` | 19,402 | `b42f92f2f6143b4cf79eaee3ed55f7e01deba2a6fc4d77164529c4ef3b19e42c` |
| **Preprints.org** | `projects/benchmark/submission/preprints/source_bundle.zip` | 818,200 | `aec6d84e5f02dbc30710ffc7d019070f6d2de137ec3cbace7933514e8e187f62` |
| **Preprints.org** | `projects/benchmark/submission/preprints/SHA256SUMS.txt` | 361 | `5e8b60e5889aa93a14047d9eb2ea84ae3a64ea3dcf14516e0a701ebebf9034fd` |
| **Applied Sciences** | `projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.pdf` | 366,225 | `6cd073eda041b9a9767f994e32cdc4f38efd2a0303939e246c759a69e0ac8026` |
| **Applied Sciences** | `projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.tex` | 71,888 | `55c268a3f2f0776f1ff1662e3f66f2331d1d25ff3717bdd814cba216a83b7022` |
| **Applied Sciences** | `projects/benchmark/submission/applsci/references.bib` | 19,402 | `b42f92f2f6143b4cf79eaee3ed55f7e01deba2a6fc4d77164529c4ef3b19e42c` |
| **Applied Sciences** | `projects/benchmark/submission/applsci/source_bundle.zip` | 1,854,183 | `15a1d5cae582ce8673b5dcbdc07a12650b8d78008bb5b9419c4d41e38393cbbc` |
| **Applied Sciences** | `projects/benchmark/submission/applsci/SHA256SUMS.txt` | 373 | `148623e3e5f8cc343e221a2669bb175aa0544f2827c52c3eed4dac98442881bd` |

---

## 3. A23 Evidence Artifacts Checksum Table

| Gate | Artifact Path | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | ---: | :--- |
| A23 | `projects/benchmark/publication/a23/a22_archived_evidence_manifest.json` | 7,620 | `a5f9aeba3270d6bd5bb7584532c6d7e842d134134161125eaebbb6f0e8164fd3` |
| A23 | `projects/benchmark/publication/a23/a22_review_findings.csv` | 2,829 | `caec690cef0975c0ef61ada7ca46f111dce5e46ec6c96089cb757fdd8b84b739` |
| A23 | `projects/benchmark/publication/a23/A23_CORRECTION_NOTE_ON_A22.md` | 4,146 | `05d0eb280a9581cc2151a5199d8e8fe5d467c709973c926c755def1f42342f0e` |
| A23 | `projects/benchmark/publication/a23/dataset_graph_identity_canonical.csv` | 12,106 | `f616c1c931392c8f57213764917c26d7bb1cab8f631773828a28cb4b11faf047` |
| A23 | `projects/benchmark/publication/a23/dataset_graph_identity_canonical.md` | 4,356 | `a95e040c4ea55ee23631e041fe246d14cefa6f78d1da53ef4a7c7eb7bf367d8c` |
| A23 | `projects/benchmark/publication/a23/dataset_graph_identity_gate.json` | 400 | `b96b36da3e321303f2dc4f73a633a8bc1734a524d86d746985a1e93516290364` |
| A23 | `projects/benchmark/publication/a23/ablation_provenance_noncausal.csv` | 2,032 | `30bca3524f052aee1aa4b3e7a1f88943f5646483dcc081e544235158677f9525` |
| A23 | `projects/benchmark/publication/a23/ablation_provenance_noncausal.md` | 1,634 | `814194df69b5d39c8b4f39a274b5b8f675816f3aa35a495df0d5a78c3aa30284` |
| A23 | `projects/benchmark/publication/a23/a23_g2_prohibited_phrase_scan.json` | 1,304 | `7047dc42f6f1a54b5d7e2eee18c1231d2695f88586875999fdbb244f3f686352` |
| A23 | `projects/benchmark/publication/a23/a23_stat_view_membership.csv` | 1,753 | `ae127cb8669e137e4a82b717ccb2870b876ce0b0c2edd6ecaf598bf95ae72574` |
| A23 | `projects/benchmark/publication/a23/s3_label_provenance_gate.csv` | 2,026 | `06c540662770fdb83ebde7d40d2601ae34c791ad78343cde7751d064669930a5` |
| A23 | `projects/benchmark/publication/a23/gadnr_failure_evidence_note.md` | 3,193 | `b069e81a5d1c8e88502a51803da98b72a12b8bc3333c3477516345eefc5e8508` |
| A23 | `projects/benchmark/publication/a23/a23_g4_asymmetric_orientation_test.json` | 702 | `56d6dea402a0e5983fbda0fc36bfc191528e81fed3628bf21860642a6e30e382` |
| A23 | `projects/benchmark/publication/a23/g1_exact_loss_canonical_trace.md` | 6,395 | `f6be3bd73b0830cbd8a5cf0754d1c9f413d2486cc84e65d88e6850230ba21e0e` |
| A23 | `projects/benchmark/publication/a23/a23_claims_to_evidence.csv` | 5,029 | `49170cd78204a5f8a57bc9a593c61e51c37fe3795170a49952c01df62277dbd7` |
