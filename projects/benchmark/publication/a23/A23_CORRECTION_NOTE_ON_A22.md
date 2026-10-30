# A23 Correction Note on Historic A22 Submission Readiness Report

- **Date:** 2026-10-30
- **Historic A22 Tag Preserved:** `benchmark-a22-submission-2026-10-30` (`7e764da3a55d9d13ab50a2dbebab13670946a6cd`)
- **Post-Tag ORCID Commit:** `93e4c82790b9ac1a6b12f4eddf037858c83921c3`
- **Untouched Historic Report:** `projects/benchmark/publication/a22/A22_SUBMISSION_READINESS_REPORT.md`
- **Archived A22 Snapshot Directory:** `projects/benchmark/publication/a23/archive_a22_snapshot/`

## 1. Purpose of This Note

In accordance with Work Order `DLG-BENCH-WO-A23-FINAL-PROVENANCE-ONLY`, the signed A22 report (`projects/benchmark/publication/a22/A22_SUBMISSION_READINESS_REPORT.md`) and historic Git tag `benchmark-a22-submission-2026-10-30` are preserved untouched as an immutable audit record. This correction note documents why a post-tag commit caused SHA-256 drift between `A22_SUBMISSION_READINESS_REPORT.md` and the working-tree submission bundles, and summarizes the six documentary/provenance corrections completed in **A23**.

## 2. Root Cause of the A22 Report vs. Package SHA-256 Discrepancy (`A22-F1`)

1. At commit `7e764da3a55d9d13ab50a2dbebab13670946a6cd` (tagged `benchmark-a22-submission-2026-10-30`), the compiled manuscripts had SHA-256 digests:
   - `DLG-Benchmark_A16_Preprints.pdf`: `334cd3cc6287fd92ff30628d50d998d66b62ccbc761586c43e1601b62a9cfbc5`
   - `DLG-Benchmark_A16_AppliedSciences.pdf`: `54684e5245238eb797a2acde19a01ba7c6c4a46b2f6c9a5e6d74ffca02e207a8`
   These exact digests were recorded in `A22_SUBMISSION_READINESS_REPORT.md`.
2. Immediately afterward, commit `93e4c82790b9ac1a6b12f4eddf037858c83921c3` updated author ORCID metadata in `DLG-Benchmark_A16_Preprints.tex` and `DLG-Benchmark_A16_AppliedSciences.tex`, recompiled the PDFs and `source_bundle.zip` archives, and updated `submission/preprints/SHA256SUMS.txt` and `submission/applsci/SHA256SUMS.txt`, **without** regenerating `A22_SUBMISSION_READINESS_REPORT.md`.
3. In **A23**, all SHA-256 digests in `submission/preprints/SHA256SUMS.txt`, `submission/applsci/SHA256SUMS.txt`, `publication/a23/A23_SHA256SUMS.txt`, and `publication/a23/A23_SUBMISSION_READINESS_REPORT.md` are computed in a single post-build pass from the final frozen files on disk, eliminating any post-report rebuild drift.

## 3. Summary of A22 Independent Review Findings & A23 Resolutions

| Finding ID | Severity | Summary of Issue | A23 Resolution Artifact |
| :--- | :---: | :--- | :--- |
| `A22-F1` | P0 | Post-report ORCID rebuild caused SHA-256 mismatch between `A22_SUBMISSION_READINESS_REPORT.md` and `submission/*/SHA256SUMS.txt`. | `publication/a23/A23_SHA256SUMS.txt`, `A23_SUBMISSION_READINESS_REPORT.md` |
| `A22-F2` | P0 | `a22_dataset_topology_diagnostics.csv` used DGraphFin's `evaluation_nodes` (`367,702`) instead of `graph_nodes` (`1,225,601`) and divided mismatched upstream denominators (`node_retention_ratio > 1`). | `publication/a23/dataset_graph_identity_canonical.csv`, `dataset_graph_identity_canonical.md` |
| `A22-F3` | P0 | Over-causal mechanistic interpretation of the Elliptic permutation/zero-vector ablation in Table A3 caption, Limitations item 7, and `a22_ablation_provenance_audit.csv`. | `publication/a23/ablation_provenance_noncausal.csv`, `a23_g2_prohibited_phrase_scan.json` |
| `A22-F4` | P0 | `a22_stat_view_membership.csv` described View S3 as all real entity fraud labels, overlooking that `BitcoinOTC` uses canonical injected node anomaly labels (`180/6,005 = 3.00%`) on a real trust graph. | `publication/a23/s3_label_provenance_gate.csv`, `a23_stat_view_membership.csv` |
| `A22-F5` | P0 | GADNR failure narrative needed explicit separation between Flickr-Syn 8-GB OOM, historical `NO_APPROVED_RAW_RUN` failures on 4 datasets, and the synthetic rank-deficient covariance illustration. | `publication/a23/gadnr_failure_evidence_note.md` |
| `A22-F6` | P1 | `a22_exact_loss_source_trace.md` wrote `Z^T A[:,i] = (AZ)_i` (valid only for symmetric `A`), used shorthand paths `dlg_gnn/model/*`, and needed explicit $N=40$ CPU fixture scope. | `publication/a23/g1_exact_loss_canonical_trace.md`, `a23_g4_asymmetric_orientation_test.json` |
