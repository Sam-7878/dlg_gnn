# DLG-GNN Benchmark A18 — Independent Evidence Review

**Review date:** 2026-10-28 (Asia/Seoul)  
**Inputs examined:** user-uploaded `243_A18.zip` and `submission.zip`; publicly accessible `Sam-7878/dlg_gnn` A18 numerical CSVs and audit code on GitHub `main`.  
**Decision:** **Scientific repairs substantially successful; submission status HOLD pending narrowly scoped A19 evidence/provenance closure.**  
**Not claimed:** An independent re-execution of private 105 `scores.npz` arrays or 455 source metric JSONs; these were not included in the two uploaded ZIPs.

## 1. Independently confirmed successes

- **Submission artifacts exist and render:** Preprints PDF 21 pages, MDPI PDF 18 pages; Figure 1 has been reworked to show the two-stage GCN/local-feature pretraining and full-graph operations; Table 3 and Table A2 are present. The architecture figure no longer includes the old PageRank/topology regularization nodes.
- **Table 3 arithmetic:** The published `publication/a18/alert_budget_per_seed_verified.csv` contains exactly 210 rows (3 datasets × 7 models × 5 seeds × 2 budgets). Independent arithmetic checks of `TP`, `K`, `M_test`, per-seed precision and recall find **0 violations**. The 15 dataset-seed evaluation populations are internally consistent across model identities. The 42 grouped aggregates include Ethereum DOMINANT P@1%=0.489655… and R@1%=0.0120538…, corresponding to the manuscript's 0.4897 and 0.0121.
- **F1 table arithmetic:** `f1_per_seed_verified.csv` contains 455 values and 91 model–dataset pairs, each with five seeds. Independently recomputing means and sample SDs reproduces examples printed in Appendix Table A2: Elliptic DLG-Aug = 0.223936 ± 0.016390; BitcoinOTC DOMINANT = 0.932903 ± 0.006025; Ethereum OCGNN = 0.643673 ± 0.055984; BSC DLG-Base = 0.465969 ± 0.023352.
- **Dual source bundles:** In the uploaded `submission.zip`, nested Preprints and MDPI `source_bundle.zip` files include correctly named A16 TeX sources whose bytes match the standalone uploaded TeX files. Both PDFs opened and have the claimed page counts.
- **Scientific framing:** DLG-Base's gate, pretrained and frozen local representations in DLG-Aug, unsquashed dot-product decoding, support-aware results, CONAD diagnostic-only policy and the lack of universal augmentation benefit are more defensible than in prior manuscript versions.

## 2. Submission-relevant unresolved issues

### H1 — G1 proves *metric JSON internal arithmetic*, not independent reconstruction from score arrays

In `projects/benchmark/scripts/a18_gate_g0_g1_g2.py`, lines ~183–189 check `scores.npz` existence/hash; lines ~200–265 read `test_total`, `test_positives`, `alert_budgets`, `precision`, `recall`, and (optionally) `true_positives` from `metrics.json` and check their algebraic consistency. **The script never loads the raw score or test-label arrays to independently recover top-K memberships/TPs.** The 210/210 arithmetic PASS is genuine but does not, by itself, establish a raw-score-to-Table-3 re-evaluation. Correct either the claim or the verification implementation. Prefer a bounded CPU-only raw-score recomputation for 105 preserved crypto runs, without retraining.

### H2 — F1's *validation-thresholded test* interpretation is not verified for 420 of 455 rows

The generated `f1_per_seed_verified.csv` has **420 empty `threshold` fields**; only the 35 LANL rows carry threshold numbers. The current A18 source reader's fallback precedence is:

1. `thresholded.f1`
2. `validation_f1`
3. `f1`
4. `test_f1`

It checks equality with the frozen registry F1 but **does not verify that the F1 came from predictions on the test mask using a threshold selected only on the validation mask**. In particular, `validation_f1` could denote F1 **on validation**, not test. The table caption presently says “Validation-thresholded Test F1 Score; Thresholds selected solely on validation splits.” This may be correct for the underlying runner, but is **not established by the current A18 evidence**. A19 must identify each record's originating metric field and threshold-generation code or soften the caption for unverified runs; do not invent missing threshold values.

### H3 — A18 source map contradicts actual evaluation populations and support history

`243_A18/a18_authoritative_source_map.md` lists:

- Ethereum `M=584, N=1428`; however, A18 per-seed CSV gives `N_test=2893` and seed-specific positives 1143, 1246, 1153, 1184, 1189 (mean 1183).
- Polygon `M=120, N=4641`; however, A18 per-seed CSV gives `N_test=471` and positives 16, 11, 7, 18, 9 (mean 12.2).
- BSC `N_test=1500`, mean positives 221, is consistent.

The source map also names unsupported pairs involving OCGNN/CoLA and “BHP” that do not match the frozen 13+1 contract, and describes historical raw scores as present where the project's Data Availability statement explicitly says they were not stored for 350 old runs. This **metadata page must not ship as the authoritative source map** in its current form. Generate it from machine-readable data, not static prose.

### H4 — Frozen public tag does not contain A18 evidence

Both submitted Data Availability documents point to `benchmark-a16-submission-2026-10-28`. That tag exists, but `projects/benchmark/publication/a18/alert_budget_per_seed_verified.csv` is **not available at that tag**; it is present on `main`. A18 manuscript, metrics audit, and raw-source trace must be pinned to a **new versioned, publicly reachable commit/tag/release**, or to an immutable archive, before a reader can reproduce A18 rather than A16.

### H5 — Reports/cryptographic digests have multiple incompatible authorities

The A18 **Execution Report** and uploaded `243_A18/submission_gate_report.md` list different digest sets for all six core submission files. Independent SHA-256 of actual ZIP contents matches **`submission_gate_report.md` and `publication/a18/final_artifact_hashes.sha256`**, not the Execution Report's ledger. For instance:

| Artifact | Actual uploaded SHA-256 prefix | A18 Execution Report prefix |
|---|---|---|
| Preprints PDF | `b865b3f41315` | `3dfa53ebefd6` |
| Preprints ZIP | `fcb017697437` | `ae7da5fc2748` |
| MDPI PDF | `615de37923bd` | `16866fcb7dd6` |
| MDPI ZIP | `b100fbdf08a2` | `4c718a7c29be` |

Also, `submission/preprints/PREPRINT_GATE_REPORT.md` and `submission/applsci/SUBMISSION_GATE_REPORT.md` remain **A13** reports, even though their folders now contain A18 documents; the separate old `_A16.md` reports likewise should not be treated as current. Keep old reports in a history folder and publish exactly one current A18/A19 gate report and checksum ledger.

### M1 — Residual vocabulary and portability

The Preprints PDF page 11 still says “local ego-net augmentation” in the Results discussion, although the now-corrected Methods and code trace explicitly state **no ego-net extraction occurs**. Replace with “pretrained local-feature augmentation.” A18 scripts hardcode `/mnt/d/_Work/goat_bank/dlg_gnn` or a Windows `D:/` fallback, preventing clean-clone reproduction of the audit on a third-party machine. Make the new reproduction path repository-relative, leave historical script hashes intact, and document local-only raw data prerequisites.

## 3. What the reviewer should NOT demand

- No new 13 × 7 × 5 training campaign, ROCm, fresh baselines, or GPU memory framework.
- No retroactive fabrication of 350 unarchived historical score arrays.
- Do not treat missing threshold metadata as proof of leakage; record uncertainty until original code/config resolves it.
- Do not declare `metrics.json` self-consistency incorrect: it is valid but narrower evidence than raw-score re-evaluation.
- Do not retune threshold on test labels or change score direction after examining held-out labels.

## 4. Final judgment

The A18 repair of Tables 3 and A2 is materially stronger than A17 and the manuscript fits the topic of the Applied Sciences anomaly-detection Special Issue. However **the submission package should remain HOLD** until H1/H2 claims are properly qualified or verified, H3/H5 contradictory documentation is fixed, and H4's immutable release points to the actual A18 evidence. These are bounded **CPU-only evidence/editorial/release tasks**, not a scientific benchmark extension.
