# DLG-GNN Benchmark A19 — Final Evidence and Release Closure Work Order

**Document ID:** `DLG-BENCH-WO-A19-FINAL-EVIDENCE`  
**Date:** 2026-10-28 (Asia/Seoul)  
**Parent:** A18 Critical Evidence Repair; accompanying A18 independent review  
**Target:** Preprints.org → Applied Sciences, *Machine Learning and Its Application for Anomaly Detection*  
**Priority:** Journal-level scientific honesty, **no GNN retraining**, no additional dataset/model experiments

> **Final stop rule:** After the six gates below pass, freeze the verified manuscript and submit. Additional model comparisons and broad benchmark development are out of scope. If source evidence is unavailable, narrow the claim rather than manufacture it.

## Gate A19-G1 — Make Table 3 verification genuinely source-level, or state its limit

**Current finding:** the A18 script hashes `scores.npz` but reads all `K`, `TP`, `precision`, `recall`, `M_test` from `metrics.json`; 210 successful invariants prove internal arithmetic, not actual top-K ranking reproduction.

**Required work:**

1. Using the preserved 105 crypto `scores.npz` files and the *same frozen* label/test-mask arrays, reconstruct `N_test`, `M_test`, `K`, TP@1%, TP@5%, P@K and R@K for seeds 42–46.
2. Explicitly specify `K` rounding convention, score orientation, node-ID ordering, evaluation population, tie-breaking and whether the ranking is restricted to test nodes.
3. Assert per-run identity against `metrics.json` and `alert_budget_per_seed_verified.csv` using exact integer TP and K; fail on mismatched population/indices. Do **not** relabel the test population or invert score direction to improve performance.
4. If raw array schema/mask provenance is insufficient for this bounded CPU-only check, keep A18's Table 3 only as **metric-JSON-derived** and disclose that raw rank re-evaluation was not independently established. Do not claim a raw-score reconstruction PASS.

**Outputs:** `a19_alert_budget_raw_recheck.csv`, `a19_alert_budget_raw_recheck.md`, optional `a19_alert_budget_raw_recheck.py`, and source-hash manifest.

**GO:** all checked records match; alternatively unsupported raw recheck is explicitly reported and the level of evidence stated accurately. **NO-GO:** unverifiable Table 3 values promoted as raw-recomputed.

## Gate A19-G2 — Prove or qualify the 455 F1 metrics' evaluation semantics

**Current finding:** `f1_per_seed_verified.csv` contains 455 source-linked values but **420 missing thresholds**; the A18 extraction function accepts `validation_f1` before generic `f1` or `test_f1`.

**Required work:**

1. For each campaign (`A08`, A04/A05/Round5, LANL), record the *exact* JSON field used for F1 and the original scorer/threshold selection code path.
2. Distinguish `F1_on_validation`, `F1_on_test_at_validation_threshold`, `F1_on_test_at_default_threshold`, and `UNKNOWN` (no inferred labels).
3. Verify test-mask evaluation and selection of thresholds on validation-only data wherever original code and logs allow. Do not infer threshold selection merely from metric naming, and do not tune thresholds on test labels.
4. Retain the 455 numerical F1 values if they are genuine historical reported metrics, but change Appendix Table A2's **“Validation-thresholded Test F1”** caption unless the strong statement is demonstrated for every reported group. The safe fallback is **“Source-reported F1 (mean ± SD across five seeds); threshold provenance varies by campaign; see supplementary provenance table.”** If a field is actually validation F1, label the group explicitly and do **not** call it test F1.

**Outputs:** `a19_f1_semantics_by_campaign.csv`, `a19_f1_provenance_gate.md`, updated Table A2/Methods/Limitations if necessary.

**GO:** title/Methods wording matches the proven metric semantics; every campaign classified. **NO-GO:** 420 unknown thresholds silently labeled validation-selected test thresholds.

## Gate A19-G3 — Regenerate one correct source map; eliminate wrong evaluation populations

Replace hand-maintained `a18_authoritative_source_map.md` from immutable machine-readable inputs.

**Mandatory checks:**

- Ethereum `N_test=2893`, per-seed `M_test=[1143,1246,1153,1184,1189]`, mean 1183; not `N=1428, M=584`.
- BSC `N_test=1500`, mean positive count 221.
- Polygon `N_test=471`, per-seed `M_test=[16,11,7,18,9]`, mean 12.2; not `N=4641, M=120`.
- Model support statuses must match frozen `support_14x7.csv`; remove references to `BHP` and unsupported model identities not in the frozen contract.
- Clearly distinguish **105 crypto raw scores stored** from **350 historical scores not archived**; never claim nonexistent historical raw scores.
- Match machine-readable evaluation CSV, Table 1, Table 2, Table 3, official support table and Data Availability.

**Output:** `a19_authoritative_source_map.md`, `a19_source_map_consistency.json`, provenance for its generation.

**GO:** zero cross-file disagreement; old map archived as nonpublication historical error record.

## Gate A19-G4 — Correct release pin and submission package evidence

**Current finding:** `benchmark-a16-submission-2026-10-28` does **not** include A18's corrected Table 3/F1 evidence; the latter is on `main`.

**Required work:**

1. Commit final corrected A18/A19 code, aggregate CSVs, manuscript TeX, figures, integrity reports and public reproduction instructions.
2. Create **new** immutable project-scoped Git tag (e.g. `benchmark-a19-preprint-2026-10-28`) and push the tag and commit. Preserve prior A16/A17 tags unchanged.
3. Verify from an external, unauthenticated GitHub query that the new tag exposes the correct `publication/a18` or `publication/a19` data and the exact manuscript revision.
4. Put the **predeclared new tag name** into both manuscripts and Data Availability *before* committing; create the final commit and attach the tag to it. Verify and record the actual full SHA in the GitHub Release notes and an external submission receipt. To avoid an impossible self-referential Git hash, **do not require a tracked manuscript to contain the hash of its own final commit**. If a full hash is wanted inside the manuscript, cite an earlier pinned *evidence snapshot commit* and state the distinction explicitly. No `main` branch substitutes or future placeholders.
5. Optional GitHub Release/Zenodo DOI; do not falsely call a tag alone a GitHub Release with binary assets.

**GO:** public tag/commit matches a working fresh-clone reference and includes the needed A18/A19 evidence.

## Gate A19-G5 — One canonical digest ledger and final editorial sweep

**Current finding:** A18 Execution Report digests disagree with the bytes in the uploaded `submission.zip`, while `243_A18/submission_gate_report.md` matches the uploaded bytes. Additionally, current `submission/*/*GATE_REPORT.md` default files still describe A13.

**Required work:**

1. From the actual final disk bytes, regenerate **one** SHA-256 ledger for both PDFs, both stand-alone TeXs and both `source_bundle.zip` files.
2. Regenerate the execution report, both current submission gate reports and release metadata from that one ledger. Archive older A13/A16 reports in `history/` rather than leaving an obsolete `*GATE_REPORT.md` as the default current report.
3. Recheck the `source_bundle.zip` TeX byte identity, clean-unpack compilation, BibTeX, undefined citations, and PDF page counts.
4. Replace residual “local ego-net augmentation” in Results with “pretrained local-feature augmentation”; retain the accurate “no explicit ego-net induction” methodological statement.
5. Run an actual visual inspection of the key Figure 1, Table 3, Table A2 and final pages for both PDFs.
6. Remove absolute Windows/WSL developer-specific workspace paths from the **current** reviewer reproduction command; derive repository root from `Path(__file__)` and use data paths supplied by a manifest/environment variable. Preserve old code as historical if needed.

**Outputs:** `A19_SUBMISSION_GATE_REPORT.md`, `A19_SHA256SUMS.txt`, `A19_PDF_VISUAL_QA.md`, fresh-clone command log.

**GO:** every checksum and current gate file matches actual upload bytes; no stale submission documents remain in the current package.

## Gate A19-G6 — Stop and hand off to publication

Run protocol-aware checks covering:

- unchanged 13+1 × 7 × 5 primary/external contract, 84/91 support and 455 successes;
- unchanged AP/ROC and paired DLG-Aug/Base delta results;
- Table 3 exact arithmetic (and truthful source-verification tier);
- F1 source values, five-seed SD, and accurately described threshold/test semantics;
- graph input, evaluation population and label/provenance identities;
- no unverified causal statements, no invented baseline results;
- two final TeX/PDF/ZIP snapshots and one frozen public Git tag.

**Success condition:** `PREPRINT READY` is appropriate only after **G1–G5** pass. Then submit to Preprints.org, obtain a distinct Benchmark preprint DOI, disclose preprint status to *Applied Sciences*, and stop benchmark engineering. **No A20 experimental cycle** is authorized by this work order.

---

## Practical execution order (minimum effort)

`G2 (F1 semantics) → G1 (score-level alert check) → G3 (source map) → G5 (editorial/checksums) → G4 (new public tag) → G6 (submission)`

The order prioritizes scientific correctness over cosmetic cleanup. All actions are bounded CPU/data/document/release operations; no new GPU training is needed.
