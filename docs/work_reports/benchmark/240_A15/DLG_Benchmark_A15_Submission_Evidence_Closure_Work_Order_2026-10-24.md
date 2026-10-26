# DLG-GNN Benchmark — A15 Submission Evidence Closure Work Order
## A15 — Final Test-Population Integrity, Report Reconciliation, and Public Release Repair

**Document ID:** DLG-BENCH-WO-A15-2026-10-24  
**Date:** 2026-10-24 (Asia/Seoul)  
**Parent:** A14 Final Campaign Coherence Work Order / A14 execution report  
**Scope:** Minimal preprint/journal correctness closure. **No broad neural retraining, no new baselines or datasets.**  
**Submission target:** Preprints.org first; *Applied Sciences* SI “Machine Learning and Its Application for Anomaly Detection” second.

> **A15 is not a new benchmark project.** A14 resolved the cryptocurrency Table 4 versus Figure 3 conflict. A15 resolves a more consequential inconsistency in the AP **random/no-skill baseline** and several stale report/release claims before publishing a permanent preprint DOI.

---

# 0. Frozen contract and stop rules

Keep unchanged unless *actual evidence corruption* is demonstrated:

- 13 primary + LANL external datasets;
- seven primary detector identities; five seeds 42–46;
- 84/91 complete primary support pairs; 420 primary runs + 35 LANL runs;
- original A08 clean cryptocurrency campaign as canonical, superseding A03; A05 BitcoinOTC campaign; historical Round5 plus LANL per recorded provenance;
- original AP/ROC/F1 run-level values from approved canonical records;
- CONAD reference diagnostic-only policy;
- honest disclosure of unavailable historical raw anomaly-score arrays.

**Explicitly prohibited:** rerunning 455 jobs to recover historical raw scores, adding ROCm, optimizing GADNR, tuning model hyperparameters, switching datasets, post-hoc score flipping, silently changing model runs to improve performance.

A targeted data/label-mask extraction is allowed; **GPU retraining is not required** to verify label prevalence.

---

# 1. A15-G1 (HARD) — Define and verify evaluation population + prevalence

## 1.1 Create canonical denominator ledger

**Inputs:** A14 `canonical_selected_run_registry.csv`, A08 `approved_registry.json`, historical run metric JSON/label or mask metadata, frozen loaders and dataset manifests.

For each dataset and seed create:

```text
publication/a15/evaluation_population_by_seed.csv
```

Required columns:

```text
dataset,seed,campaign_id,graph_nodes,graph_edges,feature_dim,
model_input_positive_count,model_input_population,
evaluation_mask_definition,evaluation_total,evaluation_positive,
evaluation_prevalence,source_label_hash,split_hash,
source_path,verification_level
```

**Checks:**

1. `evaluation_prevalence == evaluation_positive / evaluation_total` within numerical tolerance;
2. every model evaluated on the same `(dataset,seed)` uses the same designated labels/split/metric population, or discrepancies are explicitly recorded and do not support a common AP no-skill baseline;
3. distinguish **source/full-graph prevalence**, **injected anomaly prevalence**, **training/validation/test prevalence**, and **labeled evaluation prevalence**;
4. where raw labels/masks are not archived, recover counts from the loader using the frozen dataset construction and seed, or from credible persisted run metrics. Do not manufacture counts;
5. separately document split variability across seeds.

## 1.2 Crypto hard cross-check

A14 approved A08 registry already includes:

```text
Ethereum: test_total = 2,893; seed-wise prevalence mean ≈ 0.408918
BSC:      test_total = 1,500; seed-wise prevalence mean ≈ 0.147333
Polygon:  test_total =   471; seed-wise prevalence mean ≈ 0.025902
```

These values do **not** agree with Table 2's old `π_test` values `0.0195 / 0.0210 / 0.0052` and may differ from Table 1 global/injected values. This inconsistency is a **publication blocker**. Verify test mask and label unit; do not simply copy the above means without checking that A08 APs actually use those masks.

## 1.3 Choose and document paper presentation

Preferred:

- Table 1: `Graph N/E/F`, `Source/injected label prevalence` with an unambiguous heading and denominator;
- Table 2: `Evaluation no-skill AP baseline`, defined as the **mean test/evaluation positive rate across the same five seeds**, with a footnote describing within-dataset seed variation;
- Results prose: comparisons of AP with prevalence must use this verified Table 2 baseline, not global source prevalence;
- Appendices: seed-level evaluation-positive and evaluation-total data.

If historical denominator cannot be authenticated for a dataset, mark `NOT_VERIFIED`, remove unsubstantiated numeric AP-vs-no-skill comparisons for that dataset, and explain the limitation. **Never substitute a default fraction**.

## 1.4 Acceptance

- [ ] 14 datasets have honest denominator statuses, with no two differently scoped quantities both labeled “test prevalence”;
- [ ] all crypto per-seed prevalence agrees with the masks used by their AP metric computation;
- [ ] Table 1/Table 2 headings, footnotes and Discussion are aligned;
- [ ] empirical no-skill claims are recomputed or removed where unverifiable;
- [ ] no new model training.

---

# 2. A15-G2 (HARD) — Reconcile all main AP/support claims to the canonical registry

From `canonical_selected_run_registry.csv` regenerate:

```text
publication/a15/main_ap_mean_sd.csv
publication/a15/support_14x7.csv
publication/a15/conditional_ap_14.csv
```

Assert:

- all 14 datasets × seven planned detectors represented in support matrix;
- complete supported cell contains seeds `{42,43,44,45,46}` and five metrics;
- 84/91 primary, seven unsupported, seven external supported;
- **actual missing model–dataset pairs:**
  - `GADNR`: Elliptic, DGraphFin, Yelp-Syn, Flickr-Syn, Reddit-Syn (five);
  - `AnomalyDAE`: Yelp-Syn, Reddit-Syn (two);
- no zero or fabricated metric for failed cell;
- failure **reason** is taken from source failure registry, not inferred solely from absence of success;
- Table 2, Figure 2, Abstract, captions and main-text support paragraph reflect these exact counts;
- Table 4/Figure 3 remain at their currently verified A08-clean deltas.

Replace the existing manuscript sentence that says GADNR fails on seven graphs while also listing two AnomalyDAE failures. Do not add new supported cells or change model performance.

**Acceptance:** complete per-cell audit CSV; zero unsupported/performance contradictions; cross-artifact checks include Tables 1, 2 and 4, not just Table 4.

---

# 3. A15-G3 — Correct operational claims and report content

## 3.1 Replace the wrong A14 execution report Table 2.4

The A14 report's 14-row “Final Harmonized Delta Table” is **not** the real 13+1 dataset table. It includes Tolokers, Questions, T-Finance, T-Social and YelpChi, and misreports Ethereum/BSC/Polygon/BitcoinOTC/LANL Base/Aug APs.

Regenerate **the full section from** `publication/a14/conditional_ap_14.csv` or a new verified A15 copy. Delete all non-contract dataset names. Add a script-side guard enforcing the exact 13+1 dataset set. Update the `A14_FINAL_SIGNOFF.md` with a versioned correction notice or create a new `A15_FINAL_SIGNOFF.md`; **do not silently rewrite immutable historical releases**.

## 3.2 Remove unsupported mechanisms in descriptive rows

Rewrite overly causal phrasings in Table 4 and surrounding text:

```text
BAD: "localized contract similarity enhances scam discovery"
BETTER: "DLG-Aug has higher mean AP in this constructed similarity-graph evaluation"

BAD: "clustered subgraphs benefit from ego-net pretraining"
BETTER: "positive ΔAP observed; mechanism not isolated by this comparison"
```

Use “exact full-graph execution supported under the reported envelope”, not “scalable ego-net lift”, for DGraphFin unless timing/complexity evidence directly supports a stronger claim.

**Acceptance:** no external dataset substitution in any A15 execution report and no causal wording inconsistent with the available diagnostics.

---

# 4. A15-G4 — Fix Git identity and publish reproducibility location

The public annotated tag exists and currently resolves as follows:

```text
Tag name:       benchmark-a14-submission-2026-10-24
Tag object:     346dc59674e1d893a1a1317d14d21224bbfdde2d
Target commit:  19442b4612dcd5dca3c5cdef5a8b37575cabd5db
```

The old uploaded `public_release_verification.json` cites a different object/target, and the A14 execution report includes older `bd06fd...` / `b9e417...` commits. Resolve identities **from the current remote Git ref**, not local HEAD.

**Rules:**

1. Do not move/rewrite the existing A14 tag to retroactively match an inaccurate report;
2. put the accurate tag-target commit in a new versioned A15 release manifest;
3. if manuscript source changes, create **a distinct new project-scoped A15 tag**, push, verify remotely, record tag object + target commit;
4. optionally publish a GitHub Release and, if feasible, deposit a frozen archive DOI; at review time, the A14 tag existed but no published GitHub Release by that tag was found;
5. no future-placeholder DOI may appear in final paper or cover letter;
6. ensure a fresh clone can fetch the cited tag and access the exact reproduction files.

Output:

```text
publication/a15/remote_git_identity_verified.json
publication/a15/release_manifest.json
```

**Acceptance:** a reader can resolve the manuscript's exact public ref and confirm all cited files exist there.

---

# 5. A15-G5 — Reissue checksums from the actual final bytes

The uploaded submission bundles have internally consistent `SHA256SUMS.txt` values, but the A14 execution report, signoff and `public_release_verification.json` cite older hashes.

Examples from uploaded bytes:

```text
Preprints PDF:
671b0c80944de257595409b4cd59c1f88aa86da6aab3b43bf5dd59b936d0fe1c

MDPI PDF:
e43270c3c8a7d2e513923c107674121506c7d18b83871cf7f71a8a94a44efb38

Preprints source ZIP:
5ca731a678fc89f57d5ea194bdbdf7107a492c6adef746fd6f2ec1a1bfc4f5c3

MDPI source ZIP:
1122e1a982f700b4b75f7473a3ad7e17e3f0338aa573696522d3674bc41a990a
```

These apply only to the uploaded **A14** bytes, *not* automatically to newly compiled A15 files.

## Procedure

1. Build final TeX/PDF, do not edit them afterward;
2. zip only the final TeX, bibliography, figures, and required template resources;
3. compute SHA-256 from actual disk bytes **after** final assembly;
4. generate submission `SHA256SUMS.txt`, release manifest, execution report and signoff programmatically from that same inventory;
5. independently unzip into a clean temporary directory, compile and inspect the output;
6. test a deliberately corrupted hash is rejected by the release validator.

**Acceptance:** every reported size/hash equals the published release asset's actual bytes; no stale report/signoff entries.

---

# 6. A15-G6 — Extend final validator and PDF inspection

The final validator must hard-fail on:

```text
wrong dataset names in report or manuscript
Table1/Table2 prevalence scope mismatch
per-seed eval positive count and denominator mismatch
Table2 AP values not sourced from canonical registry
unsupported pair counted as a supported run
GADNR failure count != 5 or AnomalyDAE failure count != 2 (for A14 frozen primary)
report Table4 values differ from canonical Table4
Figure3 data differ from Table4
report/signoff SHA-256 differs from final actual bytes
tag target differs from asserted commit
unverified mechanism asserted as causal
```

Perform mutation tests including a changed cryptocurrency `π_test`, changed Table 2 AP, inserted non-contract dataset, wrong GADNR failure count, stale hash and wrong Git tag target.

PDF inspection checkpoints:

- inspect both 18-/16-page equivalents after new build (page counts may change);
- verify Table 1 prevalence labels/denominators, Table 2 no-skill comparison and any changed Results paragraphs;
- verify Figure 3 still legible, legend not covering data;
- check citation/reference integrity, AI/data-availability disclosures, author and submission metadata.

Output:

```text
publication/a15/evidence_denominator_audit.md
publication/a15/report_manuscript_consistency.json
publication/a15/negative_controls_a15.json
publication/a15/final_submission_readiness.md
```

---

# 7. Minimal execution order (no extensive rerun)

```text
[1] Freeze A14 bytes as a rollback archive
[2] Extract/verify evaluation denominators, priority crypto 3
[3] Reconcile graph vs evaluation prevalence across all 14
[4] Rebuild Table1/Table2 AP-baseline claims from the correct population
[5] Verify 84/91 support and correct the 5+2 failure description
[6] Reissue correct campaign/report delta table (same 13+1)
[7] Reword unsupported causal operational interpretations
[8] Repair provenance/tag identity; prepare new A15 public ref if paper changed
[9] Rebuild Preprints and MDPI documents
[10] Build ZIPs; then compute final SHA-256 inventory
[11] Run extended negative controls and independent PDF visual review
[12] Preprints.org deposit, obtain distinct benchmark DOI
[13] Disclose preprint DOI in Applied Sciences submission metadata and submit
```

## Final stop rule

**If denominator reconstruction is complete (or unverifiable baselines are transparently removed), no new detector training is required.** After G1–G6 pass, stop development. Do not add an A16 general benchmark cycle merely for manuscript polish.

---

# 8. Go / No-Go

**GO for Preprints.org** only when:

- [ ] test/population prevalence is honestly labeled and AP baseline matches evaluated labels;
- [ ] Table 2, Table 4, Figures 2–3 and all associated claims derive from the selected campaign;
- [ ] A15 report contains only the actual frozen 13+1 datasets and accurate values;
- [ ] signed/public Git identity is independently resolvable;
- [ ] actual ZIP/PDF SHA-256 matches all documentation;
- [ ] clean compilation and visual check succeed.

**GO for Applied Sciences** after Preprints.org registration details are recorded and the MDPI journal package passes the same A15 gates. Journal acceptance cannot be guaranteed by passing an internal validator.
