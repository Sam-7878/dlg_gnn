# DLG-GNN Benchmark — A14 Independent Evidence Review

**Review ID:** DLG-BENCH-REVIEW-A14-2026-10-24  
**Date:** 2026-10-24 (Asia/Seoul)  
**Scope:** Uploaded `DLG_Benchmark_A14_Execution_Report_2026-10-24.md`, `scripts.zip`, `a14.zip`, `a14_paper.zip`, `submission.zip`; public GitHub A14 tag and related API records.  
**Decision:** **A14 substantially fixes Figure 3 versus Table 4, but the two submissions are still HOLD for a small number of source/claim integrity corrections. No broad model retraining is warranted.**

## 1. Independently verified strengths

1. The canonical registry uploaded in `a14.zip` contains **455 rows**, 14 unique datasets, seven model names and exactly five seeds for each supported model–dataset pair. The supported primary pairs are **84/91** (420 successful runs), plus seven LANL pairs (35 runs).
2. The 70-row `paired_ap_by_seed.csv` and 14-row `conditional_ap_14.csv` reproduce the core Figure 3 and Table 4 direction: **Elliptic +0.0350; Ethereum +0.0253; BSC +0.0021; Polygon −0.0124; Reddit-Syn −0.0656; LANL −0.0252**. The 14-row table embedded in the actual A14 Preprints TeX matches the canonical CSV at reported precision. These numbers were recalculated against the uploaded row data, not merely accepted from the report.
3. The A14 Figure 3 is legible at independent raster rendering, and no large negative BSC bar is erroneously clipped (the canonical BSC difference is small and positive). The separately uploaded Preprints and MDPI PDFs render and have 18 and 16 pages, respectively. The two source bundles include the corresponding final TeX, figures and bibliography; the TeX inside each ZIP is byte-identical to the final TeX outside it.
4. The canonical selector labels the crypto A08 campaign as the replacement for obsolete A03 runs and records **105 superseded historical rows**. This provenance distinction materially improves interpretability.
5. An actual *public Git tag* named `benchmark-a14-submission-2026-10-24` is reachable. The corresponding source commit and tag need more careful identification (Section 4).

These findings support reusing the A14 model outputs and approved campaign logic, subject to the checks below.

## 2. Publication blocker: misleading evaluation prevalence denominators

The **actual A14 manuscript** contains two different columns both called or interpreted as **Test Prevalence / empirical no-skill AP baseline**. Table 1 (dataset properties) and Table 2 (AP comparison) disagree for **13 of 14 datasets**; Elliptic is the sole numerical match.

| Dataset | Table 1 “Test Preval.” | Table 2 π_test (AP no-skill baseline) | Inconsistency |
|---|---:|---:|---|
| Elliptic | 0.0976 | 0.0976 | match |
| DGraphFin | 0.0127 | 0.0134 | mismatch |
| BitcoinOTC | 0.0300 | 0.0890 | mismatch |
| Ethereum | 0.4184 | 0.0195 | **major** |
| BSC | 0.1476 | 0.0210 | **major** |
| Polygon | 0.0261 | 0.0052 | **major** |
| Yelp-Syn | 0.0100 | 0.1113 | **major** |
| Amazon-Syn | 0.0300 | 0.0718 | mismatch |
| Reddit-Syn | 0.0200 | 0.0345 | mismatch |
| Flickr-Syn | 0.0200 | 0.0500 | mismatch |
| Cora-Syn | 0.0299 | 0.0500 | mismatch |
| CiteSeer-Syn | 0.0298 | 0.0500 | mismatch |
| PubMed-Syn | 0.0300 | 0.0500 | mismatch |
| LANL-RedTeam | 0.0180 | 0.0019 | **major** |

It is possible that Table 1 describes full-graph/global prevalence or injection targeting while Table 2 inherited a distinct historical label population. If so, those figures must be **explicitly named and separately defined**. They cannot both be presented as the same test-set prevalence, especially while Table 2 calls π_test the no-skill AP baseline.

### Direct run evidence for crypto test prevalence

The canonical uploaded registry has explicit `test_total`, `test_positive` and `test_prevalence` for the A08 crypto seeds, agreeing across the seven detectors *within the same seed*. The mean across seeds 42–46 is:

| Dataset | Actual per-seed test population | Mean recorded test prevalence | Table 2 claimed no-skill baseline |
|---|---:|---:|---:|
| Ethereum | 2,893 | **0.408918** (range 0.395092–0.430695) | 0.0195 |
| BSC | 1,500 | **0.147333** (range 0.135333–0.160000) | 0.0210 |
| Polygon | 471 | **0.025902** (range 0.014862–0.038217) | 0.0052 |

For an ordinary average precision computed over that labeled test set, the empirical random-ranking expectation is approximately the positive proportion of **that test set**. Thus the published no-skill claims for these three datasets do not agree with the reported executed evaluation masks. In particular, Ethereum DLG-Aug mean AP ≈0.4359 is close to 0.4089, not enormously above a 0.0195 baseline.

**Required:** Recover the exact evaluation masks/label counts (or authoritative per-run evaluation metadata) for all datasets; state whether graph, injected-label, train/validation/test, or labeled-evaluation prevalence is displayed. Regenerate AP-baseline comparisons and all interpretation dependent on them. Do not alter approved detector APs merely to force agreement. If a subset's denominator cannot be recovered, remove an unsupported numeric no-skill comparison rather than guess.

## 3. Internal report and methods consistency issues

### 3.1 The A14 execution report's Section 2.4 table is not the canonical registry

The report includes a 14-row “Final Harmonized Delta Table” with data not matching `conditional_ap_14.csv`. Examples:

| Row in report | Report Base / Aug AP | Actual uploaded canonical Base / Aug AP |
|---|---|---|
| Ethereum | **0.6277 / 0.6530** | **0.4106 / 0.4359** |
| BSC | **0.6384 / 0.6405** | **0.3390 / 0.3411** |
| Polygon | **0.6726 / 0.6602** | **0.0854 / 0.0731** |
| BitcoinOTC | **0.3325 / 0.3341** | **0.8274 / 0.8211** |
| LANL | **0.0135 / 0.0134** | **0.1367 / 0.1114** |

The same report lists **Tolokers, Questions, T-Finance, T-Social, YelpChi**, none of which belongs to the frozen 13+1 dataset contract. This looks like a copy/paste or stale-generated section and must be reissued. Crucially, **the actual A14 TeX Table 4 and Figure 3 *do* use the correct frozen dataset names and CSV deltas**. Do not conflate report corruption with a demonstrated table/figure corruption.

### 3.2 Support explanation is numerically inaccurate

The canonical registry's actual seven missing primary pairs are:

- **GADNR (5):** Elliptic, DGraphFin, Yelp-Syn, Flickr-Syn, Reddit-Syn.
- **AnomalyDAE (2):** Yelp-Syn, Reddit-Syn.

The submitted paper currently says **“GADNR fails ... on seven graphs, while AnomalyDAE ... on Yelp-Syn and Reddit-Syn”**, which suggests at least nine missing pairs. The A14 campaign selection rationale similarly mislists GADNR failures on supported Cora/CiteSeer/Amazon instead of the actual five. Rebuild both prose and failure reasons from the support/run registry and raw failure taxonomy, not hand-typed sentences. A lack of a successful row alone does not prove that its failure was OOM rather than a numerical/backend limitation.

### 3.3 Overly causal operational characterizations

The A14 `table_conditional_delta.tex` contains statements including “localized contract similarity enhances scam discovery”, “clustered subgraphs benefit from ego-net pretraining”, and “massive financial graph demonstrates scalable ego-net lift.” These go beyond the numerical ΔAP evidence when topology/mechanism diagnostics are not directly established. Prefer observational wording (“higher mean AP in this evaluation”, “full-graph run supported within the declared envelope”).

### 3.4 Scope of the numerical validator

The A14 validator checks the full 14-row *conditional-delta table* against its CSV, but does not independently validate **Table 1 vs Table 2 prevalence**, all Table 2 AP values, every support status, the numeric claims in the execution report, or release checksums. Its six negative controls are useful but cannot justify describing the entire manuscript as fully cross-artifact verified.

## 4. Public tag, release identity and hash manifests

The public GitHub API confirms the existence of annotated tag `benchmark-a14-submission-2026-10-24`. The tag object is **`346dc59674e1d893a1a1317d14d21224bbfdde2d`** and resolves to **commit `19442b4612dcd5dca3c5cdef5a8b37575cabd5db`**. The repository also contains earlier commits `bd06fd453f6fed3c113dd891a26d8f069c1544ee` and `b9e417718c03631f7750ceb84629187577abaf0f`, but neither is the current A14 tag target.

By contrast, the uploaded A14 execution report header gives `b9e417...`, Section 0 gives `bd06fd...`, its user's pasted summary gave `19442b4`, and the archived `public_release_verification.json` claims a different annotated-tag object and target (`67dea...` / `b9e417...`). **The public tag exists, but the archived verification JSON and report are stale.**

GitHub's API returns `404` for a **published GitHub Release by that tag** at the time checked. A public tag is still usable, but a published Release and DOI must not be claimed unless created and verified.

### Reported hash values do not match the actual uploaded submission bundle

Example independent disk SHA-256 checks:

| Artifact | Actual uploaded SHA-256 | A14 execution report SHA-256 |
|---|---|---|
| Preprints PDF | `671b0c80944de257595409b4cd59c1f88aa86da6aab3b43bf5dd59b936d0fe1c` | `aa948c2275811c7ff10aee75a401416e83d8e57849c065f4d8e87e14ba69f9ba` |
| MDPI PDF | `e43270c3c8a7d2e513923c107674121506c7d18b83871cf7f71a8a94a44efb38` | `ee95a940428ef5dc100f22c608f3f87f9eb4bc268d3744957e84cbef130f1484` |
| Preprints source ZIP | `5ca731a678fc89f57d5ea194bdbdf7107a492c6adef746fd6f2ec1a1bfc4f5c3` | `bf44083758b998cf7aa6541f92e399bf080ee1ea30c00c0f997c679234ddbe23` |
| MDPI source ZIP | `1122e1a982f700b4b75f7473a3ad7e17e3f0338aa573696522d3674bc41a990a` | `d041e2eb98b0f77d332aa6c382352bfa32e3ffcb24d2ef07cfc8430e62d4c0c1` |

The actual uploaded `submission/*/SHA256SUMS.txt` and `publication/a14/final_artifact_hashes.sha256` **do match the uploaded bytes** for these artifacts. The inconsistency is chiefly the report/signoff/public-release JSON, not evidence that the ZIP itself is corrupted. Reissue the report/signoff from the latest manifest, with one resolved Git identity, rather than revising hash strings manually.

## 5. Gate assessment

| A14 gate | Independent result |
|---|---|
| G1 canonical 455 run selection | **PASS at record-selection level**; original per-run metadata not all re-verified |
| G2 14-dataset deltas | **PASS**; actual canonical CSV, figure, and TeX align |
| G3 Figure 3 rendering | **PASS** for inspected vector figure, with no BSC clipping |
| G4 literal manuscript validation | **PARTIAL**; prevalence / other claims not covered |
| G5 reproduction entrypoint | **PARTIAL**; relative-path code exists, clean independent full invocation not repeated here |
| G6 public git identity | **PARTIAL**; tag exists, commit/tag report mismatch; no published GitHub Release verified |
| G7 submission packaging | **PASS for archive-content correspondence**; report hash manifests stale |
| G8 overall sign-off | **HOLD** pending prevalence, support statement, report hashes and Git identity |

## 6. Final recommendation

Do not begin another model benchmark cycle. Perform the narrowly scoped **A15 Submission Evidence Closure Work Order**. Its hard scientific gate is the evaluation-mask/test-prevalence denominator; its editorial/provenance gates are the erroneous A14 report section, the support/failure prose, Git identity and hash reconciliation. Once all pass, regenerate PDF and source bundles, check byte hashes, and then proceed to Preprints.org → Applied Sciences. This review makes no claim to have independently re-executed the 455 neural training runs.
