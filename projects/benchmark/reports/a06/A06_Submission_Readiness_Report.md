# A06 submission readiness report

Date: 2026-10-05 (Asia/Seoul)  
Target: *Applied Sciences*, Special Issue “Graph Neural Networks: Theory, Methods and Applications”  
Decision: **LOCAL PACKAGE COMPLETE; EXTERNAL PUBLICATION GATE OPEN — NOT YET READY TO SUBMIT**

## Scope and scientific checks

A06 only transforms the frozen A05 release. It adds no datasets, detectors, model training, hyperparameter search, ROCm runs, or DGraphFin rerun. The A05 G4-JOURNAL validator was rerun in `.venv_cuda` and returned **PASS, 23/23, zero blockers**. The 13 primary datasets plus LANL, seven functioning primary configurations, diagnostic-only PyGOD CONAD path, 80/91 supported primary pairs, 435 successful records, and S1–S5 results are preserved. In particular, S3 is a mixed-label six-graph view with `p=0.082284`, interpreted as non-significant.

The A06 builder verified the SHA-256 values and approved run IDs of 60 existing A03 contract-graph metric JSONs before producing the 1%/5% alert-budget table. The table consistently covers DOMINANT, CoLA, OCGNN, and DLG-Base across Ethereum, BSC, and Polygon. A05-approved DLG-Aug repaired records lack these fixed-budget fields and the historical prediction arrays were not retained, so DLG-Aug is omitted rather than estimated. The operating points are labeled retrospective diagnostics, not deployment guarantees. `evaluation/benchmark/v2/paper_ready_a06/claims_to_evidence_a06.csv` maps 12 main claims to paper locations and approved evidence; all listed source paths exist.

The article now has current GAD context, exactly three stated contribution axes, explicit GenAI disclosure, a clear distinction between frozen-primary support and post-primary AnomalyDAE recovery, the (O(N^2)) AnomalyDAE arithmetic limit, qualified memory claims, conditional DLG guidance, a usage guide, and limitations on mixed campaign provenance and Elliptic mechanism inference. No universal superiority or production deployment claim is made.

## Package verification

| Item | Result |
|---|---|
| A05 journal gate | PASS 23/23 |
| Evidence ZIP | 645 hashed payload files plus manifest; all archive bytes and SHA-256 values independently rechecked |
| Evidence ZIP SHA-256 | `459a3721efd1c0e17e49315ca31e92331ae754ae1683068303bdc793a5feab08` |
| MDPI source ZIP | 22 files; independently extracted and compiled to a 13-page PDF |
| MDPI source ZIP SHA-256 | `6b1be835de67dde60f96dd94294ff36fc2a2f74b378266c373fe5218b1325ece` |
| Neutral PDF | 13 pages; SHA-256 `2972ecb1e95541fc540949b9d328087b2971333830faa4e30a6a3e69fa49a722` |
| MDPI PDF | 13 pages; SHA-256 `015f52291c3d6cb399552c2b6dd60404dc021da818b98e94f26b3d193ea2339a` |
| PDF log and visual review | PASS; all 13 pages of each PDF reviewed; see `A06_Final_PDF_Review.md` |

The archive contains the A05 publication manifest, approved run registry, canonical dataset manifest, protocol, environment locks, table/statistics generation scripts, exact-equivalence reports, diagnostic records, and new BitcoinOTC and targeted Ethereum/DGraphFin score evidence. Dataset provider terms still govern the original graph files. Historical raw predictions and per-run package locks are incomplete, and complete bitwise replay of the legacy campaign is not claimed.

## Submission gate

| Gate | Status | Basis / remaining action |
|---|---|---|
| Scientific benchmark | **PASS** | Frozen A05 G4-JOURNAL 23/23. |
| Evidence/provenance | **PASS WITH DISCLOSED LEGACY LIMITATIONS** | 645-file hash manifest verified; historical limitations stated in article and availability text. |
| Manuscript claims | **PASS** | 12 claim links checked; S3, support, projected time, exactness, and alert-budget wording audited. |
| Recent literature context | **PASS** | Established and 2023–2026 GAD methods added without claiming they were run. |
| Data/code location | **HOLD** | The local evidence ZIP has no public persistent release URL/tag/DOI. Public GitHub `main` resolves to `92a4f667eb0ae71be375867f7b0c435f31ad122d`; local working source is based on `8a238d96ea1231a69821228db6d17d811fd1125b` with uncommitted A05/A06 changes. Neither is a published A06 source release. Publish an immutable source/evidence release, verify access and hashes, then insert its URL/tag/commit and DOI if available into the manuscript and availability statement. |
| GenAI disclosure | **PASS, subject to author confirmation** | Methods, Acknowledgments and submission note identify OpenAI ChatGPT/Codex, assistance scope, verification and author responsibility. |
| PDF visual review | **PASS** | Both 13-page PDFs inspected page by page; no clipping, missing citations, or misplaced tables. |
| MDPI metadata | **PREPARED / HOLD** | Title, abstract, authors, affiliations, ORCIDs, funding, contributions, declarations, cover letter and SI field prepared. Coauthor sign-off, current official MDPI template comparison, public evidence link, and Benchmark preprint DOI/version are pending. |

**Final status:** The scientific and local document work is complete. The A06 stop rule requiring an actual persistent evidence location and preprint record has **not** passed. Do not upload this candidate to the journal while the availability text says the release URL will be inserted later. The preceding DLG-GNN preprint DOI `10.20944/preprints202609.0848.v1` belongs to a different article and must not be used as the Benchmark DOI.

## Final external sequence

1. Obtain both authors' final approval of the manuscript, GenAI disclosure, contributions, funding, and cover letter. Confirm whether a Benchmark preprint already exists, avoiding a duplicate record.
2. Commit the A05/A06 source and publish the verified evidence ZIP and source under an immutable release/tag; optionally archive it for a DOI. Verify public download and SHA-256 against this report. The repository's `.gitignore` excludes `docs/work_reports/` and `*.csv`, so the intended manuscript/report/claims files require explicit `git add -f` (or a deliberate ignore-rule change) when making that commit. Do not assume the present local files are already in Git.
3. Insert the real public URL, tag, commit and DOI (if issued) into the generated manuscript text, `data_availability_statement.md`, and submission metadata. Rebuild both PDFs, refresh both ZIPs, and repeat the PDF/hash review after those text changes.
4. Register the Benchmark manuscript at Preprints.org or update its existing record; record its distinct DOI/version. Insert that preprint disclosure in the journal submission metadata and cover letter.
5. Check the Special Issue and current official MDPI LaTeX template immediately before upload. The packaged local MDPI class identifies itself as the **12 September 2024** version; compatibility with the current official template remains unverified. Recompile and review if the class changes.
6. Submit the authors' approved, updated package to *Applied Sciences*. No further benchmark campaign is required by the present evidence.

The release and publication steps are external, public actions. Their absence is a real gate failure, not a reason to reopen the frozen experiments.
