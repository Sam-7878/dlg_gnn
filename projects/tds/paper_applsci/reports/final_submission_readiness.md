# TDS Applied Sciences final submission readiness

Date: 2026-10-20 (Asia/Seoul).

**Development/evidence revision: completed. Submission: HOLD for author-owned metadata and public-release closure.** No journal submission, Git push, public release or coauthor approval is asserted.

The executable work order is the 314 task in `docs/work_reports/tds/314_grok_review/`. The title is narrowed to **A Validity-First Evaluation of Graph Neural Networks under Temporal Distribution Shift** because the actual primary unit is a contract-graph observation rather than a transaction-level online stream. The contribution is evaluation methodology, with implementation-specific findings.

## Completed evidence

- All 24,316 packed graph identities and upstream contract-label joins checked.
- Sample, label, full-history degree, capped topology, state timing and timestamp semantics made explicit.
- Timestamp ties: 3,910 participating samples / 24,316 (16.08%).
- Test horizon: 18 h 57 min 10 s; 3,648 contracts, 107 positives, only one Polygon positive.
- Per-chain/split/slice composition and contract/new-positive-contract overlap outputs complete.
- All 3,648 inclusive held-out source reconstructions match both packed feature and edge-index tensors exactly.
- Strict earlier-time features and topology evaluated on 20 frozen original checkpoints; mean AP changes are at most 0.0032 in absolute value, with no seed changing by more than 0.0053.
- Seven supplemental conditions × five seeds: **35 completed runs**, each with selected checkpoint, validation/test output, full neural history or LR solver record, and hashes.
- Matched loss matrix: GIN weighted BCE/focal AP 0.4524/0.4885; GraphSAGE weighted BCE/focal AP 0.3114/0.2992.
- Summary LR/MLP AP 0.3634/0.4751; no-chain GIN AP 0.3273. A unique message-passing-superiority claim is withdrawn.
- Original seven comparisons recomputed with standard tied-score AP, 10,000 ordinary bootstrap + 10,000 stratified bootstrap + 10,000 randomizations each. Raw/Holm values both preserved. A separate six-contrast exploratory family is clearly labeled.
- Fitted temperatures and bin support audited; mixed miscalibration wording replaces unsupported uniform confidence-direction claims.
- Top-1%, top-5%, top-50 and top-100 operating-budget metrics complete; F1@0.5 remains secondary.
- Official current MDPI distribution downloaded and hashed; real CTAN soul dependency included. Five editable figures (PDF+SVG) and their data/code supplied.
- MDPI article **13 pages** and supplementary document **4 pages** compile. Final logs have no unresolved citations/references, overfull/underfull boxes or remaining warnings. Selected final pages are manually reviewed in `reports/pdf_review/`; full numerical validation is in `reports/validation.json`.
- Eight scientific-invariant tests pass. Original primary result files and the prior reviewed manuscript are preserved.
- The local source ZIP compiles from a fresh extraction without cached bibliography; both compiled documents have identical extracted text to the final local PDFs. Evidence ZIP includes the new source, raw predictions, checkpoints and retained primary evidence; no upstream multi-gigabyte transaction ZIP is embedded. CRC checks and SHA-256 values are recorded in `packages/package_manifest.json`.

The original convergence audit has an explicit evidence tier: selected epochs and patience code imply stopping before epoch 20 for all original temporal/fraud baselines, but their full trajectories were not retained. Original GIN seed 27 may have hit the ceiling. All new neural controls have full histories and an extended predeclared ceiling; no original curve is fabricated. No claim of exhaustive historical convergence evidence is made.

## Scientific interpretation

The frozen original GIN remains stronger than the three controlled original comparators. The matched controls show that GIN's GraphSAGE gap survives the two evaluated losses, but summary MLP is competitive (exploratory GIN-BCE versus MLP adjusted p=0.1184). Chain covariates and structural summaries contribute substantially. Focal GIN raises AP while worsening ECE. The revised conclusion is therefore about an auditable information/evaluation contract and ranking–calibration separation, rather than a novel architecture or universal temporal-memory fragility.

The original AP tie bug affects resampled intervals, not the saved original point estimates. Its correction is disclosed in both manuscript and supplement. The primary MC-versus-temperature comparison has Holm p=0.0920, and ensemble-versus-ensemble has p=0.1279. Neither is represented as familywise significant.

## Submission gates still requiring the authors

| Requirement | Status and required evidence |
|---|---|
| TDS-specific CRediT allocation and final approval | **PENDING**. Confirm each author's actual roles and approval of the revised scientific interpretation, AI assistance disclosure and cover letter. Other-project approval is not reused as TDS approval. |
| Corresponding author / affiliations / ORCID | **PENDING**. Existing TDS names, Ajou affiliation and emails are retained. Confirm the corresponding author and preferred name spellings. Candidate ORCIDs exist in another approved manuscript, but are not inserted without TDS identity confirmation. |
| Funding | **PENDING OFFICIAL RECORD**. Preserve `IITP-2026-RS-2021-II211835`. Published papers corroborate the spelling; the actual award agreement or official award record is still needed. No guessed change is made. |
| Conflict-of-interest / submission declarations | **PENDING**. The authors must provide the actual declaration, originality and concurrent-submission statements. IRB and consent are not applicable to this public-data analysis. |
| Data/code availability | **PENDING PUBLIC IDENTITY**. Actual repository and upstream URLs are stated. This local TDS revision has no published immutable release/tag. Approve/deposit the intended files and verify unauthenticated retrieval before replacing the explicitly pending Data Availability text. |

The pending fields are visible in the article, rather than silently replaced with invented author claims. Final author information and release details can be inserted without opening a new model-development round.

## Optional work and stop rule

No multi-origin backtest, leaked-snapshot benchmark or MC-variance association is added. The sub-day test horizon and retrospective label limitations are stated. The strict-time diagnostic did not reveal a contract bug requiring primary reruns. Summary-model competitiveness is handled by narrowing the claim. Further experiments are not proposed solely to improve a favorable score; follow the work order's stop rule.

Historical Scam Gate A and Cross-Layer Gate B remain false and are outside this manuscript's primary evidence. Historical Main Gate M v8 remains an archived completeness decision, not a substitute for the submission gates above.
