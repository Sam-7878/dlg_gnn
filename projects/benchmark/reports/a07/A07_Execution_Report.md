# A07 manuscript and repository finalization report — 2026-10-05

## Result

**Local A07 candidate complete; public submission gate HOLD.** No model training, dataset addition, detector addition or score change was performed. The frozen A05/A06 evidence remains the scientific source. A06 rollback copies remain at `docs/work_reports/benchmark/229_claude_review/rollback_a06/`.

- Related Work now covers graph imbalance/fraud learning, dynamic graphs, scalable GNN execution, and benchmark methodology. The manuscript cites 44 references; unverified DR-GAD was excluded.
- Appendix K includes four S1/S2 PR-AUC Wilcoxon rows, with Holm adjustment checked against the frozen family. All four adjusted comparisons fail to establish DLG-Aug superiority over DLG-Base or DOMINANT. The paper does not interpret non-rejection as equivalence.
- BSC augmentation degradation, the smaller Ethereum Base advantage, BSC/GADNR BitcoinOTC seed spread, and the correct S3 provenance (five source/native-label graphs plus injected-label BitcoinOTC) are explicit.
- The MDPI candidate is 14 pages. F1 is a separate one-page supplement; ROC-AUC remains in Appendix L. The neutral candidate retains the F1 table in its appendix. No numeric cell was removed from the release evidence.
- Root README/INSTALL/CITATION, three architecture documents, reproduction command metadata, four `projects/` facades, wrappers, environment notes, data acquisition matrix and lightweight CI matrix were added/updated. Shared scientific imports were not moved.
- New local Git commits: `de5500c`, `429a840`, `92c6865`, `2d51c16`. Project-scoped **local candidate** tag `benchmark-v2.0.0-a07-candidate-r2` resolves to `2d51c162c7f72d89de19b4f78fb1b3059ca92999`. It has not been pushed or released publicly. The historical `v1.0.0-preprint` was not changed.

## Verification evidence

| Check | Result |
|---|---|
| Bundled ZIP manifest | PASS: 645 payload files and archive SHA-256 verified |
| Frozen support/registry | PASS: 13 primary, seven inferential models, 80/91 supported, 435 approved successes |
| Archived exact sparse/fused/AnomalyDAE reports and CONAD policy | PASS from frozen hash-checked records; not a new model rerun |
| S1/S2 Holm recalculation and A06 alert budget source rebuild | PASS; alert CSV SHA-256 `7d64d0733fd1090890aafffa5934fde09b483e546762d0a780cf86cf3c0cd131` |
| Manuscript/repository consistency gate | PASS; release identity absent in candidate PDF, as expected |
| Four project verify modes | PASS in local research environment; DLG fixed forward/backward/gate/exact-sparse and PyGOD tests, StreamMC replay/bounded state, and TDS graph fixtures executed with frozen output hashes |
| Local integration tests | PASS: 10 DLG/StreamMC tests under `.venv_cuda` with `PYTHONPATH=src` |
| Fresh checkout at final candidate tag | PASS: system-Python benchmark verify and `.venv_cuda` paper; 44 refs, four Holm rows, 14 claim links. `paper` elapsed 2.79 s. System-Python four-project CI matrix also passed integrity checks (scientific smokes skipped without optional packages). |
| PDF build and full visual review | PASS as a candidate: 14 MDPI pages and one F1 supplement page; no overfull or undefined-reference warnings. See `A07_Final_PDF_Review.md`. |
| Candidate package | PASS: 95 payload files, ZIP SHA-256 `562981c32263f76234e95eae6a2e9ead7a358cbf7dc7cd7cf2a7d05c7632876a`; the historical package now resides at ignored `local_storage/benchmark/backups/A07_candidate_review_package.zip` |

The original 2026-10-05 candidate PDFs had MDPI SHA-256 `3a279a9a4f9e5c1a620dacf048c432061670d4ed70b721ea24ab660063b0821b`, neutral SHA-256 `9dee19da603e1087b10d985f5b48653c7423a3420121cf9a88ebcc021e9cdd1f`, and F1 supplement SHA-256 `53f55fe1b5e29bfd3b8dcaada237a3c85362a4ab283142309d635cb130c25ae6`. The 2026-10-06 project-layout rebuild has new PDF hashes only because the automatic print date changed; see the updated PDF review for exact current hashes and normalized-text comparison.

## Public release and submission gate

The GitHub Release, archive DOI, Preprints.org deposit and Applied Sciences submission have **not** occurred. `gh` CLI is absent on this WSL host, and `git push --dry-run origin HEAD:refs/heads/main` failed because HTTPS GitHub write credentials are unavailable (`could not read Username`). More importantly, the current PDF still promises a future release URL. The A07 work order forbids depositing that PDF. Before external publication, obtain author/coauthor signoff, publish a project-scoped public tag and evidence asset, record the actual URL/commit/DOI if available, rerun manuscript generation, compile and review the new PDF, then deposit the preprint and submit to the journal. The local candidate tag and package do not satisfy the public Data Availability requirement.

The source manifest reveals a separate reproducibility limit: Ethereum/BSC/Polygon `*_hybrid_graph.pt` inputs have frozen hashes but no confirmed public acquisition route or redistribution permission. This is now stated in `projects/benchmark/DATASETS.md` and the candidate manuscript. Full independent reruns of those three graphs remain conditional on separate data access; reviewer fast verification and paper arithmetic are supported without them.

Eight preexisting modified A05 source-snapshot files remain unstaged in the working tree. They were not added to A07 commits or silently substituted into the frozen evidence ZIP.
