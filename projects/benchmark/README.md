# Benchmark v2 evidence and reproduction

**Published evidence:** A09 supporting evidence is publicly released at [benchmark-a09-2026-10-10](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a09-2026-10-10), scientific commit `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d`. Earlier `NOT_CLEARED` or local-candidate statuses describe pre-publication snapshots. This immutable release identity is authoritative for the published assets; manuscript submission remains a separate author action. The [A10 minor submission finishing report](reports/minor_submission_a10/FINAL_HANDOFF.md) records the later author-local Keywords correction without changing the release or scientific results.

## Current science: completed A08; A09 submission closure

A08 replaces the label-informed Ethereum/BSC/Polygon hybrid inputs with raw-transfer-derived, label-independent contract features and a declared binary Euclidean 5-neighbor similarity graph. The actual provider populations are 14,464 / 7,499 / 2,353 contracts, respectively, with eight features. See [the correction protocol](protocols/A08_DATA_REPAIR_AMENDMENT.md), [input evidence](evidence/a08_data_repair/) and [execution report](reports/a08_data_repair/NEW_CAMPAIGN_REPORT.md).

The corrected campaign completed105/105 fresh runs: seven detectors and seeds42–46 on all three corrected chains. Current support counts, rankings and paper claims must come from the completed A08 registry. **Current A08 G0–G10 and end-to-end validation passed (FINAL_PASS). A09 C1–C6 closure is SUBMISSION_READY**, with user-confirmed coauthor approval, immutable public evidence, independent retrieval/numeric replay and final private PDF/source review. Actual bindings and limits are in `reports/a08_data_repair/FINAL_AUDIT_A08.md` and `FINAL_ACCEPTANCE_A08.json`. The old archive is preserved as historical evidence.

`verify`, `tables` and author-local `paper` select A08 when its config is present and fail explicitly if current evidence is incomplete. `--revision legacy-a07` selects historical reproduction explicitly. A08 `verify` requires the new curated numeric bundle and checks actual payload bytes and input/split/config/run identities; restricted raw-score recomputation and complete scientific/PDF acceptance remain separate. Unsubmitted A08 manuscript sources, writers and PDFs remain author-local.

## Preserved historical A05–A07 evidence

The historical frozen evidence contains **13 primary datasets**, seven functioning inferential models, **80/91** supported primary pairs and **435** approved primary/external successful records. These are historical counts, not A08 acceptance targets. CONAD is diagnostic only. BitcoinOTC uses controlled node anomaly injection on a real trust graph. Historical crypto metrics are excluded from current corrected primary inference.

## Reproduction modes

1. `./projects/benchmark/reproduce.sh verify --revision legacy-a07` checks the historical public numeric ZIP’s **641** payload hashes, support/run counts, archived numerical qualifications and a tiny independent Gram identity. Current A08 `verify` instead requires `evidence/a08_public_numeric_evidence.zip` and its separate outer checksum record. Standard Python suffices for either integrity mode; no GPU or raw provider data is required.
2. `./projects/benchmark/reproduce.sh tables --revision legacy-a07` performs historical verification and regenerates its frozen numeric tables plus seeded Friedman permutation statistics (200,000 permutations/view). NumPy and SciPy are required. Outputs: `evidence/astra_revision/`. Current A08 `tables` requires its approved registry.
3. `paper` is author-local only: unsubmitted TeX/Bib/PDF, A06 manuscript input and manuscript writers are intentionally excluded from Git. Missing private inputs yield an explicit message. It is not a public checkout reproduction promise.
4. `full` is a guard for manual campaign orchestration. It exits without training. Full neural reruns need source data, appropriate permissions, GPUs and campaign-specific environments. Corrected A08 crypto construction has complete source-derived identities and repeat/counterfactual checks; provider redistribution permission is not asserted.

Use `.venv_cuda` for local scientific work. CI uses standard Python for integrity, without claiming neural retraining. See [data and construction](DATASETS.md), [environment provenance](ENVIRONMENT.md), [revision audit](reports/astra_revision/README.md) and [checkout verification](FRESH_CLONE_VERIFICATION.md).

## Public evidence versus private originals

`evidence/public_numeric_evidence.zip` is a publication derivative of the local immutable A05/A06 ZIP. It removes four manuscript-writing/packaging scripts and retains all other payload bytes unchanged. Its manifest records original archive hash, excluded paths and every retained payload hash. `expected_outputs.json` records both archive identities. A measured-memory figure PDF inside this evidence ZIP is supporting evidence, not the manuscript PDF.

The previous GitHub commit already contained the original evidence ZIP and A07 writer. Removing them from current tracking cannot remove past Git history. No history rewrite has been performed. Historical `v1.0.0-preprint` refers to an earlier ten-dataset paper, not this candidate. Scientific code and supporting evidence are published at [benchmark-a09-2026-10-10](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a09-2026-10-10). Current post-publication closure is recorded separately under reports/submission_closure_a09/; no platform submission has been executed.

## A09 final closure

See [current closure manifest](reports/submission_closure_a09/submission_closure_manifest.json), [actual public retrieval](reports/submission_closure_a09/public_retrieval_verification.json) and [final 52-page review](reports/submission_closure_a09/Final_PDF_Review_A09.md). The immutable release contains the earlier reviewed companion snapshot. These later outer records do not alter its ZIP bytes or A08 science. Current reports on main may postdate the pinned scientific release.
