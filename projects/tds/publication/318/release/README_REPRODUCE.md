# TDS317 evidence, staged by R318

Status: RELEASE_PENDING_AUTHOR_AUTHORIZATION. This directory has not been published.
The ZIP is the exact R317 candidate (SHA-256 below), with no numerical repackaging:

`a4f05674eef4d05913d897d3caa8e085026e7939e0d1d2c82c171d95a672e218`

## Numeric replay (CPU; no raw data, weights or paper sources)

Verify SHA256SUMS_RELEASE.txt from repository root, extract the ZIP into an empty
directory, then run `python replay.py` in that extracted directory. NumPy, pandas,
SciPy and scikit-learn are required; archived `environment.json` records the actual
runtime. R318 observed 110 test arrays, 95 validation-only calibrators and maximum
absolute point/calibrated metric error 1.1102230246251565e-16. This was a local
clean-directory replay, not an anonymous public download or full retraining.

## Source and figure/table reconstruction

`source/` contains repository-shaped R314/R317 code and tests, with R315/R316 gates.
`model_summary.csv`, `modelwise_calibration.csv`, `statistics_and_holm.csv`,
`operational_metrics.csv` and the sorting/lifetime CSVs feed the numerical tables.
`source/projects/tds/scripts/revision317/tables.py` creates those tables. Set its
module OUT variable to a writable copy of this extracted data directory before
calling main(); never regenerate into the sealed evidence directory. The selection
figure code is in `analyze.py:diagnostics`; its exact ECDF input is
`sample_time_lineage.csv`, and its prevalence input is
`first_last_ordering_diagnostics.csv`. The decision tree is a source-defined
procedural figure; no new fitted or empirical result is embedded in it. Archived
figure PDFs/PNGs and editable numeric tables are included. Full diagnostics also
refer to author-local graph inputs; numeric replay alone does not regenerate them.

## Full training reconstruction (separate, upstream-license-dependent)

Obtain upstream data from the GoG repository under its access/license terms, then
use the reconstruction and training source with graph-building inputs and the
documented PyTorch/PyG runtime. Raw 268-million-row transfer archives, contract
address mappings and weights are excluded. Old 18-regression cold tests require
PyTorch/PyG; new R317 integration tests additionally require author-local artifacts.
R318 ran no detector fit and did not repeat the raw-archive scan.

## Rights and identifiers

215 CSVs retain sample_id. Those IDs and timestamps can link to source contracts;
the package is pseudonymous, not anonymous, and has no formal privacy guarantee.
Read NOTICE.md and UPSTREAM_LICENSE_CC_BY_NC_SA_4_0.txt. Numeric derivatives do not
inherit an unrestricted repository software license. Manuscript licensing is a
separate author decision. Publication awaits rights and exact-file author approval.

There is no TDS release URL/tag/DOI yet. Local success must never be reported as
anonymous public verification. After approved publication, download the actual
immutable asset without login into another empty directory, verify these hashes,
run replay, and retain HTTP, asset, commit and replay receipts separately.
