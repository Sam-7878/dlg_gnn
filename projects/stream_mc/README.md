# DLG-SelectiveStream / StreamMC

Active author-local closeout: `DLG-SelectiveStream-R03` (candidate, not approved).
The public scientific/evidence revision remains `DLG-SelectiveStream-R02`.
Original scientific run: `selectivestream_r01_20261011`, not relabeled replica.
See [R02 reproduction](REPRODUCE_R02.md) and [R02 handoff](docs/r02/HANDOFF_R02.md).
See [R03 closeout](REPRODUCE_R03.md) for new CPU/access diagnostics and the
preprint-first Applied Sciences target; no new fits or GPU replays are required.
StreamMC is the project name; deterministic GIN → margin routing → eligible
GATv2 target-star inference is primary. MC dropout is a separately calibrated
ablation. The scientific scope is **retrospective contract snapshots and
bounded-state systems replay**, not prospective fraud detection or certified
production safety.

Project-owned configurations, runners, protocols, numeric evidence and closure
live here. Reusable modules remain in `src/gog_fraud/streaming`; similarity
relations use `src/gog_fraud/data/level2/relation_builder.py`. No
`*_hybrid_graph.pt` input is used. Original R4 results are not overwritten.

The R01 campaign retrains 25 independent models: pooled GIN and GATv2, and
source-only GIN for each of three held-out chains, each with five seeds.
Provider Category 0 is positive; other provider categories are research
negatives, not independently proven benign. Polygon test has no positives.

Read [REPRODUCE.md](REPRODUCE.md), [CHANGELOG_R01.md](CHANGELOG_R01.md),
[CLAIM_LEDGER.md](CLAIM_LEDGER.md) and [CLOSURE_R01.md](CLOSURE_R01.md).
Git distributes the release candidate as `evidence/r01_numeric_evidence.zip.part001`
and subsequent ordered parts of at most90MiB each. The verifier assembles the
ignored local `evidence/r01_numeric_evidence.zip`; its companion JSON records
exact part/full bytes and hashes. Actual R02 hosted assets and access checks
are documented separately in `evidence/release_manifest_r02.json` and
`evidence/access_verification_r02.json` after publication; local bytes alone
are not an external-access check.

```bash
source /mnt/d/_Work/goat_bank/.venv_cuda/bin/activate
./projects/stream_mc/reproduce.sh verify
./projects/stream_mc/reproduce.sh tables
# Explicit opt-in; expensive, independent outputs, no original results replaced:
./projects/stream_mc/reproduce.sh full --confirm-full
```

`verify` recomputes AP/confusion counts and exact paired statistics from
pseudonymous raw predictions, in addition to ZIP/source checks. `tables`
writes recomputed numeric tables without importing the unpublished manuscript.
`full` reconstructs disclosed bounded graph inputs and retrains under a new
reproduction identity. It does not silently claim bit-identical model files or
reconstruct missing raw edge/label timestamps. `paper` requires author-local
sources and a real TeX/BibTeX installation. `--revision legacy-a07` explicitly
selects the old smoke-check facade, not current paper evidence.

Unsubmitted TeX/Bib/PDFs and private submission ZIPs are excluded from Git.
Scientific code and a curated raw/numeric/input/model archive are reviewable;
address mappings, duplicate large working files and local dependencies are not
duplicated into Git. The user authorized R02 scientific-code/evidence publication
to the designated repository; paper publication/submission and exact-version
all-author approval remain separate OPEN gates.
