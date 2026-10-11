# DLG-SelectiveStream / StreamMC

Active revision: `DLG-SelectiveStream-R01`, run `selectivestream_r01_20261011`.
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
The release candidate is `evidence/r01_numeric_evidence.zip`; its companion
JSON records exact bytes and hashes. This is a **local** release candidate, not
an already uploaded or publicly hosted release.

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
duplicated into Git. Nothing here authorizes pushing or submitting externally.
