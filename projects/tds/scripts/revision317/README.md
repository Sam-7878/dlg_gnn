# TDS317 reproducible workflow

Use the repository parent's current `dev_env.md`: Ubuntu WSL `Ubuntu`,
`/mnt/d/_Work/goat_bank/.venv_cuda/bin/python`. Recorded execution used Ubuntu26.04.1,
Python3.14.4, PyTorch2.14.1+cu130 and RTX3090 `cuda:1` (24GB). The shared environment
was not upgraded. LightGBM4.6.0 is installed with `--no-deps --target
projects/tds/.deps/317`; the dependency directory stays ignored.

From the repository root:

```bash
python projects/tds/scripts/revision317/freeze.py
python projects/tds/scripts/revision317/lineage.py
python projects/tds/scripts/revision317/protocol.py
python projects/tds/scripts/revision317/horizon.py
python projects/tds/scripts/revision317/train.py --cohort original --arms gat_time,gat_no_time,summary_lr,lightgbm,isolation_forest
python projects/tds/scripts/revision317/train.py --cohort horizon7 --arms summary_lr,summary_mlp,gin_bce,gat_time
python projects/tds/scripts/revision317/analyze.py
python projects/tds/scripts/revision317/tables.py
python projects/tds/scripts/revision317/reports.py
python projects/tds/scripts/revision317/replay.py
```

G1 lineage and the material diagnostic decision precede fitting. The locally fixed
protocol has five seeds and no test-directed search. Horizon construction is
conditional on the selection finding, uses empirical archive support, includes
inactive contracts, and does not infer label discovery time. Do not pair old/new
test populations. Repeat fits resume only from hash-matched completed runs; they
do not overwrite frozen R314–316 sources/results. Raw ZIPs default to the external
`_data/gog_round7_upstream/Token Data`, or set `TDS_RAW_ROOT`.

`replay.py` is a standalone CPU-only point-metric and validation-calibration replay;
it does not require raw data, weights, or private manuscripts. `package.py` stages
only explicitly selected numerical outputs, hashes them, creates a local candidate
and cold-replays it from an empty temporary directory. `build.py` separately packages
private venue sources and compiles each article/supplement from an empty directory.
The private shared manuscript materializer is intentionally excluded from public code.

`reports.py` writes public evidence reports and source identities. `finalize.py`
verifies original seals and exact current PDFs, and writes private approval identities
plus a fail-closed public readiness state. It does not create GitHub releases, author
signatures, DOIs or platform receipts. No publication authorization is implicit.

Author-confirmation follow-up: `python projects/tds/scripts/revision317/confirm_authors.py
--rebuild` applies the recorded private A16-based user attestation, rebuilds the
two venue sources and records fresh hashes. Later `finalize.py` runs preserve that
confirmation instead of recreating the older empty approval record. Confirmed
author statements and independent grant/signature/release evidence are distinct.
The original numerical candidate and its source snapshot remain unchanged.
