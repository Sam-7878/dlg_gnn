# R01 reproduction levels

Run from the `dlg_gnn` repository root, inside WSL distribution **Ubuntu**.
The recorded environment is Ubuntu 26.04.1 LTS, Python 3.14.4,
PyTorch 2.14.1+cu130, CUDA 13.0, PyG 2.8.0.post1, PyGOD 1.1.0;
training and replay select CUDA device 0, RTX 3090 24 GB. `environment.json`
and `environment.lock` record actual versions; settings are not a promise of
bitwise equivalence on other hardware.

```bash
cd /mnt/d/_Work/goat_bank/dlg_gnn
source /mnt/d/_Work/goat_bank/.venv_cuda/bin/activate
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

From Windows, prefix commands with `wsl.exe -d Ubuntu --cd
/mnt/d/_Work/goat_bank/dlg_gnn`; do not launch scientific work in Windows Python.
Serialize WSL starts and GPU campaigns.

## 1. Public integrity and arithmetic (standard library only)

```bash
python projects/stream_mc/scripts/r01_public.py --mode verify
python projects/stream_mc/scripts/r01_public.py --mode tables
```

Checks: ZIP hash, safe unique paths, exact manifest membership and payload
hashes, 25 model identities, source-only target exclusion, unique paired
contract keys, AP (grouped ties), confusion counts/F1/precision/recall/MCC,
and exact two-sided McNemar with Holm family of five new seed tests. The five
historical R4 paired tests are independently recomputed as a separate family.
Numeric regeneration goes to `results/public_recomputed_r01`, not the frozen
release. These checks do **not** independently validate provider label truth,
missing edge/label availability, author approval, or deployment safety.

## 2. Disclosed bounded-input retraining (GPU; explicit opt-in)

Install the pinned optional project-local PyArrow dependency as shown in
section3 before retraining: output predictions are Parquet even though the
disclosed bounded inputs themselves use CSV/NPZ.

```bash
python projects/stream_mc/scripts/r01_full_reproduce.py --confirm-full
# Smaller audit: one seed across two pooled backbones and three LOCO fits
python projects/stream_mc/scripts/r01_full_reproduce.py --confirm-full --seed 11
```

The archive contains flat bounded directed edge lists, node counts, split
metadata, labels/cutoffs, stable ordered aliases and a transformation manifest.
It does not use torch pickle for graph inputs. Graph/node/edge order and label
support are preserved; float32 degree features are rebuilt. Original
lexicographic reference tie order is preserved by checked zero-padded aliases.
Reference file and selected-ID hashes change with that identity transformation;
it is not falsely called the original run. Outputs use
`selectivestream_r01_public_input_reproduction` in a separate directory.
Compare scores/counts under a declared tolerance, not model ZIP bytes (training
time metadata can differ). This reproduces retained snapshots, not the
unavailable historical raw-event-to-cache construction.

The dataset-derived inputs retain provider attribution and CC BY-NC-SA 4.0
identification. Consult upstream terms; pseudonyms are linkable hashes, not a
claim of anonymization. Raw account mappings/provider transfer ZIPs are not
mirrored. Provider: https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs

The same archive includes pseudonymous retained event inputs and numeric frozen
reference tensors. Optional replay reconstructs alias model/reference artifacts
under a new identity; it does not pretend the aliased file hash is original:

```bash
python projects/stream_mc/scripts/r01_public_replay.py --confirm-full \
  --population prefix --policy margin
python projects/stream_mc/scripts/r01_public_replay.py --confirm-full \
  --population long --policy full
```

The long command really processes100000 events; new timing values are expected.
Replays go to `results/public_alias_replay`, never the approved R01 directory.

## 3. Original author-local campaign (no missing-input fallback)

The fixed config references retained `bounded_graphs.pt` and
`raw_events_100000.parquet` under `archive/stream_mc/results/sci_v3_submission`.
Exact identities are in `execution_plan.json` and `data_audit.json`.
They are local scientific inputs, not hybrid graphs. If unavailable, fail;
do not invent source timestamps or silently use Benchmark results.

Parquet support was added project-locally without modifying the shared venv:

```bash
python -m pip install --only-binary=:all: --no-deps \
  --target projects/stream_mc/.deps -r projects/stream_mc/requirements-parquet.txt
python projects/stream_mc/scripts/r01_inventory.py
python projects/stream_mc/scripts/r01_data_audit.py
python projects/stream_mc/scripts/r01_train.py --campaign all
python projects/stream_mc/scripts/r01_campaign.py --stage all
```

Import the project common module before pandas for local optional PyArrow.
Successful models record executed training source hashes; four critical files
are frozen in `training_sources.zip`. Resume requires config and completed
artifact hashes to match. Changed scientific config needs a new run identity.

Campaign `all` runs source freeze, tests, negative fixtures, raw statistics,
stress/restart, 540 actual offline budget timings, 15 prefix jobs and three
100,000-event jobs, followed by numeric figures/tables and clean TeX builds.
Prefix/long jobs use separate OS processes. Direct `r01_replay.py --mode all`
is **not** that separate-process repetition protocol. Input preload and source
population counters are distinct from bounded core state.

Post-timing closure runs `r01_campaign.py --stage final`: checkpoint identity
and fallback-field amendments, tests, actual nonzero negative CLI fixtures,
the three actual crash fixtures,90 matched-control timing runs,180 selected
pipeline parity checks, event-source/day-block diagnostics, numeric inputs,
three clean paper builds and all-page rendering. V1 measured runtime sources
are preserved separately; amended code is not retroactively labeled timed.

Offline-v1 omitted the requested deterministic/thread setup. After an actual
default-versus-deterministic diagnostic, `--stage repair` reexecutes540 batch
measurements with explicit `_offline_v2` measurement identity, then numeric
audits/generation/build/render. `runtime/offline_v2` is the active frontier;
root `runtime/budget_*` is superseded diagnostic history, not overwritten.
V2 completion verifies config/source/artifact hashes before a resume and does
not describe verified existing measurements as a new timing run. Trained
models and the already deterministic prefix/long replay are not retrained or
retimed by this repair. Source archives identify v1 and v2 separately.

## 4. Private paper build and publication packaging

```bash
python projects/stream_mc/scripts/r01_generate.py --paper
python projects/stream_mc/scripts/r01_validate.py --mode fixtures
python projects/stream_mc/scripts/r01_validate.py --mode paper
python projects/stream_mc/scripts/r01_release.py
python projects/stream_mc/scripts/r01_validate.py --mode release
```

Common science is in `paper/current/r01/common`; neutral preprint and
free-format journal wrappers share it. The supplement has its own references.
Clean builds discard old auxiliary artifacts, invoke BibTeX, reject undefined
references/citations and overfull boxes, and emit PDF hashes. All-page visual
review is separately recorded, never inferred from compiler success.

Public bundles exclude paper sources/PDFs. Private preprint/journal packages
include source, bibliography, figures, generated tables and separate PDFs.
Manifest excludes itself/checksum sidecars to avoid cyclic hashes. PDF metadata
may change on recompilation; scientific numeric inputs remain frozen.

## Limitations and approval

Public arithmetic and a clean local copy are implementation-side verification,
not an independent external reviewer or a successful public re-download.
No hosted release URL, paper DOI, peer review or acceptance is asserted for R01.
Funding/author identifiers come from the user-supplied Benchmark manuscript;
approval of Benchmark does not approve this exact StreamMC version.
Gate P/J remain HOLD until exact-version author approval and publication steps.
Submission-time official instructions/deadline must be rechecked; this run
did not submit, publish, push or email anything.
