# DLG-SelectiveStream R02 reproduction

R02 verifies/reaggregates the original `selectivestream_r01_20261011` campaign,
not the five-model seed11 public-input replica. No neural fit or 100,000-event
replay is repeated. The only new measurement is
`selectivestream_r02_20261012_state_probe`, a cheap boundary fixture. Original
training/runtime-v1/offline-v2 identities remain distinct.

## Four levels

1. Published arithmetic: standard-library archive/part/payload/source checks,
   all 25 models, 280 primary test metric rows and separate five-seed exact/Holm
   families. `--full-policy` additionally recomposes all 6,065 policy/control
   rows and 40 MC rows from public CSV/NPZ using scientific dependencies.
2. Retained-input retraining/replay: optional explicit opt-in using disclosed
   R01 bounded edge lists/aliases; always new model/reference/timing identities.
   See [original commands](REPRODUCE.md), level 2. This does not reconstruct
   missing raw-event-to-cache history, edge times or label discovery times.
3. Author-local original verification: all 939 original targets and original
   Parquet/reference/checkpoint/provider inputs. R02 sidecars fail on missing
   inputs; never silently substitute Benchmark. Original hashes checked before/
   after; recorded bootstrap and numeric inputs regenerated in an R02 shadow.
4. Author-local paper: actual clean TeX/BibTeX builds, negative fixtures, every-
   page rendering and private packages. Rendering is not visual/author approval.
   Unsubmitted paper source/PDFs remain excluded from Git and public assets.

## Path-independent public commands

Extract scientific-source ZIP into any new directory and cd to its root. Put
three R01 binary parts and `r01_release.json` in `projects/stream_mc/evidence/`:

```bash
python3 -m venv .verify-venv
.verify-venv/bin/python projects/stream_mc/scripts/r02_public.py
export PYTHONPATH="$PWD/src"
# Requires recorded scientific dependencies, not the empty stdlib venv:
python projects/stream_mc/scripts/r02_public.py --full-policy
```

The verifier atomically assembles the exact ZIP from SHA-checked parts. Outputs
go to `results/public_recomputed_r02`. Actual unauthenticated release-download
and isolated-path checks are recorded in `evidence/access_verification_r02.json`
when executed. An implementer-controlled clean directory is not peer review.
Neural scores were float32: restore this dtype from shortest round-tripping CSV
text before calibration; directly parsing as float64 changes threshold ties.

Recorded environment: WSL distro **Ubuntu**, Ubuntu 26.04.1 LTS, Python 3.14.4,
PyTorch 2.14.1+cu130, CUDA 13.0, PyG 2.8.0.post1, PyGOD 1.1.0, RTX 3090.
Use existing root `.venv_cuda`; do not mutate it. Optional pinned PyArrow is
project-local `.deps` (R01 instructions). Cross-version/device identity not
promised.

## Original-input verification commands

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python projects/stream_mc/scripts/r02_handoff.py
python projects/stream_mc/scripts/r02_run.py numeric
python projects/stream_mc/scripts/r02_run.py systems
python projects/stream_mc/scripts/r02_run.py tests
python projects/stream_mc/scripts/r02_run.py tables
python projects/stream_mc/scripts/r02_run.py public
python projects/stream_mc/scripts/r02_run.py paper
```

Do not run legacy R01 main writers for R02: they target immutable R01 outputs.
R02 uses explicit destination adapters. Actual failed new adapter attempts
(MC key, pressure-header count, generator input, CSV dtype) are retained, not
altered R01 science. Logs contain real exits. Original amendment history stays
in `REPRODUCE.md`, `CHANGELOG_R01.md` and frozen runtime archives, not the main
scientific contribution. Tests remain descriptive after historical test
inspection/shared validation selection; no retrospective preregistration.
