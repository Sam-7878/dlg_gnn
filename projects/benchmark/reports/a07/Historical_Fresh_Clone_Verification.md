# Historical checkout records; superseded visibility policy

# A07 fresh-checkout reviewer test

A new local Git clone was created at `/tmp/dlg_gnn_a07_fresh` from the repository (no working-tree files copied). The tested commit was `429a8407f7435b6b93cf0275be091e7a845d8b10` on Linux WSL2 kernel 6.18.33.2. The integrity-only command used system Python 3.14.4; paper regeneration used the author's `.venv_cuda` Python 3.14.4 with SciPy 1.18.1.

| Command in clean checkout | Exit | Observed result |
|---|---:|---|
| `python3 scripts/reproduce_project.py --project benchmark --mode verify` | 0 | 645 ZIP payload hashes, 13 graphs, seven models, 80 supported pairs, 435 approved successes, archived equivalence PASS; 0.15 s |
| `/mnt/d/_work/goat_bank/.venv_cuda/bin/python scripts/reproduce_project.py --project benchmark --mode paper` | 0 | 44 cited references, four Holm rows, 14 evidence claims, automatic consistency gate PASS; 2.71 s |
| `python3 scripts/reproduce_project.py --project {dlg_gnn,benchmark,stream_mc,tds} --mode verify` (four invocations) | 0 each | All integrity checks pass. Optional scientific smokes skip under system Python without scientific packages. |

The clean-checkout paper output hashes at the tested commit were: A07 TeX `a98a577f950f5f8452f454c45bac22c7e542189a9161f14a4506a1a846d23cf4`, bibliography `1c75476b33fd8b8f63de870a87f69339e69326664c9f5705d0eae22cfa49bc95`, pairwise CSV `51daa126c05861518b169b5553ad203b0eeed9052e90884456df206e49aa046b`, alert-budget CSV `7d64d0733fd1090890aafffa5934fde09b483e546762d0a780cf86cf3c0cd131`. Subsequent generator edits add the crypto-data access limitation and separate F1 supplement; the final local candidate will be tested again at its project-scoped tag before publication.

This fast test proves integrity and paper arithmetic from frozen results. It does not redownload third-party data or re-execute the neural benchmark; the three constructed crypto graphs do not currently have a documented public download.

## Project-layout migration check — 2026-10-06

A separate shallow, no-local clone at `/tmp/dlg-layout-dGLKgD/repo` checked out
commit `2ad775672a51e3b832fe004a55be107366b16994` (WSL2, Python 3.14.4,
SciPy 1.18.1). The clone occupied 118 MB. It omitted local build caches,
checkpoints, offline wheels and the 2.9 GB pre-rename environment backup.

| Command | Exit | Result |
|---|---:|---|
| Four `/mnt/d/_work/goat_bank/.venv_cuda/bin/python scripts/reproduce_project.py --project <name> --mode verify` invocations | 0 each | Benchmark: 35 current paper hashes and 645 frozen ZIP payload hashes PASS; DLG/StreamMC/TDS source and deterministic scientific fixture checks PASS |
| `/mnt/d/_work/goat_bank/.venv_cuda/bin/python scripts/reproduce_project.py --project benchmark --mode paper` | 0 | 44 references, four Holm rows, 14 claim links, consistency PASS; 2.93 s |

Paper output hashes: A07 TeX `97b8b89bbdacd3d029af31f9e663f5c55d75e29aeae7482ad998a8a1b855ff46`, F1 TeX `0e541810ad7380c409b74b81649fb3cbd529ec4d146f9b7057e9c232bca11055`, bibliography `1c75476b33fd8b8f63de870a87f69339e69326664c9f5705d0eae22cfa49bc95`, pairwise CSV `51daa126c05861518b169b5553ad203b0eeed9052e90884456df206e49aa046b`, alert-budget CSV `7d64d0733fd1090890aafffa5934fde09b483e546762d0a780cf86cf3c0cd131`.
