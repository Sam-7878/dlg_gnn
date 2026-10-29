# R316 numerical evidence — local candidate

`TDS_316_numeric_release_candidate.zip` is manuscript-free and locally cold-tested.
Its exact SHA-256, source archive identities, file hashes and unchanged65 vectors
are recorded in release_manifest.json. The originally reviewed R315 archive, the
relocated R315 derivative and this candidate are distinct; their score payloads
match exactly. Only documentation and publication-gate code/tests are added.

After fresh extraction run `python replay.py . --statistics` and
`python -m pytest tests -q`. The final archive has18 tests: eight scientific,
three R315 guards and seven R316 guards. Full GPU retraining is not performed or
guaranteed. Source/packed identities and public replay are different levels.

Raw GoG transactions, addresses, checkpoints, manuscript source/styles/logos and
author records are excluded. Software license and upstream NC-SA data terms are
retained without automatic relicensing. Obtain explicit rights and exact-package
approval before publishing. A real immutable TDS tag/commit/URL and anonymous
download/replay are still required. No push, tag or release was performed.
