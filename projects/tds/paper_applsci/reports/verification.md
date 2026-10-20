# Verification record

- Command: `/mnt/d/_Work/goat_bank/.venv_cuda/bin/python -m pytest projects/tds/tests/test_revision314.py -q`
- Final outcome: **8 passed**, 29.33 seconds. There are 32 non-failing upstream PyTorch/PyG future/deprecation warnings in the Python 3.14 environment. No test failure is suppressed.
- Tests cover threshold-grouped AP, bootstrap duplicate scores, Holm correction, no-chain invariance, label-independent TGN predict-then-update, weighted-focal/BCE equivalence at gamma zero, operating budgets/tie ordering, and an empty strict-time graph.
- `py_compile` succeeds for the revision modules. Five materialized original-source files exactly match their recorded original Git source SHA-256 values.
- Full numerical validation: `reports/validation.json`, status PASS. Original 88 artifact hashes and 20 checkpoint hashes remain valid; 35 new runs and 3,648 exact source reconstructions pass checks.
- Final manuscript and supplementary LaTeX logs contain no warnings, unresolved references, underfull or overfull boxes. PDFs are 13 and 4 A4 pages.
- Manual visual review: manuscript title page, table/statistics page 7, and MC/reliability page 9. Titles, table values, SD bars, calibration-bin support, author-pending notice and manuscript-status footer are visible, without clipping or overlap in those inspected pages. No claim of full manual inspection of every page is made.
- Actual source-ZIP clean compilation passes for both PDFs in a fresh temporary directory with no cached bibliography. Extracted document text equals that of the corresponding local final PDF. Binary PDF identity is not claimed because creation timestamps/trailer IDs can differ. See `reports/clean_source_compile.json`.
- The two ZIP archives are local author handoff artifacts only; package sizes, CRC tests and SHA-256 values are recorded in `packages/package_manifest.json`. No new public upload or submission was made.
