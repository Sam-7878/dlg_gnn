# R322 execution and replay

Run from the repository root in the documented Ubuntu Python environment.
The GPU runs selected cuda:1 on the RTX 3090 host.

- `unsupervised.py COHORT DETECTOR --seed SEED --out DIRECTORY` runs one fixed PyGOD contract adapter. Run both archived and seven-day cohorts, all three detectors, and seeds 321, 17, 27, 37, 47. Keep score arrays and logs private.
- `tree_grid.py`, `original_neural.py --arm gin_size` and `original_neural.py --arm gin_sum` run the archive controls. `neural_grid.py COHORT ARM` runs one of four validation-only grid panels.
- `counterparty.py` requires the licensed source archives and the original private lineage. It exports only aggregate intervals. Its grouping is a proxy using the dominant retained endpoint within each chain.
- `windows.py` rebuilds eligible one-, three- and fourteen-day cohorts. Then run `window_models.py DAYS ARM` for each new day and each of node_lr, mlp, gin_size. Run `window_summary.py` after all nine fits to add the preserved seven-day comparator and render the figure.
- `replay.py EXTRACTED_PACKAGE` validates the manifest and recomputes ranking and capture metrics. The private manuscript scripts stage, clean-build and hash both candidates. Approval precedes deposit and publication.
