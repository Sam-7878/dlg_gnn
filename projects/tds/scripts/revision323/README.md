# R323 TDS evaluation

R323 uses the preserved R322 split, graph caches and matched validation search.
Run from the repository root with the CUDA Python environment.

- `refits.py DAY ARM --seed SEED --device cuda:1` extends the 1-, 3-, and
  14-day window controls for `gin_size` and `mlp` with seeds 17, 27, 37, 47.
  The seven-day comparators and the earlier seed-7 controls are preserved.
- `refits.py 1 ARM --cohort COHORT --seed SEED --device cuda:1` confirms
  the short-grid-selected setting with a 60-epoch ceiling and patience 5.
  The day argument is unused when a cohort is supplied. Run for
  `original` and `horizon7`, both neural arms and seeds 7, 17, 27.
  The runner writes scores and validation history to the local private tree.
- `check_diagnostics.py REPLAY_ROOT` verifies the prevalence-matched draws,
  their top-1% budget statistics and the retrospective activity strata.
  The replay bundle also contains the independent score and hash verifier.

The release review candidate preserves test order for exact paired replay.
Aggregates under `results/323` exclude individual contract scores and IDs.
