# Post-hoc achieved-budget matching

All 90 nearest achieved-fraction pairs are recomputed from active offline-v2's
540 timings. Per seed/frozen margin validation target, choose entropy/random/
learned-benefit's closest achieved test fraction; ties prefer smaller frozen
validation budget. Accept absolute gap <= 0.05; retain unmatched cells; do not
refit thresholds, impute pairs or choose a new primary by test performance.

The main q=0.25 table shows matched/excluded seed counts, fraction gap, AP
difference and both measured ms/contract means over matched pairs only.
Quota matching is not exact latency equality or confirmatory preregistration.
Canonical composition and active selected-execution float-tie AP differences
remain separately identified in `selected_execution_parity.csv`.
