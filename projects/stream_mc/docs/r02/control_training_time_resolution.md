# Light-head fitting time: ACCEPTED_LIMITATION

Original head fitting wall time was unrecorded: `NA (unrecorded)`. No estimate
or new timing is relabeled original. R02 takes the work order's scope-limited
closure path: **no training-efficiency/total-training-cost superiority claim**.

All 90 control runs pair scores and costs from the same actual execution;
three repeats averaged within each of five seeds. Active path parameters are
not resident harness weights. Relational MLP/kNN include local encoding/search.
Recorded original neural loop timing is distinct. Better simple-control F1 is
preserved, not removed to favor the primary.

Verification boundary: the 90 measured AP values are actual per-run scalar
records computed by `r01_controls_timing.py`; the batch timing trace does not
persist those per-repeat probability vectors. R02 verifies batch sums/counts,
source provenance, zero prediction disagreement and the original score guard,
then reaggregates same-run scalar AP/cost records. It does not falsely call
measured AP independently reconstructed from a missing probability vector.
Canonical control predictions/calibration/metrics are separately raw-recomputed
for all five seeds. Measured cost is not replaced by canonical cost, and rank-
bit identity is not assumed within the 1e-6 numerical guard.
