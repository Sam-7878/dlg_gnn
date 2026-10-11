# Frozen method and execution populations

Config: `configs/r01.json`; coefficients/weights/thresholds and hashes are
per model in `policy_fit.json`. Shared implementation, not manuscript-only code.

Primary: two-layer GIN width32, ReLU/dropout0.2, mean+max64 readout/32-wide head;
degree3 log1p input. Five epochs/batch128 Adam0.001 weighted BCE. Freeze local
before two-layer/four-head width32 GATv2 relational training (20epochs/batch256),
input/hidden LayerNorm, ELU/dropout0.2, target linear head. Local GATv2 has
matched width/budget. No early stopping/scheduler. Class weight train neg/pos.
Independent stars use raw local probability plus embedding; calibrated values
are for routing/fusion only.

Exact Euclidean k8 shared retrieval uses cutoff<=target, self exclusion and
deterministic distance/identity ties. Bidirectional target-star similarity edges
are not payments or bridges. No hybrid or Benchmark feature input. References
are immutable training features, potentially larger than the hot cache.

PlattC1 source validation. Fusion w in {0,.25,.5,.75,1}; F1 ties choose largest
threshold. Margin ranks -abs(p1-local_tau), validation higher-quantile budgets
{0,.1,.25,.5,.75,1}; direct/deep share final threshold. q0/q1 forced endpoints
before eligible fallback. Entropy, random3 and validation-crossfit Ridge
benefit controls use same frozen budgets. Achieved test fractions can differ.

MC T1 deterministic; T3/5/8 dropout-only, stateful layers frozen, each T gets
source-validation fit. T1 error AUROC entropy; T>1 population variance.
Empty/nonfinite local abstains; no reference, nonfinite deep or soft timeout
uses calibrated local at the SAME final threshold. Timeout is not cancellation
or an SLA. Cache misses reload immutable numeric references.

Populations:540 batch128 contract policy runs (5seeds×36policies×3timings),
prefix500 after25warm with5process repeats each local/margin/full, and long100000
once per policy. CUDA sync included. Event scoring also includes feature
updates. Decode/sort excluded; logging/checkpoint included in throughput.
Contract AP frontiers never use event timings as their contract cost.

Active batch measurement: `runtime/offline_v2`,540 actual new runs with
deterministic algorithms and one CPU thread. V1 omitted that setup in its
offline process and is retained as superseded history; train/event replay
already enabled it. An actual default/deterministic diagnostic exposed local
embedding/kNN sensitivity. We do not repair score mismatch by loosening the
guard or substitute old costs for the reexecuted v2 frontier.

Resident cap5000, node/edge128, TTL90event-days, cache10000/64MiB, queue512 retry,
trace buffer1000. Immutable arrays/index, GPU/model, source preload, input
accounting and checkpoint copies separate. Logical payload budget is not exact
RSS; RSS/GPU allocated/reserved measured. No whole-process O(1) assertion.
Payload accounting is a fixed-width ASCII field budget for tested identifiers,
not an arbitrary Unicode/Python allocation bound: identifier limits count
characters. Reference search embeddings and cKDTree data share memory; their
reported component byte counts must not be added as unique resident bytes.
Synthetic pressure is not fraud evidence. Restart covers abrupt local exit
before/during/after fsynced replacement; distinguish score tolerance, bits,
labels, state/cursor and replay loss. No power-loss/distributed certification.
