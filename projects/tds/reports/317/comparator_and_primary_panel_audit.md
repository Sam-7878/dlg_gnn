# Comparator and primary panel audit

The primary comparison uses the original matched GIN+BCE, GraphSAGE+BCE, summary
LR and MLP; new GATv2 time/no-time, LightGBM and IF controls use the same test IDs.
Actual feature and loss differences are declared rather than hidden. The separate
archival GIN/GAT/chain-recurrent/focal-SAGE and averaging panel is moved to the
Supplement. In particular, archivalGIN AP0.4648 is never substituted for matched
GIN+BCE AP0.4524 in the paired MLP comparison (AP0.4751).

New GATv2 meanAP time0.311875 vs no-time0.389731; five seeds each, same weightedBCE,
60-epoch ceiling/patience5, hidden48, batch64 and AdamW settings. Removing time also
omits its16 inputs from the classifier head; this is a branch/head ablation, not a
claim to isolate extrapolation causality. Original test normalized float32 time
is[1.0036826,1.0040067], all outside train[0,0.99999976]; initial maximum harmonic
frequency gives about4radians of phase extension.

The former TGN-style model is a **chain-level recurrent-state comparator**:
three48-dimensional states indexed by chain; reset at every epoch and partition;
logit from current summary+previous state; then GRU update from summary+time;
no label input; frozen weights in held-out traversal; no cross-partition warm-start.
It is not original TGN node memory. The executable label-independence test is retained.

All optional lightweight items were EXECUTED: one LightGBM4.6.0 configuration,
unlabeled mixed-train IsolationForest with fixed negative-score orientation,
95 modelwise validation-only TS calibrators, chain-specific AP, train-chain prior,
fixed budgets/tie bounds, and 2,000-draw hour-block sensitivity. No full original
TGN/DyGLib retraining, second horizon grid or test-directed tuning is added.
IF AP0.016989 is reported without reversing scores after seeing labels; probability
metrics are inapplicable. Deterministic LR/LightGBM repetitions have no seed spread
and do not remove sampling uncertainty. All45 new fit records retain test_access_count1.

The finite-seven-day panel has LR AP0.906827, MLP0.958154, GIN0.635791,
time-GATv20.612597 at test prevalence0.561621. This reinforces competitive simple
controls on a different retrospective task, not a paired gain over the old test.

Evidence: `panel_comparability.csv`, `run_registry.csv`, `experiment_preregistration.json`,
`temporal_input_ranges.csv`, `temporal_input_ablation.csv`, `state_semantics_audit.json`.
