# R01 claim ledger

Machine-readable exact claim/scope, artifact hash, command, disposition and reviewer: `evidence/claim_ledger_r01.csv`.

- snapshot_scope: Retrospective snapshots, not strict online detection (retained); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/temporal_audit.json`.
- primary: Five-seed means from paired contract predictions (changed_by_new_run); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/aggregate_metrics.csv`.
- source_only: Target excluded from source-only fit and selection (new); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/experiment_registry.jsonl`.
- statistics: Exact paired correctness and descriptive paired intervals (changed_by_new_run); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/paired_statistics.csv`.
- risk: Denominator-aware direct risk, no formal safety (new); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/risk_counts.csv`.
- polygon: Zero positive support; positive metrics undefined (retained); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/per_seed_metrics.csv`.
- bsc_cases: Descriptive false negatives without causal attribution (new); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/bsc_error_cases.csv`.
- mc: Task-limited dropout ablation, primary deterministic (new); `projects/stream_mc/results/selectivestream_r01_20261011/analysis/mc_metrics.csv`.
- budget: Actual selected execution, not a deep-fraction latency proxy (new); `projects/stream_mc/results/selectivestream_r01_20261011/runtime/offline_v2/budget_runtime.csv`.
- boundaries: Declared resident/cache/queue bounds tested, not whole-process O(1) (new); `projects/stream_mc/results/selectivestream_r01_20261011/stress/bounded_state_audit.json`.
- restart: Local abrupt-file failure scope only (new); `projects/stream_mc/results/selectivestream_r01_20261011/stress/restart_audit.json`.
- build: Actual compiler/visual checks, not author approval (new); `projects/stream_mc/results/selectivestream_r01_20261011/audits/build_audit.json`.

Removed: strict as-of/live safety, production readiness, uniform superiority, Polygon positive detection and generalized MC claims. Approval/publication remain HOLD; see CLOSURE_R01.md.
