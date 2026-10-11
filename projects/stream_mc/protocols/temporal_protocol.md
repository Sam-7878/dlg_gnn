# Temporal and label protocol — R01

Chosen branch: retrospective contract-snapshot evaluation plus bounded-state
systems replay. Strict as-of fraud detection and live safety claims are removed.

| Field | Definition and availability |
| --- | --- |
| event_time | Retained replay transfer timestamp; UTC epoch seconds |
| snapshot_cutoff | Retained cached metadata event_end; replay event timestamp |
| feature_max_event_time | Unknown for cached snapshots; latest accepted replay event |
| label_available_time | Unknown; no fabricated interpolation |
| model_fit_data_max_time | Maximum source-train snapshot cutoff |
| calibration_data_max_time | Maximum source-validation snapshot cutoff |
| policy_fit_data_max_time | Source-validation maximum; calibration partition reused |
| reference_cutoff | Each immutable train snapshot cutoff; must be <= target |

Model and calibration maxima exceed some target cutoffs. Retrieval's zero
future-reference violations do not repair those paths. Counts are per model
and summed explicitly across repeated fits, not independent subjects. Cached
edge timestamp lineage cannot be reconstructed from retained topology alone.

Pooled: all chains permitted in train/validation, separate per-chain test.
LOCO: independent local+relational fits on other two chains; target excluded
from gradients, calibrators, fusion, thresholds, routing and reference pool.
No target adaptation. Test labels are only evaluation labels. No chain-qualified
contract overlap exists among inherited partitions. Their chain-specific
chronology is not a global as-of split.

Category0 maps to positive. Other categories are research negatives, not
confirmed benign; unmatched labels are errors, not zero. All cached mappings
were checked against the local provider file. Availability dates are unknown.
Polygon test has zero positives.

Replay sorting: event_time, block_number, transaction_index, stable source ID.
Event IDs are hashed to bounded width. Late resident input is quarantined;
duplicate recognition covers only the resident edge-ID window. No live
finality/confirmation rule is evaluated. Copied contract labels do not identify
each transfer as fraud.

Evidence: `analysis/temporal_audit.json`, `temporal_audit_rows.csv`,
`split_support.csv`, `split_overlap.json`, `data_audit.json`, input hash and
per-model policies. Unknown means unverifiable, not zero.
