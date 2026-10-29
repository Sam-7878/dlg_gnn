# Collection-end and sample-time audit

**Decision: SELECTION_EFFECT_SUPPORTED.** Claims are restricted to sampling-associated
composition in a retrospective archive. No identified censoring mechanism or
prospective label availability is asserted.

All 24,316 contracts were independently streamed from the hash-matched three ZIPs:
268,282,924 rows, all valid under the original endpoint check. Recovered last times
match every packed cutoff; source label Category0 joins one-to-one. First-time
sorting uses the same full population and count sizes, before any new model fitting.
The preregistered descriptive trigger was an absolute >=5 percentage-point
prevalence change in any split; it is not a significance test.

| Sorting | Train | Validation | Test |
|---|---:|---:|---:|
| Last transfer |39.8508%|8.4453%|2.9331%|
| First transfer |14.7935%|71.3189%|56.9901%|

Pooled median observed lifetimes: positive0.670625days; other608.447361days.
Ethereum medians0.524583/717.142940days. These are observed spans, not survival
estimates under a known censoring model. First-time class quantiles, split-specific
first/last/activity/chain summaries and endpoint proximity are in the linked CSVs.

The pinned official README documents End Month2024-02. Empirical maximum is
1709251199, 2024-02-29T23:59:59Z. The inspected collector's generic end_block and
API termination loop do not establish a precise archive stopping policy.
`collection_boundary_time` therefore remains unknown; the observed maximum is
stored separately. Fraud discovery time is unknown for every contract.

Seven-day eligibility retains24,179 and removes137 (112positive/25other).
Thirty-day eligibility retains23,739 and removes577 (422positive/155other).
Inclusion does not require later activity. The new `GoG-317-first7d-v1` rebuilds
all feature degrees and tail128 topology <=cutoff; every sample has >=1 valid
transfer and all24,179 maximum input timestamps satisfy cutoff. Its split is
16,925/3,627/3,627, positives2,507/2,542/2,037. Test prevalence56.1621%.
New and old tests are not paired and have distinct identities.

Public `sample_time_lineage.csv` is identifier-reduced, not formally anonymous.
CSV is used instead of Parquet because PyArrow is absent; no environment upgrade
was required. The address-bearing mapping and source snapshots remain private.
Zero chain-qualified contract overlaps do not establish wallet independence;
cross-chain address-string overlaps are24/30/36 for train--val/train--test/val--test.

Evidence: `results/317/selection_decision.json`, `sample_time_lineage.csv`,
`first_last_ordering_diagnostics.csv`, `lifespan_by_class.csv`,
`pooled_time_quantiles.csv`, `fixed_horizon_eligibility.csv`, `horizon7_*`.
