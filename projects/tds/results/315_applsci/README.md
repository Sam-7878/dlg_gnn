# TDS task 315 numeric outputs

This is the author's task-specific result location, now under `projects/tds/results/`.
The immutable R314 inputs
remain in their original frozen locations. `input_hash_manifest.json` seals their
identities before bounded 315 edits; primary and supplemental groups are distinct.

`predictions/` contains 65 identifier-reduced, frozen-order test arrays. Float
round-trip and original ID/time/label alignment are verified before replacing
identifiers by row IDs. `public_prediction_lineage.json` binds the derivatives to
their original CSV hashes. The AP tables and representative statistics replay are
recomputed from existing outputs, without new training. Numeric derivatives retain
GoG attribution and noncommercial/share-alike redistribution review requirements.

See `projects/tds/publication/315/` for the actual local candidate inventory and
license review, and `projects/tds/reports/315/` for audit and readiness results.
These local files have not been published by the agent. Raw data, checkpoints,
manuscript sources and private author records are excluded from this directory.

R314 experiment files previously in paper/applsci were relocated to
`projects/tds/results/314_applsci/`. The earlier `projects/results/tds/315_applsci/`
location is superseded. `publication/layout/path_migrations.json` preserves path
lineage without changing the original input seal.
