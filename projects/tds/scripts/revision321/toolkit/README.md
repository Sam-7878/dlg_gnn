Install from this directory with `python -m pip install .`.

```sh
tds-audit ordering example.csv
tds-audit cutoff example.csv
tds-audit counts example.csv --features retained_nodes
```

The ordering output lists train, validation and test after time and sample-ID
sorting. Cutoff exits nonzero for a feature input timestamp after its cutoff.
Counts fits on the train split and evaluates on test. For tie-aware score
replay, use `tds-audit ap scores.csv` with `label` and `score` columns.
