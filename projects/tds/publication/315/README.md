# TDS 315 publication candidate

Status: **local staging, unpublished**. Proposed tag: `tds-315-submission-2026-10-29`.
The shared repository is https://github.com/Sam-7878/dlg_gnn. A repository URL and
the Benchmark releases do not establish a public TDS release.

Current storage: scientific inputs in `projects/tds/results/314_applsci/` and task
315 outputs in `projects/tds/results/315_applsci/`. Manuscripts are separately
generated for Preprints.org and Applied Sciences. The latest venue target and
layout supersede earlier 315 metadata; see `projects/tds/reports/315/venue_split/`.

The generated candidate contains 65 identifier-reduced, aligned test prediction
CSVs (30 primary, 35 supplemental), scalar tables, corrected statistics, source
hash lineage, scientific implementations and the eight scientific invariant
tests. No TeX/Bib/PDF, manuscript writer, raw transaction record, contract address,
checkpoint or author approval record is included. Numeric vectors retain the
original labels, score values and ordering, using zero-based row IDs. They are
research evidence; identifier reduction is not a formal anonymization guarantee.

After extracting `TDS_315_numeric_candidate.zip` into an empty directory:

```bash
python replay.py .
python replay.py . --statistics
python -m pytest tests/test_revision314.py -q
```

`replay.py` needs NumPy, pandas and scikit-learn; the invariant tests additionally
need PyTorch/PyG. Exact executed versions are in `environment.json`. The statistics
option replays all three 10,000-replicate protocols for the fixed MC-ensemble vs
deep-ensemble comparison, including tied-score AP. It is not a full replay of all
13 comparisons. All seven supplemental mean/sample-SD AP pairs are recomputed.

The upstream contract Category 0 is mapped to the positive class. Contracts, not
transactions, are the labeled unit. The terminal held-out panel has 3,648 samples
and 107 positives over 18 h 57 min 10 s. Full-history degree inputs precede the
128-edge topology cap. See `dataset_protocol.md` and `reproduction_levels.md`.

Source transformations are listed explicitly. The historical row-wise AP helper
is preserved for provenance; replay replaces it with threshold-grouped AP. The
eight invariant test bodies are unchanged. The metric-only `outputs.py` excludes
the manuscript writer. Source changes never replace frozen scientific outputs.

Before publication, obtain author redistribution approval, review the manifest,
commit the approved public subset, choose an unused immutable tag and verify the
exact download anonymously. Then record its actual commit, URL and archive hash.
No push, release, DOI assignment or journal submission is performed by staging.
