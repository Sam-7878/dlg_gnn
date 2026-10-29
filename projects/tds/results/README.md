# TDS numeric results

Future TDS tasks write numerical outputs under this directory, in a named task
folder. They do not write data into `paper/` or the former `projects/results/tds/`.

- `main_final_v2/`: frozen historical primary panel, unchanged.
- `314_applsci/`: preserved seven-arm experiments, histories, numeric tables,
  corrected statistics and original audit inputs relocated from paper/applsci.
- `315_applsci/`: identifier-reduced public prediction vectors, AP checks,
  input hashes, source lineage and task 315 derivatives.
- `316_dual_submission/`: R315 input seal, unchanged numeric identities, final
  artifact hashes and immutable platform-state history. No new training.
- `historical_reviewed_draft/data/`: three earlier figure-input CSVs moved from
  paper/current, preserved without changing their values.

Git-eligible numeric outputs include reduced score arrays, aggregate statistics,
training histories, protocol/scaler metadata and hashes. Identifier-bearing
original vectors, raw-source audit tables, checkpoint tensors and private audit
screenshots remain ignored. Public reports are under `projects/tds/reports/`.
Unsubmitted manuscript/source ZIPs and author records remain private. The upstream
GoG license and publication-candidate redistribution review still apply; local
Git eligibility is not a claim that a public release has already been published.
