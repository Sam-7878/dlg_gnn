# TDS Applied Sciences revision 314

The working manuscript is `applsci_article.tex`, compiled to `applsci_article.pdf`.
`supplement/supplementary.tex` compiles to the supplementary PDF. The reviewed
earlier manuscript in `../paper/current/` is preserved. This revision does not
claim a journal submission, coauthor approval or new public release.

The primary frozen panel in `../results/main_final_v2/` is preserved. New runs are
kept under `runs/`, strict-time diagnostic predictions under `supplement/`, and
audits and readiness decisions under `reports/`.

## Reproduce

From the `dlg_gnn` repository root, use a Python environment containing PyTorch,
PyG, NumPy, pandas, SciPy, scikit-learn, Matplotlib and pytest. The exact local
versions are in `reports/runtime.json` and `supplement/runtime_lock.json`.
The current workspace uses `/mnt/d/_Work/goat_bank/.venv_cuda/bin/python`; this
is an execution path, not a portable installation requirement.

```bash
python projects/tds/scripts/revision314/audit.py
python projects/tds/scripts/revision314/train.py --device cuda:1
python projects/tds/scripts/revision314/strict_time.py --device cuda:0
python projects/tds/scripts/revision314/statistical_audit.py
python projects/tds/scripts/revision314/control_statistics.py
python projects/tds/scripts/revision314/outputs.py
python -m pytest projects/tds/tests/test_revision314.py -q
```

The train script resumes only hash-verified completed runs; it does not silently
retrain or overwrite a successful test panel. The supplemental protocol is
frozen before new held-out evaluation. The fixed primary graph is read from
`archive/tds/data/gog_scimain_v1/graph.pt` and checked against its historical hash.
The strict-time script reads retained official chain ZIPs in
`/mnt/d/_Work/_data/gog_round7_upstream/Token Data/transactions/` in this workspace.
For a different installation, set `TDS_RAW_ROOT` to the official distribution
directory containing `labels.csv` and `transactions/{chain}.zip`. The actual
absolute source paths are recorded in the audit. Historical source needed for
the audit is included under `scripts/revision314/original_source/`, so a source
ZIP does not depend on a full Git checkout. The scripts do not repair the
dangling historical ext4 symlinks.

```bash
cd projects/tds/paper_applsci
latexmk -pdf -interaction=nonstopmode -halt-on-error applsci_article.tex
cd supplement
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The official 18 June 2025 MDPI class is the current distribution actually
downloaded on 2026-10-20. A genuine CTAN soul bundle is included for the missing
local TeX dependency. The original official EPS logos and their existing
byte-source-matched PDF conversions are provided; no fake style stub is used.
Standard TeX dependencies are required. Editable figures are available as
PDF/SVG, with generation code in `outputs.py` and CSV data alongside the paper.

## Interpretation and submission

The new degree-summary MLP obtains AP comparable to the GIN, and chain removal
has a substantial effect. The manuscript therefore makes an evaluation-methodology
claim, not a unique message-passing or novel-architecture claim. The corrected
statistics use threshold-grouped AP in bootstrap samples. All raw original
statistics remain unchanged in the historical panel.

See `reports/final_submission_readiness.md` before submission. Author-specific
declarations, award-agreement verification and the TDS revision's public release
identity remain to be confirmed. A public Git URL for other repository content
does not establish publication of these new artifacts.
