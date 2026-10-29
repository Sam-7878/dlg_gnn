# Current TDS publication revision: 316

Start with [work_report_316.md](work_report_316.md) and the exact gates in
[submission_readiness_316.json](submission_readiness_316.json).
The current manuscript sources are private under `paper/submission_316/`; R315
manuscripts, results and reports remain preserved without regeneration.

| Asset | Location |
|---|---|
| Shared scientific source | `paper/submission_316/shared/sections/manuscript_body.tex` |
| Neutral preprint | `paper/submission_316/preprints/preprint_article.pdf` |
| MDPI article | `paper/submission_316/mdpi_applsci_anomaly/applsci_article.pdf` |
| Source ZIPs and platform metadata | `submission/preprints/316/`, `submission/applsci/316/` (private) |
| Numeric identities/audits and state history | `results/316_dual_submission/` |
| Numeric candidate/manifest | `publication/316/` |
| Author proof forms and QA images | `submission/author_records/316/`, `private/316/` (private) |

The work-order diagram's numeric data/manifests and audit reports are mapped
outside paper/, preserving the author's explicit paper/data separation. LaTeX
table fragments/figures and source metadata remain paper inputs; raw results,
checkpoints, audit bundles and author approvals are not copied into paper/.

Active target: Applied Sciences Special Issue “Machine Learning and Its Application
for Anomaly Detection”, after actual Preprints.org posting and verified DOI/status.
The reported deadline2026-12-31 is distinguished from the inaccessible live portal.
Local builds/replay PASS; final author, rights, public-release and portal facts
remain separate. No training, public push, posting or submission was performed.

From the repository root using the existing environment:

```bash
python projects/tds/scripts/revision316/freeze.py
python projects/tds/scripts/revision316/build.py
python projects/tds/scripts/revision316/qa.py
python projects/tds/scripts/revision316/numeric_candidate.py
python projects/tds/scripts/revision316/finalize.py
```

Run dependent commands sequentially. Private paper source is required for
author-local builds; public numeric replay needs only its separate archive.
