# TDS paper workspace

Current revision: `submission_317/`, with one canonical scientific source
`shared/manuscript_body.tex`, `shared/references.bib`, and `shared/declarations.tex`.
Edit those shared sources and run `shared/prepare.py`; then use
`python projects/tds/scripts/revision317/build.py` from the repository root.
Preprints article: `submission_317/preprints/preprint_article.pdf`.
MDPI article: `submission_317/mdpi_applsci_anomaly/applsci_article.pdf`.
Each has its own four-page Supplement and source ZIP. CSVs, arrays and experiments
are in `../results/317/`, not paper/. Author records/source deliveries are in the
ignored `../submission/*/317/` paths. Public work report: `../reports/317/README.md`.
The older source trees below are preserved history, not the active manuscript.

The preserved R314 input is in `applsci/`. The active 315 submission candidate is
`applsci/submission_315/applsci_article.tex`, with its compiled PDF beside it.
The current public report is `../reports/315/venue_split/README.md`.
The entire paper tree stays author-local. The old `paper_applsci` prefix occurs
in sealed historical ZIPs; it is not a second working manuscript.

Current workflow: edit private `common/315/`, then generate both
`preprints/submission_315/` and `applsci/submission_315/`. Source ZIPs and approval
records are outside paper/ in private submission areas; CSVs, prediction vectors
and experiment artifacts are in `projects/tds/results/<task>/`. Current report:
`projects/tds/reports/315/venue_split/README.md`. The original R314 manuscript
remains here as historical source, but its numeric experiments were relocated.

`current/main.tex` and `current/manuscript_body.tex` preserve the earlier reviewed
source. `current/wrapper_elsarticle.tex`, `current/wrapper_ieee.tex` and
`current/tds.tex` are venue/rendering wrappers around the same body.
Figures, source CSVs, bibliography, the Makefile and existing PDFs moved
together so relative LaTeX inputs remain valid. The TDS `verify` facade checks
only deterministic graph construction and risk fixtures; it does not certify
the paper's full evaluation.
