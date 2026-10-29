# TDS paper workspace

The preserved R314 input is in `applsci/`. The active 315 submission candidate is
`applsci/submission_315/applsci_article.tex`, with its compiled PDF beside it.
The public closure report is `../reports/315/submission_readiness_315.md`.
The entire paper tree stays author-local. The old `paper_applsci` prefix occurs
in sealed historical ZIPs; it is not a second working manuscript.

`current/main.tex` and `current/manuscript_body.tex` preserve the earlier reviewed
source. `current/wrapper_elsarticle.tex`, `current/wrapper_ieee.tex` and
`current/tds.tex` are venue/rendering wrappers around the same body.
Figures, source CSVs, bibliography, the Makefile and existing PDFs moved
together so relative LaTeX inputs remain valid. The TDS `verify` facade checks
only deterministic graph construction and risk fixtures; it does not certify
the paper's full evaluation.
