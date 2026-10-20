# TDS paper workspace

The Applied Sciences revision is now in [../paper_applsci/applsci_article.tex](../paper_applsci/applsci_article.tex),
with [its compiled PDF](../paper_applsci/applsci_article.pdf) and
[submission-readiness report](../paper_applsci/reports/final_submission_readiness.md).
It supersedes the reviewed draft scientifically and includes the Round 314 audits and controls.

`current/main.tex` and `current/manuscript_body.tex` preserve the earlier reviewed
source. `current/wrapper_elsarticle.tex`, `current/wrapper_ieee.tex` and
`current/tds.tex` are venue/rendering wrappers around the same body.
Figures, source CSVs, bibliography, the Makefile and existing PDFs moved
together so relative LaTeX inputs remain valid. The TDS `verify` facade checks
only deterministic graph construction and risk fixtures; it does not certify
the paper's full evaluation.
