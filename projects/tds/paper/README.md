# TDS paper workspace

`current/main.tex` and `current/manuscript_body.tex` are the active paper
source. `current/wrapper_elsarticle.tex`, `current/wrapper_ieee.tex` and
`current/tds.tex` are venue/rendering wrappers around the same body.
Figures, source CSVs, bibliography, the Makefile and existing PDFs moved
together so relative LaTeX inputs remain valid. The TDS `verify` facade checks
only deterministic graph construction and risk fixtures; it does not certify
the paper's full evaluation.
