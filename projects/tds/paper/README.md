# TDS paper workspace

Current final candidate: `submission_318/`, identity
`FINAL_CANDIDATE_318_RECONCILED`. Common source is in its `shared/` directory;
the venue source trees have byte-identical scientific body, Bib, declarations,
tables and figures. Frozen R317 science is preserved; R318 improves checklist
layout and clarifies pseudonymous identifiers/reproduction scope.
Delivered articles/source ZIPs/Supplements and metadata are in private
`../submission/{preprints,applsci}/318/`. Each article is 13 pages; each Supplement
is 4 pages. Current hashes: `../reports/318/final_candidate_ledger.md` and
`../publication/318/SHA256SUMS_FINAL.txt`. Author authority is the user-designated
R317 `author_confirmation_from_a16.json`; exact-file unsigned forms are in
`../submission/author_records/318/`. Public report: `../reports/318/README.md`.
CSV arrays/experiments stay in `../results/317/`; R318 audits stay in `../results/318/`.
Use `../scripts/revision318/` for this bounded closure. R317 generation scripts
target the preserved older candidate; they are not the current submission writer.
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
