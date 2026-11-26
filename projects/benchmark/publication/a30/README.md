# A30 manuscript release

A30 is the corrected release of the A29 manuscript. Text, build date, front matter and sign-off procedure changed; the data
and all numbers are those of A29 (`projects/benchmark/results/a29`, documented in `publication/a29/README.md`).

- Built files: `projects/benchmark/paper/a30/` (two PDFs with their source bundles, editor inquiry, cover letter, portal sheet)
- [Manuscript hash list](A30_MANUSCRIPT_HASHES.sha256): two PDFs, two source bundles, editor inquiry (stage-1 sign-off)
- `A30_COVER_LETTER_HASH.sha256`: written after the preprint DOI is filled into the cover letter (stage-2 sign-off)
- [Sign-off form](A30_AUTHOR_SIGNOFF.md): hashes filled in by `scripts/a29_finalize.py`; check with `python a29_finalize.py verify`
- [PDF quality assurance](A30_PDF_QA.md), [work order response](A30_WORK_ORDER_RESPONSE.md), [receipts](A30_SUBMISSION_RECEIPTS.md),
  [reviewer Q&A](REVIEWER_QA.md)
- `A30_RESULTS_INDEX.json`: SHA-256 of every summary file in `results/a29`; `A30_FREEZE.json`: build record and list hash

Rebuild: set `FREEZE_DATE` in `scripts/a29_common.py`, then `a29_build_manuscript.py --final` → `a29_pdf_qa.py` →
`a29_finalize.py`. A rebuild changes the hashes, so approvals must be renewed (mark old rows "superseded").
