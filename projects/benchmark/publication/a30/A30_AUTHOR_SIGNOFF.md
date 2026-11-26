# A30 author approval

Release A30, frozen 2026-11-26. The SHA-256 values below are written by `scripts/a29_finalize.py`; do not edit them by hand.
Fill in only the approval and date columns. `python a29_finalize.py verify` checks every signed row against the actual files
(hash of the approved list = actual hash, date no later than the day of the check). If any listed file changes, its hash list
changes and the approval must be renewed. Rows that turn out to be wrong are kept and marked "superseded", never deleted.

## Stage 1 — manuscript (both authors, before posting the preprint)

Approves [A30_MANUSCRIPT_HASHES.sha256](A30_MANUSCRIPT_HASHES.sha256): both PDFs, both source bundles and the inquiry to the editorial office.
The approval covers the manuscript and appendix of both versions, including the title; authorship, affiliations, author
contributions, funding, conflicts of interest and the generative-AI disclosure; the Data Availability Statement and the
internal-only scope of per-node scores, labels, split masks and node identifiers; posting the Preprints.org version under
CC BY 4.0 and submission to Applied Sciences, Special Issue C67H3U9QGT.

| Author | Written approval (yes/no) | Date (YYYY-MM-DD) | Approved list SHA-256 |
| --- | --- | --- | --- |
| Seong-Su Park | | | `aef8af5c38348b3ed2fd18275c01dd9764f675a8708ab430313d36a236bcbd29` |
| Ki-Hyung Kim, corresponding author | | | `aef8af5c38348b3ed2fd18275c01dd9764f675a8708ab430313d36a236bcbd29` |

## Stage 2 — cover letter (corresponding author, just before the SuSy submission)

Approves [A30_COVER_LETTER_HASH.sha256](A30_COVER_LETTER_HASH.sha256): the cover letter with the preprint URL, DOI and version filled in, and the
corresponding-author confirmations (exclusive consideration, any prior MDPI submission, APC responsibility).

| Author | Written approval (yes/no) | Date (YYYY-MM-DD) | Approved list SHA-256 |
| --- | --- | --- | --- |
| Ki-Hyung Kim, corresponding author | | | (run `python a29_finalize.py cover-letter` after the DOI is filled in) |
