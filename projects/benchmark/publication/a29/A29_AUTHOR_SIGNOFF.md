# A29 final author approval (sign once, after the final freeze)

Approve only the files whose SHA-256 values are listed in [A29_SUBMISSION_HASHES.sha256](A29_SUBMISSION_HASHES.sha256).
If any listed file changes after signing, the hashes change and the approval must be renewed.

The approval covers:

- the manuscript and appendix (Applied Sciences and Preprints.org versions), including the new title;
- authorship, affiliations, author contributions, funding, conflicts of interest and the generative-AI disclosure;
- the Data Availability Statement and the internal-only scope of per-node scores, labels, split masks and node identifiers;
- posting the Preprints.org version under CC BY 4.0, and submission to Applied Sciences, Special Issue C67H3U9QGT;
- the cover letter and the pre-submission inquiry to the editorial office.

| Author | Written approval (yes/no) | Date | Hash manifest SHA-256 approved |
| --- | --- | --- | --- |
| Seong-Su Park — **superseded: hash mismatch** (see note) | yes | 2026-11-27 | [4a561ed83a9da583683a2e443f945b46020a0b50c3b57f14b4330558e5d9e825](./A29_SUBMISSION_HASHES.sha256) |
| Ki-Hyung Kim, corresponding author | | | |

Corresponding-author confirmations before submission (cover letter): exclusive consideration, disclosure of any prior MDPI
submission, APC responsibility (check the current Applied Sciences APC).

**Superseded (recorded 2026-11-26, A30-04).** The Park row above approves `4a561ed8…`, but the SHA-256 of
[A29_SUBMISSION_HASHES.sha256](A29_SUBMISSION_HASHES.sha256) is `16daf5385975ae90efb2bbb927512694ce066ca31e98af3e113f920679a9305a`
(also in `A29_FREEZE.json`), and `4a561ed8…` matches no A29 file. The row is kept as a record and is not a valid approval.
The A29 files are replaced by the A30 release; sign the A30 form (`publication/a30/A30_AUTHOR_SIGNOFF.md`), whose hashes are
filled in by `scripts/a29_finalize.py` and checked by `a29_finalize.py verify`.
