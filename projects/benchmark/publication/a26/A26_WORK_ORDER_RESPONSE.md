# A26 work-order response

Prepared 6 November 2026 KST for author review. The source and PDF pairs in [A26 submission hashes](A26_SUBMISSION_HASHES.sha256) supersede the A25 submission copies. The scientific release contains 455 successful selected evaluations and 84 of 91 supported primary pairs. Both public integrity and scalar-table replay modes passed again on the pinned A24 checkout. [Replay output](scientific_release_replay.log)

| Work item | Disposition | Evidence |
|---|---|---|
| W-01 | Original S1–S4 MC counts reproduced. Independent samplers and exact finite counts corroborate the four results. Both reference probabilities and the ratio are in Table 5. | [Statistical audit](A26_SCIENTIFIC_AUDIT.md) |
| W-02 | Audited 35 BitcoinOTC plus five DGraphFin AnomalyDAE archived arrays. Corrected the 310 unavailable count in both manuscripts and the access record. | [Access record](A26_EVIDENCE_ACCESS.md) |
| W-03 | Verified all 105 score-bound cryptocurrency inputs and APs. Updated graph types, tensor IDs and historical dimension reconciliation. | [Input audit](A26_SCIENTIFIC_AUDIT.md) |
| W-04 | Refit 60 feature-only evaluations with exact score vector matches and reproduced ten matched Elliptic DLG alert results. The abstract and discussion distinguish AP from top-1% precision. | [Orientation audit](A26_SCIENTIFIC_AUDIT.md) |
| W-05 | The complexity table shows Reddit-Syn DLG-Base measured allocated peak 13.968 GiB and points to the reserved peak 17.375 GiB. Illustrative estimates are labeled in decimal GB. | [PDF review](A26_PDF_QA.md) |
| W-06 | Table 8 separates historical and matched LANL preprocessing and AP results. | [PDF review](A26_PDF_QA.md) |
| W-07 | 200-word abstract starts from the cross-domain motivation and explains execution and alert-budget precision. | [Portal metadata](../../paper/a26/A26_PORTAL_METADATA.md) |
| W-08 | Replaced internal study names in the manuscript prose and tables. The immutable public release identifier remains in the availability statement. The 70-epoch Base control uses global epochs. | [PDF review](A26_PDF_QA.md) |
| W-09 | Separated research use in Methods from editorial assistance in Acknowledgments. | [PDF review](A26_PDF_QA.md) |
| W-10 | Consolidated the contributions into four distinct items. | [PDF review](A26_PDF_QA.md) |
| W-11 | Merged three pairs of panels, validated 280 merged cells in each version and ordered Appendix A1–A5. Current Elliptic interpretation uses the matched controls. | [PDF review](A26_PDF_QA.md) |
| W-12 | Replaced the blanket tail explanation with results for each finite view. The ratios persist with zero ties. | [Statistical audit](A26_SCIENTIFIC_AUDIT.md) |
| W-13 | RTX 3090 seed-42 diagnostics are reported below. Figure 2, Section 5.1 and the complexity table describe the historical causes as unestablished. | [Diagnostic status](gadnr_diagnostic_status.json) |
| W-14 | Corrected P@1%, SNAP link and funding format. Printed and linked both ORCIDs in both variants. Both authors share the Department of Computer Engineering, Graduate School affiliation. | [Portal metadata](../../paper/a26/A26_PORTAL_METADATA.md) |
| W-15 | Replaced unsupported prevalence thresholds with the benchmark test prevalences from Table 1. | [PDF review](A26_PDF_QA.md) |
| W-16 | Cover letter and Editorial Office inquiry are drafted. The expected submission date, email receipt, coauthor approvals, Benchmark DOI and platform receipts are pending. | [Cover letter](../../paper/a26/applsci/COVER_LETTER_A26.md) · [Inquiry](../../paper/a26/SI_EDITOR_INQUIRY_A26.md) · [Receipts](A26_SUBMISSION_RECEIPTS.md) |

## GADNR diagnostic boundary

The seed-42 tests used the archived full-batch Round 5 configuration, its 50-epoch budget and the current locked CUDA environment. The 300-second diagnostic cap was per graph in a fresh process. These attempts are separate from the five-seed support table. A timeout identifies the cap reached on this diagnostic, without an exception or a diagnosis of the original run.

| Graph | RTX 3090 diagnostic | Historical evidence |
|---|---|---|
| Elliptic | Timed out at the 300 s diagnostic cap | Cause unestablished |
| DGraphFin | Timed out at the 300 s diagnostic cap | Cause unestablished |
| Yelp-Syn | Timed out at the 300 s diagnostic cap | Cause unestablished |
| Reddit-Syn | Timed out at the 300 s diagnostic cap | Cause unestablished |

Raw logs and setup attempts are under `local_storage/benchmark/a26`. The initial launch lacked the repository root in the import path. The reported diagnostic statuses come from the corrected launch.

## Author handoff

Review the A26 PDF and supplementary evidence hashes before recording both author approvals. The authors deposit the Benchmark preprint and use its public DOI, version and URL to finalize the journal cover letter and SuSy submission. [Author approval](A26_AUTHOR_SIGNOFF.md) · [Receipts](A26_SUBMISSION_RECEIPTS.md)
