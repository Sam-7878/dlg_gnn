# A24 Submission Readiness Report

**Technical status: A24-PREPRINT-READY / A24-JOURNAL-REVIEWER-STRENGTHENED.**

Scientific release: https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1
Annotated tag: `benchmark-a24-supplementary-2026-10-30-r1`. Scientific commit: `9e7e3a940d7e754b497043cf3959557c82773928`.
Original A23 tag/commit and111 frozen artifacts remain unchanged. Primary scores, seeds, support and rankings remain455successful records =420primary+35external,84/91primary pairs. New results are separate supplementary campaigns.

## Corrections that materially improve the paper

The S1 prose now agrees with the recomputed source:n=8,k=7,Q=29.0357,W=0.6049. Four1,000,000-permutation views reproduce the frozen exceedance counts exactly. All new uncertainty tables and paired-delta figures use sampleSD(ddof=1). S3's best mean rank is accompanied by its non-significant p≈0.1921; view-specific Holm families and the old S5 naming collision are documented.

Old hybrid dimensions were erroneously reused in A23's crypto graph table. Three independent raw builds match the actual score-bound A08 features/edges. EthereumN14464/E95098,BSC7499/48462,Polygon2353/15368,all8features,exact symmetric Euclidean5-neighbor similarity,no transfer-edge union.105selected score/input bindings were verified. Metadata changes do not replace metrics.

The historical diagnostic `ValF1` variable was actually TEST F1 after a restricted validation-quantile search. That column is omitted rather than promoted into a validation claim. A23TableA3 is nowA24TableA5;both PDFs show all last ROC-AUC values. Recommendations distinguish execution from quality; related work and zero-residual numerical conventions are corrected. Both foundation BibTeX entries already matched the user's supplied202609.0848 citation. Only the inherited cover letter contained a different stale DOI;the new cover letter removes that inconsistency. No foundation BibTeX change or benchmark DOI assignment is claimed.

## Targeted supplementary findings

60feature-only evaluations completed;ECOD LANL meanAP0.2007. LANL's historical code used additional log1p/train-set standardization,so historical input reproduction is explicitly NOT MATCHED. Fair within-A24 raw-feature comparisons use new BaseAP0.1022/Aug0.0852. This qualification is explicit;no post-hoc transform change or retraining occurred. Other historical values retain their evidence tier.

25matched Elliptic controls completed:BaseAP0.0687,alignedAug0.1037,Permuted0.1081,Zero0.0859,Base70.0638. Results weaken a node-alignment explanation;they do not prove a capacity/regularization cause.

Four full-budget resource cases and four separately labelled pilots completed. DGraphFin/DOMINANT50epochs allocated3.764GiB,reserved4.088GiB,fit11.54s. Reddit/Base allocated13.968GiB,reserved17.375GiB,fit203.83s. These are separate RTX3090 measurements,not historical speedup replacements. RSS is process lifetime high-water;post-run idle telemetry is not a training power trace. FullCora2708float32 checks passed for loss,node scores,allparametergradients and oneAdamupdate acrossDOMINANT/Base/Aug,atol1e-5+rtol1e-4|reference|;no general trajectory theorem is claimed.

30real-domain alert evaluations reuse10Elliptic control vectors and add20fits.45newGNN training runs excluding resources remain below the55run cap.113new vector sets were independently recomputed into AP/ROC/TP/P/R/lift and actual validation-F1 checks. OptionalG9was not attempted;unsupported historical cells remain unchanged.

## Actual release and reproducibility checks

Both r1 release assets were retrieved anonymously withHTTP200 and exactSHA256matches. The initial empty A24 release is preserved;GitHub immutable-release settings required draft-first asset upload forr1. A23 was not modified. The r1 scientific payload is frozen;later outer closure reports/qualification notes are separately committed and do not change its bytes.

Remote fresh r1 clone:`verify`exit0,`tables`exit0. Public`paper`correctly exits withA24_PRIVATE_INPUT_REQUIRED;this is an availability boundary,not a public paper reproductionPASS. Author-local`paper`actually built both final PDFs,and each source ZIP rebuilt the same PDF byte-for-byte in an empty directory. Numerical/source-location mutations18plus native-PDF last-cell deletions2were rejected. All53pages were rendered and visually reviewed;no finaloverfull hbox or unresolved citations/references remain. Private source/PDF identities are inA24_SHA256SUMS.txt,not uploaded manuscript contents.

## Gate decisions

- G0: PASS
- G1: PASS
- G2: PASS
- G3: PASS
- G4: PASS
- G5: PASS_QUALIFIED_LANL_MATCHED_TO_A24_ONLY
- G6: PASS
- G7: PASS
- G8: PASS_SEPARATE_NEW_PROTOCOL_WITH_HISTORICAL_LIMIT
- G9: OPTIONAL_NOT_ATTEMPTED
- G10: PASS

## Submission handoff and practical limits

Platform deposit/journal submission:NOT_EXECUTED. No new benchmark preprint DOI is asserted. Use the private publisher-neutral A24 PDF/source bundle for Preprints.org first;once assigned,add the actual DOI to the journal disclosure. The Applied Sciences target is the work-order SI C67H3U9QGT. Automated official-page verification returnedHTTP429,so the current deadline and platform forms require the normal author-side check. New A24 coauthor sign-off is not fabricated from the old A09 approval. The paper/source/covers retain author metadata and funding while disclosing the actual AI-assisted coding,orchestration,analysis and writing scope.

G9 and further model suites are not required for this completed scope. Do not extend the campaign or alter the primary frozen leaderboard for submission. Raw data access and author-local new score arrays remain subject to provider terms. A release is not a journal acceptance or platform deposit.
