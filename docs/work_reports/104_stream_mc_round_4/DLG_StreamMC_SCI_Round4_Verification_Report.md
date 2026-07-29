# DLG-StreamMC SCI Round 4 Verification Report

**Dataset v2: PASS**  
**Leakage: PASS**  
**Real PyGOD: PARTIAL (6/8 full 5-seed models)**  
**Main 5-Seed: PASS**  
**MC Sensitivity: PASS**  
**Routing: PASS**  
**Calibration: PASS**  
**100k Streaming: PASS**  
**Temporal: PASS_WITH_RESTRICTIONS**  
**Cross-Chain: PASS_WITH_RESTRICTIONS**  
**Statistics: PASS**  
**Paper-Eligible Records: 340**  
**Paper Revision Gate: OPEN_WITH_RESTRICTIONS**

## Executive decision

Round 4 produced real prediction artifacts and opens quantitative paper revision with restrictions. The result is not an unrestricted claim of superiority. DLG/StreamMC main, MC, routing, calibration, 100k resource, rolling temporal, held-out cross-chain and statistical experiments ran from the leakage-audited SCI v2 dataset. Legacy numeric mapping remains Appendix-only.

## Experiment inventory

- Unique main records: 200 (DLG 80; full PyGOD 120)
- Unique MC records: 140 (4 scopes × 5 seeds × 7 T values)
- Routing/calibration: 16/16 records
- Rolling temporal: 15 folds
- Cross-chain: 18 train/test/feature settings
- Streaming/resource: 5 independent 100k-event LPP variants, all with 100% coverage and OOM 0
- Statistical tests: 11 records
- In-scope tests: 120 passed, 0 failed, 0 errors
- Unresolved CRITICAL issues: 0

## Main quantitative snapshot (pooled, mean of five seeds)

See `results_sci_v2/tables/main_metrics_ci.csv` for 95% CIs. Pooled DLG-Full-Fusion ROC-AUC is approximately 0.840; the corresponding PR-AUC is approximately 0.392. These values are generated from raw prediction vectors, not demo constants.

## Restrictions

- Polygon fixed temporal test has 0 fraud samples; ROC-AUC/PR-AUC are undefined.
- Polygon rolling fold 5 has 0 fraud samples.
- DONE is transductive-only and is not directly comparable to the inductive temporal protocol.
- GAAN is incompatible with PyGOD 1.1.0 on the SCI feature graph; GUIDE exceeded the 180 s graphlet budget.
- Real analyst time was not measured; routing claims are simulated queue-volume claims only.
- Leakage-safe out-of-fold PyGOD train scores for legacy-score augmentation were not generated.

## Gate interpretation

`OPEN_WITH_RESTRICTIONS` permits quantitative Results/Abstract revision only with all restrictions disclosed. It does not permit claims about measured analyst productivity, complete legacy reproduction, or defined Polygon fixed-holdout fraud metrics.
