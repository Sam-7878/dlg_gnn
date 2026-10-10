# A08 shared API and descendant project impact

The hierarchical graph-target interface is unchanged. The new contract-node adapter is a separate benchmark path; the static all-population scaler and random node masks are not adopted for temporal Stream/TDS tasks.

Actual verification: 15 existing graph-level unit tests passed in the earlier 32-test adapter suite. Seven additional stateful-stream replay and temporal-integrity tests pass. The source files in `audit/family_impact.json` match the pre-A08 Git HEAD exactly.

Stream replay imports StreamingDataset/FraudDataset and its source-label/global-graph interfaces. It does not import the new A08 node adapter, raw builder or score evaluator. The temporal split, rolling-origin and temporal-leakage modules remain unchanged. These API regressions do not certify their old scientific inputs or results.

The earlier hybrid results and first unqualified Level2 migration cannot be imported as corrected evidence into DLG-GNN, Stream or TDS. Downstream empirical reuse requires explicit source-feature/relation/label and temporal fit-population audits against each project's input manifest. Current historical family outputs remain preserved, with no claim of family submission readiness or automatic expansion to the new raw-contract population.
