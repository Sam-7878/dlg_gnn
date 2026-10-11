# Risk and uncertainty specification

For direct D, escalated E, truth y and final decision h:

| Quantity | Numerator | Denominator |
| --- | --- | --- |
| coverage | count(D) | N |
| direct selective error | count(D and h != y) | count(D) |
| direct fraud FNR | count(D and y=1 and h=0) | count(D and y=1) |
| population direct fraud miss | count(D and y=1 and h=0) | count(y=1) |
| false omission | count(D and y=1 and h=0) | count(D and h=0) |

Raw numerator/denominator are retained. Zero denominator is NA. AP is
average_precision_score, not trapezoidal PR-AUC. Positive ranking/recall/F1/
ROC-AUC on zero-positive slices are NA; FP/FPR/specificity remain valid.
Precision is zero with false alerts and NA with no alerts.

Seed SD: sample SD over five fits on the same test. Random router and runtime
replicates are averaged within seed first. Bootstrap: 2,000 class-stratified
paired contract resamples per fitted seed, conditional on that model;
dependence remains. The500-draw UTC-month sensitivity has one block and is
degenerate, not uncertainty evidence. Added post-hoc500-draw UTC-day sensitivity
has few blocks and is exploratory, not reliable dependence correction.
McNemar: exact two-sided correctness discordance, Holm across five new
pooled-GIN tests; historical R4 is a separate family. Analyses follow historical
test inspection and are descriptive, not preregistered F1 significance tests.

Calibration: ten equal-width bins, counts/positives and Wilson intervals;
Brier/NLL/ECE separately all/direct/escalated. Better ECE does not guarantee
routing quality. BSC cases include scores and routing/reference support, not
unmeasured causal explanations. Validation alert quantiles and same-test
prevalence reweighting at1/5/10% are exploratory, not new temporal cohorts.

Corrections/harms compare to local at the SAME final threshold. Unexecuted
selective relational scores are counterfactual full-run results, not actual
selected work and never charged as selected computation.
