# Main Timestamp-GNN Publication Readiness

Gate M: **False**. The existing five frozen checkpoints and 3,648
aligned held-out predictions support a valid five-model deep ensemble, 10,000-resample paired
bootstrap/randomization comparisons, six sequential test slices, and four generated figures.

Deep Ensemble: AUC-PR **0.4902**, ECE **0.1676**, NLL
**0.2316**. MC Dropout Ensemble T=10: AUC-PR **0.4721**, ECE
**0.1285**, NLL **0.1892**.

Gate M remains closed because the frozen packed training/validation dataset and upstream SCI v2
derivative are absent. TGN-style, TGAT-style, and FraudSAGE cannot be trained on the identical split;
validation-only temperature scaling cannot be fitted; the temperature-inclusive reliability figure
and all required paired comparisons are consequently incomplete. No test-set calibration fitting or
cross-dataset baseline substitution was used.
