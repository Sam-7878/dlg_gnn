# Seven-day GNN/MLP gap: R319 evidence

The independent raw source scan reproduced all 24,316 original and 24,179 seven-day graph tensors exactly (zero mismatches), including edge multiplicity/order, labels, chain and cutoff. Input times never exceed the registered cutoff. This rules out a detected R317 source-to-tensor graph-builder discrepancy as the explanation for the frozen inputs; it does not prove all possible implementation bugs absent.

Five frozen seeds on the same seven-day split give:

- chain_prior: AP 0.6713, ROC-AUC 0.6832
- gat_time: AP 0.6126, ROC-AUC 0.5674
- gin0: AP 0.6418, ROC-AUC 0.5621
- gin_size: AP 0.9445, ROC-AUC 0.9110
- gin_sum: AP 0.9529, ROC-AUC 0.9220
- lightgbm: AP 0.9632, ROC-AUC 0.9445
- mlp_full: AP 0.9582, ROC-AUC 0.9389
- mlp_no_size: AP 0.9054, ROC-AUC 0.8883
- mlp_topology_only: AP 0.9548, ROC-AUC 0.9313
- negative_only_iforest: AP 0.6068, ROC-AUC 0.6268
- summary_lr: AP 0.9068, ROC-AUC 0.9012

GIN-0 has mean AP 0.6418; adding explicit cutoff-admissible graph size reaches 0.9445, while sum pooling reaches 0.9529. The 24-feature MLP reaches 0.9582; removing explicit counts/density reaches 0.9054. The topology-summary MLP includes counts/density and reaches 0.9548. A single retained-node-count LR reaches AP 0.8943. The output strongly supports explicit size/readout information as a major contributor to the original GIN/MLP gap. It does not isolate the only causal factor: sum pooling changes capacity, tabular summaries expose degree statistics differently, and model optimization may still contribute. GIN and MLP never received identical raw information or parameter budgets. The 7-day test has 56.16% positive prevalence, and the labels are retrospective.

Each run's complete epoch train/validation loss, AP, ROC-AUC, selected epoch, parameter count, positive weight, checkpoint hash and validation/test prediction hash is retained. R319 test scores were read once after validation-selected checkpoints. The original R317 GIN and MLP scores remain unchanged. The separate four-comparison family is in `new_statistics_319.csv`; non-significance is not equivalence.
