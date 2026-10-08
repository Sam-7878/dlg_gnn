# A05 AnomalyDAE dense versus row-block equivalence

**Gate: PASS**

The candidate uses the existing exact row-block implementation with block size 256. No sampled edges or nodes are used.

## Cora-Syn (torch.float32)

Status: **PASS**; nodes: 2708; device: cuda:0.

- Max score difference: 1.90735e-06; gradient pass: True; strict update pass: True.
- Max post-Adam parameter difference: 4.24404e-06; max ROC/PR difference: 2.34977e-06.
- Final Spearman: 0.999999999; ROC-AUC: 0.929822 / 0.929824; PR-AUC: 0.214831 / 0.214831.

## BitcoinOTC (torch.float32)

Status: **PASS**; nodes: 6005; device: cuda:0.

- Max score difference: 3.8147e-06; gradient pass: True; strict update pass: True.
- Max post-Adam parameter difference: 8.07876e-06; max ROC/PR difference: 2.49826e-06.
- Final Spearman: 1.000000000; ROC-AUC: 0.890115 / 0.890115; PR-AUC: 0.202607 / 0.202605.

## Cora-Syn (torch.float64)

Status: **PASS**; nodes: 2708; device: cuda:0.

- Max score difference: 7.10543e-15; gradient pass: True; strict update pass: True.
- Max post-Adam parameter difference: 2.71831e-14; max ROC/PR difference: 0.
- Final Spearman: 1.000000000; ROC-AUC: 0.929822 / 0.929822; PR-AUC: 0.214831 / 0.214831.

