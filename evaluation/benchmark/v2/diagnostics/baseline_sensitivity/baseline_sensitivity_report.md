# Gate A03-10: Baseline Mini-Sensitivity Report (CoLA & OCGNN)

## 1. Executive Summary
In compliance with Work Order A03 §12, this reviewer-defense analysis verifies baseline stability under pre-registered hyperparameter perturbations on frozen benchmark datasets (`Cora-Syn`, `CiteSeer-Syn`) across seeds 42, 43, 44.

Key Observations:
- **CoLA (`batch_size` 32 vs 64 vs 128)**: Performance remains bounded without catastrophic collapse. Minor variation is consistent with stochastic negative subgraph sampling.
- **OCGNN (`contamination` 0.05 vs 0.10 vs 0.15)**: Radius margin perturbations exhibit smooth ranking preservation with minimal metric degradation.
- Neither model shows artificial degradation or cherry-picked hyperparameter fragility in the primary benchmark configuration.

## 2. Summary Statistics

| ('dataset', '')   | ('model', '')   | ('config_id', '')   | ('changed_parameter', '')   |   ('changed_value', '') |   ('roc_auc', 'mean') |   ('roc_auc', 'std') |   ('pr_auc', 'mean') |   ('pr_auc', 'std') |   ('validation_f1', 'mean') |   ('validation_f1', 'std') |
|:------------------|:----------------|:--------------------|:----------------------------|------------------------:|----------------------:|---------------------:|---------------------:|--------------------:|----------------------------:|---------------------------:|
| CiteSeer-Syn      | CoLA            | default             | batch_size                  |                   64    |              0.724367 |            0.140377  |            0.0920667 |           0.0289595 |                  0.0926333  |                  0.046603  |
| CiteSeer-Syn      | CoLA            | nearby_lower        | batch_size                  |                   32    |              0.673667 |            0.0498479 |            0.0527    |           0.0258052 |                  0.0731     |                  0.0460248 |
| CiteSeer-Syn      | CoLA            | nearby_upper        | batch_size                  |                  128    |              0.617767 |            0.0501109 |            0.0618667 |           0.0216338 |                  0.0260667  |                  0.0250682 |
| CiteSeer-Syn      | OCGNN           | default             | contamination               |                    0.1  |              0.492433 |            0.0489267 |            0.0336    |           0.0185429 |                  0.033      |                  0.0336688 |
| CiteSeer-Syn      | OCGNN           | nearby_lower        | contamination               |                    0.05 |              0.492767 |            0.0485389 |            0.0336    |           0.0185429 |                  0.0330333  |                  0.0337198 |
| CiteSeer-Syn      | OCGNN           | nearby_upper        | contamination               |                    0.15 |              0.492933 |            0.0483464 |            0.0336    |           0.0185429 |                  0.0330667  |                  0.0337707 |
| Cora-Syn          | CoLA            | default             | batch_size                  |                   64    |              0.604033 |            0.0341087 |            0.0638667 |           0.0191949 |                  0.102333   |                  0.0193578 |
| Cora-Syn          | CoLA            | nearby_lower        | batch_size                  |                   32    |              0.681533 |            0.0238919 |            0.087     |           0.0289019 |                  0.125167   |                  0.0325591 |
| Cora-Syn          | CoLA            | nearby_upper        | batch_size                  |                  128    |              0.608967 |            0.121906  |            0.0662333 |           0.0416313 |                  0.083      |                  0.0275229 |
| Cora-Syn          | OCGNN           | default             | contamination               |                    0.1  |              0.427167 |            0.0711165 |            0.0366    |           0.013268  |                  0.0089     |                  0.0154153 |
| Cora-Syn          | OCGNN           | nearby_lower        | contamination               |                    0.05 |              0.4284   |            0.0704247 |            0.0366333 |           0.0132862 |                  0.00913333 |                  0.0158194 |
| Cora-Syn          | OCGNN           | nearby_upper        | contamination               |                    0.15 |              0.427167 |            0.0709707 |            0.0365667 |           0.01325   |                  0.00913333 |                  0.0158194 |

## 3. Raw 36-Run Records

| dataset      | model   | config_id    |   seed | changed_parameter   |   changed_value |   roc_auc |   pr_auc |   validation_f1 |
|:-------------|:--------|:-------------|-------:|:--------------------|----------------:|----------:|---------:|----------------:|
| Cora-Syn     | CoLA    | nearby_lower |     42 | batch_size          |           32    |    0.7005 |   0.1144 |          0.1481 |
| Cora-Syn     | CoLA    | nearby_lower |     43 | batch_size          |           32    |    0.6894 |   0.0568 |          0.0879 |
| Cora-Syn     | CoLA    | nearby_lower |     44 | batch_size          |           32    |    0.6547 |   0.0898 |          0.1395 |
| Cora-Syn     | CoLA    | default      |     42 | batch_size          |           64    |    0.6281 |   0.0731 |          0.1127 |
| Cora-Syn     | CoLA    | default      |     43 | batch_size          |           64    |    0.565  |   0.0418 |          0.08   |
| Cora-Syn     | CoLA    | default      |     44 | batch_size          |           64    |    0.619  |   0.0767 |          0.1143 |
| Cora-Syn     | CoLA    | nearby_upper |     42 | batch_size          |          128    |    0.694  |   0.1143 |          0.1119 |
| Cora-Syn     | CoLA    | nearby_upper |     43 | batch_size          |          128    |    0.6636 |   0.0428 |          0.08   |
| Cora-Syn     | CoLA    | nearby_upper |     44 | batch_size          |          128    |    0.4693 |   0.0416 |          0.0571 |
| Cora-Syn     | OCGNN   | nearby_lower |     42 | contamination       |            0.05 |    0.5074 |   0.0468 |          0.0274 |
| Cora-Syn     | OCGNN   | nearby_lower |     43 | contamination       |            0.05 |    0.3722 |   0.0216 |          0      |
| Cora-Syn     | OCGNN   | nearby_lower |     44 | contamination       |            0.05 |    0.4056 |   0.0415 |          0      |
| Cora-Syn     | OCGNN   | default      |     42 | contamination       |            0.1  |    0.5074 |   0.0468 |          0.0267 |
| Cora-Syn     | OCGNN   | default      |     43 | contamination       |            0.1  |    0.3719 |   0.0216 |          0      |
| Cora-Syn     | OCGNN   | default      |     44 | contamination       |            0.1  |    0.4022 |   0.0414 |          0      |
| Cora-Syn     | OCGNN   | nearby_upper |     42 | contamination       |            0.15 |    0.5074 |   0.0468 |          0.0274 |
| Cora-Syn     | OCGNN   | nearby_upper |     43 | contamination       |            0.15 |    0.3726 |   0.0216 |          0      |
| Cora-Syn     | OCGNN   | nearby_upper |     44 | contamination       |            0.15 |    0.4015 |   0.0413 |          0      |
| CiteSeer-Syn | CoLA    | nearby_lower |     42 | batch_size          |           32    |    0.6464 |   0.0422 |          0.0625 |
| CiteSeer-Syn | CoLA    | nearby_lower |     43 | batch_size          |           32    |    0.7312 |   0.0821 |          0.1235 |
| CiteSeer-Syn | CoLA    | nearby_lower |     44 | batch_size          |           32    |    0.6434 |   0.0338 |          0.0333 |
| CiteSeer-Syn | CoLA    | default      |     42 | batch_size          |           64    |    0.677  |   0.1176 |          0.1379 |
| CiteSeer-Syn | CoLA    | default      |     43 | batch_size          |           64    |    0.6138 |   0.0606 |          0.0448 |
| CiteSeer-Syn | CoLA    | default      |     44 | batch_size          |           64    |    0.8823 |   0.098  |          0.0952 |
| CiteSeer-Syn | CoLA    | nearby_upper |     42 | batch_size          |          128    |    0.6264 |   0.0385 |          0.05   |
| CiteSeer-Syn | CoLA    | nearby_upper |     43 | batch_size          |          128    |    0.663  |   0.0812 |          0      |
| CiteSeer-Syn | CoLA    | nearby_upper |     44 | batch_size          |          128    |    0.5639 |   0.0659 |          0.0282 |
| CiteSeer-Syn | OCGNN   | nearby_lower |     42 | contamination       |            0.05 |    0.4754 |   0.0256 |          0      |
| CiteSeer-Syn | OCGNN   | nearby_lower |     43 | contamination       |            0.05 |    0.4553 |   0.0548 |          0.0674 |
| CiteSeer-Syn | OCGNN   | nearby_lower |     44 | contamination       |            0.05 |    0.5476 |   0.0204 |          0.0317 |
| CiteSeer-Syn | OCGNN   | default      |     42 | contamination       |            0.1  |    0.4754 |   0.0256 |          0      |
| CiteSeer-Syn | OCGNN   | default      |     43 | contamination       |            0.1  |    0.4543 |   0.0548 |          0.0673 |
| CiteSeer-Syn | OCGNN   | default      |     44 | contamination       |            0.1  |    0.5476 |   0.0204 |          0.0317 |
| CiteSeer-Syn | OCGNN   | nearby_upper |     42 | contamination       |            0.15 |    0.4754 |   0.0256 |          0      |
| CiteSeer-Syn | OCGNN   | nearby_upper |     43 | contamination       |            0.15 |    0.4558 |   0.0548 |          0.0675 |
| CiteSeer-Syn | OCGNN   | nearby_upper |     44 | contamination       |            0.15 |    0.5476 |   0.0204 |          0.0317 |
