# Related-work fidelity checks

Primary sources checked 2026-10-30. Descriptions below identify mechanisms, without adding these models to the seven-model primary leaderboard.

| Method | Supported mechanism | Primary source |
|---|---|---|
| CoLA | Target-node / sampled-local-subgraph contrast; repeated scoring | https://arxiv.org/abs/2103.00113 |
| CONAD | Anomaly-inspired graph augmentations with contrastive and reconstruction training; the audited PyGOD loss-path finding is implementation-specific | https://github.com/zhiming-xu/conad |
| PREM | Preprocessing plus ego-neighbor matching, reducing training-time propagation | https://arxiv.org/abs/2310.11676 |
| TAM | Local affinity learned using truncated graph views | https://proceedings.neurips.cc/paper_files/paper/2023/hash/9b905031125e56a557db38dff4fa8d21-Abstract-Conference.html |
| GADAM | Local inconsistency learning and adaptive message passing | https://proceedings.iclr.cc/paper_files/paper/2024/hash/257a7d7934bad9b3a1774e8994e2a0dc-Abstract-Conference.html |
| ADA-GAD | Anomaly-denoised autoencoder pretraining with augmentations, followed by detection on the original graph | https://arxiv.org/abs/2312.14535 |
| Isolation Forest | Isolation-based outlier detection | https://research.monash.edu/en/publications/isolation-forest/ |
| ECOD | Dimension-wise empirical cumulative distributions and aggregated tail probabilities | https://arxiv.org/abs/2201.00382 |

The Gram identity is standard algebra; the paper's contribution is its objective-preserving implementation and empirical numerical/execution qualification. Full Cora checks concern evaluated loss, score, gradients and one Adam step, not an arbitrary long training trajectory. The zero-residual derivative convention is explicitly distinguished from the undefined dense square-root derivative at zero. Neither the GADNR synthetic covariance example nor a missing original run record proves the historical cause of that missing run.
