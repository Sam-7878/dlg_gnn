# Numerical Replay Audit — R320

Independent verification performed on frozen raw scores and logs located in `projects/tds/results/319`.

## 1. Cohort and Test Sample Support
- **7-Day Test Total Samples**: $N = 3,627$
- **Positive Count**: $2,037$
- **Positive Prevalence**: 56.1627% (matches reported 56.16%)
- **Chain Breakdown**:
  - Ethereum: $n = 2,637$, positives $= 1,800$ (prevalence 68.26%)
  - BSC: $n = 818$, positives $= 226$ (prevalence 27.63%)
  - Polygon: $n = 172$, positives $= 11$ (prevalence 6.40%)
- **Audit Outcome**: 100% match.

## 2. Model Performance on 7-Day Cohort (5-seed Means)
| Model Architecture | Recomputed AP | Manuscript AP | Recomputed ROC-AUC | Seed Count | Status |
|---|---|---|---|---|---|
| GIN-0 (Mean-Max) | 0.6418 (std 0.0630) | 0.6418 | 0.5621 | 5 | EXACT MATCH |
| GIN-size (Explicit Size) | 0.9445 (std 0.0183) | 0.9445 | 0.9110 | 5 | EXACT MATCH |
| GIN-sum (Sum Readout) | 0.9529 (std 0.0010) | 0.9529 | 0.9220 | 5 | EXACT MATCH |
| Summary MLP (Full 24) | 0.9582 (std 0.0026) | 0.9582 | 0.9389 | 5 | EXACT MATCH |
| MLP (No Explicit Size) | 0.9054 (std 0.0135) | 0.9054 | 0.8883 | 5 | EXACT MATCH |
| MLP (Topology Only) | 0.9548 (std 0.0022) | 0.9548 | 0.9313 | 5 | EXACT MATCH |
| LightGBM (Val Selected) | 0.9632 (std 0.0000) | 0.9632 | 0.9445 | 5 (deterministic) | EXACT MATCH |
| Summary LR | 0.9068 (std 0.0000) | 0.9068 | 0.9012 | 5 (deterministic) | EXACT MATCH |
| GATv2 + Absolute Time | 0.6126 (std 0.0645) | 0.6126 | 0.5674 | 5 | EXACT MATCH |
| Chain Prior | 0.6713 | 0.6713 | 0.6832 | 1 | EXACT MATCH |
| Isolation Forest (Unsup) | 0.6068 | 0.6068 | 0.6268 | 1 | EXACT MATCH |

## 3. Retrospective Diagnostics & Observed Span
- Single retained-node-count LR (admissible): **AP 0.8943** (matches 0.8943)
- Full-archive observed span oracle (future): **AP 0.9464** (matches 0.9464)
- Subgroup conditioning on $L_i$:
  - $L < 1$d ($n=1,162$, 1,147 pos, 98.71%): GIN AP 0.9837, MLP AP 0.9995
  - $1 \le L < 7$d ($n=536$, 511 pos, 95.34%): GIN AP 0.9432, MLP AP 0.9966
  - $L \ge 7$d ($n=1,929$, 379 pos, 19.65%): GIN AP 0.2827, MLP AP **0.5834**
- Audit Outcome: 100% exact match.

## 4. Exploratory Inference Family (4 Comparisons)
| Comparison | Recomputed Delta AP | 95% Bootstrap CI | Holm p-value | Status |
|---|---|---|---|---|
| GIN-size vs GIN-0 | +0.3027 | [+0.2858, +0.3200] | 0.0010 | EXACT MATCH |
| GIN-sum vs GIN-0 | +0.3111 | [+0.2931, +0.3292] | 0.0010 | EXACT MATCH |
| MLP Full vs MLP No Size | +0.0528 | [+0.0434, +0.0623] | 0.0010 | EXACT MATCH |
| MLP Full vs MLP Topo Only | +0.0034 | [+0.0017, +0.0050] | 0.0010 | EXACT MATCH |

## 5. Probability Calibration Audit
- GIN-size: Raw ECE 0.0975 -> TS ECE 0.0854; Raw NLL 0.3812 -> TS NLL 0.3758
- Summary MLP: Raw ECE 0.0643 -> TS ECE 0.0464; Raw NLL 0.3246 -> TS NLL 0.3213
- LightGBM: Raw ECE 0.0531 -> TS ECE 0.0530; Raw NLL **0.3039 -> TS NLL 0.3054** (Worsens under TS; preserved truthfully)

Audit Outcome: **ZERO DISCREPANCIES DETECTED**. All claims in the manuscript are fully substantiated by local raw prediction evidence.
