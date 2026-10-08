# Main Paper Publication Readiness — Gate v8

## Outcome

- Main Paper Gate M v8: **TRUE**
- Scam/GraphRAG Gate A: **FALSE**
- Full Cross-Layer Gate B: **FALSE**
- Dataset branch: **A — exact GoG-SCIMain-v1 recovery**

Gate M has no blocking requirements. Gate A/B remain closed because there are not at least 300
independent double human annotations, independently verified benign controls are insufficient, and
authorized real wallet transaction hashes/block timestamps/complete cross-layer cases are absent.

## Evidence completeness

- 24,316/24,316 historical source files verified.
- Four/four packed benchmark hashes reproduced.
- Chronological split: 17,021 train / 3,647 validation / 3,648 test.
- Held-out positives: 107; prevalence: 0.029331.
- Exhaustive derivative audit: 24,316 records, zero future-edge violations.
- Checkpoints: 20 verified (5 proposed + 5 each TGN/TGAT/fraud-oriented GraphSAGE).
- Raw prediction files: 50, all identity-aligned and hashed.
- Validation-only temperature scaling: 5/5 seeds.
- Required comparisons: 7/7, each with 10,000 ordinary bootstrap, 10,000 class-stratified
  bootstrap, and 10,000 paired-randomization replicates.
- Temporal analysis: six 608-event bins × five methods; every bin has at least five positives.
- Reliability figure: complete.
- Latency A and B: explicitly separated.

## Main five-seed results

| Method | AUC-PR | ECE | NLL |
|---|---:|---:|---:|
| CausalLocalGIN deterministic | 0.4648 ± 0.0687 | 0.1672 ± 0.0524 | 0.2365 ± 0.0610 |
| MC dropout T=10 | 0.4416 ± 0.0855 | 0.1285 ± 0.0436 | 0.1935 ± 0.0473 |
| Temperature-scaled CausalLocalGIN | 0.4648 ± 0.0687 | 0.0357 ± 0.0093 | 0.1070 ± 0.0113 |
| TGAT-style temporal attention | 0.3412 ± 0.0474 | 0.1013 ± 0.0161 | 0.1688 ± 0.0127 |
| TGN-style event memory | 0.2324 ± 0.0093 | 0.0380 ± 0.0074 | 0.1244 ± 0.0133 |
| Fraud-oriented GraphSAGE | 0.3084 ± 0.0334 | 0.2595 ± 0.0182 | 0.3548 ± 0.0247 |

The proposed deterministic model exceeds TGN, TGAT, and fraud-oriented GraphSAGE by mean AUC-PR
0.2323, 0.1236, and 0.1564 respectively; ordinary and class-stratified 95% intervals exclude zero,
and paired-randomization p-values are 0.0001. MC dropout improves calibration but lowers ranking.
Temperature scaling preserves per-seed ranking and provides the strongest single-model calibration.
The temperature-scaled deep ensemble has aggregate AUC-PR 0.4941 and NLL 0.0986, but it is one
fixed aggregate rather than five independent runs.

## Manuscript status

`docs/papers/_43_01_GraphRAG/graphrag.tex` was updated with comparator, calibration, statistical,
temporal, reliability, provenance, latency, and Gate v8 results. Static validation found balanced
LaTeX environments/braces, complete label/reference and citation/bibliography mappings, a present
figure, and no stale single-event latency or global-gate-false statement. PDF compilation was not
run because no `pdflatex`, `latexmk`, `tectonic`, `xelatex`, or `lualatex` executable is installed in
the Ubuntu 24.04 environment.

