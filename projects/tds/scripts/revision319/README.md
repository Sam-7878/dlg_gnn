# R319 execution order

Use `/mnt/d/_Work/goat_bank/.venv_cuda/bin/python` from the Ubuntu environment, with the repository root as working directory. The 3090 was used as `cuda:1` at execution time; device order can differ on another machine.

1. `protocol.py` seals R314–R318 baselines and R319 hypotheses.
2. `builder_audit.py` rebuilds every original and 7-day tensor from licensed raw GoG ZIPs. Fitting stops if `seven_day_builder_equivalence.json` is not `PASS`.
3. `diagnostics.py --phase existing` computes observed-span subgroup and calendar composition reports; `--phase features` fits cutoff-admissible count baselines.
4. `neural.py --arm ARM --device cuda:1` runs each of six preregistered arms for seeds 7, 17, 27, 37, 47. `controls.py` selects the LightGBM grid using validation, then evaluates the chain prior and negative-only Isolation Forest.
5. `aggregate.py` checks score identity, computes modelwise calibration, operating metrics, chain support and the separate exploratory family. `figure.py` renders vector figures.
6. The private `paper/submission_319/update.py` materializes the shared scientific source for both venues; `build.py` independently builds each source ZIP and verifies cold extraction/compilation.

Numerical results are local candidates. This sequence does not grant approval to release, post or submit either paper.
