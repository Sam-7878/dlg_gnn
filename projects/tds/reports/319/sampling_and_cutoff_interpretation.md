# Three estimands and first-transfer composition

| Evaluation contract | Unit and ordering | Train / validation / test positive fraction | Inference scope |
|---|---|---|---|
| Archived last-transfer | Same 24,316 completed-history contracts ordered by final valid transfer | 39.85% / 8.45% / 2.93% | Retrospective archived comparison |
| First-transfer reorder | Same population, same completed records, ordered by first valid transfer | 14.79% / 71.32% / 56.99% | Sorting and label-composition sensitivity |
| First-plus-seven-day | New 24,179-eligible population, rebuilt inputs at seven days, own split | 14.81% / 70.09% / 56.16% | Fixed-observation sensitivity under retrospective labels |

The manuscript cannot make paired AP tests across these three populations. Source-first/last timestamps and the 7-day builder are verified; label discovery timestamps and the collection stopping policy are unavailable. Thus the evidence supports sampling-associated composition differences and does not distinguish collection effects from behavior change.

`first_time_monthly_composition.csv` and `first_time_quarterly_composition.csv` report UTC period, chain, support, positives, Wilson 95% interval, span quantiles, archive-endpoint distance, 7/30-day eligibility, and source-category count. For example, all-chain 2023Q1 contains 1,187 contracts with 28.39% positives (Wilson 25.90–31.02%), while 2023Q3 contains 2,903 contracts with 70.03% (68.34–71.67%). These are composition descriptions, not proof that incidence changed. The vector figure `results/319/figures/first_time_quarterly_composition.pdf` visualizes periods with sufficient support and retains sparse periods in the CSV.

Recommended protocol for new work: prespecify observation unit, cutoff, available label time, and split; report chain/activity/count controls and support; then use a fixed-horizon sensitivity where source reconstruction permits. The seven-day cohort is not a unique correct split and no classifier is recommended for operational fraud alerts from these retrospective labels.
