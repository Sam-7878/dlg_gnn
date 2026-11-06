# A26 statistical and input audits

## W-01 and W-12 — Friedman permutation audit

All four results use five-seed mean AP per model and dataset. SciPy 1.18.1 reproduces each Q from the source AP matrix using average ranks. There are no ties and the tie factor is 1. Each draw permutes model labels independently within each dataset. The two implementations compare doubled integer rank sums using an inclusive upper tail, which avoids float rounding at the boundary. The first implementation matches all original counts. The second uses a row-wise shuffle and consumes the random stream differently even with the same initial seed.

| View | Q | W | Original b | Independent b | MC p | χ² p | Iman–Davenport p | χ² / MC | Exact p | χ² / exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 29.0357 | 0.6049 | 3 | 0 | 3.999996e-06 | 5.98974098e-05 | 3.33306822e-07 | 14.97 | 1.50418989e-06 | 39.820 |
| S2 | 23.7538 | 0.4568 | 20 | 17 | 2.0999979e-05 | 8.94808183e-05 | 5.20948661e-06 | 4.26 | 1.44177382e-05 | 6.206 |
| S3 | 6.1333 | 0.2556 | 192107 | 192095 | 0.192107808 | 0.189409678 | 0.185879032 | 0.99 | 0.192254241 | 0.985 |
| S4 | 21.8286 | 0.7796 | 0 | 5 | 9.99999e-07 | 0.000216797341 | 1.36108889e-07 | 216.80 | 2.78692719e-06 | 77.791 |

Both MC implementations use one million draws and initial seed 20261024. The exact calculation fixes the first permutation by model-label symmetry, sorts partial rank sums to reuse equivalent states, and counts each next assignment with its multiplicity. It prunes branches only when the rearrangement upper bound falls below the observed rank-sum statistic. Seven exhaustive small-fixture threshold checks cover the full, empty and intermediate tails. Every exact probability lies within the separately calculated 95% binomial interval for each sampler. The intervals are kept separate because both implementations used the same starting seed. The ratios therefore persist even under exact counting with no ties. For these block and model counts, the bounded finite null has lower tail mass at the observed high Q of S1, S2 and S4, and slightly higher tail mass at S3. The F reference helps in the three extreme cases but remains an approximation. The omnibus decisions and all primary ranks are unchanged. [Statistics results](statistics_audit.json) · [Monte Carlo source](statistics.py) · [Exact enumerator](exact_null.cpp) · [Exact inputs](exact_input.txt)

## W-02 and W-03 — Score archive and graph provenance

| Graph | Earlier N / E | Score-bound ID | Actual N / E / d | Verified primary score and AP matches |
|---|---:|---|---:|---:|
| Ethereum | 14,385 / 115,065 | `ethereum_contract_clean_v1` | 14,464 / 95,098 / 8 | 35 |
| BSC | 7,481 / 59,832 | `bsc_contract_clean_v1` | 7,499 / 48,462 / 8 | 35 |
| Polygon | 2,303 / 18,411 | `polygon_contract_clean_v1` | 2,353 / 15,368 / 8 | 35 |

The older counts came from the hybrid graph inventory. The corrected graph metadata describes the tensors already evaluated by all 105 selected cryptocurrency runs. The full contract population and the binary symmetrized k=5 feature-neighbor rule define a different graph from the old hybrid input. The audit compared each frozen tensor against the independently reconstructed feature and edge arrays retained from the earlier raw build and then checked each run input manifest, raw score hash, metric hash, labels, split masks and recomputed test AP. The APs agree because the scores were already computed on these exact tensors. [Graph and score inventory](provenance.json) · [Verification source](provenance.py) · [Score archive scope](A26_EVIDENCE_ACCESS.md)

## W-04 — Native score orientation

The audit refit both feature-only estimators over the full observable feature tensor for all six graphs. For IForest the anomaly score is `-score_samples(X)`. For ECOD it is the fitted `decision_scores_`. Each original vector matched exactly, including all cells below ROC-AUC 0.5. Evaluation reused the original target labels and test masks. Negation is a diagnostic counterfactual and was never selected from test labels. DLG applies no sign transform to its per-node reconstruction error. The ten matched Elliptic DLG arrays reproduce the AP, ROC-AUC and top-1% alert metrics under that convention. Archived scores are available for those repeats rather than the original historical DLG runs. The matched Elliptic DLG-Aug lift of 0.4046 gives no evidence of useful top-1% triage. The above-prevalence AP difference is 0.0061, which exceeds the 0.0048 sample SD, so the suggested “within one SD” wording would be incorrect. [Estimator replay](score_audit.json) · [Replay source](scores.py) · [Isolation Forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) · [ECOD source](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ecod.html)


## Reproduce the audits

From the repository root, using the locked benchmark Python:

```bash
audit=projects/benchmark/publication/a26
python "$audit/statistics.py" "$PWD" /tmp/a26_mc.json
g++ -O3 -std=c++17 "$audit/exact_null.cpp" -o /tmp/a26_exact_null
/tmp/a26_exact_null < "$audit/exact_input.txt"
PYTHONPATH="$PWD/projects/benchmark/scripts:$PWD/src" python "$audit/provenance.py" /tmp/a26_provenance.json
PYTHONPATH="$PWD/projects/benchmark/scripts:$PWD/src" python "$audit/scores.py" /tmp/a26_scores.json
```

The Friedman audit needs the selected scalar run registry and membership table. The orientation and provenance scripts also require the retained private inputs and the supplementary PyOD environment. They write their outputs to the paths supplied and do not modify the frozen run registry.
