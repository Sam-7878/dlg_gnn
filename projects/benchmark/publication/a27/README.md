# A27 Public Audit & Reproduction Package

This directory (`projects/benchmark/publication/a27/`) contains the open-source statistical, score-orientation, and provenance audit scripts and inputs referenced in the Data Availability Statement of *“Graph Anomaly Detection Across Domains: Predictive Quality and Exact Execution”* (A27 revision).

## Included Public Audit Files (A27-04)

1. `statistics.py` — Reproduces the Friedman statistic $Q$, Kendall's $W$, tie-correction verification (`ties = 0`), $\chi^2$ and Iman–Davenport $F$ reference $p$-values, both $B=10^6$ within-block Monte Carlo permutation streams (seed `20261024`), 95% binomial tail confidence intervals, exact tail ratios $p_{\chi^2}/p_{\mathrm{exact}}$, and the View S2 two-sided exact Wilcoxon signed-rank post-hoc comparisons with Holm step-down adjustment (`statistics_audit.json`).
2. `exact_null.cpp` — C++17 branch-and-bound enumerator that counts the exact number of favorable rank-sum assignments over all $(k!)^{n-1}$ tie-free permutations for Views S1–S4.
3. `exact_input.txt` & `exact_counts.txt` — Input $(n, k, \text{cutoff})$ triples and exact 128-bit integer numerator/denominator counts for Views S1–S4.
4. `scores.py` — Audits the fixed higher-is-more-anomalous estimator conventions (`-det.score_samples(x)` for `sklearn.ensemble.IsolationForest` and `det.decision_scores_` for `pyod.models.ecod.ECOD` in `projects/benchmark/scripts/a24_run_studies.py:78-79`), verifies exact vector identity against the 60 stored `.npz` arrays, and computes the 12-cell rescoring and standardized feature-space dispersion ratios $\mathrm{RMS}(|z|)_{\mathrm{pos}}/\mathrm{RMS}(|z|)_{\mathrm{neg}}$ (`score_audit.json`).
5. `provenance.py` — Verifies the 105 cryptocurrency score vectors and tensor bindings, the 40 archived rerun score vectors (35 `BitcoinOTC` and 5 `DGraphFin` AnomalyDAE), the 310 historical runs with no stored `.np*` score files on disk, the 113 supplementary score vectors breakdown, and the `DGraphFin` three-level population reconciliation (`provenance.json`).

## Reproduction Commands

From the repository root (`dlg_gnn`):

```bash
audit=projects/benchmark/publication/a27

# 1. Friedman MC + View S2 exact Wilcoxon audit (uses public scalar registry only)
python "$audit/statistics.py" "$PWD" /tmp/a27_statistics_audit.json

# 2. Exact finite Friedman permutation enumeration (uses public exact_input.txt only)
g++ -O3 -std=c++17 "$audit/exact_null.cpp" -o /tmp/a27_exact_null
/tmp/a27_exact_null < "$audit/exact_input.txt"

# 3. Feature-only orientation & dispersion audit (requires local feature tensors and saved scores)
PYTHONPATH="$PWD/projects/benchmark/scripts:$PWD/src" python "$audit/scores.py" /tmp/a27_score_audit.json

# 4. Score archive & tensor provenance audit (requires local score archive)
PYTHONPATH="$PWD/projects/benchmark/scripts:$PWD/src" python "$audit/provenance.py" /tmp/a27_provenance.json
```

