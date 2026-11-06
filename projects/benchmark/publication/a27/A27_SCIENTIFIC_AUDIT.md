# A27 Scientific and Provenance Audit

## 1. A27-01 & Section 5.4 — Complete 12-Cell Feature-Only Orientation Audit and Call-Site Verification

The supplementary feature-only evaluation call site is `projects/benchmark/scripts/a24_run_studies.py` lines 78–79:

```python
det = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1) if model == 'IForest' else ECOD(n_jobs=1)
det.fit(x)
s = -det.score_samples(x) if model == 'IForest' else np.asarray(det.decision_scores_)
```

In `scikit-learn`, `IsolationForest.score_samples(X)` returns the opposite of the Liu et al. (2008) anomaly score ($-2^{-E(h(x))/c(n)} \in [-1, 0]$, where lower values are more abnormal), so `-det.score_samples(x)` $= +2^{-E(h(x))/c(n)} \in (0, 1]$ is the standard higher-is-more-anomalous anomaly score (matching `PyOD` `IForest.decision_scores_`). In `PyOD`, `ECOD.decision_scores_` is the non-negative empirical tail negative log-probability sum (higher values indicate more extreme empirical tails).

| Dataset | Estimator | $N$ | $d$ | Native AP | Negated AP | Native ROC-AUC | Negated ROC-AUC | $\mathrm{RMS}(\|z\|)_{\mathrm{pos}} / \mathrm{RMS}(\|z\|)_{\mathrm{neg}}$ | $\overline{\max_j\|z_{ij}\|}_{\mathrm{pos}} / \overline{\max_j\|z_{ij}\|}_{\mathrm{neg}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elliptic | IForest | 46,564 | 165 | 0.0525 | 0.6444 | 0.1056 | 0.8944 | 0.58 (0.51/0.89) | 0.59 (2.31/3.91) |
| Elliptic | ECOD | 46,564 | 165 | 0.0568 | 0.2399 | 0.1865 | 0.8135 | 0.58 (0.51/0.89) | 0.59 (2.31/3.91) |
| DGraphFin | IForest | 1,225,601 | 17 | 0.0130 | 0.0131 | 0.4972 | 0.5028 | 1.00 (0.93/0.92) | 0.92 (1.53/1.66) |
| DGraphFin | ECOD | 1,225,601 | 17 | 0.0089 | 0.0201 | 0.3460 | 0.6540 | 1.00 (0.93/0.92) | 0.92 (1.53/1.66) |
| Ethereum | IForest | 14,464 | 8 | 0.3525 | 0.5402 | 0.3940 | 0.6060 | 0.97 (0.89/0.92) | 0.96 (1.44/1.50) |
| Ethereum | ECOD | 14,464 | 8 | 0.3507 | 0.4560 | 0.4242 | 0.5758 | 0.97 (0.89/0.92) | 0.96 (1.44/1.50) |
| BSC | IForest | 7,499 | 8 | 0.2740 | 0.0949 | 0.7345 | 0.2655 | 1.35 (1.15/0.85) | 1.26 (1.77/1.41) |
| BSC | ECOD | 7,499 | 8 | 0.2223 | 0.1043 | 0.6660 | 0.3340 | 1.35 (1.15/0.85) | 1.26 (1.77/1.41) |
| Polygon | IForest | 2,353 | 8 | 0.0902 | 0.0170 | 0.7816 | 0.2184 | 1.26 (1.12/0.88) | 1.22 (1.68/1.38) |
| Polygon | ECOD | 2,353 | 8 | 0.0650 | 0.0190 | 0.6980 | 0.3020 | 1.26 (1.12/0.88) | 1.22 (1.68/1.38) |
| LANL-RedTeam | IForest | 16,694 | 13 | 0.1746 | 0.0117 | 0.7819 | 0.2181 | 5.07 (1.64/0.32) | 3.93 (4.00/1.02) |
| LANL-RedTeam | ECOD | 16,694 | 13 | 0.2007 | 0.0122 | 0.7186 | 0.2814 | 5.07 (1.64/0.32) | 3.93 (4.00/1.02) |

Across all 12 dataset–estimator cells, native $\mathrm{ROC\text{-}AUC} > 0.5$ occurs **if and only if** the standardized attribute dispersion ratio $\mathrm{RMS}(\|z\|)_{\mathrm{pos}} / \mathrm{RMS}(\|z\|)_{\mathrm{neg}} > 1$ (`BSC`: `1.35`, `Polygon`: `1.26`, `LANL-RedTeam`: `5.07`). On `Elliptic` (`0.58`) and `Ethereum` (`0.97`), positive targets are more tightly clustered near the interior of raw attribute space than negative nodes, causing generic unsupervised tail-/isolation-based detectors to assign lower outlier scores to positive targets.

## 2. A27-02 — DGraphFin Three-Level Population and Prevalence Reconciliation

- **Raw upstream DGraphFin dataset**: `3,700,550` nodes and `4,300,999` directed edges (`2,474,949` nodes are unlabeled background accounts).
- **Model-input tensor (`DGraphFin.pt`)**: $N = 1,225,601$ labeled nodes, $E = 746,271$ directed edges, $d = 17$ features, containing $y_{\mathrm{pos}} = 15,509$ total positive fraud nodes ($\pi_{\mathrm{graph}} = 15,509 / 1,225,601 = 0.012654 \approx 0.0127$). Dense float32 adjacency memory is $4N^2 = 6,008,391,244,804$ bytes ($6.008\text{ TB} \approx 6.0\text{ TB}$ in Table 6).
- **Official split partition of the $1,225,601$ model-input tensor**:
  - `train_mask`: `857,899` nodes with `10,857` positives (`1.2655%` $\approx 0.0127$)
  - `val_mask`: `183,862` nodes with `2,326` positives (`1.2651%` $\approx 0.0127$)
  - `test_mask`: `183,840` nodes with `2,326` positives ($\pi_{\mathrm{test}} = 2,326 / 183,840 = 0.012652 \approx 0.0127$)
  - Combined `val+test` evaluation pool: `367,702` nodes (`183,862 + 183,840`) with `4,652` positives (`4,652 / 367,702 = 0.012652` $\approx 0.0127$). Dividing the `val+test` positive count (`4,652`) by the full `train+val+test` node count (`1,225,601`) mixes numerators and denominators across splits.

## 3. A27-03 & A27-11 — Friedman Exact Null Tail Probabilities and View S2 Post-Hoc Wilcoxon Audit

| View | $Q$ | $W$ | Original $b$ | Independent $b$ | $p_{\mathrm{MC}}$ | $p_{\mathrm{exact}}$ | $p_{\chi^2}$ | $p_F$ | $p_{\chi^2}/p_{\mathrm{MC}}$ | $p_{\chi^2}/p_{\mathrm{exact}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 29.0357 | 0.6049 | 3 | 0 | 4e-06 | 1.50419e-06 | 5.98974e-05 | 3.33307e-07 | 14.97 | 39.82 |
| S2 | 23.7538 | 0.4568 | 20 | 17 | 2.1e-05 | 1.44177e-05 | 8.94808e-05 | 5.20949e-06 | 4.26 | 6.21 |
| S3 | 6.1333 | 0.2556 | 192107 | 192095 | 0.192108 | 0.192254 | 0.18941 | 0.185879 | 0.99 | 0.99 |
| S4 | 21.8286 | 0.7796 | 0 | 5 | 9.99999e-07 | 2.78693e-06 | 0.000216797 | 1.36109e-07 | 216.80 | 77.79 |

### View S2 ($n=13, k=5$) Post-Hoc Paired Two-Sided Wilcoxon Signed-Rank Tests (`zero_method="wilcox"`)

| Comparison | $n_{\mathrm{eff}}$ (ties) | Wins / Losses (`DLG-Aug` vs baseline) | Wilcoxon $W$ | Unadjusted $p_{\mathrm{raw}}$ | Holm $p_{\mathrm{adj}}$ |
|---|---:|---:|---:|---:|---:|
| `DLG-Aug` vs `DOMINANT` | 13/13 (0) | +7/-6 | 42.0 | 0.839355 | 1.000000 |
| `DLG-Aug` vs `CoLA` | 13/13 (0) | +13/-0 | 0.0 | 0.000244 | 0.000977 |
| `DLG-Aug` vs `OCGNN` | 13/13 (0) | +12/-1 | 4.0 | 0.001709 | 0.005127 |
| `DLG-Aug` vs `DLG-Base` | 13/13 (0) | +7/-6 | 44.0 | 0.946045 | 1.000000 |

## 4. A27-07 — BitcoinOTC Tensor Construction and Raw-to-Evaluated Graph Transformations

Verified in `projects/benchmark/evidence/submission_closure_a09/historical_sources/benchmark_8x10_pipeline.py` lines 270–342:
- Raw SNAP Bitcoin-OTC (`soc-sign-bitcoinotc`) has `5,881` active user IDs and `35,592` signed temporal rating edges spanning integer node index range `1..6,005`.
- Taking the final PyG `BitcoinOTC(edge_window_size=10)` snapshot, adding self-loops with edge coalescing (`_repackage_graph`), and injecting PyGOD contextual and structural outliers (seed 42, `180` positive nodes, $180 / 6,005 = 0.0300$) yields the evaluated tensor $N=6,005$, $E=7,409$, $d=67$ ($64$-dim Node2Vec plus $3$ degree statistics).

## 5. A27-10 & A27-12 — Direct RTX 3090 Measurement (`Reddit-Syn / DLG-Aug`) and Elliptic ROC-AUC Rounding

- **Direct RTX 3090 measurement (`Reddit-Syn / DLG-Aug`, seed 42, 20 local + 50 global epochs)**: fit time `203.30 s`, score time `2.491 s`, mean global epoch time `3.813 s`, peak allocated VRAM `13.848 GiB` (`14,869,192,704` bytes), peak reserved VRAM `17.461 GiB` (`18,748,538,880` bytes). Both Table 6 and Table 9 now report the directly measured `13.848 GiB` (`13.848/17.461 GiB` allocated/reserved), superseding the earlier 5-seed summary log figure (`13.968/17.375 GiB`).
- **Elliptic DLG-Aug ROC-AUC**: Primary historical 5-seed mean is `0.570530` (reported as `0.5705` in Tables A1 and A5); fresh RTX 3090 matched control 5-seed mean is `0.570550` (`2.0e-5` difference across the `0.57055` half-way rounding boundary). Table S4 is unified to `0.5705` with an explicit caption note.
