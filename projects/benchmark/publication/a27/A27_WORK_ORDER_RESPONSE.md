# A27 Work Order Response (`A27-01` through `A27-12` & Section 5 Audit)

## 1. Final A27 Submission File Hashes (`SHA-256`)

| File | Path | SHA-256 |
|---|---|---|
| `DLG-Benchmark_A27_Preprints.pdf` | `projects/benchmark/paper/a27/preprints/DLG-Benchmark_A27_Preprints.pdf` | `894efa1ce3f97fb880458782227c62550da5a7cef935de9006e0293368598720` |
| `DLG-Benchmark_A27_Preprints.tex` | `projects/benchmark/paper/a27/preprints/DLG-Benchmark_A27_Preprints.tex` | `8edef7c5dcab105320bd08005ed82446f277c44e58e516cfec679c1dfda7f529` |
| `source_bundle.zip` (Preprints) | `projects/benchmark/paper/a27/preprints/source_bundle.zip` | `07cba0aad2883b5f6e94c00bcf2fa5a4a4d9a2a258c589597905a887ce75f0a1` |
| `DLG-Benchmark_A27_AppliedSciences.pdf` | `projects/benchmark/paper/a27/applsci/DLG-Benchmark_A27_AppliedSciences.pdf` | `8c88a2526dc6f4e4581944417551e0f40f1559655c18e0753670c73354d80b06` |
| `DLG-Benchmark_A27_AppliedSciences.tex` | `projects/benchmark/paper/a27/applsci/DLG-Benchmark_A27_AppliedSciences.tex` | `f989d3cb5400ae168bc84377ecba8663bd39022fac61227df613d09bf58c34d9` |
| `source_bundle.zip` (Applied Sciences) | `projects/benchmark/paper/a27/applsci/source_bundle.zip` | `f46f8b9ba9089c5520095971d3667d80d270fb81a5099b4ffb56440ea60cf940` |

Both clean double-builds (`clean1`, `clean2`) under `SOURCE_DATE_EPOCH=1793923200` produced byte-identical PDFs with **zero** LaTeX warnings (`Overfull`, `undefined`, `Float too large`, `multiply defined`).

---

## 2. Item-by-Item Resolution (`A27-01` – `A27-12`)

### P0 — Scientific Interpretation & Data Integrity

1. **`A27-01` (Feature-only score orientation & complete 12-cell audit):**
   - **Pipeline call site verified:** `projects/benchmark/scripts/a24_run_studies.py` lines 78–79:
     ```python
     det = IsolationForest(n_estimators=100, max_samples=256, max_features=1., bootstrap=False, contamination='auto', random_state=seed, n_jobs=4) if model == 'IForest' else ECOD(contamination=.1, n_jobs=1)
     det.fit(x)
     s = -det.score_samples(x) if model == 'IForest' else np.asarray(det.decision_scores_)
     ```
   - **API convention:** In `scikit-learn`, `IsolationForest.score_samples(X)` returns $-2^{-E(h(x))/c(n)} \in [-1, 0]$ (lower = more abnormal), so `-det.score_samples(x)` $= +2^{-E(h(x))/c(n)} \in (0, 1]$ is the exact higher-is-more-anomalous Liu et al. (2008) anomaly score (matching `PyOD` `IForest.decision_scores_`). In `PyOD`, `ECOD.decision_scores_` is the non-negative empirical tail negative log-probability sum (higher = more extreme tail).
   - Appendix Table A3 reports all 12 estimator–dataset cells. Elliptic and Ethereum combine below-0.5 native ROC-AUC with lower positive-class standardized feature dispersion. These diagnostics establish an association across the evaluated datasets. Estimator APIs and exact vector replays establish the native score convention.

2. **`A27-02` (DGraphFin population and prevalence reconciliation across all three levels):**
   - Verified directly on `DGraphFin.pt` in `provenance.py` (`dgraphfin_reconciliation` in `provenance.json`):
     - **Raw upstream DGraphFin:** `3,700,550` nodes and `4,300,999` directed edges (`2,474,949` unlabeled background accounts).
     - **Model-input tensor (`DGraphFin.pt`):** $N = 1,225,601$ labeled nodes, $E = 746,271$ directed edges, $d = 17$ features, containing $y_{\mathrm{pos}} = 15,509$ total positive fraud nodes ($\pi_{\mathrm{graph}} = 15,509 / 1,225,601 = 0.012654 \approx 0.0127$; $4N^2 = 6.008\text{ TB} \approx 6.0\text{ TB}$ in Table 6).
     - **Split partition of `DGraphFin.pt`:** `train_mask` has `857,899` nodes (`10,857` positives, `1.27%`), `val_mask` has `183,862` nodes (`2,326` positives, `1.27%`), and `test_mask` has `183,840` nodes (`2,326` positives, $\pi_{\mathrm{test}} = 2,326 / 183,840 = 0.012652 \approx 0.0127$).
     - **Combined `val+test` evaluation pool:** `367,702` nodes (`183,862 + 183,840`) with `4,652` positives (`4,652 / 367,702 = 0.012652 \approx 0.0127`). Earlier text in `DATASETS.md` cited only the `4,652` positives in the `367,702` `val+test` pool without listing the `10,857` positives in `train_mask`. Table 1 (`^\dagger`), Section 1, Section 5.1, Section 6, and `DATASETS.md` now state all three population levels explicitly.

3. **`A27-03` (Friedman Table 5 exact DP tail probabilities $p_{\mathrm{exact}}$ and $p_{\chi^2}/p_{\mathrm{exact}}$):**
   - Table 5 (`tab:friedman_stats`) and Section 5.4 now report both $p_{\mathrm{exact}}$ (`1.5e-06`, `1.44e-05`, `0.192`, `2.79e-06`; full precision `1.50419e-06`, `1.44177e-05`, `0.192254`, `2.78693e-06`) and the exact tail ratio $p_{\chi^2}/p_{\mathrm{exact}}$ (`39.82`, `6.21`, `0.99`, `77.79`) alongside $p_{\mathrm{MC}}^\dagger$ and $p_{\chi^2}/p_{\mathrm{MC}}$, with an explicit footnote explaining the $10^{-6}$ Monte Carlo grid resolution and 95% binomial CIs at $B=10^6$.

---

### P1 — Repository Tracking, Claims & Metadata Consistency

4. **`A27-04` (Git tracking of audit scripts and Data Availability path):**
   - The five requested audit source/input files and README are downloadable and byte-identical at the pinned public commit. The manuscript links directly to that commit. No A27 release tag was found. [Public audit files](https://github.com/Sam-7878/dlg_gnn/tree/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27)

5. **`A27-05` (Section 5.3 causal attribution & portal abstract versions):**
   - Added the `Aug-Permuted` (`0.1081`) vs `Aug` (`0.1037`) and `Aug-Zero` (`0.0859`) vs `Base` (`0.0687`) causal attribution explanation directly to Section 5.3 immediately after Table 4.
   - The manuscript and portal share a 200-word abstract. A separate Editorial Office summary is at most 250 words and uses the same findings.

6. **`A27-06` (Section 5.2 items & Table 2 LANL protocol note):**
   - A27 Section 5.2 contained four interpretive items. A28 adds the separate synthetic benchmark item. In A27, Table 2 (`tab:pr_auc_main`) marks `LANL-RedTeam^\dagger` with a footnote cross-referencing Table 8 (`tab:lanl_protocols`) for the historical $\log(1+x)$ vs matched raw-feature protocols.

7. **`A27-07` (BitcoinOTC $N=6,005, E=7,409$ derivation & raw-to-evaluated edge transformations):**
   - Verified in `projects/benchmark/evidence/submission_closure_a09/historical_sources/benchmark_8x10_pipeline.py` lines 270–342: raw SNAP Bitcoin-OTC (`5,881` active user IDs, `35,592` signed temporal edges across node IDs `1..6,005`) $\to$ final PyG temporal window (`edge_window_size=10`) + self-loops with coalescing + PyGOD seed-42 outlier injection (`180` positives) = $N=6,005, E=7,409, d=67$. Documented in Table 1 (`^\ddagger`), Appendix Table A4 caption, and `DATASETS.md`.

8. **`A27-08` (MDPI front-matter `16, 0` citation fix, Featured Application, Ref [37] access date, Funding grant number, Table A5 caption):**
   - Replaced `\articlenumber{0}` with `\articlenumber{}\n\firstpage{1}\n\continuouspages{yes}` so the *Applied Sciences* left column renders `Appl. Sci. 2026, 16, 1–27.` instead of `16, 0.`.
   - Added `\featuredapplication{...}` to `DLG-Benchmark_A27_AppliedSciences.tex`.
   - Restored `(accessed on 10 October 2026)` on Reference [37] (`snapbitcoinotc`).
   - Updated `\funding{...}` to include `under grant number IITP-2026-RS-2021-II211835`.
   - Updated Appendix Table A5 (`tab:app_ablation_elliptic`) caption to cross-reference Supplement Table S4 (`tab:a24_controls`).

---

### P2 — Layout, Hardware Measurement & Precision Polish

9. **`A27-09` (Table 4 / Table 5 float ordering):**
   - Changed Figure 3 and Table 4 (`tab:conditional_delta`) to `[H]`. Verified in `qa.py` that in both `Preprints` and `AppliedSciences`, `Figure 3` appears on page 11, `Table 4` on page 12, and `Table 5` on page 12 (`page_map['Table 4'] <= page_map['Table 5']`).

10. **`A27-10` (Direct RTX 3090 measurement of `Reddit-Syn / DLG-Aug` in Table 6 & Table 9):**
    - Executed a fresh full-budget measurement of `Reddit-Syn / DLG-Aug` (seed 42, 20 local + 50 global epochs) on the RTX 3090 (`local_storage/benchmark/a27/measure_reddit_dlg_aug.py` $\to$ `reddit_dlg_aug_measured.json`):
      - `fit_seconds`: `203.30 s`, `score_seconds`: `2.491 s`, `mean_global_epoch_seconds`: `3.813 s`
      - Peak allocated VRAM: `13.848 GiB` (`14,869,192,704` bytes); peak reserved VRAM: `17.461 GiB` (`18,748,538,880` bytes).
    - Updated Table 6 (`13.848 GiB (meas.)`), Table 9 (`Reddit-Syn / DLG-Aug` row `13.848/17.461`), and Section 6.2.

11. **`A27-11` (View S2 post-hoc Wilcoxon signed-rank statistics):**
    - Computed and reported in `statistics.py`, `statistics_audit.json`, and Section 5.4 (`n_eff = 13/13`, `zero_method = 'wilcox'`, 0 ties):
      - `DLG-Aug vs DOMINANT`: $+7/-6$, $W=42.0$, $p_{\mathrm{raw}} = 0.8394$, Holm $p_{\mathrm{adj}} = 1.0000$
      - `DLG-Aug vs DLG-Base`: $+7/-6$, $W=44.0$, $p_{\mathrm{raw}} = 0.9460$, Holm $p_{\mathrm{adj}} = 1.0000$
      - `DLG-Aug vs CoLA`: $+13/-0$, $W=0.0$, $p_{\mathrm{raw}} = 0.00024$, Holm $p_{\mathrm{adj}} = 0.00098$
      - `DLG-Aug vs OCGNN`: $+12/-1$, $W=4.0$, $p_{\mathrm{raw}} = 0.00171$, Holm $p_{\mathrm{adj}} = 0.00513$

12. **`A27-12` (Elliptic `DLG-Aug` ROC-AUC rounding unification):**
    - Unified Supplement Table S4 (`tab:a24_controls`) from `0.5706` to `0.5705` (matching Tables A1 and A5) and noted the exact 6-decimal means (`0.570550` fresh matched control vs `0.570530` historical primary) in the Table S4 caption.

---

## 3. Resolution of Section 5 ("데이터로 확인 불가") Items

1. **Breakdown of the 113 supplementary vector sets (`local_storage/benchmark/a24/runs/`):**
   - Verified in `provenance.py` (`supplementary_113_breakdown` in `provenance.json`) and documented in the Data Availability Statement and `A27_EVIDENCE_ACCESS.md`:
     - `60` feature-only runs (`A24_SUPPLEMENTARY_FEATURE_ONLY`: 6 datasets $\times$ 2 models $\times$ 5 seeds)
     - `25` matched Elliptic control runs (`A24_CANONICAL_CONTROL`: 5 models $\times$ 5 seeds)
     - `20` DGraphFin/LANL alert-reproduction runs (`A24_ALERT_REPRODUCTION`: 2 datasets $\times$ 2 models $\times$ 5 seeds)
     - `4` full-declared-budget resource runs (`A24_RESOURCE_FULL_DECLARED`)
     - `4` one-epoch pilot resource runs (`A24_RESOURCE_ONE_EPOCH_PILOT`)
2. **All 12 cells of `IForest` and `ECOD` across 6 datasets:**
   - Included in full in Appendix Table A3 (`tab:orientation_audit`), `score_audit.json`, and `A27_SCIENTIFIC_AUDIT.md`.
3. **Verification of the 310 missing per-node score runs (`275` primary + `35` LANL):**
   - Verified in `provenance.py` (`missing_310_audit` in `provenance.json`): all 310 scalar JSON records exist on disk (`archive/benchmark/outputs/sci_round5_final/raw` and `archive/benchmark/outputs/sci_defense_extension_real/benchmark/raw`), contain `0` `.npy`/`.npz` score path fields, and have `0` `.np*` files in their directories.
4. **Actual `IsolationForest` / `ECOD` pipeline code path:**
   - Verified at `projects/benchmark/scripts/a24_run_studies.py` lines 78–79 and cited in Section 7.2, Section 7.3, Appendix Table A3, and `A27_SCIENTIFIC_AUDIT.md`.


## Codex closure

The two rebuilt packages supersede the AntiGravity PDF and ZIP hashes. The Reddit-Syn DLG-Aug measurement adds timing and memory evidence. The 113 archived supplementary arrays refer to the earlier study. For the four historical GADNR cells the cause remains unestablished. Three later diagnostics reached the 300 s cap. The Reddit-Syn watchdog recorded CPU preprocessing and the terminal record is missing. The fixed leaderboard remains at 455 successful evaluations and 84/91 primary pairs.

Author approval of the current hashes and the corresponding author’s decision on permitted derived-score access remain pending. The author will record the preprint DOI and journal receipt after deposit and submission. [Sign-off](A27_AUTHOR_SIGNOFF.md) · [Receipts](A27_SUBMISSION_RECEIPTS.md)

## A28 cross-reference

The matched Elliptic control was independently refitted for the same configuration and five seeds. The means round independently to 0.5705 (primary) and 0.5706 (matched). [A28 seed-level verification](../a28/elliptic_roc.json) · [Current submission status](../a28/A28_WORK_ORDER_RESPONSE.md)
