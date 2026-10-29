# Gate L1: Report Numeric Reconciliation Report

**Document ID:** `DLG-BENCH-GATE-L1-NUMERIC-RECONCILIATION`  
**Date:** 2026-10-29  
**Parent Work Order:** `DLG-BENCH-WO-A21-LEDGER-ONLY`  
**Status:** **PASS** (Report Documentation Fully Reconciled with Frozen Manuscript Tables)  

---

## 1. Context and Objective

During independent submission review of A20 (`DLG_Benchmark_A20_Independent_Submission_Review_2026-10-29.md`, Finding F1), an inconsistency was noted in the internal prose description of Gate G4 within `projects/benchmark/publication/a20/a20_editorial_diff.md`:
- The diff report narrative mistakenly conflated Polygon's test prevalence ($\pi_\mathrm{test} = 0.0259$) and an erroneous absolute delta number (`+0.0094`) with Elliptic's relative improvement figure (`+50.95%` / `+50.97%`).
- Importantly, independent review confirmed that the actual manuscript tables (Table 2 and Table 4 in both `DLG-Benchmark_A16_Preprints.tex` and `DLG-Benchmark_A16_AppliedSciences.tex`) **were already correct** and did not contain `+0.0094`.

The objective of Gate L1 is to reconcile the internal documentation before archive closure, providing explicit before $\to$ after quotations while keeping the frozen scientific manuscript tables unchanged.

---

## 2. Before $\to$ After Reconciliation in `a20_editorial_diff.md`

### Before (Errant Text in A20 Narrative):
```markdown
- **Problem in A19/A16:** Manuscript emphasized relative percentage gain (+50.97%) without giving equal weight to absolute \Delta AP (+0.0094), and overclaimed that BSC disproved universal benefit.
- **A20 Resolution:**
  - Primary presentation led with absolute metric difference \Delta AP = +0.0094, placing relative percentage gain as secondary context.
  - Contextualized \Delta AP against test prevalence baseline (\pi_test = 0.0259).
```

### After (Reconciled Text in `a20_editorial_diff.md`):
```markdown
- **Problem in A19/A16:** Manuscript emphasized relative percentage gain (+50.95%, Elliptic) without giving equal weight to absolute \Delta AP (+0.0350, from 0.0687 to 0.1037), and overclaimed that BSC disproved universal benefit.
- **A20 Resolution:**
  - Primary presentation led with absolute metric difference \Delta AP = +0.0350 (on Elliptic), placing relative percentage gain (+50.95%) as secondary context.
  - Contextualized \Delta AP against test prevalence baseline (Elliptic \pi_test = 0.0976, test positives M_test = 926 of N_test = 9,487).
  - Separately documented Polygon where \pi_test = 0.0259 (M_test \approx 13 of N_test = 494) with DLG-Base AP = 0.0854, DLG-Aug AP = 0.0731, and \Delta AP \approx -0.0124.
```

---

## 3. Authoritative Empirical Source Values Cross-Check

The table below cross-checks the relevant datasets across Table 2 (Overall Performance), Table 4 (Conditional AP Deltas), and `canonical_selected_run_registry.csv`:

| Dataset | Metric / Property | DLG-Base | DLG-Aug | Absolute $\Delta\mathrm{AP}$ | Relative Gain | Test Prevalence ($\pi_\mathrm{test}$) | $M_\mathrm{test} / N_\mathrm{test}$ |
|---|---|---|---|---|---|---|---|
| **Elliptic** | Average Precision (AP) | 0.0687 | 0.1037 | **+0.0350** | **+50.95%** | **0.0976** (9.76%) | 926 / 9,487 |
| **Polygon** | Average Precision (AP) | 0.0854 | 0.0731 | **−0.0124** | −14.40% | **0.0259** (2.59%) | ~13 / 494 |
| **Ethereum** | Average Precision (AP) | 0.8173 | 0.8425 | **+0.0253** | +3.08% | **0.4089** (40.89%) | 884 / 2,162 |
| **BSC** | Average Precision (AP) | 0.2087 | 0.2108 | **+0.0021** | +1.01% | **0.1473** (14.73%) | 221 / 1,500 |
| **Reddit-Syn** | Average Precision (AP) | 0.2085 | 0.1429 | **−0.0656** | −31.46% | **0.0326** (3.26%) | ~358 / 10,996 |

---

## 4. Verification Verdict

- [x] Errant `+0.0094` removed from internal documentation.
- [x] Elliptic correctly anchored to absolute $\Delta\mathrm{AP} = +0.0350$ and $\pi_\mathrm{test} = 0.0976$.
- [x] Polygon correctly anchored to $\Delta\mathrm{AP} \approx -0.0124$ and $\pi_\mathrm{test} = 0.0259$.
- [x] Primary scientific tables (Table 2, Table 4) and raw run registries remain strictly frozen and unedited.
- **GATE L1 STATUS: PASS**
