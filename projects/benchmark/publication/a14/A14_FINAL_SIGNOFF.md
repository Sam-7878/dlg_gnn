# DLG Benchmark A14 Campaign Coherence Final Sign-Off

**Document Identifier:** `A14_FINAL_SIGNOFF.md`  
**Date:** 2026-10-24  
**Target Milestone:** Benchmark Preprint (Preprints.org) $\to$ MDPI *Applied Sciences* Special Issue (`C67H3U9QGT`)  
**Git Release Tag:** `benchmark-a14-submission-2026-10-24`  
**Git Commit Hash:** `bd06fd453f6fed3c113dd891a26d8f069c1544ee`  
**Status:** **ALL GATES (A14-G1 through A14-G8) FULLY PASSED AND VERIFIED**

---

## 1. Executive Summary Across Gates (A14-G1 to A14-G8)

| Gate | Title | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- | :---: |
| **A14-G1** | Canonical Approved-Run Selection | Resolve dual campaigns; canonical 455 runs (420 primary + 35 LANL) | `a14_select_canonical_registry.py`, SHA-256 hash locked | **PASSED** |
| **A14-G2** | Source-Derived Delta Table | 14-dataset paired AP; exact arithmetic $\Delta\mu = \mu_1 - \mu_0$; TeX auto-gen | `a14_compute_deltas.py`, `table_conditional_delta.tex` | **PASSED** |
| **A14-G3** | Verified Visual Layout & Support Matrix | Figure 3 unclipped; x-limits `[-0.112, 0.065]`; fail-closed support matrix | `a14_generate_figures.py`, visual inspection via `pdftoppm` | **PASSED** |
| **A14-G4** | Whole-Doc Numerical Validator & Negative Controls | Literal `.tex` parsing vs CSV; 6 negative controls rejected | `a14_validate_final_manuscripts.py`, `negative_controls_a14.json` | **PASSED** |
| **A14-G5** | Fast Benchmark Reproduction Tool | Portable, non-hardcoded verification script (`--mode all/tables/figures`) | `a14_verify_release.py`, verified execution | **PASSED** |
| **A14-G6** | Version Control & Public Release Tag | Git commit, tag `benchmark-a14-submission-2026-10-24`, push to origin | Remote verification `refs/tags/benchmark-a14-submission-2026-10-24` | **PASSED** |
| **A14-G7** | Submission Packaging & Clean Unpack Build | Preprints & MDPI ZIP bundles; isolated unpack; 0 undef refs | `a14_package_submission.py`, `unpack_build_summary.json` | **PASSED** |
| **A14-G8** | Final Sign-Off Document | Formal sign-off on dual stage readiness, scope adherence, no causal claims | This document (`A14_FINAL_SIGNOFF.md`) | **PASSED** |

---

## 2. Root Cause Analysis & Resolution of Figure 3 vs. Table 4 Discrepancy

### 2.1 Problem Statement in A13
In A13, an independent evidence review revealed a severe cross-artifact contradiction:
- **Table 4 (Text/Table):** Sourced from the A08 clean repaired run campaign (`outputs/benchmark/benchmark_a08_crypto_clean_v1`), reporting Ethereum $+0.0253$, BSC $+0.0021$, and Polygon $-0.0124$.
- **Figure 3 (Graphic):** Ingested an uncurated historical registry (`evidence/astra_revision/approved_run_registry.csv`), which retained uncorrected runs from `outputs/benchmark/a03_production` (Ethereum $-0.0015$, BSC $-0.1983$, and Polygon $+0.0055$).
- **Impact:** Opposite signs for Ethereum and BSC, completely conflicting narratives between text and figures, and severe visual clipping of BSC in Figure 3 due to arbitrary fixed axis limits.

### 2.2 Root Cause Identification
1. The historical file `evidence/astra_revision/approved_run_registry.csv` was treated as an unverified canonical ground truth by figure generation scripts without filtering for the approved A08 clean rerun supersessions.
2. In Figure 3, the x-axis limits were hardcoded to `[-0.04, 0.04]`, clipping the historical BSC bar ($-0.1983$) entirely off the plot area.

### 2.3 Comprehensive Resolution in A14
1. **Canonical Selection Pipeline:** Implemented [a14_select_canonical_registry.py](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/scripts/a14_select_canonical_registry.py), which applies deterministic selection rules:
   - For all cryptocurrency datasets (Ethereum, BSC, Polygon), the repaired runs (`benchmark_a08_crypto_clean_v1`) strictly supersede the old `a03_production` runs.
   - Generated [canonical_selected_run_registry.csv](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a14/canonical_selected_run_registry.csv) (455 runs: 420 primary across 13 datasets + 35 LANL runs).
   - Generated [campaign_conflict_report.csv](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a14/campaign_conflict_report.csv) and [campaign_selection_rationale.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a14/campaign_selection_rationale.md) documenting all 105 superseded runs.
2. **Harmonized Comparison:**

| Dataset | Metric | A13 Inconsistent Figure 3 (`a03_production`) | Canonical A14 (Figure 3, Table 4, Text) (`a08_crypto_clean_v1`) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Ethereum** | Mean $\Delta\mathrm{AP}$ | $-0.0015$ | **$+0.0253$** | **Harmonized Positive** |
| **BSC** | Mean $\Delta\mathrm{AP}$ | $-0.1983$ (Clipped off-screen) | **$+0.0021$** (Visible, within bounds) | **Harmonized Positive** |
| **Polygon** | Mean $\Delta\mathrm{AP}$ | $+0.0055$ | **$-0.0124$** | **Harmonized Negative** |

All figures, tables, text sentences, and statistical assertions across both manuscripts now strictly reflect the canonical values.

---

## 3. Visual Layout & Architectural Graphic Verification

Visual rendering was audited via high-resolution PDF rendering (`pdftoppm -r 300`):
1. **Figure 3 (`fig_conditional_delta.pdf`):**
   - Axis limits dynamically set with safety margin to `[-0.112, 0.065]`.
   - BSC ($+0.0021 \pm 0.0048$), Ethereum ($+0.0253 \pm 0.0045$), and Polygon ($-0.0124 \pm 0.0031$) are completely visible.
   - Margin adjustments (`left=0.22`, `right=0.92`, `bottom=0.12`, `top=0.92`) completely prevent label clipping.
   - Dataset names and numeric value annotations have no collisions or overlap.
2. **Figure 2 (`fig_support_matrix.pdf`):**
   - Strictly validates 84 primary supported, 7 external, and 7 unsupported cells. Fail-closed assertion passed.
3. **Figure 1 (`fig_dlg_architecture.pdf`):**
   - Vector graphics rendered cleanly with crisp fonts and no overlapping boxes.
4. **Compiled Manuscript Layout:**
   - Both Preprints (18 pages) and MDPI Applied Sciences (16 pages) compile cleanly with zero overflow warnings, zero `?` citation marks, and zero broken internal cross-references.

---

## 4. Negative Control Validation Summary

The cross-artifact validator (`a14_validate_final_manuscripts.py`) executed 6 deliberate negative mutation tests against an isolated copy of the manuscript and data artifacts:

| Test ID | Negative Mutation Target | Injected Value | Validator Behavior | Result |
| :---: | :--- | :--- | :--- | :---: |
| **NC-1** | BSC sign in Table 4 | Changed $+0.0021$ to $-0.0021$ | Trapped: Table value mismatch vs CSV | **PASS (Caught)** |
| **NC-2** | Ethereum sign in Table 4 | Changed $+0.0253$ to $-0.0253$ | Trapped: Table value mismatch vs CSV | **PASS (Caught)** |
| **NC-3** | Reddit base AP typo | Changed $0.4044$ to $0.4045$ | Trapped: Exact 4-decimal mismatch | **PASS (Caught)** |
| **NC-4** | Stale Git tag string | Injected unreleased tag string | Trapped: Missing canonical A14 release tag | **PASS (Caught)** |
| **NC-5** | Unqualified causal claim | Injected *"DLG inherently solves"* | Trapped: Forbidden causal phrase detected | **PASS (Caught)** |
| **NC-6** | Figure source corruption | Corrupted CSV delta column | Trapped: Figure checksum mismatch | **PASS (Caught)** |

**Conclusion:** All 6 negative controls failed fast with informative exceptions. The validator was verified to be fail-closed and sensitive.

---

## 5. Artifact Manifest & SHA-256 Hashes

| Artifact Category | Relative File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Canonical Registry** | `projects/benchmark/publication/a14/canonical_selected_run_registry.csv` | `45d4ca7b4a9b010aa779e7cb60a1816e0f1c08e98039a119d008a1d1cd5db907` |
| **Paired CSV** | `projects/benchmark/publication/a14/paired_ap_by_seed.csv` | `b6807f7c46f6630f9d98cf71fa088469c4701dd92fa942be6a427ae3e2fa46b3` |
| **Summary Delta CSV** | `projects/benchmark/publication/a14/conditional_ap_14.csv` | `295a0924734563a340b07b34b1509a25b16f39446a0c5fe674b02d847c293fe2` |
| **Table 4 LaTeX** | `projects/benchmark/paper/a14/tables/table_conditional_delta.tex` | `cb552f440f353a3b5a1b3fc1418cefa5e46be9f2682974cc5bbf729ec0b1f7b9` |
| **Figure 3 PDF** | `projects/benchmark/paper/a14/figures/fig_conditional_delta.pdf` | `e2a4be6ce67657d47225c5a0fb70cb5c1894d348a4be0ebaa942a632b70f0322` |
| **Preprint Source ZIP**| `projects/benchmark/submission/preprints/source_bundle.zip` | `bf44083758b998cf7aa6541f92e399bf080ee1ea30c00c0f997c679234ddbe23` |
| **Preprint PDF** | `projects/benchmark/submission/preprints/DLG-Benchmark_A14_Preprints.pdf` | `aa948c2275811c7ff10aee75a401416e83d8e57849c065f4d8e87e14ba69f9ba` |
| **MDPI Source ZIP** | `projects/benchmark/submission/applsci/source_bundle.zip` | `d041e2eb98b0f77d332aa6c382352bfa32e3ffcb24d2ef07cfc8430e62d4c0c1` |
| **MDPI PDF** | `projects/benchmark/submission/applsci/DLG-Benchmark_A14_AppliedSciences.pdf` | `ee95a940428ef5dc100f22c608f3f87f9eb4bc268d3744957e84cbef130f1484` |
| **MDPI Cover Letter** | `projects/benchmark/submission/applsci/COVER_LETTER.md` | `9d5598696ab1ce502b4fcbc4fe0c239d5e3ef54ce3da2f04e1e355c7a40b7100` |
| **Validation Script** | `projects/benchmark/scripts/a14_validate_final_manuscripts.py` | `0d06114eb9aa6cfaad9001b97cf86a11e5a5c68bfa93ca33c9aa5d27b1eeaf3c` |
| **Reproduction Script**| `projects/benchmark/scripts/a14_verify_release.py` | `43282245b7863f699fc7102e1cbb3d67f4057e6ae5d10d9be173d120a169b5fe` |

---

## 6. Dual Submission Readiness Sign-Offs

### Stage 1: PREPRINT READY (Preprints.org)
- [x] Dedicated Preprints LaTeX manuscript (`DLG-Benchmark_A14_Preprints.tex`) compiled cleanly to 18-page publication PDF (`DLG-Benchmark_A14_Preprints.pdf`).
- [x] Complete self-contained source bundle (`source_bundle.zip`, 117,721 bytes) verified in clean unpack directory.
- [x] Clear preprint data availability and distinct Benchmark Preprint DOI statement included.
- [x] Automated preprints gate passed: [PREPRINT_GATE_REPORT_A14.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/PREPRINT_GATE_REPORT_A14.md).

**Sign-off:** **`A14 PREPRINT READY`**

---

### Stage 2: APPLIED SCIENCES SUBMISSION PACKAGE READY (MDPI)
- [x] Official MDPI LaTeX template (`DLG-Benchmark_A14_AppliedSciences.tex`, `mdpi.cls`) compiled cleanly to 16-page publication PDF (`DLG-Benchmark_A14_AppliedSciences.pdf`).
- [x] Complete self-contained source bundle (`source_bundle.zip`, 1,153,703 bytes) verified in clean unpack directory with all EPS/PDF logos and BST styles.
- [x] Comprehensive Cover Letter targeting Special Issue *"Machine Learning and Its Application for Anomaly Detection"* (Special Issue ID: `C67H3U9QGT`, deadline 2026-12-31).
- [x] Seven dedicated Applied Sciences references added and verified in citations.
- [x] Automated submission gate passed: [SUBMISSION_GATE_REPORT_A14.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/SUBMISSION_GATE_REPORT_A14.md).

**Sign-off:** **`A14 APPLIED SCIENCES PACKAGE READY`**

---

## 7. Scope Adherence & Methodological Transparency Declaration

1. **No Broad Retraining:** The benchmark results remain 100% frozen. No model was retrained; no hyperparameters were altered.
2. **Canonical Run Integrity:** Exactly 455 runs (420 primary across 13 datasets + 35 LANL runs) are approved. All rankings, Friedman test statistics ($p < 0.05$), and Nemenyi post-hoc CD values remain exactly as established in approved records.
3. **Scientific Humility & Honest Framing:**
   - Zero claims of universal superiority or intrinsic structural causation.
   - The DLG condition is explicitly presented as a modular, structural augmentation whose benefit is conditional on graph diameter, homophily, and feature density.
   - Explicit acknowledgment that raw score arrays for older historical runs were summarized prior to A12, and that canonical paired evaluation is strictly documented at the seed summary level.
4. **Permanent Public Traceability:** The repository is tagged at `benchmark-a14-submission-2026-10-24` and pushed to remote `origin`. All artifacts, scripts, and figures can be verified with `python a14_verify_release.py --mode all`.
