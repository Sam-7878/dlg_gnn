# DLG-GNN Benchmark A20 Editorial and Evidence Diff Report

**Document ID:** `DLG-BENCH-DIFF-A19-TO-A20`  
**Date:** 2026-10-29  
**Scope:** Pre-submission Editorial, Provenance, and Evidence Alignment (No New Benchmark Cycles, No Model Retraining)  
**Authoritative Input Registry:** `projects/benchmark/publication/a20/a20_input_freeze.json`  
**Final Checksum Ledger:** `projects/benchmark/publication/a20/a20_final_sha256.txt`  

---

## 1. Executive Summary of Changes

The A20 revision strictly adheres to the work order `DLG-BENCH-WO-A20-EDITORIAL-ONLY`. It executes an editorial and evidence reconciliation pass addressing all findings from the A19 Independent Review and the peer review report on `DLG-Benchmark_A16_Preprints.pdf` (`claude_review.docx`).

**Core Scientific Contract:** 100% frozen. 14 datasets (13 primary + LANL), 7 primary models + CONAD diagnostic, 5 seeds (42–46), 455 total runs (420 primary + 35 LANL). No scores, weights, or seeds were modified or rerun.

---

## 2. Gate-by-Gate Diff Summary

### Gate G0 — Snapshot and Immutable Input Registry
- Recorded baseline SHA-256 digests for all A19 submission packages and empirical evidence files.
- Generated `projects/benchmark/publication/a20/a20_input_freeze.json` sealing 455 run artifacts, denominator audits, and hardware logs.

### Gate G1 — BSC Source-Map Contradiction Resolved (P0 / HARD)
- **Problem in A19:** Narrative report erroneously stated BSC test positive counts as `[215, 237, 217, 206, 230]`, whereas certified raw-recheck CSV and JSON have `[215, 240, 232, 215, 203]` (both means = 221.0).
- **A20 Resolution:**
  - Audited `a19_alert_budget_raw_recheck.csv` directly. Confirmed true seed-indexed counts: seed 42 = 215, seed 43 = 240, seed 44 = 232, seed 45 = 215, seed 46 = 203.
  - Published `a20_source_map_consistency.json` and `a20_authoritative_source_map.md`.
  - Executed equality assertion script verifying 0 mismatches across CSV, JSON, and manuscript text.

### Gate G2 — Claude R1: Bug Exclusion & Score Availability Disclosure (P0 / HARD)
- **Problem in A19/A16:** Manuscript text claimed that an independent audit of Elliptic and DGraphFin "ruled out implementation defects or sign inversion bugs." However, continuous per-node score arrays were not recorded at runtime for historical baselines.
- **A20 Resolution:**
  - Removed all absolute claims that software defects or sign errors were ruled out.
  - Rephrased Claim 3 to focus strictly on operational alert-budget evaluation and historical provenance boundaries.
  - Disclosed truthfully in Data Availability and Section 4.2 that per-node score arrays are preserved only for the 105 cryptocurrency evaluations, while the 350 historical baseline runs archived summary metrics only.
  - Compiled `a20_claims_to_evidence.csv` mapping all 10 core claims to explicit empirical evidence tiers.

### Gate G3 — Claude R2: Practitioner Decision Matrix (Table 7) (P0 / HARD)
- **Problem in A19/A16:** Table 7 mixed computational feasibility with predictive accuracy, presenting methods as "recommended" for large graphs where their actual performance was near or below random guessing (e.g., DOMINANT on DGraphFin with ROC-AUC $\approx 0.54$, CoLA on Elliptic with AP below prevalence).
- **A20 Resolution:**
  - Redesigned Table 7 with two distinct columns: *Computational Feasibility* and *Empirical Predictive Accuracy*.
  - Added sample size coverage ($n=1, 3$).
  - Clarified that DGraphFin Base/DOMINANT are executability-supported starting points near no-skill prevalence (not accurate fraud detectors), and CoLA is a comparator below prevalence on Elliptic.
  - Caveated AnomalyDAE on Reddit as unsupported in 24GB matrix.

### Gate G4 — Claude R3: Effect-Size Interpretation and Uncertainty (P0 / HARD)
- **Problem in A19/A16:** Manuscript emphasized relative percentage gain ($+50.97\%$) without giving equal weight to absolute $\Delta\mathrm{AP}$ ($+0.0094$), and overclaimed that BSC disproved universal benefit.
- **A20 Resolution:**
  - Primary presentation led with absolute metric difference $\Delta\mathrm{AP} = +0.0094$, placing relative percentage gain as secondary context.
  - Contextualized $\Delta\mathrm{AP}$ against test prevalence baseline ($\pi_\mathrm{test} = 0.0259$).
  - Softened text to state that BSC results "do not support a universal benefit under the studied conditions," rather than claiming "universal disproof."
  - Replaced unmeasured causal claims (e.g., "local features preserve structural fidelity") with observational descriptions of performance on citation networks.
  - Re-rendered Figure 3 (`fig_conditional_delta`) with descriptive title `Observed Mean Paired Difference (Delta AP) with Paired SE` instead of `Verified Canonical Ground Truth`.

### Gate G5 — Claude R4: Separating Measured VRAM, Profiles, and Theoretical Bounds (P0 / HARD)
- **Problem in A19/A16:** Manuscript described dense matrix requirements ($4N^2$) as if the models were executed and "crashed" with 217 GB VRAM.
- **A20 Resolution:**
  - Compiled `a20_memory_evidence_classes.csv` establishing 5 rigorous provenance tiers: `THEORETICAL_DENSE_MATRIX_LOWER_BOUND`, `ILLUSTRATIVE_PROFILE_ESTIMATE`, `MEASURED_ALLOCATED_VRAM`, `OBSERVED_OOM`, and `NOT_RUN_OR_UNSUPPORTED`.
  - Qualified Table 6 and Section 6: Reddit dense matrix requires $4N^2 \approx 217$ GB (DGraphFin 6.0 TB), mathematically exceeding 24GB capacity rather than "crashed"; 5.5 GB cited as "illustrative $\approx 5.5$-GB memory profile"; caveated AnomalyDAE row-block encoder OOM.

### Gate G6 — Claude R5: Split-Generator Audit and Small-Budget Uncertainty (P0 / HARD)
- **Problem in A19/A16:** Footnotes claimed splits were "random stratified cross-seed splits," but cryptocurrency splits were actually unstratified permutations.
- **A20 Resolution:**
  - Audited `evaluation/benchmark/v2/scripts/a04_build_dataset_manifest.py` (lines 98–105). Confirmed cryptocurrency splits use pure random permutation (`np.random.RandomState(seed).permutation(n)`) without stratification, explaining seed-dependent positive variation ($M_\mathrm{test} \in [7, 18]$ on Polygon).
  - Published `a20_split_protocol_audit.md`.
  - Corrected Table 1 and Table 2 footnotes to "seed-dependent evaluation splits (stratified for synthetic graphs; seeded random non-stratified 60/20/20 splits for cryptocurrency graphs)".
  - Added explicit discussion of Polygon small-budget discrete sensitivity ($K=5, \Delta P = 0.20$ per TP) in Table 3 caption and Section 4.3 text.

### Gate G7 — Editorial Prose, Duplicates, and Formatting (P1)
- **Duplicate Paragraphs Removed:**
  - Deleted duplicate CONAD exclusion paragraph in Section 2.1.
  - Deleted duplicate Wilcoxon minimum $p$ (0.0625) paragraph in Section 4.3.
- **Citation and BibTeX Fixes:**
  - Corrected `Fern'andez` to `Fern{\'a}ndez` in `references.bib`.
  - Corrected in-text reference `ADAG` to `ADA-GAD`.
  - Fixed MDPI citation header to `Park, S.-S.; Kim, K.-H.`.
  - Softened phrasing around Heidrich et al.
- **LANL Threshold Audit & Caption:**
  - Audited LANL run logs (`defense_real_rebuild`); compiled `a20_lanl_f1_threshold_audit.md`.
  - Qualified Table A2 caption regarding single-threshold recording. Added Elliptic all-positive baseline ($2\pi/(1+\pi) \approx 0.1778$).
- **Architecture Alignment:**
  - Aligned Section 3.1 DLG-Base description with the two-stage sequential GCN + fusion gate in Section 3.2.
  - Re-rendered Figure 1 (`fig_dlg_architecture`): removed incorrect $N \le 4.6 \times 10^3$ cryptocurrency limit and batching reference; labeled transductive full-graph structure correctly.

### Gate G8 — Statistical Bounds Phrasing (P1)
- **Monte Carlo Tail Probability:** Clarified Section 4.4 MC permutation phrasing: with $b=0$ extreme permutations out of $B=10^6$, the standard add-one estimator yields $p_\mathrm{MC} \approx 1.0 \times 10^{-6}$ with a one-sided 95% Clopper-Pearson upper confidence bound of $\approx 3.0 \times 10^{-6}$.
- **Equivalence Claim Softened:** Stated $p = 0.1921$ on synthetic graphs as "no statistically significant difference detected under the five-seed protocol," removing claims of proven equivalence.

### Gate G9 — Release Verification, PDF Recompilation, and Git Gate (P0 / HARD)
- Full recompilation via TeX Live 2025: Preprints (20 pages), Applied Sciences (19 pages).
- 0 undefined references, 0 undefined citations.
- 0 forbidden internal tokens across all 39 PDF pages.
- Source bundles built and verified 100% byte-for-byte identical to standalone source files.
- Manifests generated: `SHA256SUMS.txt` in both submission packages and `a20_final_sha256.txt`.
- Git release tag target: `benchmark-a20-preprint-2026-10-29`.
