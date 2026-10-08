# P1 Audit Report 01: Final Scientific Wording Audit

## 1. Executive Summary
- **Objective**: Verify that scientific terminology across abstract, introduction, and methods has been rigorously remediated to prevent misinterpretation of baseline architectures and sensitivity controls.
- **Scope**: `docs/papers/_42_Benchmark/DLG-Benchmark.tex`, `publication/benchmark/preprints/DLG-Benchmark-Preprint.tex`, and `publication/benchmark/mdpi/DLG-Benchmark.tex`.
- **Status**: **100% PASSED** (All remediation criteria satisfied).

---

## 2. Remediated Terminology Audit

### 2.1 DLG-Base Description
- **Previous Over-simplification**: `"matched global reconstruction baseline"`, `"matched global reconstruction model"`.
  - *Risk*: Misled readers to assume DLG-Base omitted local layers or was a purely global baseline.
- **Remediated Precision Wording**:
  - `"...while DLG-Base provides the historical non-augmentation reconstruction baseline."`
  - Accurately captures that DLG-Base is a 2-layer local GCN $\to$ 2-layer global GCN model with scalar sigmoid gating, but without the local-to-global graph augmentation input.
- **Occurrences in Manuscript Sources**:
  - `Abstract`: Verified replaced in Master, Preprints, and MDPI LaTeX sources.
  - `Introduction`: Verified consistent historical baseline terminology.

### 2.2 Capacity Controls Terminology
- **Previous Over-statement**: `Network Capacity Control (DLG-Aug-Zero)`.
  - *Risk*: Misunderstood as a full causal identification experiment or complete capacity matching.
- **Remediated Precision Wording**:
  - `Zero-Information Augmentation Control (\texttt{DLG-Aug-Zero})`.
- **Introductory Framing**:
  - Refined to explicitly clarify exploratory sensitivity probing rather than causal identification:
  > *"To probe the sensitivity of the observed differences to auxiliary information, node alignment, and optimization budget, we evaluated three targeted controls across Elliptic, DGraphFin, and LANL-RedTeam:"*

---

## 3. Preservation of Empirical Claims and Hashes (Source-of-Truth Reconciled)
- **Primary Raw Data Hash**: `39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c` (100% FROZEN).
- **Support Matrix Hash**: `c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914` (100% FROZEN).
- **45 M3 Sensitivity Controls**: Unchanged.
- **Empirical Metrics (Programmatically Verified against Frozen Artifacts)**:
  - **Elliptic**: DLG-Aug PR-AUC = $0.1037 \pm 0.0048$ vs DLG-Base = $0.0687 \pm 0.0025$ (Paired difference: $+0.0350$)
  - **DGraphFin**: DLG-Aug PR-AUC = $0.0134 \pm 0.0019$ vs DLG-Base = $0.0101 \pm 0.0002$ (Paired difference: $+0.0034$)
  - **Reddit-Syn**: DLG-Aug PR-AUC = $0.3388 \pm 0.0102$ vs DLG-Base = $0.4044 \pm 0.0104$ (Paired difference: $-0.0656$)
  - **LANL-RedTeam**: ROC-AUC: GADNR = $0.8261 \pm 0.0218$, DLG-Base = $0.7923 \pm 0.0452$, DLG-Aug = $0.7188 \pm 0.1010$; PR-AUC: DLG-Base = $0.1367$ vs DLG-Aug = $0.1114$ (DLG-Base outperforming DLG-Aug by $+0.0252$).
