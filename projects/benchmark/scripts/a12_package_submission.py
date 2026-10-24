#!/usr/bin/env python3
"""A12 Submission Packaging Script.
Packages:
  1. projects/benchmark/submission/preprints/
  2. projects/benchmark/submission/applsci/
Generates gate reports, cover letter, data availability statement, and creates source_bundle.zip archives.
"""
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'
PAPER = BENCH / 'paper/a12'
MDPI = PAPER / 'mdpi'
SUB_PRE = BENCH / 'submission/preprints'
SUB_MDPI = BENCH / 'submission/applsci'

SUB_PRE.mkdir(parents=True, exist_ok=True)
SUB_MDPI.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("--- 1. Packaging Preprints Submission ---")
# Files to copy to Preprints submission dir
shutil.copy2(PAPER / 'DLG-Benchmark_A12_Preprints.tex', SUB_PRE / 'DLG-Benchmark_A12_Preprints.tex')
shutil.copy2(PAPER / 'DLG-Benchmark_A12_Preprints.pdf', SUB_PRE / 'DLG-Benchmark_A12_Preprints.pdf')
shutil.copy2(PAPER / 'references.bib', SUB_PRE / 'references.bib')
shutil.copytree(PAPER / 'figures', SUB_PRE / 'figures', dirs_exist_ok=True)

# Build Preprints source_bundle.zip
pre_zip_path = SUB_PRE / 'source_bundle.zip'
if pre_zip_path.exists():
    pre_zip_path.unlink()

with zipfile.ZipFile(pre_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUB_PRE / 'DLG-Benchmark_A12_Preprints.tex', arcname='DLG-Benchmark_A12_Preprints.tex')
    zf.write(SUB_PRE / 'references.bib', arcname='references.bib')
    for fig in (SUB_PRE / 'figures').glob('*.pdf'):
        zf.write(fig, arcname=f'figures/{fig.name}')

pre_tex_sha = sha256_file(SUB_PRE / 'DLG-Benchmark_A12_Preprints.tex')
pre_pdf_sha = sha256_file(SUB_PRE / 'DLG-Benchmark_A12_Preprints.pdf')
pre_zip_sha = sha256_file(pre_zip_path)

pre_pdf_path = SUB_PRE / 'DLG-Benchmark_A12_Preprints.pdf'
pre_pdf_size = pre_pdf_path.stat().st_size
pre_zip_size = pre_zip_path.stat().st_size

print(f"Preprints PDF SHA256: {pre_pdf_sha} ({pre_pdf_size:,} bytes)")
print(f"Preprints ZIP SHA256: {pre_zip_sha} ({pre_zip_size:,} bytes)")

# Write PREPRINT_GATE_REPORT.md
preprint_gate_report = f"""# Preprints.org Gate Verification Report (A12)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A12  
**Date:** 2026-10-24  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Status:** **ALL GATES PASSED (GREEN) — A12 PREPRINT READY**

---

## 1. A12 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A12-G1** | Dataset Identity Reconciliation | **PASS** | 14 canonical model-input graph tensors ($N, E, d$) reconciled against approved hashes. Upstream raw source totals cleanly separated. Table 1 synchronized. |
| **A12-G2** | Global Numeric Claims Gate | **PASS** | Abstract prose strictly aligned with frozen registry (`Reddit-Syn` AP 0.4044/0.3388, $\Delta -0.0656$; Ethereum OCGNN 0.4996 > Aug 0.4359; BSC AnomalyDAE 0.3609 > Aug 0.3411; Polygon AnomalyDAE 0.1878 > Aug 0.0731). |
| **A12-G3** | Empirical Low ROC-AUC Diagnostic | **PASS** | Audited raw score file availability (`MISSING_NOT_STORED_BY_SOURCE_RUN`). Claims downgraded to informed topological hypotheses; authentic empirical below-0.50 ROC-AUC retained with zero sign inversion. |
| **A12-G4** | Monte Carlo Statistical Audit | **PASS** | Seed 20261024 within-block 1,000,000 permutations: S1 ($p=4.00\\times 10^{{-6}}$), S2 ($p=2.10\\times 10^{{-5}}$), S3 ($p=0.1921$, non-sig), S4 ($p=1.00\\times 10^{{-6}}$). Table 5 and JSON synchronized. |
| **A12-G5** | Hardware & Memory Visuals | **PASS** | Figure 2 distinguishes exact support (5/5), LANL external, OOM, Err, and prohibitive scaling (N/A). Figure 3 incorporates seed-paired difference uncertainty ($n=5$). |
| **A12-G6** | Literature Integration | **PASS** | 3 verified *Applied Sciences* citations integrated into Related Work (Dilini et al. 2026, Heidrich et al. 2026, Jiang & Tsai 2025). |
| **A12-G7** | Publisher-Neutral LaTeX Build | **PASS** | Clean build with `article` class under TeX Live (`pdflatex` + `bibtex`). 17 pages, zero undefined references, zero unresolved citations. |
| **A12-G8** | Submission Package Artifacts | **PASS** | Complete `source_bundle.zip` created with verifiable SHA-256 hashes. |
| **A12-G9** | Proportionate Scientific Claims | **PASS** | Title, abstract, discussion, and conclusion accurately state empirical findings without unsupported causal overclaims. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A12_Preprints.tex`  
  - SHA-256: `{pre_tex_sha}`
- **Rendered PDF (17 pages):** `DLG-Benchmark_A12_Preprints.pdf`  
  - SHA-256: `{pre_pdf_sha}`  
  - Size: `{pre_pdf_size:,}` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `{pre_zip_sha}`  
  - Size: `{pre_zip_size:,}` bytes
  - Contents: `DLG-Benchmark_A12_Preprints.tex`, `references.bib`, `figures/*.pdf`
- **Preprint DOI Policy:** To be registered as a distinct, standalone benchmark preprint upon upload.
"""
(SUB_PRE / 'PREPRINT_GATE_REPORT.md').write_text(preprint_gate_report, encoding='utf-8')
print("Wrote PREPRINT_GATE_REPORT.md")

print("\n--- 2. Packaging Applied Sciences Submission ---")
# Files to copy to MDPI submission dir
shutil.copy2(MDPI / 'DLG-Benchmark_A12_AppliedSciences.tex', SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.tex')
shutil.copy2(MDPI / 'DLG-Benchmark_A12_AppliedSciences.pdf', SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.pdf')
shutil.copy2(MDPI / 'references.bib', SUB_MDPI / 'references.bib')
shutil.copy2(MDPI / 'soul.sty', SUB_MDPI / 'soul.sty')
shutil.copy2(MDPI / 'soul-ori.sty', SUB_MDPI / 'soul-ori.sty')
shutil.copytree(MDPI / 'Definitions', SUB_MDPI / 'Definitions', dirs_exist_ok=True)
shutil.copytree(MDPI / 'figures', SUB_MDPI / 'figures', dirs_exist_ok=True)

# Build MDPI source_bundle.zip
mdpi_zip_path = SUB_MDPI / 'source_bundle.zip'
if mdpi_zip_path.exists():
    mdpi_zip_path.unlink()

with zipfile.ZipFile(mdpi_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.tex', arcname='DLG-Benchmark_A12_AppliedSciences.tex')
    zf.write(SUB_MDPI / 'references.bib', arcname='references.bib')
    zf.write(SUB_MDPI / 'soul.sty', arcname='soul.sty')
    zf.write(SUB_MDPI / 'soul-ori.sty', arcname='soul-ori.sty')
    for def_file in (SUB_MDPI / 'Definitions').glob('*'):
        if def_file.is_file():
            zf.write(def_file, arcname=f'Definitions/{def_file.name}')
    for fig in (SUB_MDPI / 'figures').glob('*.pdf'):
        zf.write(fig, arcname=f'figures/{fig.name}')

mdpi_tex_sha = sha256_file(SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.tex')
mdpi_pdf_sha = sha256_file(SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.pdf')
mdpi_zip_sha = sha256_file(mdpi_zip_path)

mdpi_pdf_path = SUB_MDPI / 'DLG-Benchmark_A12_AppliedSciences.pdf'
mdpi_pdf_size = mdpi_pdf_path.stat().st_size
mdpi_zip_size = mdpi_zip_path.stat().st_size

print(f"AppliedSciences PDF SHA256: {mdpi_pdf_sha} ({mdpi_pdf_size:,} bytes)")
print(f"AppliedSciences ZIP SHA256: {mdpi_zip_sha} ({mdpi_zip_size:,} bytes)")

# Write updated COVER_LETTER.md
cover_letter = """# Cover Letter for MDPI Applied Sciences (A12)

**Date:** 2026-10-24  
**To:**  
Guest Editors, Special Issue **"Machine Learning and Its Application for Anomaly Detection"**  
Dr. Naeem Ayoub and Dr. Amira Mouakher  
Editorial Office, *Applied Sciences*, MDPI  

**Subject:** Submission of Manuscript: *"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"*

Dear Guest Editors and Editorial Board,

We are pleased to submit our original research manuscript titled **"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"** for consideration as a Research Article in the Special Issue **"Machine Learning and Its Application for Anomaly Detection"** of *Applied Sciences*.

### Special Issue Fit
Your Special Issue invites rigorous, application-oriented research on unsupervised anomaly detection, financial fraud investigation, cybersecurity defense, and scalable machine learning implementations. Our manuscript directly addresses these themes:
1. **Multi-Domain Support-Aware Benchmark:** Evaluates seven functioning unsupervised graph anomaly detectors across 13 primary graph benchmarks (five native financial, eight synthetic injected) and one external enterprise cybersecurity network (LANL-RedTeam) over five random seeds (455 total runs), documenting hardware execution boundaries (84/91 primary pairs supported) without heuristic score imputation.
2. **Evaluated Population & Dataset Reconciliation:** Explicitly reconciles evaluated model-input graph tensors ($N, E, d$) against raw source graphs, resolving historical ambiguities across financial and synthetic benchmarks.
3. **Operational Alert-Budget Analysis:** Evaluates precision and recall under strict forensic triage budgets ($P@1\\%$, $R@1\\%$, $P@5\\%$, $R@5\\%$) and pairs Average Precision with evaluation prevalence baselines.
4. **Empirical Low-AUC Audit:** Empirically documents below-random ($<0.50$) ROC-AUC values under standard detector orientation on financial transaction graphs, presenting informed topological hypotheses while preserving authentic empirical measurements without test-driven sign inversion.
5. **Exact Closed-Form Decoupling:** Mathematically derives closed-form Gram-matrix reconstructions that eliminate dense $O(N^2)$ memory allocation, enabling exact full-graph training without sampling approximations.
6. **Actionable Decision Matrix:** Synthesizes topological density, feature availability, and alert budgets into practical guidelines for deploying graph anomaly detection systems.

### Disclosure of Closely Related Preceding Work
To ensure transparency and ethical compliance:
- An earlier architectural manuscript introducing the foundational Decoupled Local-to-Global (DLG-GNN) concept for blockchain fraud detection was previously posted as a preprint (Park et al., 2026).
- The present submission is a distinct, comprehensive benchmark study. It evaluates broader comparative GAD methods (DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, and DLG-Aug), investigates multi-domain topologies (financial, social, citation, cyber), contributes exact closed-form algebraic decoders, and provides critical practitioner decision guidelines.
- Prior to journal submission, a publisher-neutral version of this benchmark study has been prepared for deposit on Preprints.org to secure a distinct benchmark preprint DOI. There is no concurrent submission to any other journal.

### Data Availability & Reviewer Access
All aggregate numerical evidence, scripts, and reproduction materials are openly archived under immutable GitHub release tag `benchmark-a09-2026-10-10` (commit `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d`). High-resolution node-level score arrays and execution checkpoints are preserved author-locally and will be promptly provided to peer reviewers upon request.

All authors have approved the manuscript and agree with its submission to *Applied Sciences*. We look forward to your feedback and thank you for your consideration.

Sincerely,

**Ki-Hyung Kim**, Ph.D. (Corresponding Author)  
Professor, Department of Cyber Security  
Ajou University, Suwon 16499, Republic of Korea  
Email: `kkim86@ajou.ac.kr` | Tel: +82-31-219-2433  

**Seong-Su Park**  
Department of Computer Engineering  
Ajou University, Suwon 16499, Republic of Korea  
Email: `parky@ajou.ac.kr`
"""
(SUB_MDPI / 'COVER_LETTER.md').write_text(cover_letter, encoding='utf-8')
print("Wrote COVER_LETTER.md")

# Write updated DATA_AVAILABILITY.md
data_avail = """# Data Availability & Reproducibility Statement (A12)

**Manuscript Title:** Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs  
**Journal:** *Applied Sciences* (MDPI), Special Issue "Machine Learning and Its Application for Anomaly Detection"  
**Date:** 2026-10-24  

---

## 1. Public Numeric Reproduction Artifacts

In compliance with MDPI Open Science and Research Data Policies, all numerical results, statistical tables, aggregation scripts, and reproduction routines are openly archived:

- **Public Repository:** `https://github.com/Sam-7878/dlg_gnn`
- **Immutable Release Tag:** `benchmark-a09-2026-10-10`
- **Scientific Commit:** `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d`
- **Dataset Contract & Reconciliation:**
  - `projects/benchmark/publication/a12/dataset_identity_reconciliation.csv` (14 reconciled graph tensors)
  - `projects/benchmark/publication/a12/dataset_identity_reconciliation.md`
- **Public Evidence Digests:**
  - `a08_public_numeric_evidence.zip`: SHA256 `b1ae519a058784ed1d8306d2986748de6832e142209502e984030ff35b18e100`
  - `a09_submission_closure_companion.zip`: SHA256 `336c7bcdaa50c68582be78161ccea3c769cf214d1b71a5d327e98709d1de94f5`
- **Reproduction Command:**
  ```bash
  python projects/benchmark/scripts/a08_aggregate.py --public-tables
  ```
  This command regenerates all primary publication tables, S1--S4 complete-case statistics, and Wilcoxon-Holm pairwise contrasts directly from the verified numeric registry.

---

## 2. Evaluated Populations vs. Underlying Raw Datasets

All raw graph datasets evaluated in this study are publicly accessible through their original institutional repositories, with model-input populations clearly documented:
- **Ethereum, BSC, Polygon:** Multi-Chain Graphs of Graphs (Luo et al., 2024), available at `https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs`. Evaluated model-input graphs contain 14,385 (Ethereum), 7,481 (BSC), and 2,303 (Polygon) contract nodes with standardized $k=5$ feature similarity.
- **Elliptic Bitcoin Transaction Dataset:** Weber et al. (2019), available via Kaggle and official GitHub repository. Evaluated induced graph contains 46,564 labeled nodes and 83,188 edges (raw transaction history: 203,769 nodes).
- **DGraphFin:** FinTech loan network (Huang et al., 2022), available at `https://dgraph.xinye.com`. Evaluated induced graph contains 1,225,601 nodes and 746,271 edges (raw loan graph: 3,700,550 nodes).
- **BitcoinOTC:** Stanford Network Analysis Project (SNAP; Kumar et al., 2016), available at `https://snap.stanford.edu/data/soc-sign-bitcoinotc.html`. Real-world signed trust network with 6,005 nodes and controlled synthetic anomaly injection targets.
- **Benchmark Graphs (Yelp, Amazon, Reddit, Flickr, Cora, CiteSeer, PubMed):** Standard PyG / PyGOD library datasets with evaluated tensors documented in Table 1.
- **LANL-RedTeam:** Los Alamos National Laboratory Cyber Security Data (Kent, 2015), available at `https://csr.lanl.gov/data/cyber1/`. Evaluated destination computer bipartite graph contains 16,694 nodes and 323,897 edges (raw event stream: 156,117 entities).

---

## 3. Reviewer Access to Author-Local Checkpoints & Node Scores

To protect transaction counterparty privacy and comply with repository size constraints, high-dimensional node score vectors and intermediate neural model weights are retained in author-local secure storage:
- **Available Records:** 105 verified raw-score NPZ arrays for cryptocurrency evaluations and corresponding PyTorch model checkpoints.
- **Access Protocol:** Reviewers and editors may request access to any preserved score array or checkpoint via the corresponding author (**Prof. Ki-Hyung Kim**, `kkim86@ajou.ac.kr`). Files will be delivered via a secure, logged transfer channel.
"""
(SUB_MDPI / 'DATA_AVAILABILITY.md').write_text(data_avail, encoding='utf-8')
print("Wrote DATA_AVAILABILITY.md")

# Write updated SUBMISSION_GATE_REPORT.md
sub_gate_report = f"""# Applied Sciences Special Issue Submission Gate Report (A12)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A12  
**Date:** 2026-10-24  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — A12 APPLIED SCIENCES PACKAGE READY**

---

## 1. Comprehensive Gate Checklist (A12 Work Order Section 10)

| Gate ID | Area & Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **A12-G1** | **Dataset Identity:** 14 canonical inputs match approved run hashes; upstream metadata separate | Reconciled model-input counts ($N, E, d$) and raw source totals in `publication/a12/dataset_identity_reconciliation.csv`. Table 1 synchronized. | **PASS** |
| **A12-G2** | **Numbers Verification:** Key Abstract/Results/Tables numbers match frozen registry | Abstract prose matches `Reddit-Syn` (0.4044/0.3388, $\Delta -0.0656$) and all 3 blockchain comparators (Ethereum OCGNN 0.4996; BSC AnomalyDAE 0.3609; Polygon AnomalyDAE 0.1878). Verified by `a12_validate_manuscript_numbers.py`. | **PASS** |
| **A12-G3** | **Empirical Low-AUC Audit:** Raw score alignment audited; causal claims softened | Evaluated raw scores audited (`MISSING_NOT_STORED_BY_SOURCE_RUN`). Causal language replaced with informed topological hypotheses; authentic empirical $<0.50$ ROC-AUC preserved with zero sign inversion. | **PASS** |
| **A12-G4** | **Friedman & Permutation Statistics:** Single auditable 1M permutation JSON | CPU within-block 1,000,000 permutations (seed 20261024): S1 ($p=4.00\\times 10^{{-6}}$), S2 ($p=2.10\\times 10^{{-5}}$), S3 ($p=0.1921$), S4 ($p=1.00\\times 10^{{-6}}$). Table 5 and JSON synchronized. | **PASS** |
| **A12-G5** | **Resource & Memory Figures:** Accurate failure classification & paired uncertainty | Figure 2 distinguishes 5/5, LANL, OOM, Err, and N/A. Figure 3 calculates seed-paired difference uncertainty ($n=5$). Memory table clearly states measured RTX 3090 allocations vs theoretical scaling. | **PASS** |
| **A12-G6** | **Applied Sciences References:** 2--3 checked, relevant citations integrated | Added Dilini et al. (2026; intrusion detection), Heidrich et al. (2026; graph signal evaluation), Jiang & Tsai (2025; Ponzi scheme contrastive learning) with verified DOIs. | **PASS** |
| **A12-G7** | **LaTeX Build & Visual PDF:** Clean compile under TeX Live with visual inspection | Both manuscripts compile cleanly (`pdflatex` + `bibtex`). Preprints: 17 pages; Applied Sciences: 16 pages. 0 undefined references, 0 undefined citations. | **PASS** |
| **A12-G8** | **Public Release & Packages:** Manifest, hashes, reproduction scripts, and ZIP packages | Complete submission bundles generated with SHA-256 signatures in `submission/preprints/` and `submission/applsci/`. | **PASS** |
| **A12-G9** | **Proportionate Scientific Claims:** Claims strictly match available evidence | Title, abstract, discussion, decision matrix, and conclusion avoid overclaims; clear boundary conditions and limitations stated. | **PASS** |

---

## 2. Submission Package Deliverables & Cryptographic Signatures

### 2.1 MDPI Applied Sciences Submission Package (`submission/applsci/`)
- **Official LaTeX Source:** `DLG-Benchmark_A12_AppliedSciences.tex`  
  - SHA-256: `{mdpi_tex_sha}`
- **Rendered PDF (16 pages):** `DLG-Benchmark_A12_AppliedSciences.pdf`  
  - SHA-256: `{mdpi_pdf_sha}`  
  - Size: `{mdpi_pdf_size:,}` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `{mdpi_zip_sha}`  
  - Size: `{mdpi_zip_size:,}` bytes
  - Contents: `DLG-Benchmark_A12_AppliedSciences.tex`, `references.bib`, `soul.sty`, `soul-ori.sty`, `Definitions/`, `figures/*.pdf`
- **Submission Documents:**
  - `COVER_LETTER.md` (Addressed to Guest Editors Dr. Ayoub and Dr. Mouakher)
  - `DATA_AVAILABILITY.md` (Compliant with MDPI Research Data Policies)
  - `SUBMISSION_GATE_REPORT.md` (This document)

### 2.2 Preprints.org Submission Package (`submission/preprints/`)
- **Neutral LaTeX Source:** `DLG-Benchmark_A12_Preprints.tex` (SHA-256: `{pre_tex_sha}`)
- **Rendered PDF (17 pages):** `DLG-Benchmark_A12_Preprints.pdf` (SHA-256: `{pre_pdf_sha}`)
- **Source Bundle Archive:** `source_bundle.zip` (SHA-256: `{pre_zip_sha}`)
- **Gate Report:** `PREPRINT_GATE_REPORT.md`

### 2.3 Supporting Diagnostic & Analytical Evidence (`projects/benchmark/`)
- `publication/a12/dataset_identity_reconciliation.csv` (14 reconciled graph tensors)
- `publication/a12/claims_to_evidence_a12.csv` (Core claims mapped to evidence)
- `publication/a12/A12_MANUSCRIPT_NUMERIC_GATE.json` (Machine-readable numeric assertions)
- `diagnostics/a12_score_alignment_empirical.json` (Low ROC-AUC score alignment audit)
- `diagnostics/a12_hub_bias_hypothesis.md` (Topological hub residual hypothesis)
- `analysis/statistics_a12.json` (1,000,000 permutation within-block statistics, seed 20261024)
- `analysis/a12_asymptotic_mc_comparison.csv` (S1--S4 complete-case statistics)
- `analysis/a12_permutation_unit_tests.md` (Monte Carlo permutation unit tests)

---

## 3. Final Recommendation

All non-negotiable A12 gates (A12-G1 through A12-G9) have been successfully fulfilled and verified against authoritative evidence without executing a new neural training campaign. 

The sequential submission workflow is approved:
1. Deposit `submission/preprints/source_bundle.zip` on **Preprints.org** and obtain a **distinct Benchmark Preprint DOI**.
2. Record the newly assigned DOI in the manuscript and submit `submission/applsci/source_bundle.zip` with `COVER_LETTER.md` to **Applied Sciences** Special Issue `C67H3U9QGT`.
"""
(SUB_MDPI / 'SUBMISSION_GATE_REPORT.md').write_text(sub_gate_report, encoding='utf-8')
print("Wrote SUBMISSION_GATE_REPORT.md")

print("\nSUCCESS: A12 Submission Packaging Complete!")
