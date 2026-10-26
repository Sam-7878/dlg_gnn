# DLG-GNN Benchmark — A15 Submission Evidence Closure Final Sign-Off

**Document ID:** DLG-BENCH-SIGNOFF-A15  
**Date:** 2026-10-25 (Asia/Seoul)  
**Parent Work Order:** [DLG_Benchmark_A15_Submission_Evidence_Closure_Work_Order_2026-10-24.md](file:///d:/_Work/goat_bank/dlg_gnn/docs/work_reports/benchmark/240_A15/DLG_Benchmark_A15_Submission_Evidence_Closure_Work_Order_2026-10-24.md)  
**Submission Venues & Sequence:**
1. **Preprints.org FIRST:** Secure distinct benchmark preprint DOI.
2. **MDPI *Applied Sciences* Special Issue SECOND:** "Machine Learning and Its Application for Anomaly Detection" (Special Issue ID: `C67H3U9QGT`, deadline 2026-12-31).

---

## 1. Executive Summary & Quality Gates Assessment

All eight mandatory quality gates stipulated in the A15 Submission Evidence Closure Work Order have been executed, verified, and officially **APPROVED**:

| Gate ID | Functional Area | Verification Target & Outcome | Audit Status |
| :---: | :--- | :--- | :---: |
| **A15-G1** | **Evaluation Population & Prevalence Ledger** | Generated `evaluation_population_by_seed.csv` across all 70 evaluations ($14 \times 5$). Arithmetic identity $\mathrm{prevalence} == \mathrm{pos} / \mathrm{total}$ verified within $10^{-6}$. Cryptographic test positive rates ($\pi_{\text{test}}$) reconciled: Ethereum ($0.4089$), BSC ($0.1473$), Polygon ($0.0259$). | **APPROVED** |
| **A15-G2** | **Support Matrix & Model Counts** | Pinned `support_14x7.csv` and `main_ap_mean_sd.csv`. Verified exactly 84 primary supported pairs ($84 \times 5 = 420$ runs) + 7 external LANL pairs ($7 \times 5 = 35$ runs) = 455 total runs. Documented exact failure etiologies: 5 GADNR OOM + 2 AnomalyDAE Err = 7 unsupported pairs. Typos corrected in text. | **APPROVED** |
| **A15-G3** | **Operational Claims & Report Harmonization** | Replaced unproven causal hub assertions with observational descriptions. Completely purged non-contract datasets (Tolokers, Questions, T-Finance, T-Social, YelpChi) from the execution report, replacing Table 2.4 with the true canonical 13+1 table. | **APPROVED** |
| **A15-G4** | **Verified Public Remote Git Identity** | Project tagged under immutable release tag `benchmark-a15-submission-2026-10-25` and pushed to remote GitHub origin; verified via `git ls-remote`. | **APPROVED** |
| **A15-G5** | **Disk SHA-256 Hashes & Clean Unpack** | Checksums computed from final disk bytes. Source bundles unpacked into `/tmp/a15_clean_unpack_test` and compiled cleanly via TeX Live (18 pages Preprints, 16 pages MDPI, 0 undefined citations/references). | **APPROVED** |
| **A15-G6** | **Extended Validator & Negative Controls** | Full-document AST/regex parser evaluated all tables, citations, and metadata. 8/8 negative control injection tests trapped and rejected fail-closed. | **APPROVED** |
| **A15-G7** | **Preprints.org Dual Sign-Off** | Full neutral bundle assembled, verified, and signed off as **A15 PREPRINT READY**. | **APPROVED** |
| **A15-G8** | **Applied Sciences Dual Sign-Off** | Complete MDPI journal submission package assembled, verified, and signed off as **A15 APPLIED SCIENCES PACKAGE READY**. | **APPROVED** |

---

## 2. Definitive Correction Notice: A14 Section 2.4 Table Superseded

In the preceding A14 execution report (`DLG_Benchmark_A14_Execution_Report_2026-10-24.md`), Section 2.4 inadvertently included an older draft table containing five non-contract datasets (Tolokers, Questions, T-Finance, T-Social, YelpChi) and obsolete AP numbers.

**Formal Correction:** That draft table is entirely superseded. The single canonical 14-dataset (13 primary + 1 LANL external) representation delta table is derived directly from `projects/benchmark/publication/a15/conditional_ap_14.csv` and appears below:

| Dataset | Benchmark Domain | DLG-Base AP ($\mu \pm \sigma$) | DLG-Aug AP ($\mu \pm \sigma$) | Delta AP ($\Delta\mu$) | Paired SE | Arithmetic Status ($\mu_{\text{Aug}} - \mu_{\text{Base}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Elliptic** | Bitcoin Transaction | $0.0687 \pm 0.0039$ | $0.1037 \pm 0.0051$ | $\mathbf{+0.0350}$ | $0.0022$ | Exact Match ($+0.0350$) |
| **DGraphFin** | Credit Risk / FinTech | $0.0101 \pm 0.0004$ | $0.0134 \pm 0.0007$ | $\mathbf{+0.0033}$ | $0.0004$ | Exact Match ($+0.0033$) |
| **BitcoinOTC** | Trust Network | $0.8274 \pm 0.0034$ | $0.8211 \pm 0.0052$ | $\mathbf{-0.0063}$ | $0.0028$ | Exact Match ($-0.0063$) |
| **Ethereum** | Smart Contract (Fraud) | $0.4106 \pm 0.0326$ | $0.4359 \pm 0.0163$ | $\mathbf{+0.0253}$ | $0.0144$ | Exact Match ($+0.0253$) |
| **BSC** | Smart Contract (Fraud) | $0.3390 \pm 0.0336$ | $0.3411 \pm 0.0287$ | $\mathbf{+0.0021}$ | $0.0103$ | Exact Match ($+0.0021$) |
| **Polygon** | Smart Contract (Fraud) | $0.0855 \pm 0.0163$ | $0.0731 \pm 0.0140$ | $\mathbf{-0.0124}$ | $0.0054$ | Exact Match ($-0.0124$) |
| **Yelp-Syn** | Social / Reviews | $0.3432 \pm 0.0267$ | $0.3150 \pm 0.0248$ | $\mathbf{-0.0282}$ | $0.0122$ | Exact Match ($-0.0282$) |
| **Amazon-Syn** | Co-Purchase Network | $0.7423 \pm 0.0169$ | $0.7303 \pm 0.0142$ | $\mathbf{-0.0120}$ | $0.0032$ | Exact Match ($-0.0120$) |
| **Reddit-Syn** | Discussion Forum | $0.4044 \pm 0.0053$ | $0.3388 \pm 0.0062$ | $\mathbf{-0.0656}$ | $0.0031$ | Exact Match ($-0.0656$) |
| **Flickr-Syn** | Image Social Network | $0.7850 \pm 0.0057$ | $0.7845 \pm 0.0060$ | $\mathbf{-0.0005}$ | $0.0016$ | Exact Match ($-0.0005$) |
| **Cora-Syn** | Citation Graph | $0.8066 \pm 0.0178$ | $0.8049 \pm 0.0165$ | $\mathbf{-0.0017}$ | $0.0032$ | Exact Match ($-0.0017$) |
| **CiteSeer-Syn** | Citation Graph | $0.7466 \pm 0.0270$ | $0.7483 \pm 0.0289$ | $\mathbf{+0.0017}$ | $0.0059$ | Exact Match ($+0.0017$) |
| **PubMed-Syn** | Biomedical Citation | $0.5739 \pm 0.0162$ | $0.5752 \pm 0.0164$ | $\mathbf{+0.0013}$ | $0.0028$ | Exact Match ($+0.0013$) |
| **LANL-RedTeam** | Enterprise Cyber Net | $0.2078 \pm 0.0463$ | $0.1826 \pm 0.0401$ | $\mathbf{-0.0252}$ | $0.0126$ | Exact Match ($-0.0252$) |

---

## 3. Cryptographic Artifact Checksums (Disk-Byte SHA-256)

| Artifact Description | Local File Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Preprints LaTeX Source** | `projects/benchmark/submission/preprints/DLG-Benchmark_A15_Preprints.tex` | $56,257$ | `db69f640e759d414bddbcc242cc0c666a4a020895405506c0a251ba78d3da7d7` |
| **Preprints Compiled PDF** | `projects/benchmark/submission/preprints/DLG-Benchmark_A15_Preprints.pdf` | $564,060$ | `243af5be49cfc219141b57ce0b51b8b41573ffc76968e4192805acc17fa415ec` |
| **Preprints Source Bundle**| `projects/benchmark/submission/preprints/source_bundle.zip` | $106,590$ | `85b47f0c51534980979322540252147b381940ddad8f709922a6be975a9167ce` |
| **Applied Sciences LaTeX** | `projects/benchmark/submission/applsci/DLG-Benchmark_A15_AppliedSciences.tex` | $55,880$ | `6b4570c25c59edb8ba5c1bec23a032d4e2e7633b7f1fe93557685b74d32f5bc8` |
| **Applied Sciences PDF** | `projects/benchmark/submission/applsci/DLG-Benchmark_A15_AppliedSciences.pdf` | $344,720$ | `41327e131711e9e905b5fea525b460694a3c0de25cc81c30d05f75fbecc5e1b0` |
| **Applied Sciences Bundle**| `projects/benchmark/submission/applsci/source_bundle.zip` | $1,142,571$ | `e78f6141013aad1976f7ae827c7b99eb6aaefb9cf175f5ade7c577c08f15decb` |
| **Evaluation Pop. Ledger** | `projects/benchmark/publication/a15/evaluation_population_by_seed.csv` | $28,314$ | `bd37bb61944232ac5bc5dd142e63a5d722ff3d0004e2f65384226c9306e2e18b` |
| **Main AP Mean/SD CSV** | `projects/benchmark/publication/a15/main_ap_mean_sd.csv` | $5,560$ | `dde635d7eff2504cc0e0ddf2807ac7ca3df8a844f1f89f2830139a3b43102fa0` |
| **Support Matrix 14x7** | `projects/benchmark/publication/a15/support_14x7.csv` | $6,061$ | `bbd89bf88f05047d2f8f29c2065c73791d480bdfcadb78293f7d03a543b8ecd8` |
| **Canonical Delta CSV** | `projects/benchmark/publication/a15/conditional_ap_14.csv` | $1,706$ | `183a4634899d16248884e4f4254538e2ee586ee09e40be86e6c8f349979362e6` |
| **Canonical Registry** | `projects/benchmark/publication/a15/canonical_selected_run_registry.csv` | $671,520$ | `45d4ca7b4a9b010aa779e7cb60a1816e0f1c08e98039a119d008a1d1cd5db907` |

---

## 4. Final Dual Sign-Off

### Stage 1: PREPRINT DEPOSIT READY
- [x] Neutral LaTeX manuscript `DLG-Benchmark_A15_Preprints.tex` compiled cleanly to 18-page publication PDF (`DLG-Benchmark_A15_Preprints.pdf`).
- [x] Self-contained archive `source_bundle.zip` (106,590 bytes) independently verified via clean directory build.
- [x] Clear data availability and preprint disclosure included.

**Official Sign-Off:** **`A15 PREPRINT READY`**

---

### Stage 2: APPLIED SCIENCES SUBMISSION PACKAGE READY
- [x] Official MDPI LaTeX manuscript `DLG-Benchmark_A15_AppliedSciences.tex` compiled cleanly to 16-page publication PDF (`DLG-Benchmark_A15_AppliedSciences.pdf`).
- [x] Full source bundle `source_bundle.zip` (1,142,571 bytes) independently verified with all definitions, logos, and style files.
- [x] Cover Letter addressed to Guest Editors Dr. Naeem Ayoub and Dr. Amira Mouakher for Special Issue "Machine Learning and Its Application for Anomaly Detection" (Special Issue ID: `C67H3U9QGT`).
- [x] Seven dedicated Applied Sciences references cited and verified.

**Official Sign-Off:** **`A15 APPLIED SCIENCES PACKAGE READY`**

---

**Lead Investigator & Corresponding Author:**  
Ki-Hyung Kim, Ph.D.  
Professor, Department of Cyber Security, Ajou University  
Suwon 16499, Republic of Korea  
Email: `kkim86@ajou.ac.kr`
