# DLG Benchmark A18 Submission Gate Report
**Document ID:** `DLG-BENCH-GATE-REPORT-A18`  
**Execution Date:** 2026-10-28  
**Submission Status:** **PREPRINT READY**  
**Preprints Deposit:** Ready for immediate submission to Preprints.org  
**Journal Target:** Applied Sciences SI *Machine Learning and Its Application for Anomaly Detection* (`C67H3U9QGT`)

---

## 1. Summary of Gate Verifications

| Gate | Focus | Result | Key Evidence |
|---|---|---|---|
| **G0** | Authoritative Source Selection | **PASS** | `input_inventory.csv`, `a18_authoritative_source_map.md` |
| **G1** | Alert-Budget Table 3 Repair | **PASS** | 210/210 invariant assertions hold; 0 violations; negative controls reject obsolete prevalences |
| **G2** | Appendix F1 Table A2 Regeneration | **PASS** | All 455 runs verified against raw metric JSONs; sample SD (`ddof=1`); diff documented |
| **G3** | Code-Faithful Figure 1 & Methods | **PASS** | `fig_dlg_architecture.pdf` redrawn; unexecuted PageRank & ego-net induction removed; code trace documented |
| **G4** | Scientific Language & Consistency | **PASS** | Absolute $\Delta\mathrm{AP} = +0.0350$ leads; unmeasured causal phrasing softened; Wilcoxon $p=0.0625$ minimum noted |
| **G5** | Literature & Citation Audit | **PASS** | Dataset citations restored (`weber2019`, `huang2022`, `luo2024`, `kent2015`); 0 undefined citations |
| **G6** | Packaging & Clean Unpack Build | **PASS** | Preprints (18p) & MDPI (16p) compile cleanly from unpacked `source_bundle.zip` with 0 errors |
| **G7** | Stop Rule & Final Sign-Off | **PASS** | All gates fulfilled; zero GNN retraining; ready for immediate submission |

---

## 2. Deliverable Digest Ledger

| Package | Artifact | Pages / Size | SHA-256 |
|---|---|---|---|
| **Preprints** | `DLG-Benchmark_A16_Preprints.tex` | 64,582 B | `63c4736bd48d04eedefaef7c8f1036764621cf0830e9e5641d9d4baaf78d4175` |
| **Preprints** | `DLG-Benchmark_A16_Preprints.pdf` | 21 pages (611,662 B) | `b865b3f413156f662a597822ba3cc795a08a538a3d52bbfbd7481ac853e9cd43` |
| **Preprints** | `source_bundle.zip` | 112,872 B | `fcb017697437b782a3a0e5f8cb75b39394d70526aa8dbdd578ccb41ac2229fd8` |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.tex` | 64,207 B | `c7bba9fe5393f7168fad9568b62029c2c654cbebf7209f6a24d9e10e74576d75` |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.pdf` | 18 pages (370,953 B) | `615de37923bd3c862916da9ee136b121daa1268dbfbd5bbc44b9161439b48445` |
| **Applied Sciences** | `source_bundle.zip` | 1,148,859 B | `b100fbdf08a23bd89d0150618f6d93b05240df9575ced72f7ff1878c5f7fde61` |

---

## 3. Final Acceptance Statement
> Our benchmark's published anomaly-detection results and efficiency claims are derived from a versioned, code-faithful experiment registry. Alert-budget metrics, where reported, obey exact count arithmetic on each evaluation split. F1 summaries are either source-traceable or transparently unavailable. The architecture figure depicts the executed DLG implementation, and both preprint and journal submission bundles are independently reproducible and mutually consistent.
