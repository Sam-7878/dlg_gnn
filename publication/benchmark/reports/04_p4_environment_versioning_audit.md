# Round P4 Audit Report 04: Environment Versioning & Packaging Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite (Round P4)  
**Target Files:**
- `outputs/benchmark/manuscript_m5/provenance/frozen_execution_environment.json`
- `outputs/benchmark/manuscript_m5/provenance/current_reproduction_environment.json`
- `INSTALL.md`
- `environment.yml`
- `scripts/verify_environment.py`

---

## 1. Executive Summary

To prevent environment drift between the historical execution environment (which produced the frozen benchmark tables in the manuscript) and modern reproduction environments, Round P4 implements a two-tier provenance model:
1. **Frozen Execution Environment:** The exact hardware, OS, CUDA, Python, PyTorch, PyG, and PyGOD stack used during the original benchmark campaign.
2. **Current Reproduction Environment:** The actively verified environment used to test and validate reproduction scripts and clean unpack tests.

---

## 2. Environment Specifications Audit

| Component | Frozen Execution Environment | Current Reproduction Environment |
|---|---|---|
| **Python Version** | Python 3.12.13 | Python 3.12.13 |
| **PyTorch** | 2.5.1+cu121 | 2.5.1+cu121 |
| **PyTorch Geometric (PyG)** | 2.7.0 | 2.7.0 |
| **PyGOD** | 1.1.0 | 1.1.0 |
| **CUDA Toolkit** | CUDA 12.1 | CUDA 12.1 |
| **OS Platform** | Linux / Windows Subsystem | Linux WSL2 x86_64 |
| **Key Role** | Provenance for Tables 4–8 in paper | Live verification of reproduction pipelines |

---

## 3. Installation Documentation & Automation Audit

1. **`INSTALL.md`**:
   - Contains clean git clone instructions: `git clone https://github.com/Sam-7878/dlg_gnn.git`.
   - Explains both Conda and virtualenv setup workflows.
   - Clarifies the distinction between Mode 1 (frozen table reproduction without GPU) and Mode 2 (full re-training with CUDA).

2. **`environment.yml`**:
   - Specifies dependencies with version pins compatible with PyTorch 2.5.1 and PyGOD 1.1.0.
   - Tested and verified against standard conda environment solvers.

3. **`scripts/verify_environment.py`**:
   - Automated environment diagnostic utility.
   - Inspects installed package versions against recommended baselines and reports compatibility warnings or confirmation.

---

## 4. Automated Test Verification

- `tests/benchmark/publication_p4/test_environment_versioning_split.py` (PASS)
- `tests/benchmark/manuscript_m5/test_environment_manifest_matches_install_docs.py` (PASS)

---

## 5. Audit Verdict

**STATUS: PASS**  
Environment versioning is cleanly split, documented, and programmatically verifiable.
