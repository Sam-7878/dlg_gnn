# Gate L4: Environment Label Sanity and Provenance Audit Report

**Document ID:** `DLG-BENCH-GATE-L4-ENV-PROVENANCE`  
**Date:** 2026-10-29  
**Parent Work Order:** `DLG-BENCH-WO-A21-LEDGER-ONLY`  
**Status:** **PASS** (Wheel Runtime, Toolkit, Driver, and Legacy Environments Explicitly Disambiguated)  

---

## 1. Context and Objective

During independent review of A20 (`DLG_Benchmark_A20_Independent_Submission_Review_2026-10-29.md`, Finding F4), an environment label ambiguity was identified:
- `a20_memory_evidence_classes.csv` referenced `PyTorch 2.14 / CUDA 12.1` for RTX 3090 runs, whereas the authoritative toolchain manifest (`projects/benchmark/ENVIRONMENT.md`) specifies a `PyTorch 2.14.1+cu130` wheel family with CUDA 13.0 development lockfiles.
- Objective: Audit the underlying run lockfiles, explicitly distinguish the PyTorch wheel runtime from host drivers and legacy campaigns, and align metadata without modifying experimental execution logs.

---

## 2. Environment Audit by Campaign Era

| Campaign Era | Physical Hardware | PyTorch Wheel Runtime | CUDA Driver / Toolkit | PyG / PyGOD Version | Scope & Datasets |
|---|---|---|---|---|---|
| **A04/A05/A08 Campaign** (Modern Dedicated) | NVIDIA RTX 3090 (24 GB VRAM) | `PyTorch 2.14.1+cu130` (Python 3.14 / WSL2 Ubuntu) | CUDA 13.0 runtime / Host NVIDIA Driver 550+ | `PyG 2.8.0.post1`, `PyGOD 1.1.0` | Tier-1 Cryptocurrency (Ethereum, BSC, Polygon: 105 runs), AnomalyDAE preflight OOM tests |
| **Historical Round5 Campaign** (Legacy Multi-Env) | Legacy Environment (8 GB VRAM) | `PyTorch 2.5.1` (Python 3.12) | CUDA 12.x driver compatibility | PyG / PyGOD baseline suite | GADNR Flickr-Syn historical OOM; 280 primary synthetic/financial baselines |
| **External Defense Campaign** | Dedicated Server | `PyTorch 2.14` suite | CUDA 12.x / 13.x compatibility | Custom forensic evaluation pipeline | LANL-RedTeam (35 runs), BitcoinOTC (35 runs) |

---

## 3. Metadata Alignment

- Updated `projects/benchmark/publication/a20/a20_memory_evidence_classes.csv` to replace the imprecise `CUDA 12.1` label with `PyTorch 2.14.1+cu130 / CUDA 13.0 runtime (NVIDIA RTX 3090 24GB)`.
- Verified that historical 8GB legacy environment rows preserve `Legacy 8GB environment (PyTorch 2.5.1)`.
- Re-execution is strictly prohibited and unnecessary, as these entries describe environment metadata rather than numerical model performance.

---

## 4. Verification Verdict

- [x] PyTorch wheel runtime disambiguated from host drivers.
- [x] Legacy 8GB campaign distinguished from modern 24GB RTX 3090 campaign.
- [x] Metadata updated cleanly with zero code or score modification.
- **GATE L4 STATUS: PASS**

