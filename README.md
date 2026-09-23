# A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs

[![Preprints.org Ready](https://img.shields.io/badge/Preprints.org-Preprint_Ready-blue.svg)](https://www.preprints.org/)
[![Target Journal](https://img.shields.io/badge/Target_Journal-MDPI_Applied_Sciences-green.svg)](https://www.mdpi.com/journal/applsci/special_issues/C80IXAF9V4)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![PyG 2.7+](https://img.shields.io/badge/PyG-2.7.0-red.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official companion code, experiment runners, canonical manifests, frozen evaluation tables, and replication scripts for the benchmark paper:

> **"A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs"**  
> **Authors:** SeongSu Park$^1$ and Ki-Hyung Kim$^{2,*}$  
> $^1$ *Department of Computer Engineering, Ajou University, Suwon 16499, Republic of Korea* (`parky@ajou.ac.kr`, ORCID: [0009-0008-4056-3875](https://orcid.org/0009-0008-4056-3875))  
> $^2$ *Department of Cyber Security, Ajou University, Suwon 16499, Republic of Korea* (`kkim86@ajou.ac.kr`, ORCID: [0000-0002-2321-4475](https://orcid.org/0000-0002-2321-4475))  
> $^*$ *Corresponding author: Ki-Hyung Kim*

---

## 📌 Publication & Deposit Status

- **Preprints.org Deposit:** Prepared for Preprints.org deposit.
- **Target Journal:** *MDPI Applied Sciences* (Special Issue: *Graph Neural Networks: Theory, Methods and Applications*).
- **Official Release Asset:** `DLG_GNN_Benchmark_v1.0.0_preprint.zip` (Tag: `v1.0.0-preprint`).
- **Release State:** GitHub release publication pending; the deposit package is prepared.

---

## 📊 Frozen Benchmark Summary

This repository hosts the complete, frozen scientific artifacts and exact-sparse execution backends for large-scale graph anomaly detection:

| Benchmark Dimension | Frozen Specification |
|---|---|
| **Primary Suite** | **10 Heterogeneous Datasets** (3 real financial: *Elliptic, DGraphFin, BitcoinOTC*; 7 controlled synthetic-injection: *Yelp-Syn, Amazon-Syn, Flickr-Syn, Reddit-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn*) |
| **External Validation** | **LANL-RedTeam** authentication interaction graph (16,694 nodes, 323,897 directed stored edges, 13 features, 301 positive computers) |
| **Evaluated Detectors** | **8 Configurations** (6 established baselines: *DOMINANT, AnomalyDAE, CoLA, CONAD, GADNR, OCGNN*; 2 DLG variants: *DLG-Base, DLG-Aug*) |
| **Supported Pairs** | **71 / 80 pairs** supported exactly under declared resource limits; 9 restrictions are reported separately and are not imputed as zero scores or worst ranks |
| **Successful Primary Runs** | **355 runs** across 5 independent random model seeds (seeds 42–46) |
| **Sensitivity Controls** | **45 targeted runs** (*DLG-Aug-Zero, DLG-Aug-Permuted, DLG-Base-70*) across discrepancy graphs |
| **Scalability Backends** | Exact sparse linear structure decoder (avoids $\mathcal{O}(N^2)$ dense matrix storage with $\mathcal{O}(\|E\|d + Nd^2)$ arithmetic) & fused sparse message passing |

### Authoritative Cryptographic Hashes
- `benchmark_raw.csv`: `39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c`
- `model_dataset_support_matrix.csv`: `c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914`

---

## 🚀 Quick Reproduction Guide

The benchmark provides two reproduction pathways:

```
┌────────────────────────────────────────────────────────────────────────┐
│ MODE 1: Frozen-Artifact Fast Reproduction (0 Raw Datasets Needed)       │
│ • Validates cryptographic SHA-256 hashes of frozen raw benchmark runs  │
│ • Regenerates all manuscript performance, statistical, & control tables│
│ • Runs exact-sparse mathematical and numerical equivalence unit tests  │
│ • Execution time: ~5 seconds (no GPU or external data downloads needed)│
└────────────────────────────────────────────────────────────────────────┘
                                    │
┌────────────────────────────────────────────────────────────────────────┐
│ MODE 2: Full Benchmark Re-Execution (Raw Datasets Required)             │
│ • Primary 10-Dataset Benchmark: run_sci_round5_final.py (355 runs)     │
│ • Sensitivity Controls: run_capacity_controls_m3.py (45 runs)          │
│ • Requires downloading external datasets according to license matrix   │
└────────────────────────────────────────────────────────────────────────┘
```

### Mode 1: Fast Frozen-Artifact Reproduction
```bash
# 1. Clone repository
git clone https://github.com/Sam-7878/dlg_gnn.git
cd dlg_gnn

# 2. Run Mode 1 reproduction script (~5s, verifies hashes and generates tables)
python scripts/reproduce_frozen_artifacts.py
```

### Mode 2: Full Pipeline Re-Execution
```bash
# Set up Python environment (see INSTALL.md for detailed steps)
pip install -r requirements-core.txt

# Run full five-seed primary benchmark
python experiments/benchmark/run_sci_round5_final.py --config configs/benchmark/sci_round5_final.yaml --stage phase1

# Run pre-registered sensitivity controls
python experiments/benchmark/run_capacity_controls_m3.py
```

---

## 📦 Repository Structure

```text
dlg_gnn/
├── src/                          # Exact sparse backends, DLG models, adapters
│   ├── gog_fraud/models/         # DLG-Base, DLG-Aug, baseline adapters
│   └── gog_fraud/pipelines/      # Evaluation and verification pipelines
├── experiments/benchmark/        # Mode 2 experiment runner scripts
├── scripts/                      # Mode 1 reproduction and audit tools
├── configs/                      # Canonical configuration files
├── artifacts/                    # Small frozen derived data and canonical manifests
├── provenance/                   # Frozen-environment evidence and manifest
├── docs/                         # Architecture, mathematics, and reproduction commands
├── tests/                        # Automated benchmark/publication test suites
├── docs/papers/_42_Benchmark/    # Master LaTeX source and generated tables
├── CITATION.cff                  # Machine-readable scholarly citation
├── INSTALL.md                    # Environment setup & dependency instructions
└── README.md                     # Benchmark landing page (this document)
```

The downloadable GitHub Release archive additionally contains the same root files, the `artifacts/` and `provenance/` trees, the exact benchmark runner dependencies, and `release_metadata.json`. Raw third-party datasets and local `outputs/` are not part of either the Git repository or release archive.

---

## 📜 Dataset Provenance & Attribution

- **Public Real-Label Graphs:**
  - **Elliptic:** Bitcoin transaction graph ([Weber et al., 2019](https://arxiv.org/abs/1908.02591)), available on Kaggle.
  - **DGraphFin:** Million-scale financial user graph ([Huang et al., 2022](https://arxiv.org/abs/2207.03579)), available from FinVolution.
  - **BitcoinOTC:** Who-trusts-whom network with rating labels, available from the Stanford Network Analysis Project ([SNAP](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)).
  - **LANL-RedTeam:** Enterprise authentication log graph ([Kent, 2015](https://csr.lanl.gov/data/cyber1/)), available from Los Alamos National Laboratory.
- **Controlled Synthetic-Injection Graphs (`-Syn`):**
  - Seven benchmarks (`Yelp-Syn`, `Amazon-Syn`, `Flickr-Syn`, `Reddit-Syn`, `Cora-Syn`, `CiteSeer-Syn`, `PubMed-Syn`) are constructed from public base graphs distributed via PyGOD and SNAP, using the frozen structural and contextual anomaly-injection protocol described in the manuscript.

---

## 📖 Citation

If you use this benchmark suite, exact sparse backends, or frozen evaluation tables, please cite as follows:

```bibtex
@article{park2026dlgbenchmark,
  title={A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs},
  author={Park, SeongSu and Kim, Ki-Hyung},
  journal={Preprints.org},
  year={2026},
  note={Preprint forthcoming. Code available at \url{https://github.com/Sam-7878/dlg_gnn}}
}
```

### Relation to Preceding Work
This benchmark isolates and empirically investigates the local-to-global principle first proposed in our preceding architectural paper:
> **"DLG-GNN: Decoupled Local-to-Global Graph Neural Network for Scalable Blockchain Fraud Detection"**  
> *Preprints.org 2026*, DOI: [10.20944/preprints202609.0848.v1](https://doi.org/10.20944/preprints202609.0848.v1).

---

## 🏛️ Historical & Broader Project Context

This repository originated as part of the **GoatBank** security research initiative, which developed hierarchical local-to-global graph representations for multi-chain blockchain fraud analysis. While the present benchmark paper deliberately isolates the core static GNN anomaly detection principles for standard graph benchmarks, the wider project includes streaming replay engines, bounded state caches, and hierarchical transaction modeling components preserved under `src/` and `docs/`.

---

## 📄 License

This benchmark repository and its code are released under the [MIT License](LICENSE). Frozen benchmark outputs, evaluation tables, and pre-computed artifacts are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
