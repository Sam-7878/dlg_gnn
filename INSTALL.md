# Installation & Environment Setup Guide

This guide describes how to set up the reproducible environment for the DLG-GNN benchmark suite on Ubuntu 22.04+ (or Windows 11 WSL2) with CUDA 12.1 and PyTorch 2.5.1.

---

## 1. Prerequisites

- **Operating System**: Linux / Ubuntu 22.04 LTS (or Windows 11 WSL2)
- **NVIDIA Driver**: 530+ with CUDA 12.1 support (GPU with >=8GB VRAM recommended for re-training)
- **Python**: 3.10, 3.11, or 3.12 (frozen execution was conducted on Python 3.12.13)
- **Git**: 2.34+

---

## 2. Fast Setup via Conda / Mamba

```bash
# Clone the repository
git clone https://github.com/Sam-7878/dlg_gnn.git
cd dlg_gnn

# Create and activate conda environment
conda env create -f environment.yml
conda activate dlg_gnn

# Verify environment
python scripts/verify_environment.py
```

---

## 3. Manual Virtual Environment Setup (pip)

```bash
# 1. Clone repository and create venv
git clone https://github.com/Sam-7878/dlg_gnn.git
cd dlg_gnn
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch 2.5.1 with CUDA 12.1
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyG and compiled binary extensions
pip install torch-geometric==2.7.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install pygod==1.1.0

# 4. Install core benchmark dependencies and package in editable mode
pip install -r requirements-core.txt
pip install -e .

# 5. Verify installation
python scripts/verify_environment.py
```

---

## 4. Verification

Run the verification script to inspect installed versions against the frozen specification:
```bash
python scripts/verify_environment.py
```

Run test suites:
```bash
pytest tests/benchmark/ -v
```

---

## 5. Mode 1: Instant Frozen-Artifact Reproduction (~5s)

Mode 1 does NOT require downloading raw graphs. It verifies cryptographic SHA-256 hashes of the frozen primary raw runs and model-dataset support matrix, and regenerates all manuscript performance, statistical, and control tables:

```bash
python scripts/reproduce_frozen_artifacts.py --artifact-root outputs/benchmark/manuscript_m5/artifacts --output-dir reproduced_tables
```

---

## 6. Mode 2: Full Benchmark Re-Execution

Full re-training requires downloading raw graphs into `data/` according to individual upstream licenses (see `README.md` for dataset acquisition details):

```bash
# Primary 10-Dataset Benchmark:
python experiments/benchmark/run_sci_round5_final.py --config configs/benchmark/sci_round5_final.yaml --stage phase1

# Sensitivity Controls:
python experiments/benchmark/run_capacity_controls_m3.py --gpu 0
```
