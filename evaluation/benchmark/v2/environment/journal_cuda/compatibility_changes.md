# PyG 2.8 and Accelerator Stack Compatibility Changes Journal

**Document ID:** DLG-BENCH-V2-ENV-COMPAT-01  
**Date:** 2026-10-02 (Asia/Seoul)  
**Environment:** `.venv_cuda` (Python 3.14.4 + PyTorch 2.14.1+cu130 + PyG 2.8.0.post1 + PyGOD 1.1.0)  
**Hardware Verified:**
- GPU 0: NVIDIA GeForce RTX 3090 (24.0 GB, CC 8.6, eGPU reference)
- GPU 1: NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB, CC 8.9, diagnostic host)

---

## 1. Inventory & Classification per Work Order A02 B-02

| Symbol / Import | Location | Classification | Migration / Replacement Strategy | Equivalence Status |
|---|---|---|---|---|
| `from torch_sparse import SparseTensor, matmul as sparse_matmul` | `src/gog_fraud/models/pygod/exact_reconstruction.py` | `MIGRATE_TO_PYG28_API` | PyG 2.8 deprecated standalone `torch-sparse` in favor of native PyTorch sparse tensors. Added fallback: uses `torch.sparse_coo_tensor` and `torch.sparse.mm`. | **PASS** (max abs diff vs dense $7.6 \times 10^{-6} < 10^{-5}$) |
| `from torch_sparse import SparseTensor` | `src/gog_fraud/models/pygod/sparse_message.py` | `MIGRATE_TO_PYG28_API` | Pre-normalized transposed adjacency matrix converts to native `torch.sparse_coo_tensor` / `torch.sparse_csr_tensor`, natively consumed by `GCNConv` / PyG SpMM. | **PASS** (exact GCN forward/backward match) |
| `GADNRBase.__init__(..., tot_nodes=...)` | `pygod.detector.gadnr` | `MIGRATE_TO_PYG28_API` | Upstream PyGOD 1.1.0 passed unused `tot_nodes` argument into `**kwargs` forwarded to `MessagePassing.__init__`, which in PyG 2.8 rejects extraneous kwargs. Switched import to repo's patched `gog_fraud.models.pygod.gadnr.GADNR`. | **PASS** (100% clean initialization and scoring) |
| `NeighborSampler` | `CoLA`, `OCGNN` | `OPTIONAL_EXTENSION_REQUIRED` | Installed `pyg-lib==0.9.0+pt214cu130` from `data.pyg.org`. Ensured `edge_index` is strictly `.contiguous()` for C++ bindings. | **PASS** (Both models train and score without warnings) |
| `triton driver.c` compilation | Triton NVIDIA backend | `SYSTEM_TOOLCHAIN` | Installed `python3.14-dev` header files into Ubuntu WSL environment to allow Triton C-extension compilation. | **PASS** (Triton compiles cleanly) |

---

## 2. 8-Detector Smoke Qualification Record (Gate G0)

Both physical GPUs were subjected to the standard 8-detector smoke suite (`evaluation/benchmark/v2/scripts/01_smoke_all_detectors.py`):
- `DOMINANT`: **PASS**
- `AnomalyDAE`: **PASS**
- `CoLA`: **PASS**
- `CONAD`: **PASS**
- `GADNR`: **PASS**
- `OCGNN`: **PASS**
- `DLG-Base`: **PASS**
- `DLG-Aug`: **PASS**

All models verified for: `IMPORT_OK`, `INIT_OK`, `TRAIN_1EPOCH_OK`, `TRAIN_3EPOCH_OK`, `SCORE_OK`, `NO_NAN_INF`, and `SERIALIZE_RELOAD_OK`.
Zero silent CPU fallbacks; fail-closed execution on CUDA.
