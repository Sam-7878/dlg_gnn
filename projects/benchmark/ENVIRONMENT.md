# Benchmark environment provenance

The frozen 435 approved results combine historical Round5/A03 and later A04/A05 campaigns. **Do not attribute all historical runs to the current environment.** Each registry row has environment/source provenance fields; limitations are disclosed in the manuscript.

| Campaign | Environment evidence | Interpretation |
|---|---|---|
| Historical Round5 | Archived freeze in `archive/benchmark/provenance/` & evidence ZIP | Historical Python 3.12 / PyTorch 2.5.1 era; some records are inventory matched, not run bound |
| A03 | `environment/locks/benchmark-a03-cuda.lock.txt` | CUDA 13.0 development and qualification |
| A04/A05 | `environment/locks/benchmark-a04-cuda.lock.txt` and `.hashed.txt` | WSL2 Ubuntu 26.04, Python 3.14.4, PyTorch 2.14.1+cu130, CUDA 13.0, PyG 2.8.0.post1, PyGOD 1.1.0; use per-run evidence for exact attribution |

The active local virtual environment is `/mnt/d/_work/goat_bank/.venv_cuda`; it is outside this Git checkout and is not assumed by CI. Fast archive verification uses standard Python. Paper regeneration requires SciPy. Full training additionally requires CUDA, PyTorch/PyG/PyGOD and separately acquired graphs. The exact package inventory is bundled in the A05/A06 evidence ZIP.
