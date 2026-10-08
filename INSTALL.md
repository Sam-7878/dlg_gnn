# Installation and environment

The four projects share this repository but have different scientific dependency and data requirements. **Fast `verify`** needs Python 3.12 or newer and the standard library:

```bash
python scripts/reproduce_project.py --project benchmark --mode verify
```

The benchmark `paper` mode additionally requires SciPy for the frozen Wilcoxon calculation. On the author's WSL2 Ubuntu 26.04 system it was run with `/mnt/d/_work/goat_bank/.venv_cuda/bin/python` (Python 3.14.4, PyTorch 2.14.1+cu130, CUDA 13.0, PyG 2.8.0.post1, PyGOD 1.1.0). This local venv path is not portable. Package freezes are in `environment/locks/`; campaign attribution is documented in [benchmark environment provenance](projects/benchmark/ENVIRONMENT.md).

For full neural-model tests or training, provision the corresponding project's dependencies and source data separately. Start with the project's environment guide:

- [DLG model](projects/dlg_gnn/ENVIRONMENT.md)
- [Benchmark](projects/benchmark/ENVIRONMENT.md)
- [StreamMC](projects/stream_mc/ENVIRONMENT.md)
- [TDS](projects/tds/ENVIRONMENT.md)

Historical Round5 installation instructions belong to the archived v1 release and its frozen environment evidence. Do not use the current CUDA stack as a claim that all historical scores were generated with it.
