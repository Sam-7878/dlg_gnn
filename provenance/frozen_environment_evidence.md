# Frozen Environment Evidence

The selected Round 5 execution environment is supported by a contemporaneous machine-readable freeze generated with the primary campaign and by run logs whose tracebacks identify the same Python 3.12 virtual environment. Later P4 documents that reported Python 3.10.12, PyTorch 2.1.2, and PyG 2.5.0 were reconstructed prose without matching Round 5 evidence and are superseded.

| Field | Selected value | Primary evidence path | Evidence timestamp | Confidence | Conflicting evidence | Resolution |
|---|---|---|---|---|---|---|
| Python | 3.12.13 | `outputs/benchmark/sci_round5_final/manifests/environment_freeze.json` | 2026-08-18 08:23:55 +0900 | High | P4 prose said 3.10.12 | Select contemporaneous freeze; logs also use `.venv/lib/python3.12` |
| PyTorch | 2.5.1+cu121 | same manifest | 2026-08-18 08:23:55 +0900 | High | P4 prose said 2.1.2+cu121 | Select contemporaneous freeze |
| CUDA | 12.1 | same manifest | 2026-08-18 08:23:55 +0900 | High | None material | Retain |
| PyTorch Geometric | 2.7.0 | same manifest | 2026-08-18 08:23:55 +0900 | High | P4 prose said 2.5.0 | Select contemporaneous freeze |
| PyGOD | 1.1.0 | same manifest | 2026-08-18 08:23:55 +0900 | High | None | Retain |
| torch-sparse | 0.6.18+pt25cu121 | same manifest | 2026-08-18 08:23:55 +0900 | High | None | Retain |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU | same manifest | 2026-08-18 08:23:55 +0900 | High | None | Retain |
| OS | Linux WSL2, glibc 2.39 | same manifest | 2026-08-18 08:23:55 +0900 | High | Generic P4 Windows wording | Use recorded platform string |

The public `provenance/environment_manifest.json` is a byte-for-byte publication copy of the contemporaneous freeze. The hashed empirical result files remain independently protected by the frozen data hashes.
