# RTX 3090 eGPU Hardware Qualification and Preflight Summary (Gate G3)

- **Device**: `cuda:0` (NVIDIA GeForce RTX 3090)
- **Total Physical VRAM**: `24.0 GB`
- **Host<->Device Link**: `Thunderbolt 4 / PCIe over eGPU link`
- **H2D Transfer Rate**: `2.05 GB/s`
- **D2H Transfer Rate**: `1.61 GB/s`
- **20 GB VRAM Allocation**: `PASS`

## 5-Model Seed-42 Preflight Results

| Model | Status | Runtime (s) | Peak VRAM (MB) | Finite Loss | Finite Scores |
|---|---|---|---|---|---|
| DOMINANT | PASS | 5.66 | 24.19 | True | True |
| CONAD | PASS | 0.32 | 26.8 | True | True |
| DLG-Base | PASS | 0.15 | 25.23 | True | True |
| DLG-Aug | PASS | 0.2 | 24.17 | True | True |
| AnomalyDAE | PASS | 0.23 | 41.94 | True | True |

**Overall Gate G3 Status: PASS**
