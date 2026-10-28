# Final Submission Sign-Off (A16 Closure & A17 Gate Check)

**Document ID:** DLG-BENCH-SIGNOFF-A16-A17  
**Date:** 2026-10-28  
**Immutable Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Remote Git Tag Object:** `451326fbd8710ccbf8b2d12103e71b1822560fce`  
**Dereferenced Target Commit:** `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f`  
**Status:** **PREPRINT READY -> Preprints.org deposit -> obtain Benchmark DOI -> Applied Sciences submission**

---

## 1. Deliverable Cryptographic Digests (Verified on Disk)

| Deliverable | Path | Pages / Size | SHA-256 |
|---|---|---|---|
| **Preprints.org TeX** | `projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex` | `57,272` B | `a226b2dca3bf874f81f9c95d4e8790ced85020f9abd201d7d121653eeeba0422` |
| **Preprints.org PDF** | `projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.pdf` | `18` pages (`590,838` B) | `35343549afcda1d75768eca24ef8b146d6cdb6a765e7cfc12d4fbb8a41212a39` |
| **Preprints.org ZIP** | `projects/benchmark/submission/preprints/source_bundle.zip` | `107,005` B | `ab13612ffeaf4188dd97b3e827b1cd526c8f117a1a8e23b9bddbb04e1276d042` |
| **Applied Sciences TeX** | `projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.tex` | `56,897` B | `61a93c4ee5f1b9d7448befb495b9908857d3e04499f1901568feb000ebd37947` |
| **Applied Sciences PDF** | `projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.pdf` | `16` pages (`352,185` B) | `36c0f66a7a50e8b2eb32d5a626ae92b891dea4d30172f2f1972073822604eef2` |
| **Applied Sciences ZIP** | `projects/benchmark/submission/applsci/source_bundle.zip` | `1,142,987` B | `344f5d30e6f53d7da2242d194825444e172c36952f12c4aa30d97c66bd3fae6b` |

---

## 2. A17 Release & Editorial Gate Closure Summary

- **Gate R1 (Public Git Tag Verified):** Live remote tag `benchmark-a16-submission-2026-10-28` verified on GitHub via `git ls-remote` and GitHub REST API HTTP 200 responses. Target commit `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f` matches the immutable source tree. Recorded in `public_tag_verification_a17.json`.
- **Gate R2 (Review Language Removed):** Removed private review reference ("Claude's review noted...") and replaced with neutral, objective description of discrete within-block permutation vs asymptotic $\chi^2$ differences.
- **Gate R3 (Internal ZIP Filename Normalized):** Rebuilt both source bundles with `DLG-Benchmark_A16_Preprints.tex` and `DLG-Benchmark_A16_AppliedSciences.tex` as internal main filenames. Clean unpack build succeeds (18p Preprints, 16p MDPI, 0 undefined references/citations).
- **Gate R4 (Memory Claim Verification & Reframing):** Table 6 reframed as illustrative peak memory materialization profiles (`Profile (Cora)` and `Profile (Reddit)`) on an RTX 3090 24GB platform. Highlighted row-blocked AnomalyDAE ($B=2{,}048$) scaling behavior while accurately reflecting preflight encoder forward OOM on Reddit-Syn in the benchmark.
- **Gate R5 (Bounded Decision Matrix):** Replaced theoretical causal interpretations in Table 7 with empirical AP observations:
  - Elliptic: Highest AP on clustered transaction topology ($+50.97\%$ relative over DLG-Base).
  - Reddit-Syn: Lower AP than DLG-Base on dense community graphs ($\Delta\mathrm{AP} = -0.0656$).
  - LANL-RedTeam: Negative delta on bipartite authentication graphs ($\Delta\mathrm{AP} = -0.0252$).
- **Gate R6 (Final Stop):** All 15 extended mutation tests passed; 0 undefined references/citations; byte-for-byte identity between standalone `.tex` and zipped `.tex`. Ready for immediate deposit on Preprints.org.
