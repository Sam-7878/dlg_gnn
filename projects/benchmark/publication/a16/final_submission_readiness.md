# Final Submission Readiness Report (A16 Closure & A17 Gate Check)

**Document ID:** DLG-BENCH-READINESS-A16-A17  
**Date:** 2026-10-28  
**Benchmark Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Dual Submission Sequence:**
1. **Preprints.org FIRST:** Deposit publisher-neutral package (`18` pages) to secure distinct benchmark preprint DOI.
2. **MDPI Applied Sciences SECOND:** Submit Special Issue package (`16` pages) to Special Issue "Machine Learning and Its Application for Anomaly Detection" (Special Issue ID: `C67H3U9QGT`, deadline 2026-12-31).

---

## 1. Compliance Checklist (Work Order Gates A17-R1 through A17-R6)

| Gate | Description | Outcome | Sign-off |
|---|---|---|---|
| **Gate R1** | Remote Git Tag Independent Verification | Tag `benchmark-a16-submission-2026-10-28` dereferencing to commit `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f` verified on GitHub via `git ls-remote` and GitHub API HTTP 200. | **APPROVED** |
| **Gate R2** | Private Review-Process Language Removal | Replaced "Claude's review" with neutral statistical narrative explaining within-block permutation vs asymptotic $\chi^2$ differences. | **APPROVED** |
| **Gate R3** | Source ZIP Internal Filename Normalization | Both source bundles rebuilt with `DLG-Benchmark_A16_Preprints.tex` and `DLG-Benchmark_A16_AppliedSciences.tex` as internal main files; clean unpack succeeds. | **APPROVED** |
| **Gate R4** | Memory Claims Reframed as Profiles | Table 6 reframed from "measured VRAM" to illustrative peak memory materialization profiles on RTX 3090, correctly noting AnomalyDAE preflight OOM on Reddit-Syn. | **APPROVED** |
| **Gate R5** | Decision Matrix Bounded to Observations | Table 7 replaces unverified causal phrases ("induces over-smoothing", "ego-net assumptions fail") with concrete empirical AP observations. | **APPROVED** |
| **Gate R6** | Extended Validator & Cryptographic Digests | All 15 negative control tests pass; SHA-256 digests verified on actual disk bytes; 0 undefined references/citations. | **APPROVED** |

---

## 2. Definitive Deliverable Checksums

See `publication/a16/final_artifact_hashes.sha256` and submission `SHA256SUMS.txt` for cryptographic byte-exact records.
