# Final Submission Sign-Off (A16 Closure)

**Document ID:** DLG-BENCH-SIGNOFF-A16  
**Date:** 2026-10-28  
**Immutable Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Status:** **PREPRINT READY -> Preprints.org deposit -> obtain Benchmark DOI -> Applied Sciences submission**

---

## 1. Deliverable Cryptographic Digests (Verified on Disk)

| Deliverable | Path | Pages / Size | SHA-256 |
|---|---|---|---|
| **Preprints.org TeX** | `projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex` | `56,799` B | `ad9b1a9a30091f460f0915dab18ab321d32a94443217e7deb025ed4f0030e567` |
| **Preprints.org PDF** | `projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.pdf` | `18` pages (`589,769` B) | `e1c2e3f622a8a772b01de34bdeba7364c4844b362e71907c31d776c0d7d21de5` |
| **Preprints.org ZIP** | `projects/benchmark/submission/preprints/source_bundle.zip` | `106,914` B | `1c5b6e5624586684b2edd467258c599368527ee9b69d1a98a961e9d14769102f` |
| **Applied Sciences TeX** | `projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.tex` | `56,424` B | `ea3e3af5ca23345da4f35432d596270d36f8d2efdfeb7877389ba6d425427f0e` |
| **Applied Sciences PDF** | `projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.pdf` | `16` pages (`351,949` B) | `cc0187331ea7ad0a6dda3b2b07d15f2a7b133beea2bd45e766cc6d90239c0cb3` |
| **Applied Sciences ZIP** | `projects/benchmark/submission/applsci/source_bundle.zip` | `1,142,897` B | `e5484e7e68dccf3be02f83b3d2541c7d0aa76c091bb790c1e261246540656cb8` |

---

## 2. Gate Closure Summary (Gates G1--G6)

- **G1 (Denominator Narrative Auto-Generation):** `evidence_denominator_audit.md` and `denominator_crosscheck_a16.csv` generated directly from `evaluation_population_by_seed.csv`, `dataset_identity_reconciliation.csv`, and `main_ap_mean_sd.csv`. All 14 datasets match Manuscript Table 1 and Table 2 identically.
- **G2 (Resource Failure Provenance):** `unsupported_details` in `release_manifest.json` and all manuscript/report narratives strictly reflect the 3 failure categories in `support_14x7.csv`: 4 `NO_APPROVED_RAW_RUN` (GADNR on DGraphFin, Elliptic, Reddit-Syn, Yelp-Syn), 1 `OBSERVED_HISTORICAL_CUDA_OOM` (GADNR on Flickr-Syn, 8GB legacy GPU), and 2 `OBSERVED_CUDA_OOM` (AnomalyDAE on Reddit-Syn, Yelp-Syn in full encoder forward on 24GB RTX 3090).
- **G3 (DGraphFin Split Identity Audit):** Cryptographically verified `SAME_MASK_DIFFERENT_METADATA` (`4ebf6ded...` boolean mask hash vs `3fd8c88f...` flatnonzero index hash evaluate the identical 183,840 test nodes and 2,326 positives).
- **G4 (Concise Abstract & Backmatter):** Adopted the 172-word single-paragraph abstract in both manuscripts and updated Author Contributions and Acknowledgments / Use of Generative AI.
- **G5 (Checksum & Page Count Synchronization):** Verified byte-for-byte identity between standalone `.tex` and `.tex` inside `source_bundle.zip`; clean unpack build verified `18` pages (Preprints) and `16` pages (Applied Sciences) with 0 undefined references/citations.
- **G6 (Scientific & Submission Signoff):** Passed 12/12 negative control mutation tests.
