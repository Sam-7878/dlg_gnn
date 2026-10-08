# P1 Audit Report 07: MDPI Special Issue Submission Readiness

## 1. Executive Summary
- **Target Journal**: *MDPI Applied Sciences* (ISSN 2076-3417, Impact Factor 2.5, CiteScore 5.3).
- **Special Issue**: *"Graph Neural Networks: Theory, Methods and Applications"*
  - Special Issue URL: `https://www.mdpi.com/journal/applsci/special_issues/C80IXAF9V4`
  - Special Issue Deadline: 20 November 2026
- **Submission Strategy**:
  1. Deposit publisher-neutral preprint to **Preprints.org** (`publication/benchmark/preprints/`).
  2. Submit MDPI template package with Special Issue cover letter to **Applied Sciences** (`publication/benchmark/mdpi/`).
- **Official Readiness Declaration**: **`READY_FOR_PREPRINTS_ORG_SUBMISSION`**.

---

## 2. Dual-Submission Package Comparison

| Feature / Artifact | Preprints.org Package | MDPI Applied Sciences Package |
|---|---|---|
| **Directory** | `publication/benchmark/preprints/` | `publication/benchmark/mdpi/` |
| **Submission Archive** | `DLG_Benchmark_Preprints_Submission.zip` | `DLG_Benchmark_MDPI_Submission.zip` |
| **LaTeX Source** | `DLG-Benchmark-Preprint.tex` (Standard `article`) | `DLG-Benchmark.tex` (`Definitions/mdpi.cls`) |
| **Compiled PDF** | `DLG-Benchmark-Preprint.pdf` (32 pages) | `DLG-Benchmark.pdf` (30 pages) |
| **MDPI Branding** | **Zero** (Completely publisher-neutral) | **Included** (Official MDPI template & logos) |
| **Scientific Content** | 100% Identical | 100% Identical |
| **Accompanying Files** | `graphical_abstract.png` (300 DPI) | `cover_letter.md` |
| **Clean Compile Verified** | **Yes (0 errors)** | **Yes (0 errors)** |

---

## 3. Cover Letter Verification
`publication/benchmark/mdpi/cover_letter.md` has been prepared addressing the Editor-in-Chief and Guest Editors:
- Identifies Special Issue: *"Graph Neural Networks: Theory, Methods and Applications"*.
- Highlights 4 core contributions:
  1. Rigorous benchmark evaluating 8 detector configurations (6 baselines + 2 DLG variants) across 10 primary datasets with external LANL cybersecurity validation.
  2. Exact sparse reconstruction avoiding $\mathcal{O}(N^2)$ dense structural storage, with $\mathcal{O}(|E|d + Nd^2)$ arithmetic for the Gram-based linear decoder.
  3. Pre-registered capacity and seed sensitivity controls.
  4. Fully reproducible frozen release bundle and open artifacts.
- Declares preprint deposit to Preprints.org under CC-BY 4.0 in accordance with MDPI preprint policy.
- Affirms original work, no prior publication, author review, and ethical compliance.

---

## 4. Final Verdict
The DLG Benchmark publication preparation round P1 is fully completed and reconciled against frozen source artifacts. All automated freeze gates have passed. The manuscript and bundles are **`READY_FOR_PREPRINTS_ORG_SUBMISSION`**.
