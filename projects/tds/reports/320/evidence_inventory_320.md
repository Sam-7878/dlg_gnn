# Evidence Inventory — R320

Complete ledger of preserved experimental evidence, predictions, and manuscript packages.

## 1. Frozen Experimental Artifacts (`projects/tds/results/319`)
- **Total Registered Files**: 227 files
- **SHA256 Manifest Verification**: `SHA256SUMS_319.txt` verified against local disk: **227/227 files present, 0 mismatches**.
- **Key Categories**:
  - `builder_*_comparisons.csv`: 268,282,924 source transfer reconstruction ledger (mismatches: 0).
  - `predictions/`: 42 raw test & validation prediction CSVs across models and seeds.
  - `runs/horizon7/`: Checkpoint trajectories, training histories, and metrics for GIN-0, GIN-size, GIN-sum, MLP variants.
  - `feature_fits/`: Fitted standard scalers and degree feature parameters.
  - `new_statistics_319.csv`: Bootstrap intervals and Holm-corrected p-values for exploratory contrasts.
  - `seven_day_modelwise_calibration.csv`: Raw and temperature-scaled ECE, adaptive ECE, Brier, and NLL.

## 2. Publication Packages (`projects/tds/submission/`)
- `preprints/320/`:
  - `preprint_article.pdf`: 21-page clean build.
  - `supplement.pdf`: 5-page supplementary document.
  - `preprint_source.zip`: Complete LaTeX source bundle.
  - `metadata.json`: Exact file SHA-256 hashes and author metadata.
- `applsci/320/`:
  - `applsci_article.pdf`: 19-page MDPI Applied Sciences formatted PDF.
  - `supplement.pdf`: 5-page supplementary document.
  - `mdpi_source.zip`: Complete MDPI source bundle including official styles and definitions.
  - `cover_letter.md`: Editor cover letter for Applied Sciences Special Issue.
  - `metadata.json`: Exact file SHA-256 hashes and Special Issue URL.

## 3. Author Records (`projects/tds/submission/author_records/320/`)
- `exact_file_approval_320.md`: Author exact-file hash approval sheet (staged for final meeting signoff).
- `author_confirmation_320.json`: CRediT roles, affiliations, and ORCIDs.
- `funding_proof_320.md`: IITP-2026-RS-2021-II211835 grant verification status.
- `declarations_320.json`: Formal declarations for COI, Funding, AI assistance, and Data Availability.
