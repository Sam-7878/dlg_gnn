# P1 Audit Report 03: Preprints.org Template and Branding Audit

## 1. Executive Summary
- **Objective**: Ensure that the Preprints.org submission manuscript (`DLG-Benchmark-Preprint.tex`) and bundle (`DLG_Benchmark_Preprints_Submission.zip`) are completely publisher-neutral, eliminating all MDPI-specific logos, class files, and journal headers.
- **Status**: **100% PASSED** (0 occurrences of MDPI branding, clean article layout).

---

## 2. Branding Removal Audit

| Prohibited Asset / Macro | Purpose in MDPI | Presence in Preprints Version | Verification Method |
|---|---|:---:|---|
| `logo-mdpi.eps` | MDPI Header Logo | **ABSENT (0)** | Automated regex and zip archive inspection |
| `logo-ccby.eps` | MDPI Footer CC Logo | **ABSENT (0)** | Automated regex and zip archive inspection |
| `Definitions/mdpi.cls` | MDPI Document Class | **ABSENT (0)** | Replaced with standard `article` class |
| `\pubvolume{...}` | Journal Volume | **ABSENT (0)** | Automated string search |
| `\issuenum{...}` | Journal Issue | **ABSENT (0)** | Automated string search |
| `\articlenumber{...}` | Article Number | **ABSENT (0)** | Automated string search |
| `\datereceived{...}` | MDPI Editorial Dates | **ABSENT (0)** | Automated string search |
| `"Applied Sciences"` | Target Journal Name | **ABSENT (0)** | Full case-insensitive scan |

---

## 3. Preprints Front-Matter Structure
- **Document Class**: Standard `\documentclass[11pt,a4paper]{article}` with `geometry`, `amsmath`, `authblk`.
- **Title**: `A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs`
- **Authors**:
  - `SeongSu Park` (Department of Computer Engineering, Ajou University)
  - `Ki-Hyung Kim` (Department of Cyber Security, Ajou University; Corresponding Author)
- **Abstract & Keywords**: Formatted via standard `\begin{abstract}` ... `\end{abstract}`.
- **License**: Preprints.org applies open CC-BY 4.0 banner during ingestion.
