# P1 Audit Report 06: Preprint DOI Identity Audit

## 1. Executive Summary
- **Objective**: Prevent bibliographic conflation between the preceding foundational study and the current benchmark manuscript.
- **Preceding Study DOI**: `10.20944/preprints202609.0848.v1` (Park & Kim, Sept 2026, *DLG-GNN*).
- **Current Manuscript DOI**: `null` (Pending deposit on Preprints.org).
- **Preprint Status**: `preprint-forthcoming`.
- **Status**: **100% PASSED** (Strict DOI separation maintained across all repositories and files).

---

## 2. Integrity Verification Matrix

| Document / Asset | Field / Reference | Recorded Value | Evaluation |
|---|---|---|---|
| `outputs/benchmark/manuscript_m5/release/release_metadata.json` | `preceding_work_doi` | `10.20944/preprints202609.0848.v1` | **CORRECT** (Attributed to preceding paper) |
| `outputs/benchmark/manuscript_m5/release/release_metadata.json` | `preprint_doi` | `null` | **CORRECT** (Awaiting preprint deposit) |
| `outputs/benchmark/manuscript_m5/release/release_metadata.json` | `preprint_status` | `preprint-forthcoming` | **CORRECT** (No premature claims) |
| `CITATION.cff` | `doi` | `null` / omitted | **CORRECT** (No false top-level DOI) |
| `CITATION.cff` | `references[0].doi` | `10.20944/preprints202609.0848.v1` | **CORRECT** (Cited as prior work) |
| `publication/benchmark/preprints/references.bib` | `park2026dlg` | `10.20944/preprints202609.0848.v1` | **CORRECT** (Cited in bibliography) |
| Public Citation Metadata | Target Journal | Preprints forthcoming | **CORRECT** (Zero "Under Review" claims) |
