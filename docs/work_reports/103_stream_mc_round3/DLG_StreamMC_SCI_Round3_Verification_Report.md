# DLG-StreamMC SCI Round 3 Verification Report

## Executive decision

- Overall status: **NOT_READY**
- P0 gate: **BLOCKED**
- Paper Revision Gate: **CLOSED**
- Sample-level leakage: **PASS**
- Paper-eligible experiment records: **0**

The report distinguishes artifact validity from scientific readiness. A valid report does not open the manuscript gate.

## Dataset v2

| Chain | Built / Expected | Transaction rows | Reordered raw files | Legacy resolved | Mapping status |
|---|---:|---:|---:|---:|---|
| bsc | 7499/7499 | 121,612,480 | 6,016 | 6,837/7,481 | INCOMPLETE |
| ethereum | 14464/14464 | 81,788,211 | 8,887 | 12,329/14,385 | INCOMPLETE |
| polygon | 2353/2353 | 64,882,233 | 1,634 | 2,156/2,303 | INCOMPLETE |

- Dataset version: `gog-sci-v2.0`
- Label semantics: `RESOLVED`
- Leakage violations: `0`
- Records audited: `24316`

## Verification matrix

| Required result | Status | Evidence / reason |
|---|---|---|
| Processed Dataset v2 | PASS | 24,316/24,316 files built; failures 0 |
| Raw-to-v2 Graph Mapping | PASS | Contract-address sample IDs and source hashes embedded directly |
| Legacy Numeric Graph Mapping | INCOMPLETE | 2,847 shape-ambiguous legacy graphs remain |
| Embedded Label Orientation | PASS | 21,322 consistent, 0 reversed |
| Sample-Level Leakage | PASS | 24,316 samples; 0 violations |
| Train-Only Normalizer / Relation Pool | PASS | Fold artifacts audited; future candidate/relation count 0 |
| In-Scope Tests | PASS | 113 tests, 0 failures, 0 errors |
| Real PyGOD Pilot | NOT_RUN | P0 legacy mapping gate blocked |
| DLG/StreamMC Main 5-Seed | NOT_RUN | P0 legacy mapping gate blocked |
| MC / Routing / Calibration | NOT_RUN | Downstream of main |
| 100k Resource / Temporal / Cross-Chain / Statistics | NOT_RUN | Downstream of main |

## Label orientation decision

SCI v2 corrects the Round 2 assumption. The upstream README states that category 0 is fraud, and legacy embedded-label aggregate counts independently agree within missing-graph counts. Therefore v2 maps category 0 to binary fraud 1 and every non-zero category to binary benign 0. The upstream repository commit/tag remains unavailable and is retained as a provenance limitation.

## Implemented Round 3 components

- deterministic stable sorting with original-row tie breaking;
- order-independent duplicate-sensitive row multiset digests;
- source, sorted artifact, graph, and feature-provenance hashes;
- cutoff-bound tensor graph artifacts with edge timestamps;
- explicit contract/chain/sample identity and global/legacy mapping fields;
- train-only fixed temporal normalizers and historical relation candidate pools;
- all-sample strict leakage auditor and separate legacy compatibility auditor;
- PyGOD/PyG 2.7 compatibility fixes for DLG and nested nGNN batching;
- fail-closed report, evidence index, validation, and package generation.

## Experiment decision

Experiment status: **NOT_RUN**. Round 3 fail-closed P0 prerequisite did not pass; main experiments were not authorized by the work order.

No smoke heuristic, mock result, or hard-coded metric is promoted to a paper claim.

### Blocking condition and next action

The v2 dataset itself has complete address-based identity. The remaining blocker is compatibility with legacy numeric JSON names: edge/node shape uniquely resolves most graphs but 2,847 graphs share a shape. A stronger legacy fingerprint (canonical degree/value-feature vector digest or an authoritative historical index) is required before the work order permits pilot/main execution. After resolving it, rerun compatibility v2, require all three statuses `PASS`, then freeze a clean commit/tag and start Phase B.

## Reproducibility

- WSL2 Python: `/mnt/d/_Work/goat_bank/.venv/bin/python`
- Git SHA: `ebb38480f3c85b28158441cc0f15427128a79834`
- Dirty at report generation: `True`
- Canonical SCI v2 root: `/mnt/d/_Work/_data/GoG_sci_v2`
