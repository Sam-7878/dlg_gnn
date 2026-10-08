# Round P3 Audit Report 05: Data Availability & Repository Go-Live Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite  
**Scope:** Data Availability provenance description, GitHub repository go-live sequence, release metadata semantics, and external accessibility verification.

---

## 1. Data Availability Statement Provenance Audit

The Data Availability statement in `DLG-Benchmark.tex` (and duplicated in `DLG-Benchmark-Preprint.tex` and the MDPI bundle) was audited against source-of-truth guidelines:

> *"The seven ``-Syn'' benchmarks are constructed from public base graphs distributed through PyGOD and their original repositories using the frozen contextual and structural anomaly-injection protocol described in this paper. BitcoinOTC is obtained from the SNAP repository with real transaction rating labels. Elliptic is available on Kaggle, DGraphFin on FinVolution, and LANL-RedTeam from the Los Alamos National Laboratory Cyber Security Data repository. The complete benchmark suite, model source code, exact sparse execution backends, canonical manifests, evaluation tables, and replication scripts are openly available upon preprint release at \url{https://github.com/Sam-7878/dlg_gnn}."*

Key provenance criteria verified:
1. Distinguishes base graphs from injected anomalies (clarifying that base graphs were not originally synthetic).
2. Explicitly identifies BitcoinOTC as a real-label graph from SNAP.
3. Points to the live repository URL: `https://github.com/Sam-7878/dlg_gnn`.
4. **Automated Test:** `tests/benchmark/publication_p3/test_data_availability_syn_provenance.py` (PASS)

---

## 2. Repository Metadata & Boolean Semantics (Reconciled in Round P4)

Release metadata in `outputs/benchmark/manuscript_m5/release/release_metadata.json` was reconciled to reflect actual remote state:
- **`is_public_release`**: `false` (accurate declaration prior to external release creation on GitHub)
- **`release_state`**: `"prepared_for_public_release"`
- **`repository_url`**: `"https://github.com/Sam-7878/dlg_gnn"`
- **`version`**: `"1.0.0-preprint"`
- **`git_tag`**: `"v1.0.0-preprint"`
- **Replaced Asset**: `DLG_GNN_Benchmark_v1.0.0_preprint.zip` replaces historical `DLG_GNN_Benchmark_M5_Release.zip`

---

## 3. Repository Go-Live Verification Sequence

The repository is public at `https://github.com/Sam-7878/dlg_gnn`. The pending step is the GitHub release attachment:

```text
Step 1: Confirm GitHub repository is set to Public at https://github.com/Sam-7878/dlg_gnn (COMPLETED)
Step 2: Tag v1.0.0-preprint and publish release asset DLG_GNN_Benchmark_v1.0.0_preprint.zip (PENDING GITHUB RELEASE)
Step 3: Verify logged-out web access to release archive (BLOCKED ON STEP 2)
Step 4: Deposit neutral manuscript package (DLG_Benchmark_Preprints_Submission.zip) to Preprints.org
Step 5: Receive Preprints.org screening confirmation and official DOI
Step 6: Update MDPI cover letter with actual DOI and submit MDPI package
```

---

## 4. Audit Verdict
**STATUS: PENDING_EXTERNAL_RELEASE** (Reconciled in Round P4 Section 16; pending remote GitHub release attachment)
