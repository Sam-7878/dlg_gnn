# Round P4 Audit Report 05: GitHub Tag & Release External Verification

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite (Round P4)  
**Remote Repository:** `https://github.com/Sam-7878/dlg_gnn`

---

## 1. Executive Summary

This audit performs an external, unauthenticated query against the live GitHub repository to determine whether the public repository state matches the requirements for depositing to Preprints.org. 

---

## 2. Remote State Query Results

A live unauthenticated HTTPS query to GitHub APIs (`https://api.github.com/repos/Sam-7878/dlg_gnn`) yielded the following status:

| Checkpoint | Target State | Live Remote State | Status |
|---|---|---|:---:|
| **Repository Visibility** | Public (not private) | Public (`private: false`, HTTP 200) | **PASS** |
| **Logged-out Web Access** | Accessible to public | Accessible via standard browser | **PASS** |
| **Git Tag `v1.0.0-preprint`** | Tagged on remote | Prepared locally; not yet pushed | **PENDING_PUSH** |
| **GitHub Release** | Release published on remote | 0 releases found on remote API | **PENDING_RELEASE** |
| **Release Asset Download** | `DLG_GNN_Benchmark_v1.0.0_preprint.zip` downloadable | Asset ready locally (0.49 MB) | **PENDING_ATTACHMENT** |

---

## 3. Author Action Steps to Achieve Complete Go-Live

To transition the project state from `PENDING_GITHUB_RELEASE` to `READY_FOR_PREPRINTS_ORG_DEPOSIT`, the author must perform the following standard steps:

### Step 1: Push Commits and Tag to GitHub
```bash
# Push main branch containing clean README.md, INSTALL.md, LICENSE
git push origin main

# Push release tag
git push origin v1.0.0-preprint
```

### Step 2: Create GitHub Release on Web UI or GitHub CLI
- **Tag:** `v1.0.0-preprint`
- **Release Title:** `DLG-GNN Benchmark v1.0.0-preprint`
- **Release Notes Summary:**
  ```markdown
  ## DLG-GNN Benchmark v1.0.0-preprint
  
  This release provides the complete source code, frozen benchmark artifacts, and reproduction scripts for:
  **"Empirical Evaluation of Directed Local-Global Graph Neural Networks for Fraud and Anomaly Detection"**
  
  ### Quick Reproduction (Mode 1)
  Extract this release package and run:
  ```bash
  python scripts/reproduce_frozen_artifacts.py
  ```
  
  ### Frozen Hashes
  - `artifacts/primary/benchmark_raw.csv`: `39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c`
  - `artifacts/primary/model_dataset_support_matrix.csv`: `c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914`
  ```
- **Attach Binary Asset:**
  Upload `outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip`.

### Step 3: Update Metadata Flag & Re-run Gate Test
Once the release is live on GitHub:
1. Update `outputs/benchmark/manuscript_m5/release/release_metadata.json`:
   - `"is_public_release": true`
   - `"release_state": "public"`
2. Re-run `pytest tests/benchmark/publication_p4/test_github_remote_status_gate.py` to confirm external download verification passes.

---

## 4. Audit Verdict

**STATUS: PENDING_GITHUB_RELEASE**  
The repository is public. All assets, documentation, and tags are prepared locally. Completion of the GitHub release publish will fulfill the external release gate.
