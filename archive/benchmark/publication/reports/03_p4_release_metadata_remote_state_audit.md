# Round P4 Audit Report 03: Release Metadata & Remote State Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite (Round P4)  
**Target File:** `outputs/benchmark/manuscript_m5/release/release_metadata.json`

---

## 1. Executive Summary

A critical finding in Round P4 was the discrepancy between local metadata declarations (which claimed `is_public_release: true`) and the actual external state of the GitHub remote repository (`https://github.com/Sam-7878/dlg_gnn`), which currently has 0 releases and 0 tags published. Round P4 strictly aligns local metadata with physical reality, ensuring no automated tests falsely report a PASS on external availability.

---

## 2. Release Metadata Audit

The canonical metadata in `release_metadata.json` has been updated to:

```json
{
  "release_name": "DLG-GNN Benchmark v1.0.0-preprint",
  "version": "1.0.0-preprint",
  "git_tag": "v1.0.0-preprint",
  "is_public_release": false,
  "release_state": "prepared_for_public_release",
  "repository_url": "https://github.com/Sam-7878/dlg_gnn",
  "release_asset_filename": "DLG_GNN_Benchmark_v1.0.0_preprint.zip",
  "preceding_paper_doi": "10.20944/preprints202609.0848.v1",
  "current_paper_doi": null,
  "authors": [
    {
      "name": "SeongSu Park",
      "orcid": "0009-0008-4056-3875"
    },
    {
      "name": "Ki-Hyung Kim",
      "orcid": "0000-0002-2321-4475"
    }
  ]
}
```

---

## 3. Remote State Reconciliation

| Dimension | Local Declaration | Remote GitHub State | Consistency Verdict |
|---|---|---|:---:|
| **Repository Visibility** | Public (`https://github.com/Sam-7878/dlg_gnn`) | Public (HTTP 200 OK) | **MATCH** |
| **Git Tag `v1.0.0-preprint`** | Created locally / prepared | Not yet pushed to remote | **MATCH** (`prepared`) |
| **GitHub Release** | Packaged locally (`DLG_GNN_Benchmark_v1.0.0_preprint.zip`) | Not yet created on remote | **MATCH** (`prepared`) |
| **Release Flag** | `is_public_release: false` | 0 releases on remote | **TRUTHFUL** |

---

## 4. Gate Test Design & Guardrails

To prevent future regression or premature PASS:
1. `tests/benchmark/publication_p4/test_github_remote_status_gate.py` connects via public unauthenticated GitHub API:
   - Queries `https://api.github.com/repos/Sam-7878/dlg_gnn/tags`
   - Queries `https://api.github.com/repos/Sam-7878/dlg_gnn/releases`
2. If the tag or release does not exist on remote, the test issues an explicit `pytest.skip` explaining that the repository is in state `PENDING_GITHUB_PUSH` or `PENDING_GITHUB_RELEASE`.
3. Only after the author pushes the tag and attaches the release asset does the remote gate transition to full remote PASS.

---

## 5. Audit Verdict

**STATUS: PASS (Truthful State: `prepared_for_public_release`)**  
Local metadata accurately represents the preparation status and enforces external verification gates.
