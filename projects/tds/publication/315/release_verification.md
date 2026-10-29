# Release verification

Status: **LOCAL_STAGING_ONLY / PUBLIC GATE FAIL**.

Candidate tag: `tds-315-submission-2026-10-29`. Both local tag listing and a read-only
`git ls-remote --tags https://github.com/Sam-7878/dlg_gnn.git refs/tags/tds-315-submission-2026-10-29`
returned no matching tag on 2026-10-29. This is a point-in-time name-availability
check, not a reservation or immutable release identity.

Actual release URL: null. Published tag/commit: null. DOI: not assigned.
No public push, GitHub release, Zenodo deposition or journal submission occurred.
The existing shared repository and Benchmark tags are not TDS release evidence.

The local numeric archive CRC and SHA-256 are in candidate_archive.json. The
payload manifest records all included files and transformation lineage. Privacy
scan found no private drive/home prefix, email, contract address, recognized key
pattern or prohibited manuscript/model payload. This is a bounded pattern scan
plus inventory review, not a guarantee against every possible secret encoding.

Fresh-extraction tests and prediction replay are recorded in
`projects/tds/reports/315/candidate_local_verification.json`. These demonstrate
local portability, not anonymous download. One staged 10,000-replicate statistical
comparison also matches the corrected frozen CSV.

After explicit author approval: review the exact archive hash, license terms and
public Git diff; publish an unused immutable tag for the approved commit; retrieve
the release asset in an unauthenticated session; compare its hash; rerun replay and
scientific tests from the download; record URL, commit, date, HTTP evidence and
output hashes. Update Data Availability and final approval identities afterwards.
Unsubmitted paper sources and private author records must remain excluded.
