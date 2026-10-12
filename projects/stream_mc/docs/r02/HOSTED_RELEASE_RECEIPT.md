# Actual R02 hosted release/access receipt

Published scientific-code/evidence prerelease:
https://github.com/Sam-7878/dlg_gnn/releases/tag/stream-mc-r02-20261012

Target source commit: `07c8578571481b4b0836d38a7929a3530d27063e`.
No force push, original evidence overwrite, manuscript upload or journal/preprint
submission. Benchmark's existing latest release is not replaced.

All eight assets were actually downloaded without Authorization/authenticated
cookies and SHA/byte verified, HTTP 200. A fresh standard-library venv and full
6,065-policy/40-MC verification ran in a new `/tmp` extracted source/input path.
Full scientific dependencies explicitly reused the existing `.venv_cuda`; this
is not a freshly downloaded scientific environment or independent peer review.
Actual receipt: `evidence/access_verification_r02.json` (date 12 October 2026).

Immutable asset identities:

- R02 source ZIP: `45ecc0b07519e06106e7b1b6b9f6ed09e595df9bd96fd10473c77e167ae3d2cc`, 266,435 bytes.
- R02 verification ZIP: `8bc53b9b7312978be804653e9084fb69b473caa9e145545cb393c4ab135dc365`, 1,602,499 bytes.
- R01 assembled ZIP: `97bbdf52f19673a1d0406f4bc1593129d79d70e0b2b1b7dee3cbcbf3dc814d2f`, 258,850,451 bytes, three exact existing parts.

GitHub's draft upload responses initially contained temporary `untagged-...`
download URLs. After publishing, the helper's idempotent existing-release path
refetched the actual published asset metadata; no asset was replaced. Stable
versioned URLs, not the draft responses, were then used for the real public
downloads. For this frozen helper snapshot, that refresh precedes `--mode access`.

The source/evidence ZIPs capture the pre-hosting scientific verification state.
Final live access and final private paper build/visual/package identities are
separate post-hosting receipts in Git to avoid cyclic/self-modifying hashes.
An additional original 40-row MC ranking-definition audit was executed after
packing and is also a separate Git receipt, not falsely claimed as a ZIP member.

Original R01 local manifest/paper remain unchanged. The cheap R02 state probe's
new RSS/timing identity never replaces original v1/active-v2 measurements.
Author-final AI/CRediT/exact PDF/source/evidence approval remains OPEN.

Post-hosting `r02_bindings.py` adds metadata lineage for all 110 original model
prediction Parquet inputs (50 split inputs, 20 light-control inputs, 40 MC
inputs), each with original SHA and the distinct public CSV member/SHA. It is
an added Git-side receipt helper, not a member falsely attributed to the
already published source ZIP. Re-run it after `r02_finalize.py` to extend the
summary-level claim-to-raw CSV. It performs no fitting/scoring/timing changes.

Private paper render hygiene: after a pagination reduction from 17 to 16 pages,
the renderer correctly rejected the stale 17th image rather than accepting a
mixed image inventory. The entire old render directory was moved to ignored
`handoff/r02/visual_before_pagecount_cleanup`, then the exact final paper was
rebuilt/rendered in a fresh directory. Its failed attempt and successful final
attempt have separate real exit-code receipts. The frozen renderer assumes a
fresh output directory when pagination shrinks; preserve/move old renders first.
