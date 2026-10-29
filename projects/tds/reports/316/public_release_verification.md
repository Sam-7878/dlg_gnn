# Public release verification

Status: **LOCAL_COLD_REPLAY_PASS / PUBLIC_EVIDENCE_RELEASE_VERIFIED=false**.

Current local and remote main HEAD: `70cd82e2b0c2e93995c27384e7a776223db3e87f` at
inspection. The inherited R315 relocation/source changes are still in the working
tree. Commit identity and changed-path evidence are in git_remote_audit.json.
An anonymous GitHub release inventory was retrieved with HTTP200 and is recorded
in anonymous_release_inventory.json. Existing Benchmark/StreamMC releases are not
used as TDS evidence. No approved exact TDS tag/asset URL is verified here.

The originally reviewed numeric archive has SHA5ac20f2c...; the current relocated
R315 derivative has SHA13c70168.... Both are distinguished in the316 input seal.
The new316 candidate adds documentation and gate tests; all65 prediction vectors
are byte-identical to both predecessors. Exact archive/vector hashes are in
publication/316/release_manifest.json. No raw transactions, contract addresses,
checkpoints or manuscript sources are included.

A new temporary extraction passes hash verification, all seven supplemental AP
mean/SD pairs, one original comparison's three10000-replicate protocols and18
tests. This proves local L2 portability, not anonymous download of a published
immutable TDS asset. L0 is author-local source build; L1 has packed/artifact hash
verification; L3 independent full retraining remains unguaranteed.

Before any release: obtain explicit approval of the exact candidate and rights,
publish the approved commit/tag/asset, then download it without credentials and
repeat hash/replay/tests. Record actual URL, commit, tag, asset hash and date.
No push/tag/release/DOI operation occurred in316. No platform-approved availability
exception has been invented.
