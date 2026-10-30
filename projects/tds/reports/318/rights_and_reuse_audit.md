# Rights and reuse audit — HOLD

The [upstream repository](https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs)
describes address-named per-contract CSVs and numeric-index/address mappings. Its
[pinned LICENSE](https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs/blob/7264f1bf510f7ba4f5041ac7a29b606abc12f262/LICENSE)
is CC BY-NC-SA 4.0, independently fetched through GitHub (blob c14a34c4954e217a88d27d206752db1c79dcaf47).
The license is not an unrestricted software grant: attribution, noncommercial and
share-alike conditions apply within its scope. The [official legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
also separates copyright/database rights from privacy and other rights. This audit
identifies obligations and unresolved scope; it does not supply author/third-party permission.

Actual ZIP scan: no direct 40-hex contract addresses in CSV/JSON, no raw transfers,
address mapping table, weights or manuscript source. However 215 CSVs contain sample_id;
all 24,316 lineage IDs join one-to-one to a private contract mapping. Chain/time/category
signatures can also support joins to upstream records. Identifier-reduced means
pseudonymous/linkable here; no anonymity or privacy guarantee is claimed. Only aggregate
linkage counts are recorded publicly; no private mapping is copied into public outputs.

The exact 13-MB R317 candidate remains sealed; a byte-identical release copy, attribution
NOTICE and the upstream license are staged in publication/318/release. No new derivative
license is imposed on behalf of unconsulted rights holders. Software and numeric
derivatives have distinct license scopes; manuscript CC BY intentions do not silently
relicense upstream-derived data or figures. The existing manuscript citation/upstream
link is retained. Authors must clear final distribution scope, including any derivative
figure permissions, before pushing/tagging/releasing. No history rewrite/removal occurred.

The R317 source manifest's new_code_committed=false was a historical working-tree
snapshot. Current R314–R317 files are now checked against local HEAD and a R317 code
file is publicly readable at that commit; this is not a numerical TDS release or
author release permission. See results/318/source_commit_reconciliation.json.
