# Publication visibility and evidence identity (2026-10-09)

The author has not submitted the current manuscript and requests that its LaTeX sources stay off GitHub. `.gitignore` excludes manuscript TeX/Bib/class/style/PDF files, active `paper/current` trees, the benchmark A06 manuscript input and manuscript writers. Public project evidence is separated from these local inputs.

- Shared implementations remain in `src/`; project facades and audits remain in `projects/<project>/scripts/`.
- Public benchmark evidence: `projects/benchmark/evidence/public_numeric_evidence.zip` and `evidence/astra_revision/`. The original 645-payload frozen ZIP stays local unchanged; the public derivative retains 641 byte-identical payloads and excludes four manuscript-generation/packaging scripts. Both hashes are recorded.
- Public checkout commands: `verify` (standard Python) and `tables` (NumPy/SciPy). `paper` is author-local; full training is separate manual orchestration.
- Exact-byte `.gitattributes` prevents Git newline conversion from invalidating source/evidence hashes across Windows and WSL.
- Current index removals preserve all local files. Previously published ZIP/writer content remains in Git history; no history rewrite is authorized or performed.

Scientific integrity PASS is distinct from publication readiness. Construction of the Ethereum/BSC/Polygon hybrid artifacts remains unresolved after discovery of same-label added edges and seven constant feature columns. Original builder evidence is required before unsupervised fraud interpretation is cleared. Historical crypto Aug F1 uses a different threshold policy and remains explicitly diagnostic. Historical missing score arrays cannot be recreated from scalar metrics. These issues are tracked in the benchmark revision report.
