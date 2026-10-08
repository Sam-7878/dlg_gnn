# Frozen Data Recovery Requirement

Expected packed path: `/mnt/d/_Work/goat_bank/dlg_gnn/data/benchmark/gog_scimain_v1`. The manifest has been recovered, but `graph.pt`,
`transactions.parquet`, and `future_edge_audit.csv` are absent. Expected upstream derivative:
`/mnt/d/_Work/_data/GoG_sci_v2`; it is also absent.
The preserved manifest binds the packed graph to SHA-256 `067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd` and the
transaction metadata to `4d240fe8d5488f6f27fd1d475d039abfc96aa40aa6dc34d34a75dfa92be3df3d`, and the future-edge audit to
`395cc4fe3c0c2198fbb25368f9cf843bd9de4352efcbcba8a7d8e77fd5e43f7f`. Publication training must resume only after the
recovered/rebuilt files match these frozen hashes, or after a new explicitly versioned data freeze is
approved before looking at new test results. The missing audit must not be synthetically reconstructed
from its expected hash alone.
