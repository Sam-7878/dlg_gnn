# DLG-GNN core

Reusable model and exact execution code stays under `src/gog_fraud/`. Run `./projects/dlg_gnn/reproduce.sh verify` for source-integrity checks. When Torch/PyG/PyGOD are installed, this also checks a fixed CPU forward/backward/gate fixture, exact sparse value and gradient equivalence, a frozen output hash, and the PyGOD fit/decision/predict interface. Without those packages it reports the scientific smoke as skipped. The fast facade does not rerun the benchmark. The preceding architecture preprint has DOI `10.20944/preprints202609.0848.v1`. Paper regeneration is not yet qualified for this facade.

The published paper source/PDF reviewer copy is indexed in [paper/README.md](paper/README.md).
