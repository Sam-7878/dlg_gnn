"""Reproducible experiment entry points for DLG-GNN."""
from pathlib import Path

# Extend package __path__ to automatically include subproject experiment modules
# This enables both `from experiments.graph_rag.round7 import ...` and `from experiments.round7 import ...`
_pkg_root = Path(__file__).resolve().parent
for _subproject in ("graph_rag", "stream_mc", "dlg_gnn", "benchmark"):
    _subpath = str(_pkg_root / _subproject)
    if _subpath not in __path__:
        __path__.append(_subpath)
