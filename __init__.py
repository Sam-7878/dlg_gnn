"""dlg_gnn package root."""
from pathlib import Path

# Extend package search path to include src/ so that subpackages in src/
# can be imported either directly (e.g. `import graphrag`) or as
# submodules of dlg_gnn (e.g. `import dlg_gnn.graphrag`).
_src = str(Path(__file__).resolve().parent / "src")
if _src not in __path__:
    __path__.append(_src)
