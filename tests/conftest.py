"""Global pytest configuration for dlg_gnn repository.

Ensures both dlg_gnn root and its parent (goat_bank) are in sys.path
so all package imports (`import dlg_gnn...`, `import experiments...`, `import src...`)
resolve reliably regardless of where pytest is executed from.
"""
import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent
_dlg_gnn_root = _tests_dir.parent
_goat_bank_root = _dlg_gnn_root.parent

for _p in (str(_dlg_gnn_root), str(_goat_bank_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
