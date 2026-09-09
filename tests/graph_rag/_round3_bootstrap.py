"""Ensure Round 3 tests import this repository's experiment package."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The editable package exposes ``src/experiments`` as a namespace.  Pytest may
# discover it before the repository-level experiment entry points; discard only
# that namespace so the explicit package above is imported deterministically.
loaded = sys.modules.get("experiments")
loaded_file = str(getattr(loaded, "__file__", ""))
if loaded is not None and not loaded_file.startswith(str(ROOT / "experiments")):
    for name in tuple(sys.modules):
        if name == "experiments" or name.startswith("experiments."):
            del sys.modules[name]
