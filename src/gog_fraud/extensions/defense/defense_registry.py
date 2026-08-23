"""Isolated dataset registry for defense extension datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import torch
from torch_geometric.data import Data


DEFENSE_DATASETS = ["DARPA-TC-THEIA", "LANL-RedTeam"]
DEFENSE_DISPLAY_NAMES = {
    "DARPA-TC-THEIA": "DARPA-TC-THEIA",
    "LANL-RedTeam": "LANL-RedTeam",
}


def load_defense_dataset(name: str, base_dir: Path | str = "outputs/sci_defense_extension_real/graphs") -> Data:
    """Load canonical PyG defense graph artifact."""
    base = Path(base_dir)

    # Real D3 official graph paths
    if (base / "theia_graph.pt").exists() and name == "DARPA-TC-THEIA":
        path = base / "theia_graph.pt"
    elif (base / "lanl_graph.pt").exists() and name == "LANL-RedTeam":
        path = base / "lanl_graph.pt"
    elif name == "DARPA-TC-THEIA":
        # Fallback to legacy/processed path if specified
        path = base / "darpa_theia" / "darpa_tc_theia_e3.pt"
        if not path.exists():
            path = base / "theia_graph.pt"
    elif name == "LANL-RedTeam":
        path = base / "lanl_redteam" / "lanl_redteam_computer_graph.pt"
        if not path.exists():
            path = base / "lanl_graph.pt"
    else:
        raise KeyError(f"Unknown defense dataset: {name}. Available: {DEFENSE_DATASETS}")

    if not path.exists():
        raise FileNotFoundError(f"Defense dataset artifact not found at {path}. Run build script first.")

    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def get_defense_registry(base_dir: Path | str = "outputs/sci_defense_extension_real/graphs") -> Dict[str, Callable[[], Data]]:
    """Return dictionary of lazy dataset loaders for defense extension."""
    return {
        name: (lambda n=name: load_defense_dataset(n, base_dir))
        for name in DEFENSE_DATASETS
    }
