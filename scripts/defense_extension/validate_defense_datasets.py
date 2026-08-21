"""Dataset quality gate and leakage audit for defense extension datasets."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch_geometric.data import Data

from gog_fraud.extensions.defense.defense_registry import DEFENSE_DATASETS, load_defense_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def validate_defense_dataset(name: str, data: Data, manifest_path: Path) -> dict:
    """Run rigorous quality gate and leakage audit on a defense dataset."""
    log.info("Validating defense dataset: %s", name)

    # 1. Manifest assertion
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 2. Shape and type assertions
    assert hasattr(data, "x") and data.x is not None, "Missing feature matrix x"
    assert hasattr(data, "edge_index") and data.edge_index is not None, "Missing edge_index"
    assert hasattr(data, "y") and data.y is not None, "Missing label vector y"

    x = data.x
    edge_index = data.edge_index
    y = data.y

    n_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else x.size(0)
    assert x.size(0) == n_nodes, f"x rows ({x.size(0)}) != n_nodes ({n_nodes})"
    assert y.size(0) == n_nodes, f"y length ({y.size(0)}) != n_nodes ({n_nodes})"

    # 3. Value finiteness assertions
    assert torch.isfinite(x).all(), "Feature matrix contains NaN or Inf"
    assert not torch.isnan(y).any(), "Label vector contains NaN"
    assert edge_index.dtype in (torch.int64, torch.long), f"edge_index dtype must be int64/long, got {edge_index.dtype}"

    # 4. Edge index bounds assertions
    n_edges = edge_index.size(1)
    if n_edges > 0:
        assert edge_index.min() >= 0, f"Negative node index in edge_index: {edge_index.min()}"
        assert edge_index.max() < n_nodes, f"Out-of-bounds node index in edge_index: {edge_index.max()} >= {n_nodes}"

    # 5. Label distribution assertions
    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    assert n_pos > 0, "Zero positive labels found"
    assert n_neg > 0, "Zero negative labels found"
    assert n_pos + n_neg == n_nodes, f"Labels contain non-binary values: pos({n_pos}) + neg({n_neg}) != {n_nodes}"
    pos_ratio = n_pos / n_nodes
    log.info("%s label distribution: pos=%d, neg=%d, pos_ratio=%.4f", name, n_pos, n_neg, pos_ratio)

    # 6. Label leakage audit: compute correlation between each feature column and binary label y
    y_np = y.cpu().numpy().astype(np.float64)
    x_np = x.cpu().numpy().astype(np.float64)
    correlations = []
    for col in range(x_np.shape[1]):
        feat_col = x_np[:, col]
        if np.std(feat_col) > 1e-8 and np.std(y_np) > 1e-8:
            corr = float(np.corrcoef(feat_col, y_np)[0, 1])
        else:
            corr = 0.0
        correlations.append(corr)
        # Assert no single feature has exact trivial correlation (e.g. 1.0 or -1.0) indicating direct leakage
        assert abs(corr) < 0.99, f"Feature column {col} has suspicious correlation {corr:.4f} with ground truth label!"

    max_corr = max(abs(c) for c in correlations) if correlations else 0.0
    log.info("%s leakage audit passed (max |correlation| with y is %.4f)", name, max_corr)

    # 7. Topology metrics (homophily, isolated nodes, self-loops)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    self_loops = int(np.sum(src == dst))

    nodes_with_edges = np.unique(np.concatenate([src, dst])) if n_edges > 0 else np.array([])
    isolated_nodes = n_nodes - len(nodes_with_edges)

    # Edge homophily
    if n_edges > 0:
        same_label_edges = int(np.sum(y_np[src] == y_np[dst]))
        edge_homophily = float(same_label_edges / n_edges)
    else:
        edge_homophily = 1.0

    report = {
        "dataset": name,
        "nodes": n_nodes,
        "edges": n_edges,
        "features": int(x.size(1)),
        "positives": n_pos,
        "negatives": n_neg,
        "positive_ratio": round(pos_ratio, 6),
        "edge_homophily": round(edge_homophily, 4),
        "isolated_nodes": isolated_nodes,
        "self_loops": self_loops,
        "max_feature_label_corr": round(max_corr, 4),
        "status": "PASS",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate defense datasets quality and leakage.")
    parser.add_argument("--base-dir", type=str, default="outputs/sci_defense_extension/processed")
    parser.add_argument("--output-json", type=str, default="outputs/sci_defense_extension/manifests/quality_gate.json")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    results = {}

    for name in DEFENSE_DATASETS:
        sub_dir = "darpa_theia" if name == "DARPA-TC-THEIA" else "lanl_redteam"
        manifest_file = "darpa_theia_manifest.json" if name == "DARPA-TC-THEIA" else "lanl_redteam_manifest.json"
        manifest_path = base_dir / sub_dir / manifest_file

        data = load_defense_dataset(name, base_dir=base_dir)
        report = validate_defense_dataset(name, data, manifest_path)
        results[name] = report

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("ALL DEFENSE DATASETS PASSED QUALITY GATE.")


if __name__ == "__main__":
    main()
