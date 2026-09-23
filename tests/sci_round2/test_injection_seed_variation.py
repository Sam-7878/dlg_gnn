from scripts.benchmark_8x10_pipeline import _inject_outliers
import torch
from torch_geometric.data import Data
from gog_fraud.experiments.round2_validity import graph_fingerprints


def test_injection_seed_changes_generated_graph_or_labels():
    nodes = 120
    base = Data(x=torch.randn(nodes, 6), edge_index=torch.stack([torch.arange(nodes), torch.roll(torch.arange(nodes), -1)]), num_nodes=nodes)
    one = _inject_outliers(base.clone(), contextual_ratio=.05, structural_ratio=.05, m_clique=5, seed=41)
    two = _inject_outliers(base.clone(), contextual_ratio=.05, structural_ratio=.05, m_clique=5, seed=43)
    assert graph_fingerprints(one) != graph_fingerprints(two)

