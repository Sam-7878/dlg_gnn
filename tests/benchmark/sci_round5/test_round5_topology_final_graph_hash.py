import torch

from gog_fraud.pipelines.analyze_sci_round4c import _hash_tensor


def test_topology_hash_is_order_sensitive_and_exact():
    edge=torch.tensor([[0,1],[1,0]])
    assert _hash_tensor(edge) == _hash_tensor(edge.clone())
    assert _hash_tensor(edge) != _hash_tensor(edge.flip(1))
