import copy

import torch
from torch_geometric.nn import GCN

from gog_fraud.models.pygod.shared_reconstruction import CheckpointGCN


def test_checkpoint_gcn_forward_and_gradient_equivalence():
    torch.manual_seed(12)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    x_reference = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    x_checkpoint = x_reference.detach().clone().requires_grad_(True)
    reference = GCN(in_channels=3, hidden_channels=4, out_channels=2, num_layers=3).double()
    checked = CheckpointGCN(in_channels=3, hidden_channels=4, out_channels=2, num_layers=3).double()
    checked.load_state_dict(copy.deepcopy(reference.state_dict()))
    reference.train(); checked.train()
    y_reference = reference(x_reference, edge_index)
    y_checked = checked(x_checkpoint, edge_index)
    torch.testing.assert_close(y_checked, y_reference, rtol=1e-12, atol=1e-12)
    y_reference.square().mean().backward(); y_checked.square().mean().backward()
    torch.testing.assert_close(x_checkpoint.grad, x_reference.grad, rtol=1e-10, atol=1e-11)
    for p_reference, p_checked in zip(reference.parameters(), checked.parameters()):
        torch.testing.assert_close(p_checked.grad, p_reference.grad, rtol=1e-10, atol=1e-11)
