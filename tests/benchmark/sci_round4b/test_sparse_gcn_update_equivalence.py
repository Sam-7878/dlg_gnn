import torch

from conftest import model_pair


def test_sparse_gcn_update_equivalence(message_case):
    x, edge_index = message_case
    reference, fused, adj_t = model_pair(x, edge_index, 4)
    opt_reference = torch.optim.Adam(reference.parameters(), lr=.003)
    opt_fused = torch.optim.Adam(fused.parameters(), lr=.003)
    trajectories = [[], []]
    for _ in range(4):
        for index, (model, optimizer, graph) in enumerate(
            ((reference, opt_reference, edge_index), (fused, opt_fused, adj_t))
        ):
            optimizer.zero_grad(); loss = model(x, graph).square().mean()
            loss.backward(); optimizer.step(); trajectories[index].append(loss.detach())
    torch.testing.assert_close(torch.stack(trajectories[1]), torch.stack(trajectories[0]), rtol=1e-9, atol=1e-10)
    for expected, actual in zip(reference.parameters(), fused.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
