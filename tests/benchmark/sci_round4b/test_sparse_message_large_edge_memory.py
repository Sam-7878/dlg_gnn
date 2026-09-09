from gog_fraud.models.pygod.sparse_message import (
    estimate_coo_message_bytes,
    estimate_sparse_operator_bytes,
)


def test_sparse_operator_does_not_scale_with_edge_times_hidden_dimension():
    nodes, edges, hidden = 232_965, 114_869_737, 64
    coo = estimate_coo_message_bytes(edges, hidden)
    sparse = estimate_sparse_operator_bytes(nodes, edges + nodes)
    assert coo > 27 * 2**30
    assert sparse < 2 * 2**30
    assert sparse < coo / 10
