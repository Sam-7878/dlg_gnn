import torch
from torch_geometric.data import Batch, Data

from experiments.round5.models import FraudSAGEBaseline, TGATBaseline, class_balanced_focal_loss


def test_minimal_temporal_and_fraud_models_have_finite_forward_loss():
    graph = Data(
        x=torch.tensor([[0., 1., 1.], [1., 0., 1.]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor(1.), chain_id=torch.tensor(0), timestamp=torch.tensor(1),
    )
    batch = Batch.from_data_list([graph])
    temporal = TGATBaseline()
    fraud = FraudSAGEBaseline()
    temporal_logit = temporal(batch, torch.tensor([0.5]))
    fraud_logit = fraud(batch)
    assert torch.isfinite(temporal_logit).all() and torch.isfinite(fraud_logit).all()
    assert torch.isfinite(class_balanced_focal_loss(fraud_logit, torch.ones(1), 2.0))
