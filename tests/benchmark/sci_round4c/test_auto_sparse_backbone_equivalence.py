import copy

import torch
from torch_geometric.nn import GCN

from gog_fraud.models.pygod.sparse_message import AutoSparseFusedGCN


def test_drop_in_sparse_backbone_matches_pyg_coo():
    torch.manual_seed(9)
    x=torch.randn(8,5,dtype=torch.float64,requires_grad=True)
    edge=torch.tensor([[0,1,2,2,3,4,5,6,7],[1,2,0,3,4,5,6,7,0]])
    reference=GCN(5,7,3,out_channels=4,dropout=0).double()
    fused=AutoSparseFusedGCN(5,7,3,out_channels=4,dropout=0).double()
    fused.load_state_dict(copy.deepcopy(reference.state_dict()))
    actual=fused(x,edge);expected=reference(x,edge)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

