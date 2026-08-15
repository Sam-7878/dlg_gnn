import torch
from torch_geometric.data import Data

from gog_fraud.pipelines import analyze_sci_round4c as analysis


def test_data_freeze_hash_changes_with_graph(monkeypatch):
    data=Data(x=torch.ones(3,2),y=torch.tensor([0,1,0]),edge_index=torch.tensor([[0,1],[1,2]]),num_nodes=3)
    monkeypatch.setattr(analysis,"_datasets",lambda config:{"D":lambda:data})
    config={"datasets":["D"],"data":{"fixed_dataset_seed":42}}
    first=analysis.build_data_freeze(config)
    data.edge_index[1,1]=0
    second=analysis.build_data_freeze(config)
    assert first["freeze_hash"] != second["freeze_hash"]

