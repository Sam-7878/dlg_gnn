import numpy as np
import torch
from torch_geometric.data import Data
import gog_fraud.pipelines.run_sci_round1_benchmark as runner


def test_partition_scores_reassemble_in_original_contiguous_order(monkeypatch):
    data = Data(x=torch.arange(21).reshape(7, 3).float(), y=torch.tensor([0, 0, 0, 1, 1, 0, 1]),
                edge_index=torch.stack([torch.arange(7), torch.roll(torch.arange(7), -1)]), num_nodes=7)
    cursor = {"value": 0}
    def fake_fit(_, part, **kwargs):
        start = cursor["value"]; cursor["value"] += part.num_nodes
        return object(), np.arange(start, start + part.num_nodes), 0, 0, 0, 0, 0, 0
    monkeypatch.setattr(runner, "_fit_and_score", fake_fit)
    result = runner._fit_and_score_partitioned(object, data, partition_size=3, epochs=1, gpu=-1, model_kwargs={})
    assert result[1].tolist() == list(range(7))

