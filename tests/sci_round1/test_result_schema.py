from pathlib import Path

from gog_fraud.experiments.sci_round1 import ROUND1_REQUIRED_COLUMNS, ResultStore, experiment_key


def test_result_store_checkpoint_and_resume(tmp_path: Path):
    record = {column: None for column in ROUND1_REQUIRED_COLUMNS}
    record.update({
        "run_id": "run", "experiment_key": experiment_key(dataset="Cora", model="DLG", seed=42, variant="full", split_strategy="stratified", config_hash="abc"),
        "config_hash": "abc", "dataset": "Cora", "domain": "citation_reference",
        "domain_group": "general_graph_anomaly", "label_provenance": "synthetic_injection",
        "model": "DLG", "model_module": "m", "model_class": "C", "seed": 42,
        "variant": "full", "split_type": "stratified_node_transductive",
        "roc_auc": 0.8, "pr_auc": 0.4, "oracle_best_f1": 0.5,
        "validation_f1": 0.4, "f1_at_05": 0.3, "topk_f1": 0.4,
        "positive_ratio": 0.1, "num_nodes": 10, "num_positive": 1,
        "num_negative": 9, "status": "success",
    })
    path = tmp_path / "raw.csv"
    store = ResultStore.open(path); store.append(record)
    reopened = ResultStore.open(path)
    assert record["experiment_key"] in reopened.completed_keys


def test_partitioned_scoring_preserves_node_count(monkeypatch):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    import gog_fraud.pipelines.run_sci_round1_benchmark as runner

    data = Data(x=torch.randn(11, 3), y=torch.tensor([0] * 8 + [1] * 3),
                edge_index=torch.stack([torch.arange(11), torch.roll(torch.arange(11), -1)]), num_nodes=11)
    def fake_fit(model_class, part, **kwargs):
        return object(), np.arange(part.num_nodes, dtype=float), 1.0, .5, 10.0, 0.0
    monkeypatch.setattr(runner, "_fit_and_score", fake_fit)
    _, scores, train, inference, _, _ = runner._fit_and_score_partitioned(
        object, data, partition_size=4, epochs=1, gpu=-1, model_kwargs={})
    assert len(scores) == 11
    assert train == 3.0 and inference == 1.5
