import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from gog_fraud.pipelines.run_fraud_benchmark import _build_dataset_from_cfg, _load_config

cfg = _load_config("configs/ngnn_mc/static_benchmark.yaml")
dataset = _build_dataset_from_cfg(cfg)

train_g = dataset.train_graphs[:2]
test_g = dataset.test_graphs[:2]
dataset.train_graphs = train_g
dataset.valid_graphs = []
dataset.test_graphs = test_g

def augment_dataset_with_legacy_scores(dataset, cfg):
    from gog_fraud.adapters.legacy_adapter import LegacyAdapterConfig, LegacyBatchRunner
    import torch
    
    legacy_cfg = cfg.get("legacy", {})
    model_names = legacy_cfg.get("models", ["DOMINANT"])
    model_names = ["DOMINANT", "GAE"]  # Override for fast test
    
    chain_name = cfg.get("dataset", {}).get("chain", "polygon").lower()
    
    base_adapter_cfg = LegacyAdapterConfig(
        agg_method="max", topk=3, normalize_score=True, gpu=0, hid_dim=16,
        num_layers=2, epoch=5, lr=0.003, use_best_params=True, chain=chain_name
    )
    
    all_graphs = dataset.train_graphs + dataset.valid_graphs + dataset.test_graphs
    print(f"Running on {len(all_graphs)} graphs...")
    
    batch = LegacyBatchRunner(
        config=base_adapter_cfg,
        detector_overrides=base_adapter_cfg.detector_overrides,
        score_reduce=base_adapter_cfg.score_reduce,
        progress_every=base_adapter_cfg.progress_every
    )
    all_scores = batch.run_many(model_names=model_names, graphs=all_graphs)
    
    contract_to_scores = {getattr(g, "contract_id", None): [0.0]*len(model_names) for g in all_graphs}
    for i, model_name in enumerate(model_names):
        if model_name in all_scores:
            for r in all_scores[model_name].records:
                if r.contract_id in contract_to_scores:
                    contract_to_scores[r.contract_id][i] = float(r.score)
                    
    for g in all_graphs:
        cid = getattr(g, "contract_id", None)
        scores = contract_to_scores.get(cid, [0.0]*len(model_names))
        data = getattr(g, "graph", g)
        if hasattr(data, "x") and data.x is not None:
            score_tensor = torch.tensor(scores, dtype=torch.float, device=data.x.device).expand(data.x.size(0), -1)
            data.x = torch.cat([data.x, score_tensor], dim=-1)
            print(f"{cid} modified x shape: {data.x.shape}")
            
    return dataset

augment_dataset_with_legacy_scores(dataset, cfg)
