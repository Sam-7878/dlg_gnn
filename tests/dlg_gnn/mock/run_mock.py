import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from gog_fraud.pipelines.run_mc_benchmark import _build_dataset_from_cfg, _load_config, augment_dataset_with_legacy_scores
cfg = _load_config("configs/ngnn_mc/static_benchmark.yaml")
dataset = _build_dataset_from_cfg(cfg)
dataset.train_graphs = dataset.train_graphs[:4]
dataset.valid_graphs = dataset.valid_graphs[:4]
dataset.test_graphs = dataset.test_graphs[:4]
import pickle
with open('mock_ds.pkl', 'wb') as f:
    pickle.dump(dataset, f)
