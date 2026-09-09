sed -i 's/batch_size: 64/batch_size: 2/g' configs/ngnn_mc/static_benchmark.yaml
sed -i 's/batch_size: 128/batch_size: 2/g' configs/ngnn_mc/static_benchmark.yaml
sed -i 's/eval_chunk_size: 16/eval_chunk_size: 2/g' configs/ngnn_mc/static_benchmark.yaml
sed -i 's/train_chunk_size: 16/train_chunk_size: 2/g' configs/ngnn_mc/static_benchmark.yaml

cat << 'PY' > run_mock.py
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
PY
PYTHONPATH=./src .venv/bin/python3 run_mock.py

sed -i 's/dataset = _build_dataset_from_cfg(cfg)/import pickle; dataset = pickle.load(open("mock_ds.pkl", "rb"))/g' src/gog_fraud/pipelines/run_mc_benchmark.py

PYTHONPATH=./src .venv/bin/python3 src/gog_fraud/pipelines/run_mc_benchmark.py --config configs/ngnn_mc/static_benchmark.yaml --chain polygon --stages l1_legacy_aug,l1_l2_legacy_aug

git checkout src/gog_fraud/pipelines/run_mc_benchmark.py
git checkout configs/ngnn_mc/static_benchmark.yaml
