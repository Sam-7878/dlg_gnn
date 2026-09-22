import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_risk_encoder_checkpoint_provenance():
    directory = ROOT / "results/graphrag/round_4/risk_encoder_checkpoints"
    for seed in (7, 17, 27, 37, 47):
        manifest = json.loads((directory / f"seed{seed}.json").read_text())
        checkpoint = ROOT / manifest["checkpoint"]
        assert checkpoint.is_file()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
        assert manifest["seed"] == seed
        assert manifest["test_accessed"] is False
        assert manifest["dataset_sha256"] and manifest["context_sha256"] and manifest["config_sha256"]
        assert manifest["git_commit"] != "working_tree"

