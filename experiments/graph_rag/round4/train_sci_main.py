"""Train five real GNN checkpoints on the frozen chronological main track."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.round4.artifact_paths import (
    CHECKPOINT_DIR, CHECKPOINT_MANIFEST_DIR, DATASET_DIR, ensure_dirs,
)
from experiments.round4.data import load_packed
from experiments.round4.model import CausalLocalGIN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def predict(model, loader, device):
    model.eval(); labels = []; probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probabilities.extend(torch.sigmoid(model(batch)).cpu().tolist())
            labels.extend(batch.y.cpu().long().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def train_seed(seed: int, datasets, config: dict, dataset_manifest: dict, device) -> dict:
    set_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(datasets["validation"], batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=config["batch_size"], shuffle=False)
    model = CausalLocalGIN(config["input_dim"], config["hidden_dim"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_pos = dataset_manifest["split"]["train"]["n_positive"]
    train_neg = dataset_manifest["split"]["train"]["n_negative"]
    pos_weight = torch.tensor(train_neg / train_pos, device=device)
    best_ap = -1.0; best_epoch = 0; best_state = None; stale = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), batch.y.float(), pos_weight=pos_weight)
            loss.backward(); optimizer.step()
        valid_y, valid_p = predict(model, valid_loader, device)
        valid_ap = average_precision_score(valid_y, valid_p)
        if valid_ap > best_ap + 1e-6:
            best_ap = float(valid_ap); best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        print(f"seed={seed} epoch={epoch} val_ap={valid_ap:.6f}", flush=True)
        if stale >= config["early_stopping_patience"]: break
    model.load_state_dict(best_state); model.to(device)
    test_y, test_p = predict(model, test_loader, device)
    checkpoint_path = CHECKPOINT_DIR / f"seed{seed}.pt"
    torch.save({
        "model_state_dict": best_state, "model_class": "CausalLocalGIN",
        "model_config": config, "seed": seed, "best_epoch": best_epoch,
        "best_val_auc_pr": best_ap, "gnn_source": "real_checkpoint",
        "split_type": "chronological_real", "timestamp_source": "recorded_transaction_timestamp",
        "dataset_sha256": dataset_manifest["graph_sha256"],
    }, checkpoint_path)
    result = {
        "seed": seed, "best_epoch": best_epoch, "best_val_auc_pr": best_ap,
        "test_auc_pr": float(average_precision_score(test_y, test_p)),
        "test_auc_roc": float(roc_auc_score(test_y, test_p)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dataset_sha256": dataset_manifest["graph_sha256"],
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "seed_provenance": seed,
    }
    (CHECKPOINT_MANIFEST_DIR / f"seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "round4_sci_main_frozen.yaml")
    args = parser.parse_args(); ensure_dirs()
    config = yaml.safe_load(args.config.read_text())
    _, manifest, datasets = load_packed(DATASET_DIR)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    results = [train_seed(int(seed), datasets, config, manifest, device) for seed in args.seeds.split(",")]
    summary = {
        "gnn_source": "real_checkpoint", "split_type": "chronological_real",
        "timestamp_source": "recorded_transaction_timestamp", "seed_count": len(results),
        "seeds": [row["seed"] for row in results], "dataset_sha256": manifest["graph_sha256"],
        "mean_test_auc_pr": float(np.mean([row["test_auc_pr"] for row in results])),
        "std_test_auc_pr": float(np.std([row["test_auc_pr"] for row in results])),
        "runs": results,
    }
    (CHECKPOINT_MANIFEST_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
