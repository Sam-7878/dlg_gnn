"""Train five auxiliary RiskEncoder checkpoints with full split provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import DATASET_DIR, RESULTS_DIR, RISK_CHECKPOINT_DIR, ensure_dirs
from experiments.round4.risk_encoder import ObservableRiskEncoder


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def features(frame: pd.DataFrame) -> np.ndarray:
    timestamp = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    chain = pd.get_dummies(frame["chain_id"]).reindex(columns=["ethereum", "bsc", "polygon"], fill_value=0)
    numeric = np.column_stack((
        np.log1p(frame["num_nodes"].to_numpy(float)),
        np.log1p(frame["num_edges"].to_numpy(float)),
        np.sin(2 * np.pi * timestamp.dt.hour.to_numpy() / 24),
        np.cos(2 * np.pi * timestamp.dt.hour.to_numpy() / 24),
        timestamp.dt.dayofweek.to_numpy(float) / 6,
    ))
    return np.column_stack((numeric, chain.to_numpy(float))).astype("float32")


def train(seed: int, frame: pd.DataFrame, device) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_rows = frame[frame["split"] == "train"]; valid_rows = frame[frame["split"] == "validation"]
    x_train = features(train_rows); x_valid = features(valid_rows)
    mean = x_train.mean(0); scale = x_train.std(0); scale[scale == 0] = 1
    x_train = torch.tensor((x_train - mean) / scale, device=device)
    x_valid = torch.tensor((x_valid - mean) / scale, device=device)
    y_train = torch.tensor(train_rows["label"].to_numpy(float), dtype=torch.float32, device=device)
    y_valid = valid_rows["label"].to_numpy(int)
    model = ObservableRiskEncoder().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor((len(y_train) - y_train.sum().item()) / y_train.sum().item(), device=device)
    best_ap = -1.0; best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, 51):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x_train), y_train, pos_weight=pos_weight)
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad(): valid_p = torch.sigmoid(model(x_valid)).cpu().numpy()
        ap = average_precision_score(y_valid, valid_p)
        if ap > best_ap + 1e-6:
            best_ap = float(ap); best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= 8: break
    path = RISK_CHECKPOINT_DIR / f"seed{seed}.pt"
    context_path = RESULTS_DIR / "context_provenance.parquet"
    dataset_path = DATASET_DIR / "real_dataset_manifest.json"
    payload = {
        "model_state_dict": best_state, "model_class": "ObservableRiskEncoder",
        "seed": seed, "best_epoch": best_epoch, "best_val_metric": best_ap,
        "training_data": "chronological train split label-independent observable context",
        "target": "contract fraud label", "loss": "weighted_binary_cross_entropy",
        "train_split": "train", "validation_split": "validation", "test_accessed": False,
        "feature_mean": mean, "feature_scale": scale,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
    }
    torch.save(payload, path)
    manifest = {key: value for key, value in payload.items() if key not in ("model_state_dict", "feature_mean", "feature_scale")}
    manifest.update({
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(b"ObservableRiskEncoder-8-24-0.2-v1").hexdigest(),
        "git_commit": git_revision(),
    })
    (RISK_CHECKPOINT_DIR / f"seed{seed}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--seeds", default="7,17,27,37,47"); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); ensure_dirs()
    frame = pd.read_parquet(DATASET_DIR / "transactions.parquet")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    results = [train(int(seed), frame, device) for seed in args.seeds.split(",")]
    (RISK_CHECKPOINT_DIR / "training_summary.json").write_text(json.dumps({"seed_count": len(results), "runs": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
