"""Train the frozen comparable model suite on exact GoG-SCIMain-v1.

The runner refuses to start unless the four packed artifacts match the Round 7
hash contract. Test inference occurs exactly once after validation selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.round4.data import load_packed
from experiments.round5.analysis import classification_metrics
from experiments.round5.models import FraudSAGEBaseline, TGATBaseline, TemporalMemoryBaseline, class_balanced_focal_loss
from experiments.round7.provenance import EXPECTED_PACKED_HASHES, SEEDS, sha256_file, verify_hash_contract


MODEL_NAMES = {
    "tgat": "TGAT-style temporal attention",
    "tgn": "TGN-style event memory",
    "fraudsage": "fraud-oriented GraphSAGE",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_time(data, train_min: int, train_scale: int, device: torch.device) -> torch.Tensor:
    return (data.timestamp.to(device).float() - float(train_min)) / float(train_scale)


def _predict_static(model, dataset, device, train_min: int, train_scale: int, kind: str) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch, normalized_time(batch, train_min, train_scale, device)) if kind == "tgat" else model(batch)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(probabilities, dtype=float)


def _predict_tgn(model: TemporalMemoryBaseline, dataset, device, train_min: int, train_scale: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    memory = model.initial_memory(device)
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, memory = model.forward_event(
                batch, normalized_time(batch, train_min, train_scale, device), memory,
            )
            probabilities.append(float(torch.sigmoid(logits).item()))
    return np.asarray(probabilities, dtype=float)


def _train_static_epoch(model, dataset, optimizer, device, positive_weight: float, train_min: int, train_scale: int, kind: str) -> None:
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, normalized_time(batch, train_min, train_scale, device)) if kind == "tgat" else model(batch)
        if kind == "fraudsage":
            loss = class_balanced_focal_loss(logits, batch.y.float(), positive_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, batch.y.float(), pos_weight=torch.tensor(positive_weight, device=device),
            )
        loss.backward()
        optimizer.step()


def _train_tgn_epoch(model: TemporalMemoryBaseline, dataset, optimizer, device, positive_weight: float, train_min: int, train_scale: int, truncation: int = 64) -> None:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.train()
    memory = model.initial_memory(device)
    losses: list[torch.Tensor] = []
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        logits, memory = model.forward_event(
            batch, normalized_time(batch, train_min, train_scale, device), memory,
        )
        losses.append(F.binary_cross_entropy_with_logits(
            logits, batch.y.float(), pos_weight=torch.tensor(positive_weight, device=device),
        ))
        if index % truncation == 0 or index == len(dataset):
            torch.stack(losses).mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            memory = memory.detach()
            losses = []


def _model(kind: str, config: dict[str, Any]):
    common = {"input_dim": config["input_dim"], "hidden_dim": config["hidden_dim"]}
    if kind == "tgat":
        return TGATBaseline(**common, dropout=config["dropout"])
    if kind == "tgn":
        return TemporalMemoryBaseline(**common)
    if kind == "fraudsage":
        return FraudSAGEBaseline(**common, dropout=config["dropout"])
    raise ValueError(kind)


def train_one(
    kind: str,
    seed: int,
    datasets: dict[str, list],
    metadata: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    model = _model(kind, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    positive_weight = manifest["split"]["train"]["n_negative"] / manifest["split"]["train"]["n_positive"]
    train_times = metadata.loc[metadata.split == "train", "timestamp"]
    train_min = int(train_times.min())
    train_scale = max(1, int(train_times.max()) - train_min)
    valid_labels = metadata.loc[metadata.split == "validation", "label"].to_numpy(int)
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, config["max_epochs"] + 1):
        if kind == "tgn":
            _train_tgn_epoch(model, datasets["train"], optimizer, device, positive_weight, train_min, train_scale)
            valid_probability = _predict_tgn(model, datasets["validation"], device, train_min, train_scale)
        else:
            _train_static_epoch(model, datasets["train"], optimizer, device, positive_weight, train_min, train_scale, kind)
            valid_probability = _predict_static(model, datasets["validation"], device, train_min, train_scale, kind)
        valid_ap = float(average_precision_score(valid_labels, valid_probability))
        print(f"model={kind} seed={seed} epoch={epoch} val_ap={valid_ap:.6f}", flush=True)
        if valid_ap > best_ap + 1e-6:
            best_ap = valid_ap
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device)
    valid_probability = (
        _predict_tgn(model, datasets["validation"], device, train_min, train_scale)
        if kind == "tgn" else
        _predict_static(model, datasets["validation"], device, train_min, train_scale, kind)
    )
    # This is the sole held-out test access for the selected baseline checkpoint.
    test_probability = (
        _predict_tgn(model, datasets["test"], device, train_min, train_scale)
        if kind == "tgn" else
        _predict_static(model, datasets["test"], device, train_min, train_scale, kind)
    )
    checkpoint_dir = output_root / "checkpoints" / kind
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"seed{seed}.pt"
    torch.save({
        "model_state_dict": best_state,
        "model_kind": kind,
        "method": MODEL_NAMES[kind],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auc_pr": best_ap,
        "config": copy.deepcopy(config),
        "dataset_sha256": manifest["graph_sha256"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "train_time_min": train_min,
        "train_time_scale": train_scale,
        "test_access_count": 1,
    }, checkpoint_path)
    prediction_dir = output_root / "raw_predictions" / kind
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = {}
    for split, probability in (("validation", valid_probability), ("test", test_probability)):
        frame = metadata.loc[metadata.split == split, ["event_id", "timestamp", "label"]].reset_index(drop=True)
        frame["p_mean"] = probability
        frame["seed"] = seed
        frame["model"] = MODEL_NAMES[kind]
        path = prediction_dir / f"seed{seed}_{split}.csv"
        frame.to_csv(path, index=False)
        prediction_rows[split] = {"path": str(path), "sha256": sha256_file(path), "events": len(frame)}
    test_labels = metadata.loc[metadata.split == "test", "label"].to_numpy(int)
    return {
        "model_key": kind,
        "method": MODEL_NAMES[kind],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auc_pr": best_ap,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_access_count": 1,
        "validation_predictions": prediction_rows["validation"],
        "test_predictions": prediction_rows["test"],
        "positive_prevalence": float(test_labels.mean()),
        **classification_metrics(test_labels, test_probability),
    }


def run(dataset_root: Path, output_root: Path, kinds: list[str], seeds: list[int], device_name: str) -> dict[str, Any]:
    hash_audit = verify_hash_contract(dataset_root, EXPECTED_PACKED_HASHES)
    if not hash_audit["all_match"]:
        raise RuntimeError("exact GoG-SCIMain-v1 hash contract failed; baseline training prohibited")
    _, manifest, datasets = load_packed(dataset_root)
    metadata = pd.read_parquet(dataset_root / "transactions.parquet").sort_values(
        ["timestamp", "event_id"], kind="stable",
    ).reset_index(drop=True)
    config = yaml.safe_load((ROOT / "configs/round4_sci_main_frozen.yaml").read_text())
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        train_one(kind, seed, datasets, metadata, manifest, config, output_root, device)
        for kind in kinds for seed in seeds
    ]
    pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, dict)} for row in rows]).to_csv(
        output_root / "comparable_model_metrics_per_seed.csv", index=False,
    )
    result = {
        "dataset": manifest["dataset_name"],
        "dataset_sha256": manifest["graph_sha256"],
        "hash_contract": hash_audit,
        "device": str(device),
        "seeds": seeds,
        "models": kinds,
        "runs": rows,
    }
    (output_root / "comparable_models_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/benchmark/gog_scimain_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/main_final_v2")
    parser.add_argument("--models", default="tgat,tgn,fraudsage")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(
        args.dataset_root,
        args.output_root,
        args.models.split(","),
        [int(value) for value in args.seeds.split(",")],
        args.device,
    )
    print(json.dumps({"models": result["models"], "seeds": result["seeds"], "device": result["device"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

