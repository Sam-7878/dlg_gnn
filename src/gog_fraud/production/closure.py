"""Frozen-split production GNN training and model-agnostic cascade utilities."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig
from gog_fraud.models.level2.model import Level2Model, Level2ModelConfig
from gog_fraud.pipelines.fusion import FusionInput, WeightedSumConfig, WeightedSumFusion
from gog_fraud.pipelines.run_round4_experiments import SciV2Records, _normalize
from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics, select_f1_threshold, sha256_file


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path: Path) -> str:
    return sha256_file(path)


def _truncate_graph(raw: dict[str, Any], label: int, graph_id: int, max_edges: int, max_nodes: int) -> Data:
    x = raw["x"].float()
    edge_index = raw["edge_index"].long()
    if edge_index.shape[1] > max_edges:
        edge_index = edge_index[:, -max_edges:]
    nodes = torch.unique(edge_index.reshape(-1)) if edge_index.numel() else torch.tensor([0], dtype=torch.long)
    if 0 not in nodes:
        nodes = torch.cat((torch.tensor([0], dtype=torch.long), nodes))
    if nodes.numel() > max_nodes:
        nodes = nodes[-max_nodes:]
    keep = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes) if edge_index.numel() else torch.zeros(0, dtype=torch.bool)
    edge_index = edge_index[:, keep]
    mapping = torch.full((x.shape[0],), -1, dtype=torch.long)
    mapping[nodes] = torch.arange(nodes.numel())
    edge_index = mapping[edge_index] if edge_index.numel() else torch.empty((2, 0), dtype=torch.long)
    return Data(
        x=x[nodes],
        edge_index=edge_index,
        y=torch.tensor([float(label)]),
        graph_id=torch.tensor([graph_id], dtype=torch.long),
    )


def build_graph_cache(dataset_root: Path, cache_path: Path, max_edges: int, max_nodes: int) -> dict[str, Any]:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    records: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for chain in ("ethereum", "bsc", "polygon"):
        manifest = json.loads((dataset_root / f"manifests/{chain}.json").read_text(encoding="utf-8"))
        records.update({row["sample_id"]: row for row in manifest["records"]})
        split = json.loads((dataset_root / f"splits/{chain}_holdout_v2.json").read_text(encoding="utf-8"))
        for name in split_ids:
            split_ids[name].extend(split["groups"][name]["sample_ids"])
    graphs: dict[str, list[Data]] = {}
    metadata: dict[str, list[dict[str, Any]]] = {}
    graph_id = 0
    for split_name, ids in split_ids.items():
        graphs[split_name], metadata[split_name] = [], []
        for sample_id in ids:
            record = records[sample_id]
            raw = torch.load(record["graph_path"], map_location="cpu", weights_only=False)
            graphs[split_name].append(_truncate_graph(raw, int(record["label"]), graph_id, max_edges, max_nodes))
            metadata[split_name].append(
                {
                    "sample_id": sample_id,
                    "chain": record["chain_id"],
                    "contract_id": record["contract_id"],
                    "event_start": int(record["event_start"]),
                    "event_end": int(record["event_end"]),
                    "label": int(record["label"]),
                    "sorted_path": record["sorted_path"],
                }
            )
            graph_id += 1
    payload = {"graphs": graphs, "metadata": metadata, "max_edges": max_edges, "max_nodes": max_nodes}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def train_level1(
    graphs: list[Data],
    validation: list[Data],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[Level1Model, dict[str, Any]]:
    seed_all(seed)
    model_config = Level1ModelConfig(
        in_dim=int(config["in_dim"]), hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]), dropout=float(config["dropout"]),
        readout=str(config["readout"]), struct_dim=0, out_dim=1,
    )
    model = Level1Model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    labels = np.asarray([int(graph.y.item()) for graph in graphs])
    positives = max(1, int(labels.sum()))
    positive_weight = torch.tensor([(len(labels) - positives) / positives], device=device)
    train_loader = DataLoader(graphs, batch_size=int(config["batch_size"]), shuffle=True)
    valid_loader = DataLoader(validation, batch_size=int(config["batch_size"]), shuffle=False)
    best_state, best_f1, history = None, -1.0, []
    for epoch in range(int(config["epochs"])):
        model.train(); losses = []
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = F.binary_cross_entropy_with_logits(output.logits.view(-1), batch.y.view(-1).float(), pos_weight=positive_weight)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        valid_score, valid_label, _ = infer_level1(model, valid_loader, device, mc=1)
        threshold = select_f1_threshold(valid_label, valid_score)
        value = f1_score(valid_label, valid_score >= threshold, zero_division=0)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_f1": float(value)})
        if value > best_f1:
            best_f1 = float(value)
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"config": asdict(model_config), "history": history, "best_validation_f1": best_f1}


@torch.no_grad()
def infer_level1(model: Level1Model, loader: DataLoader, device: torch.device, mc: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples, labels, embeddings = [], [], []
    for batch in loader:
        batch = batch.to(device)
        batch_scores, batch_embeddings = [], []
        model.train(mc > 1)
        for _ in range(mc):
            output = model(batch)
            batch_scores.append(output.score.view(-1).detach().cpu().numpy())
            batch_embeddings.append(output.embedding.detach().cpu().numpy())
        samples.append(np.mean(batch_scores, axis=0))
        embeddings.append(np.mean(batch_embeddings, axis=0))
        labels.append(batch.y.view(-1).detach().cpu().numpy())
    model.eval()
    return np.concatenate(samples), np.concatenate(labels).astype(int), np.concatenate(embeddings)


def relation_edges(reference: np.ndarray, target: np.ndarray | None, k: int) -> torch.Tensor:
    n_reference = len(reference)
    finder = NearestNeighbors(n_neighbors=min(k + 1, n_reference)).fit(reference)
    _, local = finder.kneighbors(reference)
    source: list[int] = []; destination: list[int] = []
    for index, neighbors in enumerate(local):
        for neighbor in neighbors:
            if index != int(neighbor):
                source.extend((index, int(neighbor))); destination.extend((int(neighbor), index))
    if target is not None and len(target):
        _, external = finder.kneighbors(target, n_neighbors=min(k, n_reference))
        for offset, neighbors in enumerate(external):
            node = n_reference + offset
            for neighbor in neighbors:
                source.extend((node, int(neighbor))); destination.extend((int(neighbor), node))
    return torch.tensor([source, destination], dtype=torch.long)


def relation_data(
    train_embedding: np.ndarray,
    train_score: np.ndarray,
    train_label: np.ndarray,
    target_embedding: np.ndarray | None,
    target_score: np.ndarray | None,
    target_label: np.ndarray | None,
    k: int,
) -> Data:
    target_feature = None if target_embedding is None else np.concatenate((target_embedding, target_score[:, None]), axis=1)
    train_feature = np.concatenate((train_embedding, train_score[:, None]), axis=1)
    features = train_feature if target_feature is None else np.concatenate((train_feature, target_feature), axis=0)
    labels = train_label if target_label is None else np.concatenate((train_label, target_label), axis=0)
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=relation_edges(train_embedding, target_embedding, k),
        level1_label=torch.tensor(labels, dtype=torch.float32),
        y=torch.tensor([float(labels.mean() >= 0.5)]),
    )


def train_level2(data: Data, in_dim: int, config: dict[str, Any], seed: int, device: torch.device) -> tuple[Level2Model, dict[str, Any]]:
    seed_all(seed)
    model_config = Level2ModelConfig(
        in_dim=in_dim, hidden_dim=int(config["hidden_dim"]), num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]), dropout=float(config["dropout"]), edge_dim=0,
        readout="meanmax", out_dim=1,
    )
    model = Level2Model(model_config).to(device)
    graph = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    labels = graph.level1_label.view(-1)
    positives = max(1, int(labels.sum().item()))
    positive_weight = torch.tensor([(len(labels) - positives) / positives], device=device)
    losses = []
    for _ in range(int(config["epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(graph)
        loss = F.binary_cross_entropy_with_logits(output.logits.view(-1), labels, pos_weight=positive_weight)
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, {"config": asdict(model_config), "loss": losses}


@torch.no_grad()
def infer_level2(model: Level2Model, data: Data, offset: int, device: torch.device) -> np.ndarray:
    model.eval()
    return model(data.to(device)).score.view(-1)[offset:].detach().cpu().numpy()


def fuse_scores(level1: np.ndarray, level2: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    fusion = WeightedSumFusion(WeightedSumConfig(level1_weight=float(config["level1_weight"]), level2_weight=float(config["level2_weight"])))
    result = fusion(FusionInput(torch.tensor(level1), torch.tensor(level2)))
    return result.score.detach().cpu().numpy()


def train_tabular(dataset_root: Path, seed: int) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    dataset = SciV2Records(dataset_root)
    train_ids, valid_ids, test_ids = (dataset.ids("pooled", split) for split in ("train", "validation", "test"))
    train_x, train_y = dataset.arrays(train_ids); valid_x, valid_y = dataset.arrays(valid_ids); test_x, test_y = dataset.arrays(test_ids)
    train_x, valid_x, test_x = _normalize(train_x, valid_x, test_x)
    positive_weight = float((len(train_y) - train_y.sum()) / max(1, train_y.sum()))
    models = {
        "XGBoostFastTriage": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=positive_weight, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist", eval_metric="logloss"),
        "LightGBMFastTriage": lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=positive_weight, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1),
    }
    output = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        output[name] = (model.predict_proba(valid_x)[:, 1], model.predict_proba(test_x)[:, 1], valid_y)
    models["normalization"] = {"mean": dataset.arrays(train_ids)[0].mean(0), "scale": np.where(dataset.arrays(train_ids)[0].std(0) == 0, 1, dataset.arrays(train_ids)[0].std(0))}
    return models, output


def ambiguity_cutoff(scores: np.ndarray, threshold: float, budget: float) -> float:
    return float(np.quantile(np.abs(scores - threshold), budget))


def save_seed_bundle(path: Path, level1: Level1Model, level2: Level2Model, metadata: dict[str, Any], tabular: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": level1.state_dict(), "config": metadata["level1"]["config"]}, path / "level1.pt")
    torch.save({"state_dict": level2.state_dict(), "config": metadata["level2"]["config"]}, path / "level2.pt")
    joblib.dump(tabular, path / "tabular.joblib")
    atomic_json(path / "metadata.json", metadata)


def load_seed_bundle(path: Path, device: torch.device) -> tuple[Level1Model, Level2Model, dict[str, Any], dict[str, Any]]:
    l1_payload = torch.load(path / "level1.pt", map_location=device, weights_only=False)
    l2_payload = torch.load(path / "level2.pt", map_location=device, weights_only=False)
    level1 = Level1Model(Level1ModelConfig(**l1_payload["config"])).to(device); level1.load_state_dict(l1_payload["state_dict"]); level1.eval()
    level2 = Level2Model(Level2ModelConfig(**l2_payload["config"])).to(device); level2.load_state_dict(l2_payload["state_dict"]); level2.eval()
    return level1, level2, joblib.load(path / "tabular.joblib"), json.loads((path / "metadata.json").read_text(encoding="utf-8"))
