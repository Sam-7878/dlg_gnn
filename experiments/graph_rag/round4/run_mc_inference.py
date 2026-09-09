"""Run event-level MC-dropout inference on the chronological test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import (
    CHECKPOINT_DIR, DATASET_DIR, RAW_PREDICTION_DIR, RESULTS_DIR, ensure_dirs,
)
from experiments.round4.data import load_packed
from experiments.round4.model import CausalLocalGIN


def ece(labels, probabilities, bins=15):
    edges = np.linspace(0, 1, bins + 1); result = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1 else probabilities <= hi)
        if mask.any(): result += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(result)


def load_model(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]
    model = CausalLocalGIN(cfg["input_dim"], cfg["hidden_dim"], cfg["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"]); return model.to(device), checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--passes", default="1,5,10,20,30")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); ensure_dirs()
    _, manifest, datasets = load_packed(DATASET_DIR)
    metadata = pd.read_parquet(DATASET_DIR / "transactions.parquet")
    test_metadata = metadata.loc[metadata["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(datasets["test"], batch_size=128, shuffle=False)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    summary = []
    for seed in map(int, args.seeds.split(",")):
        checkpoint_path = CHECKPOINT_DIR / f"seed{seed}.pt"
        model, checkpoint = load_model(checkpoint_path, device)
        for passes in map(int, args.passes.split(",")):
            all_samples = []; all_mean = []; all_var = []; all_entropy = []; labels = []
            start = time.perf_counter()
            for batch in loader:
                batch = batch.to(device)
                if passes == 1:
                    # T=1 is the deterministic reference, not one random dropout draw.
                    model.eval()
                    with torch.no_grad():
                        mean = torch.sigmoid(model(batch))
                    samples = mean.unsqueeze(0)
                    variance = torch.zeros_like(mean)
                    eps = 1e-8
                    entropy = -(mean * torch.log(mean + eps) + (1 - mean) * torch.log(1 - mean + eps))
                else:
                    samples, mean, variance, entropy = model.forward_mc(batch, passes)
                all_samples.append(samples.cpu()); all_mean.append(mean.cpu())
                all_var.append(variance.cpu()); all_entropy.append(entropy.cpu())
                labels.extend(batch.y.cpu().long().tolist())
            elapsed = time.perf_counter() - start
            samples = torch.cat(all_samples, dim=1).numpy()
            mean = torch.cat(all_mean).numpy(); variance = torch.cat(all_var).numpy()
            entropy = torch.cat(all_entropy).numpy(); labels_np = np.asarray(labels)
            output = test_metadata[["event_id", "timestamp"]].copy()
            output["label"] = labels_np
            for index in range(passes): output[f"p_{index + 1}"] = samples[index]
            output["p_mean"] = mean; output["variance"] = variance; output["entropy"] = entropy
            output["seed"] = seed; output["T"] = passes
            out_path = RAW_PREDICTION_DIR / f"seed{seed}_T{passes}.csv"
            output.to_csv(out_path, index=False)
            predicted = (mean >= 0.5).astype(int)
            prediction_sha256 = sha256(out_path)
            summary.append({
                "seed": seed, "T": passes, "gnn_source": "real_checkpoint",
                "inference_mode": "deterministic" if passes == 1 else "mc_dropout",
                "split_type": "chronological_real", "timestamp_source": "recorded_transaction_timestamp",
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "n_test": len(labels_np), "n_positive": int(labels_np.sum()),
                "auc_pr": average_precision_score(labels_np, mean),
                "auc_roc": roc_auc_score(labels_np, mean), "f1": f1_score(labels_np, predicted),
                "brier": brier_score_loss(labels_np, mean), "ece": ece(labels_np, mean),
                "nll": log_loss(labels_np, mean, labels=[0, 1]),
                "latency_ms": elapsed * 1000, "events_per_second": len(labels_np) / elapsed,
                "raw_predictions": str(out_path.relative_to(ROOT)),
                "raw_predictions_sha256": prediction_sha256,
            })
            print(f"seed={seed} T={passes} AP={summary[-1]['auc_pr']:.6f}", flush=True)
    summary_path = RESULTS_DIR / "mc_sensitivity.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    prediction_manifest = {
        "dataset_sha256": manifest["graph_sha256"],
        "split_type": manifest["split_type"],
        "timestamp_source": manifest["timestamp_source"],
        "seed_count": len(set(row["seed"] for row in summary)),
        "passes": sorted(set(row["T"] for row in summary)),
        "entries": [{
            "seed": row["seed"], "T": row["T"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "raw_predictions": row["raw_predictions"],
            "raw_predictions_sha256": row["raw_predictions_sha256"],
        } for row in summary],
        "mc_sensitivity_sha256": sha256(summary_path),
    }
    (RESULTS_DIR / "raw_prediction_manifest.json").write_text(
        json.dumps(prediction_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
