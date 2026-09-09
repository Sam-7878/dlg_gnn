"""Validation-only temperature scaling for the recovered exact v1 panel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.round4.data import load_packed
from experiments.round4.model import CausalLocalGIN
from experiments.round5.analysis import apply_temperature, classification_metrics, fit_temperature
from experiments.round7.provenance import EXPECTED_PACKED_HASHES, SEEDS, sha256_file, verify_hash_contract


def _load_model(path: Path, device: torch.device, expected_dataset_hash: str) -> CausalLocalGIN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("dataset_sha256") != expected_dataset_hash:
        raise RuntimeError(f"checkpoint dataset mismatch: {path}")
    config = checkpoint["model_config"]
    model = CausalLocalGIN(config["input_dim"], config["hidden_dim"], config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _predict_validation(model: CausalLocalGIN, dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            values.extend(torch.sigmoid(model(batch.to(device))).cpu().tolist())
    return np.asarray(values, dtype=float)


def _load_preserved_test(path: Path, expected: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path)
    identity = frame[["event_id", "timestamp", "label"]].reset_index(drop=True)
    if not identity.equals(expected.reset_index(drop=True)):
        raise RuntimeError(f"preserved test identity mismatch: {path}")
    return frame.p_mean.to_numpy(float)


def run(dataset_root: Path, checkpoints: Path, preserved_raw: Path, output_root: Path, device_name: str) -> dict[str, Any]:
    contract = verify_hash_contract(dataset_root, EXPECTED_PACKED_HASHES)
    if not contract["all_match"]:
        raise RuntimeError("exact v1 hash contract failed; calibration prohibited")
    _, manifest, datasets = load_packed(dataset_root)
    metadata = pd.read_parquet(dataset_root / "transactions.parquet")
    validation_meta = metadata.loc[metadata.split == "validation", ["event_id", "timestamp", "label"]].reset_index(drop=True)
    test_meta = metadata.loc[metadata.split == "test", ["event_id", "timestamp", "label"]].reset_index(drop=True)
    validation_labels = validation_meta.label.to_numpy(int)
    test_labels = test_meta.label.to_numpy(int)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    raw_root = output_root / "raw_predictions"
    proposed_root = raw_root / "proposed"
    temperature_root = raw_root / "temperature_scaled"
    proposed_root.mkdir(parents=True, exist_ok=True)
    temperature_root.mkdir(parents=True, exist_ok=True)
    rows = []
    deterministic_panel = []
    mc_panel = []
    calibrated_panel = []
    for seed in SEEDS:
        checkpoint_path = checkpoints / f"seed{seed}.pt"
        model = _load_model(checkpoint_path, device, manifest["graph_sha256"])
        validation_probability = _predict_validation(model, datasets["validation"], device)
        validation_frame = validation_meta.copy()
        validation_frame["p_mean"] = validation_probability
        validation_frame["seed"] = seed
        validation_path = proposed_root / f"seed{seed}_validation.csv"
        validation_frame.to_csv(validation_path, index=False)
        deterministic = _load_preserved_test(preserved_raw / f"seed{seed}_T1.csv", test_meta)
        mc = _load_preserved_test(preserved_raw / f"seed{seed}_T10.csv", test_meta)
        temperature = fit_temperature(validation_labels, validation_probability)
        calibrated = apply_temperature(deterministic, temperature)
        calibrated_frame = test_meta.copy()
        calibrated_frame["p_mean"] = calibrated
        calibrated_frame["seed"] = seed
        calibrated_frame["temperature"] = temperature
        calibrated_path = temperature_root / f"seed{seed}_test.csv"
        calibrated_frame.to_csv(calibrated_path, index=False)
        validation_nll_before = classification_metrics(validation_labels, validation_probability)["nll"]
        validation_nll_after = classification_metrics(
            validation_labels, apply_temperature(validation_probability, temperature),
        )["nll"]
        rows.append({
            "seed": seed,
            "fit_split": "validation",
            "application_split": "test",
            "temperature": temperature,
            "validation_nll_before": validation_nll_before,
            "validation_nll_after": validation_nll_after,
            "validation_predictions_sha256": sha256_file(validation_path),
            "test_predictions_sha256": sha256_file(calibrated_path),
            **classification_metrics(test_labels, calibrated),
        })
        deterministic_panel.append(deterministic)
        mc_panel.append(mc)
        calibrated_panel.append(calibrated)
    scaling = pd.DataFrame(rows)
    scaling.to_csv(output_root / "temperature_scaling_per_seed.csv", index=False)
    calibration_rows = []
    for method, panel, fit_scope in (
        ("CausalLocalGIN deterministic", deterministic_panel, "frozen checkpoint"),
        ("CausalLocalGIN MC Dropout T=10", mc_panel, "frozen checkpoint stochastic inference"),
        ("Temperature-Scaled CausalLocalGIN", calibrated_panel, "validation only"),
    ):
        for seed, probability in zip(SEEDS, panel):
            calibration_rows.append({
                "method": method,
                "seed": seed,
                "aggregation": "single model",
                "fit_scope": fit_scope,
                "positive_prevalence": float(test_labels.mean()),
                **classification_metrics(test_labels, probability),
            })
    for method, probability, fit_scope in (
        ("Deep Ensemble", np.mean(np.stack(deterministic_panel), axis=0), "five independent frozen checkpoints"),
        ("MC Dropout Ensemble T=10", np.mean(np.stack(mc_panel), axis=0), "five independent frozen checkpoints"),
        ("Temperature-Scaled Deep Ensemble", np.mean(np.stack(calibrated_panel), axis=0), "per-seed validation temperatures then mean"),
    ):
        calibration_rows.append({
            "method": method,
            "seed": "ensemble",
            "aggregation": "five-model mean probability",
            "fit_scope": fit_scope,
            "positive_prevalence": float(test_labels.mean()),
            **classification_metrics(test_labels, probability),
        })
    calibration = pd.DataFrame(calibration_rows)
    calibration["ap_lift_over_prevalence"] = calibration.auc_pr / calibration.positive_prevalence
    calibration.to_csv(output_root / "calibration_baselines.csv", index=False)
    result = {
        "dataset_sha256": manifest["graph_sha256"],
        "fit_scope": "validation only",
        "test_source": "preserved raw predictions; no checkpoint test reinference",
        "seed_count": len(SEEDS),
        "validation_events": len(validation_meta),
        "test_events": len(test_meta),
        "test_positive": int(test_labels.sum()),
        "temperature_scaling_sha256": sha256_file(output_root / "temperature_scaling_per_seed.csv"),
        "calibration_baselines_sha256": sha256_file(output_root / "calibration_baselines.csv"),
    }
    (output_root / "calibration_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/benchmark/gog_scimain_v1")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "archive/gog_scimain_v1_preserved_panel/checkpoints")
    parser.add_argument("--preserved-raw", type=Path, default=ROOT / "archive/gog_scimain_v1_preserved_panel/raw_predictions")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/main_final_v2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.checkpoints, args.preserved_raw, args.output_root, args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

