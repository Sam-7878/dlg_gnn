"""1-Epoch Compatibility Smoke Test for Defense Extension Datasets."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from gog_fraud.extensions.defense.defense_registry import DEFENSE_DATASETS, load_defense_dataset
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDLGFull, SharedDOMINANT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SMOKE_MODELS = {
    "DOMINANT": SharedDOMINANT,
    "DLG-Base": SharedDLGBase,
    "DLG-Aug": SharedDLGFull,
}


def run_smoke_cell(dataset_name: str, model_name: str, model_cls, data, seed: int = 42) -> dict:
    """Run a 1-epoch compatibility test for a model on a defense dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gpu = 0 if torch.cuda.is_available() else -1
    kwargs = {
        "epoch": 1,
        "gpu": gpu,
        "batch_size": 0,
        "verbose": 0,
        "message_backend": "sparse_fused",
        "reconstruction_backend": "exact_sparse",
        "gradient_checkpointing": False,
        "score_chunk_size": 8192,
    }
    if model_name == "DLG-Aug":
        kwargs["l1_epochs"] = 1

    detector = model_cls(**kwargs)

    started = time.perf_counter()
    detector.fit(data)
    fit_sec = time.perf_counter() - started

    # Score inference
    scores = detector.decision_score_
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)

    y = data.y.detach().cpu().numpy().reshape(-1).astype(np.int64)

    assert len(scores) == len(y), f"Score length ({len(scores)}) != y length ({len(y)})"
    assert np.isfinite(scores).all(), "Scores contain non-finite values"

    roc_auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else 0.5
    pr_auc = float(average_precision_score(y, scores)) if y.sum() > 0 else 0.0

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "epoch": 1,
        "fit_seconds": round(fit_sec, 3),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "status": "success",
    }


def main():
    parser = argparse.ArgumentParser(description="Run defense extension smoke test.")
    parser.add_argument("--output-json", type=str, default="outputs/sci_defense_extension/smoke/smoke_results.json")
    args = parser.parse_args()

    results = []
    success_count = 0
    total_count = len(DEFENSE_DATASETS) * len(SMOKE_MODELS)

    for d_name in DEFENSE_DATASETS:
        data = load_defense_dataset(d_name)
        for m_name, m_cls in SMOKE_MODELS.items():
            log.info("Running smoke test: dataset=%s, model=%s", d_name, m_name)
            try:
                res = run_smoke_cell(d_name, m_name, m_cls, data, seed=42)
                results.append(res)
                success_count += 1
                log.info("Smoke PASS: %s/%s in %.2fs (ROC=%.4f, PR=%.4f)",
                         d_name, m_name, res["fit_seconds"], res["roc_auc"], res["pr_auc"])
            except Exception as e:
                log.error("Smoke FAIL: %s/%s error=%s", d_name, m_name, e)
                results.append({
                    "dataset": d_name,
                    "model": m_name,
                    "seed": 42,
                    "status": "failed",
                    "error": str(e),
                })

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nSMOKE SUMMARY: {success_count}/{total_count} SUCCESS")
    assert success_count == total_count, f"Smoke test failed: {success_count}/{total_count} passed"
    print("ALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
