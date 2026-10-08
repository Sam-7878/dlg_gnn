#!/usr/bin/env python3
"""01_smoke_all_detectors.py: 8-detector smoke qualification on CUDA for Work Order A02."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pygod.detector import CoLA, OCGNN
from gog_fraud.models.pygod.gadnr import GADNR
from gog_fraud.models.pygod.shared_reconstruction import (
    SharedAnomalyDAE,
    SharedCONAD,
    SharedDLGBase,
    SharedDLGFull,
    SharedDOMINANT,
)

OUTPUT_DIR = REPO_ROOT / "evaluation" / "benchmark" / "v2" / "environment" / "journal_cuda"


def create_fixed_small_graph(num_nodes: int = 60, num_features: int = 16, seed: int = 42) -> Data:
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, num_features, dtype=torch.float32)
    
    # Create simple connected random edges
    sources = []
    targets = []
    for i in range(num_nodes):
        # Ring connections
        sources.append(i)
        targets.append((i + 1) % num_nodes)
        sources.append((i + 1) % num_nodes)
        targets.append(i)
    # Random extra edges
    extra_src = torch.randint(0, num_nodes, (120,))
    extra_dst = torch.randint(0, num_nodes, (120,))
    mask = extra_src != extra_dst
    sources.extend(extra_src[mask].tolist())
    targets.extend(extra_dst[mask].tolist())

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1).contiguous()

    y = torch.zeros(num_nodes, dtype=torch.long)
    y[torch.randperm(num_nodes)[:6]] = 1  # 10% anomalies

    data = Data(x=x.contiguous(), edge_index=edge_index, y=y)
    data.num_nodes = num_nodes
    return data


def run_smoke_detector(model_name: str, model_cls, data: Data, device_idx: int) -> dict:
    res = {
        "model": model_name,
        "import_ok": True,
        "init_ok": False,
        "train_1epoch_ok": False,
        "train_3epoch_ok": False,
        "score_ok": False,
        "no_nan_inf": False,
        "serialize_reload_ok": False,
        "device": f"cuda:{device_idx}",
        "error": None,
    }

    try:
        # Common kwargs
        gpu_id = device_idx
        if model_name in ["DOMINANT", "CONAD", "DLG-Base", "DLG-Aug"]:
            kwargs = {
                "epoch": 1,
                "gpu": gpu_id,
                "batch_size": 0,
                "verbose": 0,
                "message_backend": "sparse_fused",
                "reconstruction_backend": "exact_sparse",
                "score_chunk_size": 256,
            }
            if model_name == "DLG-Aug":
                kwargs["l1_epochs"] = 1
        elif model_name == "AnomalyDAE":
            kwargs = {
                "epoch": 1,
                "gpu": gpu_id,
                "batch_size": 0,
                "verbose": 0,
                "reconstruction_backend": "chunked_exact",
                "score_chunk_size": 256,
            }
        elif model_name in ["CoLA", "OCGNN", "GADNR"]:
            kwargs = {
                "epoch": 1,
                "gpu": gpu_id,
                "verbose": 0,
            }
            if model_name == "CoLA":
                kwargs["batch_size"] = 30
            elif model_name == "OCGNN":
                kwargs["batch_size"] = 30
            elif model_name == "GADNR":
                kwargs["batch_size"] = 30

        # 1. Initialize
        detector = model_cls(**kwargs)
        res["init_ok"] = True

        # 2. Train 1 epoch
        d1 = data.clone()
        detector.fit(d1)
        res["train_1epoch_ok"] = True

        # 3. Train 3 epochs
        detector.epoch = 3
        if hasattr(detector, "l1_epochs") and model_name == "DLG-Aug":
            detector.l1_epochs = 2
        d3 = data.clone()
        detector.fit(d3)
        res["train_3epoch_ok"] = True

        # 4. Score
        score = detector.decision_function(d3)
        res["score_ok"] = True
        
        # Check finite
        if not torch.isfinite(score).all() or score.numel() != data.num_nodes:
            raise FloatingPointError(f"Scores contain NaN/Inf or count mismatch: {score.shape}")
        res["no_nan_inf"] = True

        # 5. Serialize and reload
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if hasattr(detector, "model") and detector.model is not None:
                torch.save(detector.model.state_dict(), tmp_path)
                state = torch.load(tmp_path, weights_only=True)
                detector.model.load_state_dict(state)
            res["serialize_reload_ok"] = True
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        res["status"] = "PASS"
    except Exception as e:
        res["status"] = "FAIL"
        res["error"] = f"{type(e).__name__}: {str(e)}"

    return res


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available! Failing closed as required by Gate G0.")
        sys.exit(1)

    # Use device 0 or 1. If CUDA_VISIBLE_DEVICES or args, honor it.
    device_idx = int(os.environ.get("CUDA_DEVICE", 0))
    print(f"Running 8-detector smoke tests on cuda:{device_idx} ({torch.cuda.get_device_name(device_idx)})...")

    detectors = [
        ("DOMINANT", SharedDOMINANT),
        ("AnomalyDAE", SharedAnomalyDAE),
        ("CoLA", CoLA),
        ("CONAD", SharedCONAD),
        ("GADNR", GADNR),
        ("OCGNN", OCGNN),
        ("DLG-Base", SharedDLGBase),
        ("DLG-Aug", SharedDLGFull),
    ]

    data = create_fixed_small_graph()
    results = []
    all_passed = True

    for name, cls in detectors:
        t0 = time.time()
        print(f"Testing {name:12s} ... ", end="", flush=True)
        r = run_smoke_detector(name, cls, data, device_idx)
        elapsed = time.time() - t0
        r["elapsed_sec"] = round(elapsed, 2)
        results.append(r)
        if r["status"] == "PASS":
            print(f"PASS ({elapsed:.2f}s)")
        else:
            print(f"FAIL ({r['error']})")
            all_passed = False

    summary_path = OUTPUT_DIR / f"smoke_summary_cuda{device_idx}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"all_passed": all_passed, "device": device_idx, "results": results}, f, indent=2)

    print(f"\nSmoke test summary saved to: {summary_path}")
    if not all_passed:
        print("CRITICAL: One or more detectors failed smoke qualification!")
        sys.exit(1)
    else:
        print("ALL 8 DETECTORS PASSED SMOKE QUALIFICATION!")


if __name__ == "__main__":
    main()
