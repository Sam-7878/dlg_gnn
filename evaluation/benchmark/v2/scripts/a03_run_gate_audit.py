#!/usr/bin/env python3
"""
a03_run_gate_audit.py — Gate A03-8 DLG-Base Gate Audit
Complies with Work Order A03 §10.

Runs:
  - Cora-Syn × seeds 42, 43, 44, 45, 46
  - Elliptic × seeds 42, 43, 44, 45, 46

Captures:
  - initial_alpha
  - final_alpha
  - sigmoid_initial_alpha
  - sigmoid_final_alpha
  - latent_local_norm
  - latent_global_norm
  - parameter_count

Outputs:
  - diagnostics/dlg_base_dominant/dlg_gate_10runs.csv
  - evaluation/benchmark/v2/paper_ready_a03/table_dlg_gate_audit.csv
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase
from scripts import benchmark_8x10_pipeline as legacy

OUTPUT_CSV_DIAG = REPO_ROOT / "evaluation/benchmark/v2/diagnostics/dlg_base_dominant/dlg_gate_10runs.csv"
OUTPUT_CSV_PAPER = REPO_ROOT / "evaluation/benchmark/v2/paper_ready_a03/table_dlg_gate_audit.csv"
OUTPUT_CSV_DIAG.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PAPER.parent.mkdir(parents=True, exist_ok=True)

DATA_ROOT = "/mnt/d/_Work/_data/DLG"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
gpu_id = 0 if DEVICE.startswith("cuda") else -1


def load_dataset(name: str) -> Data:
    legacy.DATA_ROOT = DATA_ROOT
    legacy.DATASET_SEED = 42
    if name == "Cora-Syn":
        return legacy.load_planetoid("Cora")
    elif name == "Elliptic":
        return legacy.load_elliptic()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    print("=== Gate A03-8: DLG-Base Gate Audit (10 Runs) ===")
    datasets = ["Cora-Syn", "Elliptic"]
    seeds = [42, 43, 44, 45, 46]
    records = []

    for d_name in datasets:
        print(f"\n--- Loading {d_name} ---")
        base_data = load_dataset(d_name)

        for s in seeds:
            print(f"Running {d_name} | DLG-Base | seed: {s}...", end=" ", flush=True)
            data = base_data.clone()
            torch.manual_seed(s)
            np.random.seed(s)

            det = SharedDLGBase(epoch=50, gpu=gpu_id, verbose=0, batch_size=0)
            
            # Initial alpha before training
            init_alpha = float(det.alpha)
            sig_init_alpha = float(torch.sigmoid(torch.tensor(init_alpha)).item())

            # Train detector
            t0 = time.time()
            det.fit(data)
            wall_time = time.time() - t0

            # Capture final alpha after training
            final_alpha = float(det.model.alpha.detach().cpu().item())
            sig_final_alpha = float(torch.sigmoid(det.model.alpha).detach().cpu().item())

            # Compute latent norms
            det.model.eval()
            with torch.no_grad():
                device = next(det.model.parameters()).device
                x_d = data.x.to(device)
                ei_d = data.edge_index.to(device)
                z_loc = det.model.local_encoder(x_d, ei_d)
                z_glob = det.model.global_encoder(z_loc, ei_d)
                norm_loc = float(torch.norm(z_loc, dim=-1).mean().cpu().item())
                norm_glob = float(torch.norm(z_glob, dim=-1).mean().cpu().item())

            param_count = sum(p.numel() for p in det.model.parameters() if p.requires_grad)

            print(f"DONE in {wall_time:.1f}s | alpha: {init_alpha:.3f} -> {final_alpha:.3f} (sig: {sig_init_alpha:.3f} -> {sig_final_alpha:.3f}) | local_norm={norm_loc:.3f}, global_norm={norm_glob:.3f}")

            records.append({
                "dataset": d_name,
                "model": "DLG-Base",
                "seed": s,
                "initial_alpha": round(init_alpha, 4),
                "final_alpha": round(final_alpha, 4),
                "sigmoid_initial_alpha": round(sig_init_alpha, 4),
                "sigmoid_final_alpha": round(sig_final_alpha, 4),
                "latent_local_norm": round(norm_loc, 4),
                "latent_global_norm": round(norm_glob, 4),
                "parameter_count": param_count,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV_DIAG, index=False)
    df.to_csv(OUTPUT_CSV_PAPER, index=False)
    print(f"\nSaved 10 gate audit records to:")
    print(f"  {OUTPUT_CSV_DIAG}")
    print(f"  {OUTPUT_CSV_PAPER}")

    # Summary
    print("\n--- Gate Audit Summary ---")
    summary = df.groupby("dataset")[["sigmoid_initial_alpha", "sigmoid_final_alpha", "latent_local_norm", "latent_global_norm"]].mean()
    print(summary)


if __name__ == "__main__":
    main()
