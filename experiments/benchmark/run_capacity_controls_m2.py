#!/usr/bin/env python3
"""
run_capacity_controls_m2.py

Executes M2 capacity and training-budget controls:
1. DLG-Aug-Zero: Zero-padded local feature augmentation [X || 0_64]
2. DLG-Base-70: DLG-Base with 70 global epochs
3. DLG-Aug-Permuted: Detached L1 embedding randomly permuted across nodes prior to concatenation

Matrix: 3 datasets (Elliptic, DGraphFin, LANL-RedTeam) x 3 controls x 5 seeds (42..46).
Note on Reddit-Syn: Option B is selected (excluded due to wall-time constraints: 114.9M edges, ~83m/run).
Status is audited as NOT_RUN.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure gog_fraud and scripts are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
DLG_SRC = REPO_ROOT / "dlg_gnn" / "src"
DLG_ROOT = REPO_ROOT / "dlg_gnn"
if str(DLG_SRC) not in sys.path:
    sys.path.insert(0, str(DLG_SRC))
if str(DLG_ROOT) not in sys.path:
    sys.path.insert(0, str(DLG_ROOT))

from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDLGFull
from gog_fraud.pipelines.run_sci_round1_ablation import _eligible_labels
from gog_fraud.pipelines.run_sci_round1_benchmark import _legacy_registries, _validation_test_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("capacity_controls_m2")

OUTPUT_DIR = DLG_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "controls"
RAW_DIR = OUTPUT_DIR / "raw"
M1_RAW_DIR = DLG_ROOT / "outputs" / "benchmark" / "manuscript_m1" / "capacity_controls" / "raw"


class SharedDLGAugZero(SharedDLGFull):
    """
    Capacity-control model: identical architecture and global input dimension (F + 64)
    to DLG-Aug, but local embedding features are replaced with zero vectors.
    """
    def process_graph(self, data):
        if hasattr(data, "_dlg_full_augmented") and data._dlg_full_augmented:
            data._reconstruction_backend = self.reconstruction_backend
            return
        self._orig_dim = data.x.size(1)
        data.dlg_original_x = data.x.clone()
        zeros = torch.zeros((data.x.size(0), self.l1_hid_dim), dtype=data.x.dtype, device=data.x.device)
        data.x = torch.cat([data.x, zeros], dim=-1)
        data._dlg_full_augmented = True
        data._reconstruction_backend = self.reconstruction_backend
        data._message_backend = self.message_backend


class SharedDLGAugPermuted(SharedDLGFull):
    """
    Permuted-control model: local pretraining is run normally for 20 epochs,
    but the extracted L1 embedding is randomly permuted across node indices
    prior to concatenation [X || H_permuted].
    """
    def __init__(self, perm_seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.perm_seed = perm_seed

    def process_graph(self, data):
        if hasattr(data, "_dlg_full_augmented") and data._dlg_full_augmented:
            data._reconstruction_backend = self.reconstruction_backend
            return
        self._orig_dim = data.x.size(1)
        data.dlg_original_x = data.x.clone()
        
        # Run standard L1 pretraining
        l1_embs = self._pretrain_level1(data)
        
        # Permute nodes deterministically using perm_seed
        gen = torch.Generator().manual_seed(self.perm_seed)
        perm = torch.randperm(data.x.size(0), generator=gen)
        l1_embs_perm = l1_embs[perm]
        
        # Augment features
        data.x = torch.cat([data.x, l1_embs_perm.to(data.x.device)], dim=-1)
        data._dlg_full_augmented = True
        data._reconstruction_backend = self.reconstruction_backend
        data._message_backend = self.message_backend


def stratified_split_indices(y: np.ndarray, seed: int, val_ratio: float = 0.2, test_ratio: float = 0.2):
    """Deterministic stratified node transductive train/val/test split matching run_defense_multiseed_real.py."""
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = np.arange(n)

    pos_idx = indices[y == 1]
    neg_idx = indices[y == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(len(pos_idx) * val_ratio))
    n_test_pos = max(1, int(len(pos_idx) * test_ratio))

    n_val_neg = max(1, int(len(neg_idx) * val_ratio))
    n_test_neg = max(1, int(len(neg_idx) * test_ratio))

    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    test_idx = np.concatenate([pos_idx[n_val_pos:n_val_pos + n_test_pos], neg_idx[n_val_neg:n_val_neg + n_test_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos + n_test_pos:], neg_idx[n_val_neg + n_test_neg:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def load_dataset(name: str, data_root: Path):
    if name == "DGraphFin":
        return load_dgraphfin_aligned(data_root / "DGraphFin/dgraphfin.npz")
    elif name == "LANL-RedTeam":
        lanl_pt = DLG_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real" / "graphs" / "lanl_graph.pt"
        if not lanl_pt.exists():
            raise FileNotFoundError(f"Canonical LANL graph not found: {lanl_pt}")
        return torch.load(lanl_pt, weights_only=False)
    else:
        registry, _ = _legacy_registries(str(data_root), 42)
        if name in registry:
            return registry[name]()
        raise KeyError(f"Unknown dataset: {name}")


def get_split(data, dataset_name: str, seed: int):
    y_all = data.y.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if dataset_name == "DGraphFin":
        val_nodes = np.flatnonzero(data.val_mask.detach().cpu().numpy())
        test_nodes = np.flatnonzero(data.test_mask.detach().cpu().numpy())
        return val_nodes, test_nodes, y_all[val_nodes], y_all[test_nodes]
    elif dataset_name == "LANL-RedTeam":
        train_idx, val_idx, test_idx = stratified_split_indices(y_all, seed, 0.2, 0.2)
        return val_idx, test_idx, y_all[val_idx], y_all[test_idx]
    else:
        eligible, y = _eligible_labels(data)
        val_local, test_local = _validation_test_indices(y, seed, 0.2, 0.2)
        return eligible[val_local], eligible[test_local], y[val_local], y[test_local]


def run_single_cell(dataset_name: str, model_name: str, seed: int, data_root: Path, gpu: int = 0):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RAW_DIR / f"{dataset_name}__{model_name}__seed{seed}.json"

    # For Elliptic and DGraphFin on DLG-Aug-Zero and DLG-Base-70, copy from M1 if present
    if not out_json.exists() and dataset_name in ["Elliptic", "DGraphFin"] and model_name in ["DLG-Aug-Zero", "DLG-Base-70"]:
        m1_file = M1_RAW_DIR / f"{dataset_name}__{model_name}__seed{seed}.json"
        if m1_file.exists():
            log.info(f"[COPY M1] Reusing validated M1 run for {dataset_name}/{model_name}/seed={seed}")
            shutil.copy(m1_file, out_json)

    if out_json.exists():
        try:
            cached = json.loads(out_json.read_text(encoding="utf-8"))
            if cached.get("status") == "success":
                log.info(f"[CACHED] {dataset_name}/{model_name}/seed={seed}")
                return cached
        except Exception:
            pass

    log.info(f"[RUNNING] {dataset_name}/{model_name}/seed={seed}")
    seed_everything(seed)
    data = load_dataset(dataset_name, data_root).clone()

    val_idx, test_idx, y_val, y_test = get_split(data, dataset_name, seed)

    common = {
        "gpu": gpu if torch.cuda.is_available() and gpu >= 0 else -1,
        "batch_size": 0,
        "verbose": 0,
        "message_backend": "sparse_fused",
        "reconstruction_backend": "exact_sparse",
        "gradient_checkpointing": False,
        "score_chunk_size": 8192,
    }

    if model_name == "DLG-Aug-Zero":
        detector = SharedDLGAugZero(epoch=50, l1_epochs=0, **common)
    elif model_name == "DLG-Base-70":
        detector = SharedDLGBase(epoch=70, **common)
    elif model_name == "DLG-Aug-Permuted":
        detector = SharedDLGAugPermuted(epoch=50, l1_epochs=20, perm_seed=seed, **common)
    else:
        raise ValueError(f"Unknown control model: {model_name}")

    started = time.perf_counter()
    detector.fit(data)
    train_sec = time.perf_counter() - started

    started = time.perf_counter()
    score = detector.decision_function(data).cpu().numpy().reshape(-1)
    infer_sec = time.perf_counter() - started

    score_val = score[val_idx]
    score_test = score[test_idx]

    roc = float(roc_auc_score(y_test, score_test))
    pr = float(average_precision_score(y_test, score_test))
    threshold_eval = evaluate_threshold_protocol(y_val, score_val, y_test, score_test)
    val_f1 = float(threshold_eval.validation_f1)
    oracle_f1 = float(threshold_eval.oracle_best_f1)

    record = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "status": "success",
        "roc_auc": roc,
        "pr_auc": pr,
        "validation_f1": val_f1,
        "oracle_best_f1": oracle_f1,
        "train_sec": train_sec,
        "infer_sec": infer_sec,
        "runtime": train_sec + infer_sec,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    log.info(f"[DONE] {dataset_name}/{model_name}/seed={seed} ROC={roc:.4f} PR={pr:.4f} F1={val_f1:.4f}")
    return record


def generate_controls_latex_table(df_summary: pd.DataFrame) -> str:
    """
    Generates LaTeX table comparing DLG-Base (50 ep), DLG-Aug (20+50 ep),
    DLG-Aug-Zero, DLG-Base-70, and DLG-Aug-Permuted.
    """
    lines = [
        r"% Auto-generated M2 Capacity and Training-Budget Controls Table",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation and Capacity Controls on Discrepancy Datasets (Elliptic, DGraphFin, and LANL-RedTeam). Results are reported as mean $\pm$ std across 5 seeds.}",
        r"\label{tab:capacity_controls}",
        r"\footnotesize",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Variant / Control} & \textbf{Epochs} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{Val F1} \\",
        r"\midrule"
    ]
    
    # We include authoritative benchmarks for reference
    authoritative = {
        "Elliptic": {
            "DLG-Base": {"ep": "50", "roc": "0.6865 ± 0.0039", "pr": "0.2917 ± 0.0084", "f1": "0.2941 ± 0.0064"},
            "DLG-Aug": {"ep": "70", "roc": "0.6863 ± 0.0082", "pr": "0.2798 ± 0.0076", "f1": "0.2906 ± 0.0068"}
        },
        "DGraphFin": {
            "DLG-Base": {"ep": "50", "roc": "0.6558 ± 0.0032", "pr": "0.0381 ± 0.0016", "f1": "0.0652 ± 0.0022"},
            "DLG-Aug": {"ep": "70", "roc": "0.6385 ± 0.0030", "pr": "0.0267 ± 0.0012", "f1": "0.0531 ± 0.0020"}
        },
        "LANL-RedTeam": {
            "DLG-Base": {"ep": "50", "roc": "0.7923 ± 0.0085", "pr": "0.1367 ± 0.0078", "f1": "0.2112 ± 0.0094"},
            "DLG-Aug": {"ep": "70", "roc": "0.7188 ± 0.0121", "pr": "0.1114 ± 0.0069", "f1": "0.1683 ± 0.0085"}
        }
    }
    
    datasets = ["Elliptic", "DGraphFin", "LANL-RedTeam"]
    controls = ["DLG-Aug-Zero", "DLG-Aug-Permuted", "DLG-Base-70"]
    
    for ds in datasets:
        # Reference authoritative models
        if ds in authoritative:
            b = authoritative[ds]["DLG-Base"]
            lines.append(f"{ds} & DLG-Base (Authoritative) & {b['ep']} & {b['roc']} & {b['pr']} & {b['f1']} \\\\")
            a = authoritative[ds]["DLG-Aug"]
            lines.append(f" & DLG-Aug (Authoritative) & {a['ep']} & {a['roc']} & {a['pr']} & {a['f1']} \\\\")
        
        for c in controls:
            sub = df_summary[(df_summary["dataset"] == ds) & (df_summary["model"] == c)]
            if len(sub) > 0:
                row = sub.iloc[0]
                ep = "70" if "70" in c else "50" if "Zero" in c else "70"
                roc_str = f"{row['roc_mean']:.4f} \\pm {row['roc_std']:.4f}"
                pr_str = f"{row['pr_mean']:.4f} \\pm {row['pr_std']:.4f}"
                f1_str = f"{row['f1_mean']:.4f} \\pm {row['f1_std']:.4f}"
                lines.append(f" & {c} & {ep} & {roc_str} & {pr_str} & {f1_str} \\\\")
        
        if ds != datasets[-1]:
            lines.append(r"\midrule")
        else:
            lines.append(r"\bottomrule")
            
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="Elliptic,DGraphFin,LANL-RedTeam",
                        help="Comma-separated list of datasets to run")
    parser.add_argument("--models", type=str, default="DLG-Aug-Zero,DLG-Base-70,DLG-Aug-Permuted")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    data_root = Path("/mnt/d/_Work/_data/DLG")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    records = []
    for ds in datasets:
        for m in models:
            for s in seeds:
                rec = run_single_cell(ds, m, s, data_root, gpu=args.gpu)
                records.append(rec)

    df = pd.DataFrame(records)
    summary_raw_csv = OUTPUT_DIR / "capacity_controls_summary_m2_raw.csv"
    df.to_csv(summary_raw_csv, index=False)
    log.info(f"Wrote raw summary of {len(df)} runs to {summary_raw_csv}")

    # Aggregated table
    agg = df.groupby(["dataset", "model"]).agg(
        roc_mean=("roc_auc", "mean"),
        roc_std=("roc_auc", "std"),
        pr_mean=("pr_auc", "mean"),
        pr_std=("pr_auc", "std"),
        f1_mean=("validation_f1", "mean"),
        f1_std=("validation_f1", "std"),
        count=("seed", "count")
    ).reset_index()

    summary_csv = OUTPUT_DIR / "capacity_controls_summary_m2.csv"
    agg.to_csv(summary_csv, index=False)
    log.info(f"Wrote aggregated summary to {summary_csv}")

    # Generate LaTeX table
    latex_table = generate_controls_latex_table(agg)
    tex_path = OUTPUT_DIR / "table_capacity_controls_m2.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    log.info(f"Wrote LaTeX table to {tex_path}")

    # Update paper generated directory
    paper_tex = DLG_ROOT / "docs" / "papers" / "_42_Benchmark" / "generated" / "table_capacity_controls.tex"
    paper_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_tex, "w", encoding="utf-8") as f:
        f.write(latex_table)
    log.info(f"Updated manuscript table at {paper_tex}")


if __name__ == "__main__":
    main()
