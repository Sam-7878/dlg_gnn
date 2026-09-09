"""
experiments/run_multiseed.py

Multi-seed main model evaluation.

Runs the full pipeline (GraphRAG + MC-nGNN + Uncertainty Fusion)
on N seeds and reports mean ± std for all metrics.

Usage:
    python experiments/run_multiseed.py --config configs/base.yaml --output results/multiseed
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_single_seed(cfg: Dict[str, Any], seed: int) -> Dict[str, float]:
    """
    Run one full experiment seed and return metric dict.

    This function integrates:
        1. GraphRAG context generation (SyntheticContextGenerator + GraphRAGRetriever)
        2. RiskEncoder (φ(r_t) → p_t^R)
        3. MC-nGNN inference (→ p̄_t, σ²_t)
        4. UncertaintyFusion (→ R_t)
        5. Evaluation (AUC-PR, AUC-ROC, F1, Recall@K, ECE)

    NOTE: For datasets where real GNN training is not available,
    this function runs the GraphRAG + risk encoding pipeline on synthetic data
    and reports retrieval-level metrics. Full GNN integration requires
    a trained Level1/Level2 checkpoint.
    """
    _set_seed(seed)
    log.info(f"[Seed {seed}] Starting experiment run ...")

    # ── Stage 1: GraphRAG retrieval pipeline ──────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from graphrag.risk_encoder import RiskEncoder
    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage
    )
    from fusion.uncertainty_fusion import UncertaintyFusion

    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    # Generate synthetic data for pipeline validation
    n_samples = cfg.get("experiment", {}).get("n_synthetic_samples", 1000)
    fraud_rate = 0.10

    rng = np.random.RandomState(seed)
    labels_np = (rng.rand(n_samples) < fraud_rate).astype(int)

    # Scenario assignment (no label leakage)
    scenarios = assign_scenarios_no_leakage(labels_np, seed=seed)

    # Generate context texts
    gen = SyntheticContextGenerator(seed=seed)
    event_ids = [f"tx_{i:06d}" for i in range(n_samples)]
    records = gen.generate_contexts(scenario_types=scenarios, event_ids=event_ids)

    # ── Stage 2: GraphRAG retrieval ───────────────────────────────────────
    kb = LocalKnowledgeBase()
    retriever_cfg = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=cfg.get("graphrag", {}).get("graph_hops", 1),
        similarity_threshold=cfg.get("graphrag", {}).get("similarity_threshold", 0.0),
    )
    retriever = GraphRAGRetriever(kb, retriever_cfg)
    extractor = RiskExtractor()

    risk_dicts = []
    for rec in records:
        evidence = retriever.retrieve(rec["context_text"])
        risk_dict = extractor.extract(
            evidence,
            event_id=rec["event_id"],
            pre_transaction_gap_sec=rec.get("pre_transaction_gap_sec", 300),
        )
        risk_dicts.append(risk_dict)

    # ── Stage 3: Risk Encoder ─────────────────────────────────────────────
    device = torch.device("cpu")
    encoder = RiskEncoder.from_config(cfg)
    encoder.eval()
    with torch.no_grad():
        _, p_risk = encoder.encode_risk_dict_batch(risk_dicts, device=device)
    p_risk_np = p_risk.numpy()

    # ── Stage 4: Simulate GNN output (placeholder until real GNN is available) ──
    # In the full pipeline, p_gnn and u_mc come from MCStreamingInference.
    # Here we simulate realistic GNN behavior based on labels + noise.
    noise_scale = 0.15
    p_gnn_np = np.clip(
        labels_np.astype(float) * 0.7 + (1 - labels_np) * 0.2 +
        rng.normal(0, noise_scale, n_samples),
        0.0, 1.0,
    )
    u_mc_np = np.clip(
        0.08 + noise_scale * rng.rand(n_samples) * (1 - np.abs(p_gnn_np - 0.5)),
        0.001, 0.5,
    )

    # ── Stage 5: Uncertainty Fusion ───────────────────────────────────────
    fusion = UncertaintyFusion.from_config(cfg)
    p_gnn_t = torch.from_numpy(p_gnn_np.astype(np.float32))
    u_mc_t  = torch.from_numpy(u_mc_np.astype(np.float32))
    p_risk_t = torch.from_numpy(p_risk_np.astype(np.float32))
    final_prob, _, _ = fusion.fuse(p_gnn_t, u_mc_t, p_risk_t)
    scores_np = final_prob.numpy()

    # ── Stage 6: Evaluation ───────────────────────────────────────────────
    try:
        auc_roc = float(roc_auc_score(labels_np, scores_np))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(labels_np, scores_np))
    except Exception:
        auc_pr = float("nan")

    # GNN-only metrics (for paired bootstrap CI)
    try:
        auc_pr_gnn_only = float(average_precision_score(labels_np, p_gnn_np))
    except Exception:
        auc_pr_gnn_only = float("nan")
    try:
        auc_roc_gnn_only = float(roc_auc_score(labels_np, p_gnn_np))
    except Exception:
        auc_roc_gnn_only = float("nan")

    # F1 at threshold 0.5
    preds = (scores_np >= 0.5).astype(int)
    f1 = float(f1_score(labels_np, preds, zero_division=0))

    # Recall@K (K = num_pos)
    k = max(int(labels_np.sum()), 1)
    topk_idx = np.argsort(scores_np)[::-1][:k]
    recall_at_k = float(labels_np[topk_idx].sum() / max(labels_np.sum(), 1))

    # Calibration metrics (ECE, Brier, NLL)
    def _ece(y, p, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (p >= bins[i]) & (p < bins[i + 1])
            if mask.sum() > 0:
                ece += (mask.sum() / len(y)) * abs(y[mask].mean() - p[mask].mean())
        return float(ece)

    eps = 1e-7
    ece_val = _ece(labels_np.astype(float), scores_np)
    brier = float(np.mean((scores_np - labels_np.astype(float)) ** 2))
    nll = float(-np.mean(
        labels_np * np.log(np.clip(scores_np, eps, 1 - eps))
        + (1 - labels_np) * np.log(np.clip(1 - scores_np, eps, 1 - eps))
    ))

    # Round 2: save raw per-event predictions for calibration + lineage tracing
    try:
        import pandas as pd
        from pathlib import Path as _Path
        raw_dir = _Path("results/raw_predictions")
        raw_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "event_id": event_ids,
            "label": labels_np,
            "score": scores_np,         # Uncertainty Fusion output
            "p_gnn": p_gnn_np,          # GNN simulation
            "u_mc": u_mc_np,            # MC uncertainty
            "p_risk": p_risk_np,        # RiskEncoder output
        }).to_csv(raw_dir / f"multiseed_seed{seed}_preds.csv", index=False)
    except Exception as save_err:
        log.warning(f"[Seed {seed}] Could not save raw predictions: {save_err}")

    metrics = {
        "seed": seed,
        "gnn_source": "simulated",
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "auc_pr_gnn_only": auc_pr_gnn_only,
        "auc_roc_gnn_only": auc_roc_gnn_only,
        "f1": f1,
        "recall_at_k": recall_at_k,
        "ece": ece_val,
        "brier": brier,
        "nll": nll,
        "n_samples": n_samples,
        "n_fraud": int(labels_np.sum()),
    }
    log.info(
        f"[Seed {seed}] AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f} "
        f"F1={f1:.4f} ECE={ece_val:.4f} [gnn=simulated]"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--output",  default="results/multiseed")
    parser.add_argument("--seeds",   default=None, help="Comma-separated seeds (overrides config)")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg.get("experiment", {}).get("seeds", [7, 17, 27, 37, 47])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    for seed in seeds:
        m = run_single_seed(cfg, seed)
        all_metrics.append(m)

    # ── Aggregate ──────────────────────────────────────────────────────────
    metric_keys = ["auc_roc", "auc_pr", "f1", "recall_at_k", "ece", "brier", "nll",
                   "auc_pr_gnn_only", "auc_roc_gnn_only"]
    agg = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, float("nan")))]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            log.info(f"  {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    # ── Bootstrap CI (95%) — per-seed mean resampling ──────────────────────
    n_bootstrap = 2000
    rng_ci = np.random.RandomState(0)
    ci = {}
    for k in ["auc_roc", "auc_pr", "f1", "recall_at_k"]:
        vals = np.array([m.get(k, float("nan")) for m in all_metrics])
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            boot = np.array([
                rng_ci.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            ci[k] = {"lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
            log.info(f"  {k} 95% CI: [{ci[k]['lo']:.4f}, {ci[k]['hi']:.4f}]")

    # ── Paired bootstrap: Full model vs GNN Only (AUC-PR delta) ─────────────
    full_auc_prs = np.array([m.get("auc_pr", float("nan")) for m in all_metrics])
    gnn_only_auc_prs = np.array([m.get("auc_pr_gnn_only", float("nan")) for m in all_metrics])
    valid = ~(np.isnan(full_auc_prs) | np.isnan(gnn_only_auc_prs))
    paired_bootstrap = {}
    if valid.sum() >= 2:
        deltas = full_auc_prs[valid] - gnn_only_auc_prs[valid]
        delta_mean = float(deltas.mean())
        boot_deltas = np.array([
            rng_ci.choice(deltas, size=len(deltas), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        paired_bootstrap["full_vs_gnn_only_auc_pr"] = {
            "delta_mean": delta_mean,
            "ci_lo": float(np.percentile(boot_deltas, 2.5)),
            "ci_hi": float(np.percentile(boot_deltas, 97.5)),
        }
        log.info(
            f"  Paired bootstrap Full vs GNN-only AUC-PR: delta={delta_mean:.4f} "
            f"95%CI=[{paired_bootstrap['full_vs_gnn_only_auc_pr']['ci_lo']:.4f}, "
            f"{paired_bootstrap['full_vs_gnn_only_auc_pr']['ci_hi']:.4f}]"
        )

    # ── Run manifest ────────────────────────────────────────────────────────
    import hashlib, datetime
    import subprocess as _sp
    cfg_str = json.dumps(cfg, sort_keys=True)
    cfg_sha256 = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]
    try:
        git_commit = _sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        git_commit = "unknown"
    try:
        import sys as _sys
        python_ver = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"
        import torch as _torch
        torch_ver = _torch.__version__
    except Exception:
        python_ver = "unknown"
        torch_ver = "unknown"

    manifest = {
        "experiment": "multiseed",
        "git_commit": git_commit,
        "seeds": seeds,
        "gnn_source": "simulated",
        "config_sha256": cfg_sha256,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "command": f"python experiments/run_multiseed.py --config {args.config}",
        "python_version": python_ver,
        "torch_version": torch_ver,
    }
    manifest_dir = Path("results/run_manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"multiseed_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Run manifest saved to {manifest_path}")

    # ── Save ───────────────────────────────────────────────────────────────
    results = {
        "per_seed": all_metrics,
        "aggregate_mean_std": agg,
        "bootstrap_ci_95": ci,
        "paired_bootstrap": paired_bootstrap,
        "seeds": seeds,
        "gnn_source": "simulated",
        "manifest": manifest,
    }
    out_path = output_dir / "multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Multi-seed results saved to {out_path}")


if __name__ == "__main__":
    main()
