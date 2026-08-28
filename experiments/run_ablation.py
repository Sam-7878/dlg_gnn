"""
experiments/run_ablation.py

Ablation study: 8 variants of the dlg_gnn GraphRAG pipeline.

Variants (from configs/ablation.yaml):
    gnn_only          : GNN only (no risk branch)
    semantic_only     : Risk branch only (no GNN)
    fixed_fusion      : Fixed-weight fusion (α=0.40)
    learned_fusion    : Learned MLP fusion (no uncertainty)
    uncertainty_fusion: Proposed method (β_t = σ(λ*Ũ_t + b))
    no_graphrag       : Uncertainty fusion, GraphRAG disabled
    no_mc             : Uncertainty fusion, MC disabled
    no_streaming      : Uncertainty fusion, streaming (L2) disabled

Each variant runs on seeds = base config seeds.
Results saved as: results/ablation/ablation_results.csv + .json
"""

import argparse
import json
import logging
import random
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
    with open(path) as f:
        return yaml.safe_load(f)


def _run_variant(
    variant_name: str,
    variant_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, float]:
    """Run one ablation variant for one seed."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from graphrag.risk_encoder import RiskEncoder
    from fusion.fixed_fusion import FixedFusion
    from fusion.uncertainty_fusion import UncertaintyFusion
    from fusion.learned_fusion import LearnedFusion
    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage
    )

    _set_seed(seed)
    rng = np.random.RandomState(seed)

    n_samples = 1000
    fraud_rate = 0.10
    labels_np = (rng.rand(n_samples) < fraud_rate).astype(int)

    # ── Stage 1: Context generation ───────────────────────────────────────
    scenarios = assign_scenarios_no_leakage(labels_np, seed=seed)
    gen = SyntheticContextGenerator(seed=seed)
    records = gen.generate_contexts(scenario_types=scenarios)

    # ── Stage 2: GraphRAG (enabled/disabled by variant) ──────────────────
    graphrag_enabled = variant_cfg.get("graphrag_enabled", True)

    if graphrag_enabled:
        kb = LocalKnowledgeBase()
        retriever_cfg = RetrieverConfig(top_k=5, graph_hops=1)
        retriever = GraphRAGRetriever(kb, retriever_cfg)
        extractor = RiskExtractor()
        risk_dicts = []
        for rec in records:
            evidence = retriever.retrieve(rec["context_text"])
            rd = extractor.extract(evidence, event_id=rec["event_id"],
                                   pre_transaction_gap_sec=rec.get("pre_transaction_gap_sec", 300))
            risk_dicts.append(rd)
    else:
        # GraphRAG disabled: use zero risk vectors
        risk_dicts = [{"local_risk_score": 0.0, "confidence": 0.0,
                       "risk_type_id": 0, "context_age_sec": 0,
                       "relation_hint_id": 0} for _ in records]

    # ── Stage 3: Risk Encoder ─────────────────────────────────────────────
    device = torch.device("cpu")
    encoder = RiskEncoder.from_config(base_cfg)
    encoder.eval()
    with torch.no_grad():
        _, p_risk = encoder.encode_risk_dict_batch(risk_dicts, device=device)
    p_risk_np = p_risk.numpy()

    # ── Stage 4: Simulated GNN output ─────────────────────────────────────
    noise_scale = 0.15
    p_gnn_np = np.clip(
        labels_np * 0.7 + (1 - labels_np) * 0.2 + rng.normal(0, noise_scale, n_samples),
        0.0, 1.0,
    )
    u_mc_np = np.clip(
        0.08 + noise_scale * rng.rand(n_samples) * (1 - np.abs(p_gnn_np - 0.5)),
        0.001, 0.5,
    )

    # MC disabled → zero uncertainty
    mc_enabled = variant_cfg.get("mc_enabled", True)
    if not mc_enabled:
        u_mc_np = np.zeros(n_samples, dtype=np.float32)

    # L2 (streaming) disabled → weaker GNN signal
    streaming_enabled = variant_cfg.get("streaming_enabled", True)
    if not streaming_enabled:
        p_gnn_np = np.clip(p_gnn_np * 0.85, 0.0, 1.0)

    # ── Stage 5: Fusion ───────────────────────────────────────────────────
    p_gnn_t  = torch.from_numpy(p_gnn_np.astype(np.float32))
    u_mc_t   = torch.from_numpy(u_mc_np.astype(np.float32))
    p_risk_t = torch.from_numpy(p_risk_np.astype(np.float32))

    strategy = variant_cfg.get("fusion_strategy", "uncertainty")
    fixed_alpha = float(variant_cfg.get("fixed_alpha", 0.4))

    if strategy == "gnn_only":
        scores_np = p_gnn_np
    elif strategy == "semantic_only":
        scores_np = p_risk_np
    elif strategy == "fixed":
        ff = FixedFusion(alpha=fixed_alpha)
        scores_t, _, _ = ff.fuse(p_gnn_t, p_risk_t)
        scores_np = scores_t.numpy()
    elif strategy == "learned":
        lf = LearnedFusion()
        lf.eval()
        with torch.no_grad():
            scores_t, _, _ = lf(p_gnn_t, p_risk_t)
        scores_np = scores_t.numpy()
    else:  # uncertainty (proposed)
        uf = UncertaintyFusion.from_config(base_cfg)
        scores_t, _, _ = uf.fuse(p_gnn_t, u_mc_t, p_risk_t)
        scores_np = scores_t.numpy()

    # ── Stage 6: Metrics ──────────────────────────────────────────────────
    try:
        auc_roc = float(roc_auc_score(labels_np, scores_np))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(labels_np, scores_np))
    except Exception:
        auc_pr = float("nan")

    preds = (scores_np >= 0.5).astype(int)
    f1 = float(f1_score(labels_np, preds, zero_division=0))

    k = max(int(labels_np.sum()), 1)
    topk_idx = np.argsort(scores_np)[::-1][:k]
    recall_at_k = float(labels_np[topk_idx].sum() / max(labels_np.sum(), 1))

    return {
        "variant": variant_name,
        "seed": seed,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "recall_at_k": recall_at_k,
    }


ABLATION_VARIANTS = {
    "gnn_only":          {"fusion_strategy": "gnn_only",   "graphrag_enabled": False, "mc_enabled": True,  "streaming_enabled": True},
    "semantic_only":     {"fusion_strategy": "semantic_only","graphrag_enabled": True,  "mc_enabled": True,  "streaming_enabled": True},
    "fixed_fusion":      {"fusion_strategy": "fixed",       "graphrag_enabled": True,  "fixed_alpha": 0.40, "mc_enabled": True, "streaming_enabled": True},
    "learned_fusion":    {"fusion_strategy": "learned",     "graphrag_enabled": True,  "mc_enabled": True,  "streaming_enabled": True},
    "uncertainty_fusion":{"fusion_strategy": "uncertainty", "graphrag_enabled": True,  "mc_enabled": True,  "streaming_enabled": True},
    "no_graphrag":       {"fusion_strategy": "uncertainty", "graphrag_enabled": False, "mc_enabled": True,  "streaming_enabled": True},
    "no_mc":             {"fusion_strategy": "uncertainty", "graphrag_enabled": True,  "mc_enabled": False, "streaming_enabled": True},
    "no_streaming":      {"fusion_strategy": "uncertainty", "graphrag_enabled": True,  "mc_enabled": True,  "streaming_enabled": False},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True)
    parser.add_argument("--ablation-config", default=None)
    parser.add_argument("--output",          default="results/ablation")
    parser.add_argument("--seeds",           default=None)
    args = parser.parse_args()

    base_cfg = _load_yaml(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else base_cfg.get("experiment", {}).get("seeds", [7, 17, 27, 37, 47]))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        log.info(f"\n── Variant: {variant_name} ──────────────────────────")
        per_seed = []
        for seed in seeds:
            m = _run_variant(variant_name, variant_cfg, base_cfg, seed)
            per_seed.append(m)
            all_results.append(m)

        # Aggregate
        for metric in ["auc_roc", "auc_pr", "f1", "recall_at_k"]:
            vals = [m[metric] for m in per_seed if not np.isnan(m[metric])]
            log.info(f"  {metric:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    # Aggregate by variant
    summary = {}
    for vname in ABLATION_VARIANTS:
        rows = [r for r in all_results if r["variant"] == vname]
        summary[vname] = {}
        for metric in ["auc_roc", "auc_pr", "f1", "recall_at_k"]:
            vals = [r[metric] for r in rows if not np.isnan(r[metric])]
            summary[vname][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out_json = output_dir / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump({"per_seed": all_results, "summary": summary, "seeds": seeds}, f, indent=2)

    # CSV table
    try:
        import csv
        csv_path = output_dir / "ablation_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Variant", "AUC-ROC mean", "AUC-ROC std", "AUC-PR mean", "AUC-PR std", "F1 mean", "F1 std", "Recall@K mean"])
            for vname, stats in summary.items():
                writer.writerow([
                    vname,
                    f"{stats['auc_roc']['mean']:.4f}", f"{stats['auc_roc']['std']:.4f}",
                    f"{stats['auc_pr']['mean']:.4f}",  f"{stats['auc_pr']['std']:.4f}",
                    f"{stats['f1']['mean']:.4f}",       f"{stats['f1']['std']:.4f}",
                    f"{stats['recall_at_k']['mean']:.4f}",
                ])
        log.info(f"Ablation table saved to {csv_path}")
    except Exception as e:
        log.warning(f"CSV export failed: {e}")

    log.info(f"Ablation results saved to {out_json}")


if __name__ == "__main__":
    main()
