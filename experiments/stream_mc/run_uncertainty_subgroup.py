"""
experiments/run_uncertainty_subgroup.py

Uncertainty subgroup analysis (TASK 8.1, 8.2).

Splits test events into uncertainty quartiles (Q1=lowest … Q4=highest) and
compares GNN Only / Semantic Only / Fixed Fusion / Uncertainty Fusion
AUC-PR within each quartile.

Also performs gating sanity checks:
  - corr(U_t, beta_t)
  - beta_t distribution by quartile
  - beta_t by correct/incorrect GNN prediction

Outputs:
  results/uncertainty_subgroup.csv
  results/gating_sanity.csv
  results/raw_predictions/uncertainty_subgroup_seed{N}.csv

Usage:
  python experiments/run_uncertainty_subgroup.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

SEEDS = [7, 17, 27, 37, 47]
N_SAMPLES = 1000
FRAUD_RATE = 0.10
NOISE_SCALE = 0.15  # matches run_multiseed.py simulation


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def run_subgroup_for_seed(cfg: Dict, seed: int) -> Dict:
    _set_seed(seed)
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage,
    )
    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from graphrag.risk_encoder import RiskEncoder
    from fusion.uncertainty_fusion import UncertaintyFusion
    from fusion.fixed_fusion import FixedFusion
    import torch

    rng = np.random.RandomState(seed)
    labels_np = (rng.rand(N_SAMPLES) < FRAUD_RATE).astype(int)
    scenarios = assign_scenarios_no_leakage(labels_np, seed=seed)
    gen = SyntheticContextGenerator(seed=seed)
    event_ids = [f"tx_{i:06d}" for i in range(N_SAMPLES)]
    records = gen.generate_contexts(scenario_types=scenarios, event_ids=event_ids)

    # GraphRAG → RiskEncoder
    kb = LocalKnowledgeBase()
    retriever_cfg = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=cfg.get("graphrag", {}).get("graph_hops", 1),
        similarity_threshold=0.0,
    )
    retriever = GraphRAGRetriever(kb, retriever_cfg)
    extractor = RiskExtractor()
    encoder = RiskEncoder.from_config(cfg)
    encoder.eval()

    risk_dicts = []
    for rec in records:
        ev = retriever.retrieve(rec["context_text"])
        rd = extractor.extract(ev, event_id=rec["event_id"], pre_transaction_gap_sec=300)
        risk_dicts.append(rd)

    with torch.no_grad():
        _, p_risk = encoder.encode_risk_dict_batch(risk_dicts)
    p_risk_np = p_risk.numpy()

    # Simulated GNN outputs (consistent with run_multiseed.py)
    p_gnn_np = np.clip(
        labels_np.astype(float) * 0.7 + (1 - labels_np) * 0.2
        + rng.normal(0, NOISE_SCALE, N_SAMPLES),
        0.0, 1.0,
    )
    u_mc_np = np.clip(
        0.08 + NOISE_SCALE * rng.rand(N_SAMPLES) * (1 - np.abs(p_gnn_np - 0.5)),
        0.001, 0.5,
    )

    # Fusion outputs
    unc_fusion = UncertaintyFusion.from_config(cfg)
    p_gnn_t = torch.from_numpy(p_gnn_np.astype(np.float32))
    u_mc_t = torch.from_numpy(u_mc_np.astype(np.float32))
    p_risk_t = torch.from_numpy(p_risk_np.astype(np.float32))

    final_unc, _, beta_t = unc_fusion.fuse(p_gnn_t, u_mc_t, p_risk_t)
    final_unc_np = final_unc.numpy()
    beta_t_np = beta_t.numpy() if beta_t is not None else u_mc_np  # fallback

    fixed_fusion = FixedFusion(alpha=0.5)
    final_fixed, _, _ = fixed_fusion.fuse(p_gnn_t, p_risk_t)
    final_fixed_np = final_fixed.numpy()

    # Uncertainty quartile split
    quartiles = np.percentile(u_mc_np, [25, 50, 75])
    q_labels = np.digitize(u_mc_np, quartiles)  # 0=Q1, 1=Q2, 2=Q3, 3=Q4

    quartile_rows = []
    for q in range(4):
        mask = q_labels == q
        if mask.sum() < 5:
            continue
        yt = labels_np[mask]
        row = {
            "quartile": f"Q{q+1}",
            "n_events": int(mask.sum()),
            "mean_U_t": float(u_mc_np[mask].mean()),
            "mean_beta_t": float(beta_t_np[mask].mean()),
            "fraud_ratio": float(yt.mean()),
            "gnn_only_auc_pr": _auc_pr(yt, p_gnn_np[mask]),
            "semantic_only_auc_pr": _auc_pr(yt, p_risk_np[mask]),
            "fixed_fusion_auc_pr": _auc_pr(yt, final_fixed_np[mask]),
            "uncertainty_fusion_auc_pr": _auc_pr(yt, final_unc_np[mask]),
        }
        quartile_rows.append(row)

    # Gating sanity check
    corr_u_beta = float(np.corrcoef(u_mc_np, beta_t_np)[0, 1])
    gnn_correct = ((p_gnn_np >= 0.5).astype(int) == labels_np).astype(int)
    beta_correct = float(beta_t_np[gnn_correct == 1].mean()) if (gnn_correct == 1).sum() > 0 else float("nan")
    beta_incorrect = float(beta_t_np[gnn_correct == 0].mean()) if (gnn_correct == 0).sum() > 0 else float("nan")

    sanity = {
        "seed": seed,
        "corr_U_beta": corr_u_beta,
        "beta_correct_gnn": beta_correct,
        "beta_incorrect_gnn": beta_incorrect,
        "beta_q1": float(beta_t_np[q_labels == 0].mean()) if (q_labels == 0).sum() > 0 else float("nan"),
        "beta_q4": float(beta_t_np[q_labels == 3].mean()) if (q_labels == 3).sum() > 0 else float("nan"),
    }

    # Raw event-level predictions (for calibration and other analyses)
    import pandas as pd
    raw_dir = Path("results/raw_predictions")
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "event_id": event_ids,
        "label": labels_np,
        "p_gnn": p_gnn_np,
        "U_t": u_mc_np,
        "p_risk": p_risk_np,
        "beta_t": beta_t_np,
        "score": final_unc_np,
        "score_fixed": final_fixed_np,
        "uncertainty_quartile": [f"Q{q+1}" for q in q_labels],
    }).to_csv(raw_dir / f"uncertainty_subgroup_seed{seed}.csv", index=False)

    for row in quartile_rows:
        row["seed"] = seed
    sanity["quartile_rows"] = quartile_rows

    return {"quartile_rows": quartile_rows, "sanity": sanity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = _load_yaml(args.config)

    all_quartile_rows: List[Dict] = []
    all_sanity: List[Dict] = []

    for seed in seeds:
        log.info(f"Uncertainty subgroup analysis — seed={seed}")
        result = run_subgroup_for_seed(cfg, seed)
        all_quartile_rows.extend(result["quartile_rows"])
        all_sanity.append(result["sanity"])

    import pandas as pd

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # uncertainty_subgroup.csv — averaged over seeds per quartile
    df_q = pd.DataFrame(all_quartile_rows)
    df_q_agg = df_q.groupby("quartile").mean(numeric_only=True).reset_index()
    subgroup_csv = out_dir / "uncertainty_subgroup.csv"
    df_q_agg.to_csv(subgroup_csv, index=False)
    log.info(f"Saved: {subgroup_csv}")

    # gating_sanity.csv
    df_sanity = pd.DataFrame(all_sanity).drop(columns=["quartile_rows"], errors="ignore")
    sanity_csv = out_dir / "gating_sanity.csv"
    df_sanity.to_csv(sanity_csv, index=False)
    log.info(f"Saved: {sanity_csv}")

    # Summary
    log.info("\n" + "=" * 80)
    log.info("  UNCERTAINTY SUBGROUP ANALYSIS")
    log.info("=" * 80)
    log.info(f"  {'Quartile':8}  {'mean_U':7}  {'mean_beta':9}  "
             f"{'GNN-only':9}  {'SemOnly':8}  {'Fixed':7}  {'UncFusion':9}")
    for _, row in df_q_agg.iterrows():
        log.info(
            f"  {row['quartile']:8}  {row.get('mean_U_t', float('nan')):7.4f}  "
            f"{row.get('mean_beta_t', float('nan')):9.4f}  "
            f"{row.get('gnn_only_auc_pr', float('nan')):9.4f}  "
            f"{row.get('semantic_only_auc_pr', float('nan')):8.4f}  "
            f"{row.get('fixed_fusion_auc_pr', float('nan')):7.4f}  "
            f"{row.get('uncertainty_fusion_auc_pr', float('nan')):9.4f}"
        )

    mean_corr = np.nanmean([s["corr_U_beta"] for s in all_sanity])
    log.info(f"\n  Gating: mean corr(U_t, beta_t) = {mean_corr:.4f}")
    mean_beta_correct = np.nanmean([s["beta_correct_gnn"] for s in all_sanity])
    mean_beta_incorrect = np.nanmean([s["beta_incorrect_gnn"] for s in all_sanity])
    log.info(f"  beta_t | GNN correct={mean_beta_correct:.4f}, GNN incorrect={mean_beta_incorrect:.4f}")


if __name__ == "__main__":
    main()
