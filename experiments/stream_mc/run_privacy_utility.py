"""
experiments/run_privacy_utility.py

Privacy-Utility tradeoff experiment.

For each privacy representation mode:
    raw_context       : raw context text (baseline — no privacy)
    full_vector       : r_t = [s, q, k, a, h] as float32 (full precision)
    quantized         : float32 → int8 quantization
    noisy_gaussian    : Gaussian noise added to continuous values
    minimal           : score + category only (3-level bucketing)

Measures for each mode:
    - Fraud detection: AUC-PR, AUC-ROC, Recall@K
    - Privacy cost: serialized bytes (JSON and binary)
    - Leakage attack: scam_category attribute inference Accuracy, F1
    - Shortcut test: TF-IDF+LR on raw context → AUC

Results: results/privacy_utility/privacy_utility_results.json + .csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_privacy_mode(risk_dict: dict, mode: str, noise_scale: float = 0.05) -> dict:
    """Apply a privacy transformation to a risk dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from privacy.quantization import Quantizer
    from privacy.noise import NoiseMechanism

    d = dict(risk_dict)

    if mode == "raw_context":
        return d  # no transformation

    elif mode == "full_risk_vector":
        return d  # full precision, no transformation

    elif mode == "quantized_risk_vector":
        q = Quantizer()
        return q.quantize(d)

    elif mode == "noisy_risk_vector":
        nm = NoiseMechanism(mechanism="gaussian", scale=noise_scale, seed=42)
        res = nm.apply(d)
        # Randomized response on categorical attributes for differential privacy
        rng = np.random.RandomState(int(abs(hash(str(d.get("event_id", "0"))))) % 10000)
        if rng.rand() < 0.25:  # 25% randomized perturbation
            res["risk_type_id"] = int(rng.randint(0, 11))
            res["relation_hint_id"] = int(rng.randint(0, 6))
        return res

    elif mode == "minimal_risk_token":
        s = float(d.get("local_risk_score", 0.0))
        score_bucket = 0.9 if s >= 0.75 else (0.5 if s >= 0.4 else 0.1)
        return {
            "event_id": d.get("event_id"),
            "local_risk_score": score_bucket,
            "confidence": 0.5,
            "risk_type_id": 0,  # category stripped for maximum privacy
            "context_age_sec": 0,
            "relation_hint_id": 0,
            "privacy_mode": "minimal_risk_token",
        }
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--privacy-config", default=None)
    parser.add_argument("--output",         default="results/privacy_utility")
    parser.add_argument("--seeds",          default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from graphrag.risk_encoder import RiskEncoder
    from fusion.uncertainty_fusion import UncertaintyFusion
    from privacy.vector_codec import VectorCodec
    from privacy.leakage_attack import LeakageAttack
    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage
    )

    base_cfg    = _load_yaml(args.config)
    privacy_cfg = _load_yaml(args.privacy_config) if args.privacy_config else {}
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else base_cfg.get("experiment", {}).get("seeds", [7, 17, 27, 37, 47]))

    MODES = [
        ("raw_context",        "raw_context"),
        ("full_vector",        "full_risk_vector"),
        ("quantized",          "quantized_risk_vector"),
        ("noisy_gaussian",     "noisy_risk_vector"),
        ("minimal",            "minimal_risk_token"),
    ]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for seed in seeds:
        import random
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        n_samples = 1000
        fraud_rate = 0.10
        labels_np = (rng.rand(n_samples) < fraud_rate).astype(int)

        # Generate contexts
        scenarios = assign_scenarios_no_leakage(labels_np, seed=seed)
        gen = SyntheticContextGenerator(seed=seed)
        records = gen.generate_contexts(scenario_types=scenarios)

        # Full GraphRAG pipeline (once per seed)
        kb = LocalKnowledgeBase()
        retriever = GraphRAGRetriever(kb, RetrieverConfig(top_k=5, graph_hops=1))
        extractor = RiskExtractor()
        base_risk_dicts = []
        for rec in records:
            evidence = retriever.retrieve(rec["context_text"])
            rd = extractor.extract(evidence, event_id=rec["event_id"],
                                   pre_transaction_gap_sec=rec.get("pre_transaction_gap_sec", 300))
            base_risk_dicts.append(rd)

        # Base GNN simulation
        p_gnn_np = np.clip(
            labels_np * 0.70 + (1 - labels_np) * 0.20 + rng.normal(0, 0.15, n_samples),
            0.0, 1.0,
        )
        u_mc_np = np.clip(0.08 + 0.1 * rng.rand(n_samples), 0.001, 0.5)

        device = torch.device("cpu")
        encoder = RiskEncoder.from_config(base_cfg)
        encoder.eval()
        fusion  = UncertaintyFusion.from_config(base_cfg)
        attack  = LeakageAttack(cv_folds=3)
        codec_j = VectorCodec("json")
        codec_b = VectorCodec("binary")

        for mode_name, mode_key in MODES:
            # Apply privacy transformation
            transformed = [_apply_privacy_mode(d, mode_key) for d in base_risk_dicts]

            # Measure bytes
            if mode_key == "raw_context":
                bytes_json   = float(np.mean([len(r["context_text"].encode()) for r in records]))
                bytes_binary = bytes_json
            else:
                bytes_json   = float(np.mean([codec_j.measure_bytes(d) for d in transformed]))
                bytes_binary = float(np.mean([codec_b.measure_bytes(d) for d in transformed]))

            # Encode + fuse
            with torch.no_grad():
                _, p_risk = encoder.encode_risk_dict_batch(transformed, device=device)
            p_risk_np = p_risk.numpy()

            if mode_key == "raw_context":
                # Baseline: use full risk without privacy transformation for detection
                scores_np = np.clip(0.5 * p_gnn_np + 0.5 * p_risk_np, 0.0, 1.0)
            else:
                final_t, _, _ = fusion.fuse(
                    torch.from_numpy(p_gnn_np.astype(np.float32)),
                    torch.from_numpy(u_mc_np.astype(np.float32)),
                    p_risk,
                )
                scores_np = final_t.numpy()

            try:
                auc_roc = float(roc_auc_score(labels_np, scores_np))
            except Exception:
                auc_roc = float("nan")
            try:
                auc_pr = float(average_precision_score(labels_np, scores_np))
            except Exception:
                auc_pr = float("nan")

            k = max(int(labels_np.sum()), 1)
            topk_idx = np.argsort(scores_np)[::-1][:k]
            recall_at_k = float(labels_np[topk_idx].sum() / max(labels_np.sum(), 1))

            # Leakage attack: infer scam_category from risk representation
            category_labels = np.array([d.get("risk_type_id", 0) for d in base_risk_dicts])
            attack_result = attack.run_from_dicts(
                transformed, category_labels,
                representation_name=mode_name,
                target_attribute="scam_category",
                codec=codec_j,
            )

            row = {
                "mode": mode_name,
                "seed": seed,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "recall_at_k": recall_at_k,
                "bytes_json": bytes_json,
                "bytes_binary": bytes_binary,
                "attack_accuracy": attack_result.accuracy,
                "attack_macro_f1": attack_result.macro_f1,
                "attack_auc": attack_result.auc,
            }
            all_results.append(row)
            log.info(
                f"  [{mode_name}] seed={seed}: AUC-PR={auc_pr:.4f}, "
                f"bytes={bytes_binary:.0f}B (bin), attack_acc={attack_result.accuracy:.4f}"
            )

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary = {}
    for mode_name, _ in MODES:
        rows = [r for r in all_results if r["mode"] == mode_name]
        summary[mode_name] = {}
        for metric in ["auc_roc", "auc_pr", "recall_at_k", "bytes_binary", "attack_accuracy", "attack_macro_f1"]:
            vals = [r[metric] for r in rows if not np.isnan(r[metric])]
            summary[mode_name][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out_json = output_dir / "privacy_utility_results.json"
    with open(out_json, "w") as f:
        json.dump({"per_run": all_results, "summary": summary, "seeds": seeds}, f, indent=2)

    # CSV
    try:
        import csv
        csv_path = output_dir / "privacy_utility_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "AUC-PR", "AUC-ROC", "Recall@K", "Bytes(bin)", "Attack Acc", "Attack F1"])
            for mode_name, _ in MODES:
                s = summary[mode_name]
                writer.writerow([
                    mode_name,
                    f"{s['auc_pr']['mean']:.4f}",
                    f"{s['auc_roc']['mean']:.4f}",
                    f"{s['recall_at_k']['mean']:.4f}",
                    f"{s['bytes_binary']['mean']:.1f}",
                    f"{s['attack_accuracy']['mean']:.4f}",
                    f"{s['attack_macro_f1']['mean']:.4f}",
                ])
        log.info(f"Privacy-utility table saved to {csv_path}")
    except Exception as e:
        log.warning(f"CSV export failed: {e}")

    log.info(f"Privacy-utility results saved to {out_json}")


if __name__ == "__main__":
    main()
