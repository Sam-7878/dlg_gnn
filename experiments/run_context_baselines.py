"""
experiments/run_context_baselines.py

Context-only lexical baseline experiments (TASK 4.2, 4.3, 12.1).

This script answers the key reviewer question:
  "Is GraphRAG doing anything beyond keyword/template detection?"

Baselines:
  1. Majority classifier (always predicts majority class)
  2. Keyword-rule classifier (heuristic: count fraud-related keywords)
  3. TF-IDF + Logistic Regression
  4. TF-IDF + Linear SVM
  5. Semantic Risk Only (GraphRAG RiskEncoder output → LR/threshold)
  6. GraphRAG Retrieval Ablation:
       No Retrieval → Keyword Only → Keyword+Top-k → Keyword+1-hop BFS

Metrics per model:
  AUC-ROC, AUC-PR, Macro-F1, Balanced Accuracy, fraud_ratio

Outputs:
  results/context_baselines.csv
  results/retrieval_ablation.csv

Usage:
  python experiments/run_context_baselines.py --config configs/base.yaml
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

FRAUD_KEYWORDS = [
    "urgent", "compromised", "send", "transfer", "wallet", "usdt", "btc",
    "scam", "phishing", "verify", "account", "security", "password", "click",
    "link", "immediately", "suspended", "confirm", "alert", "prize", "won",
    "reward", "claim", "free", "limited", "offer",
]


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, balanced_accuracy_score,
    )
    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true, y_score))
    except Exception:
        auc_pr = float("nan")

    y_pred = (y_score >= threshold).astype(int)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    fraud_ratio = float(y_true.mean())

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "fraud_ratio": fraud_ratio,
    }


def run_baselines_for_seed(cfg: Dict, seed: int) -> Dict[str, Dict[str, float]]:
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    import torch

    rng = np.random.RandomState(seed)
    labels_np = (rng.rand(N_SAMPLES) < FRAUD_RATE).astype(int)
    scenarios = assign_scenarios_no_leakage(labels_np, seed=seed)
    gen = SyntheticContextGenerator(seed=seed)
    event_ids = [f"tx_{i:06d}" for i in range(N_SAMPLES)]
    records = gen.generate_contexts(scenario_types=scenarios, event_ids=event_ids)
    texts = [r["context_text"] for r in records]

    results: Dict[str, Dict[str, float]] = {}

    # ── 1. Majority classifier ────────────────────────────────────────────
    majority_score = np.full(N_SAMPLES, float(labels_np.mean()))
    results["majority_classifier"] = _compute_metrics(labels_np, majority_score)

    # ── 2. Keyword-rule classifier ────────────────────────────────────────
    def _keyword_score(text: str) -> float:
        text_lower = text.lower()
        count = sum(kw in text_lower for kw in FRAUD_KEYWORDS)
        return min(count / max(len(FRAUD_KEYWORDS) * 0.3, 1), 1.0)

    kw_scores = np.array([_keyword_score(t) for t in texts])
    results["keyword_rule"] = _compute_metrics(labels_np, kw_scores)

    # ── 3. TF-IDF + Logistic Regression ──────────────────────────────────
    n_train = int(N_SAMPLES * 0.7)
    X_train, X_test = texts[:n_train], texts[n_train:]
    y_train, y_test = labels_np[:n_train], labels_np[n_train:]

    tfidf_lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=seed, class_weight="balanced")),
    ])
    try:
        tfidf_lr.fit(X_train, y_train)
        lr_scores = tfidf_lr.predict_proba(X_test)[:, 1]
        results["tfidf_lr"] = _compute_metrics(y_test, lr_scores)
    except Exception as e:
        log.warning(f"TF-IDF+LR failed: {e}")
        results["tfidf_lr"] = {"auc_roc": float("nan"), "auc_pr": float("nan"),
                                "macro_f1": float("nan"), "balanced_accuracy": float("nan"),
                                "fraud_ratio": float(y_test.mean())}

    # ── 4. TF-IDF + Linear SVM (calibrated for probabilities) ─────────────
    tfidf_svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
        ("clf", CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, random_state=seed, class_weight="balanced"),
            cv=3,
        )),
    ])
    try:
        tfidf_svm.fit(X_train, y_train)
        svm_scores = tfidf_svm.predict_proba(X_test)[:, 1]
        results["tfidf_svm"] = _compute_metrics(y_test, svm_scores)
    except Exception as e:
        log.warning(f"TF-IDF+SVM failed: {e}")
        results["tfidf_svm"] = {"auc_roc": float("nan"), "auc_pr": float("nan"),
                                 "macro_f1": float("nan"), "balanced_accuracy": float("nan"),
                                 "fraud_ratio": float(y_test.mean())}

    # ── 5. Semantic Risk Only (GraphRAG RiskEncoder → threshold) ──────────
    kb = LocalKnowledgeBase()
    retriever_cfg_obj = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=cfg.get("graphrag", {}).get("graph_hops", 1),
        similarity_threshold=0.0,
    )
    retriever = GraphRAGRetriever(kb, retriever_cfg_obj)
    extractor = RiskExtractor()
    encoder = RiskEncoder.from_config(cfg)
    encoder.eval()

    risk_dicts = []
    for rec in records:
        evidence = retriever.retrieve(rec["context_text"])
        rd = extractor.extract(evidence, event_id=rec["event_id"], pre_transaction_gap_sec=300)
        risk_dicts.append(rd)

    with torch.no_grad():
        _, p_risk = encoder.encode_risk_dict_batch(risk_dicts)
    p_risk_np = p_risk.numpy()

    results["semantic_risk_only"] = _compute_metrics(labels_np, p_risk_np)

    # ── 6. Retrieval ablation ─────────────────────────────────────────────
    retrieval_results: Dict[str, Dict[str, float]] = {}

    # 6a. No retrieval — risk encoder on empty evidence
    def _risk_score_no_retrieval(records_list):
        rds = []
        for rec in records_list:
            # Pass empty evidence list
            rd = extractor.extract([], event_id=rec["event_id"], pre_transaction_gap_sec=300)
            rds.append(rd)
        with torch.no_grad():
            _, prisk = encoder.encode_risk_dict_batch(rds)
        return prisk.numpy()

    try:
        scores_no_ret = _risk_score_no_retrieval(records)
        retrieval_results["no_retrieval"] = _compute_metrics(labels_np, scores_no_ret)
    except Exception as e:
        log.warning(f"No-retrieval ablation failed: {e}")
        retrieval_results["no_retrieval"] = {"auc_roc": float("nan"), "auc_pr": float("nan"),
                                               "macro_f1": float("nan"), "balanced_accuracy": float("nan"),
                                               "fraud_ratio": float(labels_np.mean())}

    # 6b. Keyword only (no graph expansion)
    cfg_kw = {**cfg}
    retriever_kw_cfg = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=0,  # no graph expansion
        similarity_threshold=0.0,
    )
    retriever_kw = GraphRAGRetriever(kb, retriever_kw_cfg)
    risk_dicts_kw = []
    for rec in records:
        ev = retriever_kw.retrieve(rec["context_text"])
        rd = extractor.extract(ev, event_id=rec["event_id"], pre_transaction_gap_sec=300)
        risk_dicts_kw.append(rd)
    with torch.no_grad():
        _, p_risk_kw = encoder.encode_risk_dict_batch(risk_dicts_kw)
    retrieval_results["keyword_only"] = _compute_metrics(labels_np, p_risk_kw.numpy())

    # 6c. Keyword + Top-k (no graph traversal)
    retriever_topk_cfg = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=0,
        similarity_threshold=0.0,
    )
    retriever_topk = GraphRAGRetriever(kb, retriever_topk_cfg)
    risk_dicts_topk = []
    for rec in records:
        ev = retriever_topk.retrieve(rec["context_text"])
        rd = extractor.extract(ev, event_id=rec["event_id"], pre_transaction_gap_sec=300)
        risk_dicts_topk.append(rd)
    with torch.no_grad():
        _, p_risk_topk = encoder.encode_risk_dict_batch(risk_dicts_topk)
    retrieval_results["keyword_topk"] = _compute_metrics(labels_np, p_risk_topk.numpy())

    # 6d. Keyword + 1-hop graph expansion (full GraphRAG)
    retrieval_results["keyword_graph_1hop"] = _compute_metrics(labels_np, p_risk_np)

    return results, retrieval_results


def aggregate_seeds(per_seed: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate list of per-seed dicts → {model: {metric: {mean, std}}}."""
    all_models = list(per_seed[0].keys())
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in all_models:
        metric_vals: Dict[str, List[float]] = {}
        for sd in per_seed:
            for metric, val in sd[model].items():
                metric_vals.setdefault(metric, []).append(val)
        agg[model] = {
            m: {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
            for m, vals in metric_vals.items()
        }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = _load_yaml(args.config)

    baseline_results_all = []
    retrieval_results_all = []

    for seed in seeds:
        log.info(f"Running context baselines — seed={seed}")
        b_res, r_res = run_baselines_for_seed(cfg, seed)
        baseline_results_all.append(b_res)
        retrieval_results_all.append(r_res)

    # Aggregate
    baseline_agg = aggregate_seeds(baseline_results_all)
    retrieval_agg = aggregate_seeds(retrieval_results_all)

    # Format as flat CSV rows
    import pandas as pd

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # context_baselines.csv
    baseline_rows = []
    for model, metrics in baseline_agg.items():
        row: Dict = {"Model": model}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = round(stats["mean"], 4)
            row[f"{metric}_std"] = round(stats["std"], 4)
        baseline_rows.append(row)
    df_baseline = pd.DataFrame(baseline_rows)
    csv_path = out_dir / "context_baselines.csv"
    df_baseline.to_csv(csv_path, index=False)
    log.info(f"Saved: {csv_path}")

    # retrieval_ablation.csv
    retrieval_rows = []
    for model, metrics in retrieval_agg.items():
        row: Dict = {"Strategy": model}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = round(stats["mean"], 4)
            row[f"{metric}_std"] = round(stats["std"], 4)
        retrieval_rows.append(row)
    df_retrieval = pd.DataFrame(retrieval_rows)
    ret_csv_path = out_dir / "retrieval_ablation.csv"
    df_retrieval.to_csv(ret_csv_path, index=False)
    log.info(f"Saved: {ret_csv_path}")

    # Print summary
    log.info("\n" + "=" * 80)
    log.info("  CONTEXT BASELINE RESULTS (AUC-PR, AUC-ROC, Balanced-Acc, Macro-F1)")
    log.info("=" * 80)
    for model, metrics in baseline_agg.items():
        pr = metrics.get("auc_pr", {})
        roc = metrics.get("auc_roc", {})
        ba = metrics.get("balanced_accuracy", {})
        f1 = metrics.get("macro_f1", {})
        log.info(
            f"  {model:30s}  "
            f"AUC-PR={pr.get('mean', float('nan')):.4f}±{pr.get('std', 0):.4f}  "
            f"AUC-ROC={roc.get('mean', float('nan')):.4f}  "
            f"BalAcc={ba.get('mean', float('nan')):.4f}  "
            f"MacroF1={f1.get('mean', float('nan')):.4f}"
        )

    log.info("\n  RETRIEVAL ABLATION:")
    for strat, metrics in retrieval_agg.items():
        pr = metrics.get("auc_pr", {})
        roc = metrics.get("auc_roc", {})
        log.info(
            f"  {strat:30s}  AUC-PR={pr.get('mean', float('nan')):.4f}  "
            f"AUC-ROC={roc.get('mean', float('nan')):.4f}"
        )

    # Save JSON for figure generation
    json_path = out_dir / "context_baselines.json"
    with open(json_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "context_baselines": baseline_agg,
            "retrieval_ablation": retrieval_agg,
        }, f, indent=2)
    log.info(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
