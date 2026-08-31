# src/gog_fraud/evaluation/routing_metrics.py
"""
Routing metrics, decision flip analysis, and trace provenance for DLG-StreamMC.
Audit and evaluate selective local-to-global inference decisions against ground truth.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SampleRoutingTrace:
    sample_id: str
    chain: str
    seed: int
    policy: str
    label: int
    l1_score: float
    l1_decision: int
    mc_mean: float
    mc_variance: float
    mc_entropy: float
    mc_samples: int
    route: str  # "benign_direct", "fraud_direct", "deep_inspection"
    route_reason: str
    l2_score: Optional[float]
    l2_decision: Optional[int]
    fusion_score: Optional[float]
    fusion_decision: Optional[int]
    final_score: float
    final_decision: int


@dataclass
class FlipMetrics:
    total_samples: int
    n_direct: int
    n_deep: int
    n_direct_fraud: int
    n_direct_benign: int
    n_deep_fraud: int
    n_deep_benign: int

    # Flips on deep path: L1 vs Fusion
    n_flips: int
    flip_rate_deep: float
    flip_rate_total: float

    # Qualitative direction of flips
    wrong_to_correct: int  # Improved
    correct_to_wrong: int  # Degraded
    net_gain: int  # Improved - Degraded

    # Detailed confusion flips
    fraud_fn_to_tp: int  # Missed by L1, Caught by Fusion
    fraud_tp_to_fn: int  # Caught by L1, Missed by Fusion (Regression)
    benign_fp_to_tn: int  # False alarm in L1, cleared by Fusion
    benign_tn_to_fp: int  # True negative in L1, false alarm in Fusion

    # Routing safety / quality
    direct_exit_fnr: float
    overall_recall: float
    overall_precision: float
    overall_f1: float
    deep_route_rate: float
    direct_exit_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_entropy(prob: float, eps: float = 1e-12) -> float:
    p = float(np.clip(prob, eps, 1.0 - eps))
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def evaluate_routing_traces(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    l1_scores: Sequence[float],
    mc_means: Sequence[float],
    mc_vars: Sequence[float],
    routes: Sequence[str],
    route_reasons: Sequence[str],
    l2_scores: Sequence[float],
    fusion_scores: Sequence[float],
    threshold: float,
    chain: str = "pooled",
    seed: int = 11,
    policy: str = "dual_threshold",
    num_mc_samples: int = 8,
) -> Tuple[List[SampleRoutingTrace], FlipMetrics]:
    """
    Build auditable sample-level routing traces and compute flip metrics.
    Samples not routed to deep_inspection have NA for l2_* and fusion_*.
    """
    n = len(sample_ids)
    if not (n == len(labels) == len(l1_scores) == len(fusion_scores)):
        raise ValueError(f"Length mismatch: {n}, {len(labels)}, {len(l1_scores)}, {len(fusion_scores)}")

    traces: List[SampleRoutingTrace] = []
    
    n_direct = 0
    n_deep = 0
    n_direct_fraud = 0
    n_direct_benign = 0
    n_deep_fraud = 0
    n_deep_benign = 0

    wrong_to_correct = 0
    correct_to_wrong = 0
    fraud_fn_to_tp = 0
    fraud_tp_to_fn = 0
    benign_fp_to_tn = 0
    benign_tn_to_fp = 0
    n_flips = 0

    y_true = np.asarray(labels, dtype=int)
    final_preds = []

    for i in range(n):
        sid = str(sample_ids[i])
        y = int(labels[i])
        s_l1 = float(l1_scores[i])
        d_l1 = int(s_l1 >= threshold)

        m_mean = float(mc_means[i])
        m_var = float(mc_vars[i])
        m_ent = calculate_entropy(m_mean)

        rt = str(routes[i])
        rt_rsn = str(route_reasons[i])
        is_deep = (rt == "deep_inspection")

        if is_deep:
            n_deep += 1
            if y == 1:
                n_deep_fraud += 1
            else:
                n_deep_benign += 1

            s_l2 = float(l2_scores[i])
            d_l2 = int(s_l2 >= threshold)
            s_fus = float(fusion_scores[i])
            d_fus = int(s_fus >= threshold)

            s_final = s_fus
            d_final = d_fus

            # Flip analysis (L1 vs Fusion)
            if d_l1 != d_fus:
                n_flips += 1
                if d_fus == y:
                    wrong_to_correct += 1
                else:
                    correct_to_wrong += 1

                if y == 1:
                    if d_l1 == 0 and d_fus == 1:
                        fraud_fn_to_tp += 1
                    elif d_l1 == 1 and d_fus == 0:
                        fraud_tp_to_fn += 1
                else:
                    if d_l1 == 1 and d_fus == 0:
                        benign_fp_to_tn += 1
                    elif d_l1 == 0 and d_fus == 1:
                        benign_tn_to_fp += 1
        else:
            n_direct += 1
            if y == 1:
                n_direct_fraud += 1
            else:
                n_direct_benign += 1

            s_l2 = None
            d_l2 = None
            s_fus = None
            d_fus = None

            # Fast path exit uses MC mean score (or L1 score)
            s_final = m_mean
            d_final = int(s_final >= threshold)

        final_preds.append(d_final)

        traces.append(
            SampleRoutingTrace(
                sample_id=sid,
                chain=chain,
                seed=seed,
                policy=policy,
                label=y,
                l1_score=s_l1,
                l1_decision=d_l1,
                mc_mean=m_mean,
                mc_variance=m_var,
                mc_entropy=m_ent,
                mc_samples=num_mc_samples,
                route=rt,
                route_reason=rt_rsn,
                l2_score=s_l2,
                l2_decision=d_l2,
                fusion_score=s_fus,
                fusion_decision=d_fus,
                final_score=s_final,
                final_decision=d_final,
            )
        )

    final_preds_arr = np.asarray(final_preds, dtype=int)
    positives = int(np.sum(y_true == 1))
    tps = int(np.sum((final_preds_arr == 1) & (y_true == 1)))
    fps = int(np.sum((final_preds_arr == 1) & (y_true == 0)))
    fns = int(np.sum((final_preds_arr == 0) & (y_true == 1)))

    direct_fraud_missed = 0
    for tr in traces:
        if tr.route != "deep_inspection" and tr.label == 1 and tr.final_decision == 0:
            direct_fraud_missed += 1

    direct_fnr = (direct_fraud_missed / n_direct_fraud) if n_direct_fraud > 0 else 0.0
    overall_recall = (tps / positives) if positives > 0 else 0.0
    overall_prec = (tps / (tps + fps)) if (tps + fps) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_recall / (overall_prec + overall_recall)) if (overall_prec + overall_recall) > 0 else 0.0

    metrics = FlipMetrics(
        total_samples=n,
        n_direct=n_direct,
        n_deep=n_deep,
        n_direct_fraud=n_direct_fraud,
        n_direct_benign=n_direct_benign,
        n_deep_fraud=n_deep_fraud,
        n_deep_benign=n_deep_benign,
        n_flips=n_flips,
        flip_rate_deep=(n_flips / n_deep) if n_deep > 0 else 0.0,
        flip_rate_total=(n_flips / n) if n > 0 else 0.0,
        wrong_to_correct=wrong_to_correct,
        correct_to_wrong=correct_to_wrong,
        net_gain=(wrong_to_correct - correct_to_wrong),
        fraud_fn_to_tp=fraud_fn_to_tp,
        fraud_tp_to_fn=fraud_tp_to_fn,
        benign_fp_to_tn=benign_fp_to_tn,
        benign_tn_to_fp=benign_tn_to_fp,
        direct_exit_fnr=direct_fnr,
        overall_recall=overall_recall,
        overall_precision=overall_prec,
        overall_f1=overall_f1,
        deep_route_rate=(n_deep / n) if n > 0 else 0.0,
        direct_exit_rate=(n_direct / n) if n > 0 else 0.0,
    )

    return traces, metrics


def traces_to_dataframe(traces: Sequence[SampleRoutingTrace]) -> pd.DataFrame:
    """Convert a list of SampleRoutingTrace into a pandas DataFrame."""
    return pd.DataFrame([asdict(t) for t in traces])
