"""
tests/scam_revision/test_retrieval_metrics_gold_cases.py

Phase 2: Hand-calculated Gold Unit Tests for IR Ranking Metrics
Verifies Precision@k, Recall@k, MRR, Hit@k, and nDCG@k against 20 analytical gold ranking cases.
"""

import math
import pytest
from typing import List, Dict


def compute_gold_ir_metrics(
    retrieved_relevance: List[int],
    total_relevant_in_db: int,
    k: int = 10,
) -> Dict[str, float]:
    """
    Standard Information Retrieval metric computation for binary relevance.
    """
    k_items = retrieved_relevance[:k]
    num_retrieved = len(k_items)
    
    # 1. Precision@k (denominator is k)
    p_k = sum(k_items) / float(k)
    
    # 2. Recall@k (denominator is total_relevant_in_db)
    if total_relevant_in_db > 0:
        r_k = sum(k_items) / float(total_relevant_in_db)
    else:
        r_k = 0.0
        
    # 3. Hit@k
    hit_k = 1.0 if sum(k_items) > 0 else 0.0
    
    # 4. MRR
    mrr = 0.0
    for idx, rel in enumerate(k_items, start=1):
        if rel > 0:
            mrr = 1.0 / float(idx)
            break
            
    # 5. nDCG@k
    dcg = 0.0
    for idx, rel in enumerate(k_items, start=1):
        if rel > 0:
            dcg += 1.0 / math.log2(idx + 1)
            
    idcg = 0.0
    ideal_hits = min(k, total_relevant_in_db)
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)
        
    ndcg_k = (dcg / idcg) if idcg > 0 else 0.0
    
    return {
        f"precision@{k}": p_k,
        f"recall@{k}": min(r_k, 1.0),
        f"hit@{k}": hit_k,
        "mrr": mrr,
        f"ndcg@{k}": min(ndcg_k, 1.0),
    }


# ── 20 Analytical Gold Test Cases ──────────────────────────────────────────

@pytest.mark.parametrize(
    "case_id, retrieved, total_rel, k, expected_p, expected_r, expected_hit, expected_mrr, expected_ndcg",
    [
        # Case 1: Perfect retrieval (first 3 relevant out of 3 total)
        (1, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 3, 5, 3/5, 1.0, 1.0, 1.0, 1.0),
        # Case 2: Zero relevant items exist
        (2, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 5, 0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 3: Single relevant item at rank 1 (total 1)
        (3, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 5, 1/5, 1.0, 1.0, 1.0, 1.0),
        # Case 4: Single relevant item at rank 2 (total 1)
        (4, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 1, 5, 1/5, 1.0, 1.0, 1/2, (1/math.log2(3))/(1/math.log2(2))),
        # Case 5: Single relevant item at rank 3 (total 1)
        (5, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 1, 5, 1/5, 1.0, 1.0, 1/3, (1/math.log2(4))/(1/math.log2(2))),
        # Case 6: Single relevant item at rank 4 (total 1)
        (6, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 1, 5, 1/5, 1.0, 1.0, 1/4, (1/math.log2(5))/(1/math.log2(2))),
        # Case 7: Single relevant item at rank 5 (total 1)
        (7, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 1, 5, 1/5, 1.0, 1.0, 1/5, (1/math.log2(6))/(1/math.log2(2))),
        # Case 8: Relevant item at rank 6 (outside top-5)
        (8, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 1, 5, 0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 9: Top-10 check with hit at rank 6
        (9, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 1, 10, 1/10, 1.0, 1.0, 1/6, (1/math.log2(7))/(1/math.log2(2))),
        # Case 10: 2 items relevant, found at ranks 1 and 2 (total 4)
        (10, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 4, 5, 2/5, 2/4, 1.0, 1.0, (1/math.log2(2) + 1/math.log2(3))/(1/math.log2(2) + 1/math.log2(3) + 1/math.log2(4) + 1/math.log2(5))),
        # Case 11: 2 items relevant, found at ranks 2 and 4 (total 2)
        (11, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 2, 5, 2/5, 1.0, 1.0, 1/2, (1/math.log2(3) + 1/math.log2(5))/(1/math.log2(2) + 1/math.log2(3))),
        # Case 12: All 5 items relevant (total 5)
        (12, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 5, 5, 1.0, 1.0, 1.0, 1.0, 1.0),
        # Case 13: 5 relevant items found out of 10 total in DB
        (13, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 10, 5, 1.0, 5/10, 1.0, 1.0, 1.0),
        # Case 14: Alternating relevance [1, 0, 1, 0, 1] (total 3)
        (14, [1, 0, 1, 0, 1, 0, 0, 0, 0, 0], 3, 5, 3/5, 1.0, 1.0, 1.0, (1/math.log2(2) + 1/math.log2(4) + 1/math.log2(6))/(1/math.log2(2) + 1/math.log2(3) + 1/math.log2(4))),
        # Case 15: Worst order [0, 0, 1, 1, 1] (total 3)
        (15, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0], 3, 5, 3/5, 1.0, 1.0, 1/3, (1/math.log2(4) + 1/math.log2(5) + 1/math.log2(6))/(1/math.log2(2) + 1/math.log2(3) + 1/math.log2(4))),
        # Case 16: Empty retrieved list
        (16, [], 5, 5, 0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 17: Partial relevance with k=10
        (17, [0, 1, 0, 0, 1, 0, 0, 0, 1, 0], 5, 10, 3/10, 3/5, 1.0, 1/2, (1/math.log2(3) + 1/math.log2(6) + 1/math.log2(10))/(sum(1/math.log2(i+1) for i in range(1, 6)))),
        # Case 18: No relevance in top 10 but total_rel = 3
        (18, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, 10, 0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 19: Single hit at last position rank 10
        (19, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 1, 10, 1/10, 1.0, 1.0, 1/10, (1/math.log2(11))/(1/math.log2(2))),
        # Case 20: 10 out of 10 relevant
        (20, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10, 10, 1.0, 1.0, 1.0, 1.0, 1.0),
    ]
)
def test_gold_retrieval_cases(case_id, retrieved, total_rel, k, expected_p, expected_r, expected_hit, expected_mrr, expected_ndcg):
    metrics = compute_gold_ir_metrics(retrieved, total_rel, k=k)
    assert pytest.approx(metrics[f"precision@{k}"], abs=1e-4) == expected_p, f"Case {case_id} Precision failed"
    assert pytest.approx(metrics[f"recall@{k}"], abs=1e-4) == expected_r, f"Case {case_id} Recall failed"
    assert pytest.approx(metrics[f"hit@{k}"], abs=1e-4) == expected_hit, f"Case {case_id} Hit failed"
    assert pytest.approx(metrics["mrr"], abs=1e-4) == expected_mrr, f"Case {case_id} MRR failed"
    assert pytest.approx(metrics[f"ndcg@{k}"], abs=1e-4) == expected_ndcg, f"Case {case_id} nDCG failed"
