"""Measure controlled GraphRAG retrieval quality with val-only selection.

The benchmark's synthetic fraud scenarios map one-to-one to ScamType nodes in
the local knowledge graph.  This permits auditable Precision@k, Recall@k, MRR,
and Hit@k measurement, but only for the controlled context track.
"""
from __future__ import annotations

import csv
import itertools
import json
from datetime import datetime, timezone

import numpy as np

from experiments.round3.artifact_paths import ROUND3_REPORTS, ROUND3_RESULTS
from graphrag.local_kb import LocalKnowledgeBase
from graphrag.retriever import GraphRAGRetriever, RetrieverConfig


DATA_DIR = ROUND3_RESULTS.parents[2] / "data" / "benchmark" / "gog_microrag_stream_v1"
SCENARIO_NODE = {
    "high_yield_guaranteed_return_scam": "ST_high_yield",
    "multi_stage_grooming_scam": "ST_grooming",
    "crypto_wallet_migration_scam": "ST_migration",
    "fake_customer_support": "ST_fake_support",
    "urgent_transfer_request": "ST_urgent",
    "investment_scam": "ST_investment",
    "romance_scam": "ST_romance",
    "impersonation_scam": "ST_impersonation",
    "phishing_url_scam": "ST_phishing",
    "recovery_phrase_stealing_attempt": "ST_recovery",
}


def _ids(name: str) -> set[int]:
    return {int(value) for value in (DATA_DIR / f"{name}_ids.txt").read_text().split()}


def _contexts(ids: set[int]) -> list[dict]:
    rows = []
    with (DATA_DIR / "contexts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            node_id = int(row["event_id"].split("_")[1])
            if node_id in ids and row.get("scenario_type") in SCENARIO_NODE:
                rows.append(row)
    return rows


def evaluate(rows: list[dict], *, top_k: int, graph_hops: int, threshold: float) -> dict:
    retriever = GraphRAGRetriever(
        LocalKnowledgeBase(),
        RetrieverConfig(
            top_k=top_k,
            graph_hops=graph_hops,
            similarity_threshold=threshold,
        ),
    )
    precision, recall, reciprocal_rank = [], [], []
    for row in rows:
        expected = SCENARIO_NODE[row["scenario_type"]]
        evidence = retriever.retrieve(row["context_text"])
        retrieved = [item.node_id for item in evidence[:top_k]]
        rank = retrieved.index(expected) + 1 if expected in retrieved else 0
        precision.append((1.0 / len(retrieved)) if rank else 0.0)
        recall.append(float(bool(rank)))
        reciprocal_rank.append((1.0 / rank) if rank else 0.0)
    return {
        "n_queries": len(rows),
        "precision_at_k": float(np.mean(precision)) if precision else 0.0,
        "recall_at_k": float(np.mean(recall)) if recall else 0.0,
        "mrr": float(np.mean(reciprocal_rank)) if reciprocal_rank else 0.0,
        "hit_at_k": float(np.mean(recall)) if recall else 0.0,
    }


def main() -> None:
    ROUND3_RESULTS.mkdir(parents=True, exist_ok=True)
    ROUND3_REPORTS.mkdir(parents=True, exist_ok=True)
    validation = _contexts(_ids("valid"))
    test = _contexts(_ids("test"))
    candidates = []
    for top_k, hops, threshold in itertools.product((3, 5, 10), (0, 1, 2), (0.0, 0.2, 0.4)):
        row = {
            "split": "validation",
            "top_k": top_k,
            "graph_hops": hops,
            "similarity_threshold": threshold,
            **evaluate(validation, top_k=top_k, graph_hops=hops, threshold=threshold),
        }
        candidates.append(row)
    best = max(
        candidates,
        key=lambda row: (row["mrr"], row["recall_at_k"], row["precision_at_k"], -row["graph_hops"]),
    )
    selected_test = {
        "split": "test_selected_on_validation",
        "top_k": best["top_k"],
        "graph_hops": best["graph_hops"],
        "similarity_threshold": best["similarity_threshold"],
        **evaluate(
            test,
            top_k=best["top_k"],
            graph_hops=best["graph_hops"],
            threshold=best["similarity_threshold"],
        ),
    }
    keyword_test = {
        "split": "test_keyword_only",
        "top_k": best["top_k"],
        "graph_hops": 0,
        "similarity_threshold": best["similarity_threshold"],
        **evaluate(test, top_k=best["top_k"], graph_hops=0, threshold=best["similarity_threshold"]),
    }
    rows = candidates + [selected_test, keyword_test]
    out = ROUND3_RESULTS / "retrieval_quality.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    graph_gain = selected_test["mrr"] - keyword_test["mrr"]
    report = f"""# GraphRAG Retrieval Quality and Failure Analysis — Round 3

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Evaluation scope:** controlled, label-conditioned synthetic context only (not paper-ready detector evidence)

## Val-only selection

The search covered `top_k={{3,5,10}}`, `graph_hops={{0,1,2}}`, and
`similarity_threshold={{0.0,0.2,0.4}}`. Selection used validation MRR, Recall@k,
and Precision@k in that order. Test labels were not used for selection.

| Setting | top_k | hops | threshold | Precision@k | Recall@k | MRR | Hit@k | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val-selected/test | {best['top_k']} | {best['graph_hops']} | {best['similarity_threshold']} | {selected_test['precision_at_k']:.4f} | {selected_test['recall_at_k']:.4f} | {selected_test['mrr']:.4f} | {selected_test['hit_at_k']:.4f} | {selected_test['n_queries']} |
| Keyword-only/test | {best['top_k']} | 0 | {best['similarity_threshold']} | {keyword_test['precision_at_k']:.4f} | {keyword_test['recall_at_k']:.4f} | {keyword_test['mrr']:.4f} | {keyword_test['hit_at_k']:.4f} | {keyword_test['n_queries']} |

## Finding

Graph expansion test MRR delta versus keyword-only: **{graph_gain:+.4f}**.
The supervised TF-IDF score of 1.0 in the controlled dataset is a label-conditioned
context shortcut and must not be interpreted as real held-out transaction performance.
GraphRAG may be described as relation-aware evidence traversal, but its performance
novelty must not be claimed unless the measured graph-expansion delta is positive and
the result is replicated on label-independent context.
"""
    (ROUND3_REPORTS / "graphrag_failure_analysis.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
