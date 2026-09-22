"""Expand GraphRAG retrieval evidence without promoting it to the SCI main track."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round3.run_retrieval_quality import SCENARIO_NODE
from experiments.round4.artifact_paths import RESULTS_DIR, ensure_dirs
from graphrag.local_kb import LocalKnowledgeBase
from graphrag.retriever import GraphRAGRetriever, RetrieverConfig


def evaluate(rows: list[dict], hops: int, top_k: int = 5) -> dict:
    retriever = GraphRAGRetriever(LocalKnowledgeBase(), RetrieverConfig(top_k=top_k, graph_hops=hops))
    precision = []; recall = []; reciprocal = []; covered = []
    for row in rows:
        evidence = retriever.retrieve(row["context_text"])
        ids = [item.node_id for item in evidence]
        expected = SCENARIO_NODE[row["scenario_type"]]
        rank = ids.index(expected) + 1 if expected in ids else 0
        precision.append(1 / len(ids) if rank and ids else 0.0)
        recall.append(float(rank > 0)); reciprocal.append(1 / rank if rank else 0.0)
        covered.append(float(bool(ids)))
    return {
        "n_queries": len(rows), "precision_at_k": np.mean(precision),
        "recall_at_k": np.mean(recall), "mrr": np.mean(reciprocal),
        "hit_at_k": np.mean(recall), "coverage": np.mean(covered),
    }


def main() -> int:
    ensure_dirs()
    source = ROOT / "data/benchmark/gog_microrag_stream_v1/contexts.jsonl"
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    rows = [row for row in rows if row.get("scenario_type") in SCENARIO_NODE]
    output = []
    for hops, method in ((0, "Keyword Top-k (0-hop)"), (1, "Top-k + 1-hop graph"), (2, "Top-k + 2-hop graph")):
        output.append({
            "track": "Controlled Context-Augmentation Study", "method": method,
            "graph_hops": hops, "top_k": 5, **evaluate(rows, hops),
            "context_policy": "label-conditioned context", "paper_eligible": False,
        })
    frame = pd.DataFrame(output)
    frame.to_csv(RESULTS_DIR / "retrieval_quality_expanded_controlled.csv", index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
