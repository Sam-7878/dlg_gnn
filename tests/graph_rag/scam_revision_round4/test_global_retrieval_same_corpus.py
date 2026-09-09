import pandas as pd

from graphrag.scam_revision.round4_final_evidence import validate_same_corpus_retrieval


METHODS = [
    "BM25", "TF-IDF cosine", "LSA dense", "Hybrid lexical+dense",
    "GraphRAG 1-hop", "GraphRAG 2-hop", "Relation-filtered GraphRAG",
]


def test_all_methods_share_corpus_and_fixed_gold():
    frame = pd.DataFrame([{
        "query_id": "q1", "method": method, "candidate_corpus_sha256": "corpus",
        "relevance_set_sha256": "gold", "primary_metric": "nDCG@10",
        "n_candidates": 100, "ranking_uses_gold_membership": False,
    } for method in METHODS])
    assert validate_same_corpus_retrieval(frame)["pass"]
