"""
graphrag/__init__.py

Micro-GraphRAG module for on-device fraud risk extraction.

Pipeline:
    context_text
        ↓ query_extraction
        ↓ candidate_retrieval (cosine similarity on keyword embeddings)
        ↓ graph_expansion (BFS hop traversal on local KB)
        ↓ top-k evidence aggregation
        ↓ risk_extraction → r_t = [s_t, q_t, k_t, a_t, h_t]

Public API:
    from graphrag import LocalKnowledgeBase, GraphRAGRetriever, RiskExtractor, RiskEncoder
"""

from graphrag.local_kb import LocalKnowledgeBase
from graphrag.retriever import GraphRAGRetriever
from graphrag.risk_extractor import RiskExtractor
from graphrag.risk_encoder import RiskEncoder

__all__ = ["LocalKnowledgeBase", "GraphRAGRetriever", "RiskExtractor", "RiskEncoder"]
