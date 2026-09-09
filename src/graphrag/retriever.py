"""
graphrag/retriever.py

Micro-GraphRAG retrieval pipeline.

Pipeline:
    context_text
        ↓  query_extraction      (keyword tokenization)
        ↓  candidate_retrieval   (keyword overlap score vs KB node keywords)
        ↓  graph_expansion       (BFS neighbor traversal, graph_hops)
        ↓  evidence_aggregation  (top-k by combined score)
        → List[EvidenceItem]

Config parameters (all externalized to configs/base.yaml):
    retrieval.top_k              : int   = 5
    retrieval.graph_hops         : int   = 1
    retrieval.similarity_threshold: float = 0.0  (0 = no filtering)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from graphrag.local_kb import LocalKnowledgeBase

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrieverConfig:
    top_k: int = 5
    graph_hops: int = 1
    similarity_threshold: float = 0.0


@dataclass
class EvidenceItem:
    node_id: str
    node_type: str
    node_label: str
    score: float                          # retrieval relevance score ∈ [0, 1]
    matched_keywords: List[str] = field(default_factory=list)
    expansion_depth: int = 0              # 0 = direct match, 1+ = neighbor


# ══════════════════════════════════════════════════════════════════════════════
# Query extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "and", "or", "for",
    "of", "with", "my", "your", "our", "their", "i", "we", "you", "they", "he",
    "she", "this", "that", "will", "can", "please", "me", "am", "are", "was",
    "be", "by", "as", "from", "not", "do", "if", "so", "but", "have", "has",
    "been", "were", "which", "its", "also", "just", "all", "more", "into",
})


def extract_query_terms(text: str) -> List[str]:
    """
    Tokenize context text into meaningful query terms.
    Returns lowercased unigrams and selected bigrams, stop-words removed.
    """
    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text_lower)
    unigrams = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    # Build bigrams for multi-word cue matching
    bigrams = [f"{unigrams[i]} {unigrams[i+1]}" for i in range(len(unigrams) - 1)]

    return unigrams + bigrams


# ══════════════════════════════════════════════════════════════════════════════
# Retriever
# ══════════════════════════════════════════════════════════════════════════════

class GraphRAGRetriever:
    """
    Retrieves relevant fraud knowledge graph nodes for a given context text.

    Algorithm:
    1. Extract query terms from context_text (keyword tokenization).
    2. Score each KB node by keyword overlap ratio with query terms.
    3. Expand top candidates via BFS (graph_hops) on the knowledge graph.
    4. Re-score expanded neighbors (depth-discounted).
    5. Return top-k EvidenceItems sorted by score descending.

    This constitutes the "graph relation retrieval" required for GraphRAG —
    the retrieval uses graph traversal (step 3) to incorporate relational
    context beyond direct keyword matches.
    """

    def __init__(self, kb: LocalKnowledgeBase, config: Optional[RetrieverConfig] = None):
        self.kb = kb
        self.cfg = config or RetrieverConfig()
        # Pre-build keyword index: keyword → list of node_ids
        self._kw_index: Dict[str, List[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for node_id in self.kb.nodes:
            for kw in self.kb.get_node_keywords(node_id):
                key = kw.lower()
                self._kw_index.setdefault(key, []).append(node_id)

    def _node_overlap_score(self, node_id: str, query_terms: List[str]) -> tuple[float, List[str]]:
        """Compute keyword overlap score for a KB node against query terms."""
        node_keywords = [kw.lower() for kw in self.kb.get_node_keywords(node_id)]
        if not node_keywords:
            return 0.0, []
        query_set = set(query_terms)
        matched = [kw for kw in node_keywords if kw in query_set]
        # Jaccard-like overlap weighted by node risk_weight
        overlap = len(matched) / max(len(node_keywords), len(query_set), 1)
        score = overlap * self.kb.get_node_risk_weight(node_id)
        return float(score), matched

    def retrieve(self, context_text: str) -> List[EvidenceItem]:
        """
        Run full Micro-GraphRAG retrieval pipeline for a context text.

        Returns:
            List[EvidenceItem] — top-k evidence items sorted by score descending.
        """
        # Step 1: Query extraction
        query_terms = extract_query_terms(context_text)
        if not query_terms:
            return []

        # Step 2: Score all KB nodes (direct match)
        node_scores: Dict[str, float] = {}
        node_matched: Dict[str, List[str]] = {}
        for node_id in self.kb.nodes:
            score, matched = self._node_overlap_score(node_id, query_terms)
            if score >= self.cfg.similarity_threshold:
                node_scores[node_id] = score
                node_matched[node_id] = matched

        # Step 3: Select seed nodes for graph expansion
        sorted_seeds = sorted(node_scores, key=node_scores.get, reverse=True)
        seeds = sorted_seeds[: max(self.cfg.top_k, 3)]

        # Step 4: Graph expansion via BFS (graph_hops)
        expanded: Dict[str, tuple[float, int]] = {}  # node_id → (score, depth)
        for seed in seeds:
            expanded[seed] = (node_scores.get(seed, 0.0), 0)
            neighbors = self.kb.get_neighbors(seed, hops=self.cfg.graph_hops)
            for nbr in neighbors:
                if nbr not in expanded:
                    # Depth-discounted score from seed
                    seed_score = node_scores.get(seed, 0.0)
                    nbr_direct, nbr_matched_kw = self._node_overlap_score(nbr, query_terms)
                    # Combine: direct match + relation boost from seed
                    combined = max(nbr_direct, seed_score * 0.5)
                    if combined >= self.cfg.similarity_threshold:
                        expanded[nbr] = (combined, 1)
                        if nbr not in node_matched:
                            node_matched[nbr] = nbr_matched_kw

        # Step 5: Build EvidenceItems, sort, truncate to top_k
        evidence = []
        for node_id, (score, depth) in expanded.items():
            evidence.append(EvidenceItem(
                node_id=node_id,
                node_type=self.kb.get_node_type(node_id),
                node_label=self.kb.graph.nodes[node_id].get("label", node_id),
                score=score,
                matched_keywords=node_matched.get(node_id, []),
                expansion_depth=depth,
            ))

        evidence.sort(key=lambda e: e.score, reverse=True)
        result = evidence[: self.cfg.top_k]

        logger.debug(
            f"GraphRAGRetriever: {len(query_terms)} query terms → "
            f"{len(expanded)} candidates → top-{self.cfg.top_k} evidence"
        )
        return result
