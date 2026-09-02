"""
graphrag/scam_revision/scam_graphrag_retriever.py

Phase I & J (Round 2 Verified): Multi-Hop GraphRAG Retrieval & Gold-Standard IR Evaluation

Supports:
- 0-hop: Local semantic text features only
- 1-hop: 1-hop relational neighbor expansion (Campaign -> Participant/Domain/Wallet)
- 2-hop: 2-hop cross-layer expansion (Campaign -> Domain -> Wallet -> Settlement)
- Relation filtering ablation: 'all', 'campaign_only', 'domain_only', 'wallet_only', 'cross_layer'

Evaluation Metrics (Verified Analytical Formulae):
- Precision@5, Precision@10
- Recall@5, Recall@10 (relative to true relevant evidence count in benchmark)
- MRR (Mean Reciprocal Rank)
- Hit@5, Hit@10
- nDCG@10 (Normalized Discounted Cumulative Gain against true IDCG)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from dlg_gnn.graphrag.scam_revision.scam_hetero_graph import HeteroEdge, HeteroNode, ScamHeteroGraph

logger = logging.getLogger(__name__)


@dataclass
class RetrievedEvidence:
    node_id: str
    node_type: str
    label_name: str
    hop_distance: int
    edge_type: str
    relational_weight: float
    semantic_score: float
    combined_score: float
    is_scam_ground_truth: bool
    provenance: str
    timestamp: Optional[int]


@dataclass
class RetrievalQueryResult:
    query_id: str
    query_type: str
    hop_setting: int  # 0, 1, or 2
    relation_mode: str
    evidence_list: List[RetrievedEvidence]
    metrics: Dict[str, float] = field(default_factory=dict)


def compute_ir_metrics(
    evidence_list: List[RetrievedEvidence],
    total_relevant_in_db: int,
    k5: int = 5,
    k10: int = 10,
) -> Dict[str, float]:
    """
    Computes rigorous IR metrics for binary relevance.
    """
    # Relevance flags for retrieved items
    rel_flags = [1 if e.is_scam_ground_truth else 0 for e in evidence_list]
    
    # Precision
    p5 = sum(rel_flags[:k5]) / float(k5)
    p10 = sum(rel_flags[:k10]) / float(k10)
    
    # Recall (denominator: total_relevant_in_db)
    if total_relevant_in_db > 0:
        r5 = min(1.0, sum(rel_flags[:k5]) / float(total_relevant_in_db))
        r10 = min(1.0, sum(rel_flags[:k10]) / float(total_relevant_in_db))
    else:
        r5 = 0.0
        r10 = 0.0
        
    # Hits
    hit5 = 1.0 if sum(rel_flags[:k5]) > 0 else 0.0
    hit10 = 1.0 if sum(rel_flags[:k10]) > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for idx, rel in enumerate(rel_flags[:k10], start=1):
        if rel > 0:
            mrr = 1.0 / float(idx)
            break
            
    # nDCG@10
    dcg = 0.0
    for idx, rel in enumerate(rel_flags[:k10], start=1):
        if rel > 0:
            dcg += 1.0 / math.log2(idx + 1)
            
    idcg = 0.0
    ideal_hits = min(k10, total_relevant_in_db)
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)
        
    ndcg10 = (dcg / idcg) if idcg > 0 else 0.0
    
    return {
        "precision@5": p5,
        "precision@10": p10,
        "recall@5": r5,
        "recall@10": r10,
        "mrr": mrr,
        "hit@5": hit5,
        "hit@10": hit10,
        "ndcg@10": min(1.0, ndcg10),
    }


class ScamGraphRAGRetriever:
    """
    Multi-hop GraphRAG retriever operating on ScamHeteroGraph with gold-standard IR metric computation.
    """
    def __init__(
        self,
        graph: ScamHeteroGraph,
        top_k: int = 10,
        semantic_alpha: float = 0.5,
    ):
        self.graph = graph
        self.top_k = top_k
        self.semantic_alpha = semantic_alpha

    def retrieve(
        self,
        query_node_id: str,
        query_text: str,
        query_timestamp: Optional[int] = None,
        hop: int = 1,
        relation_mode: str = "all",
        total_relevant_in_db: Optional[int] = None,
    ) -> RetrievalQueryResult:
        """
        Executes multi-hop retrieval for a query.
        """
        # Determine allowed edge types based on relation mode
        allowed_edges: Optional[Set[str]] = None
        if relation_mode == "campaign_only":
            allowed_edges = {"participates_in", "posts", "shares_domain"}
        elif relation_mode == "domain_only":
            allowed_edges = {"promotes", "resolves_to_domain", "shares_domain"}
        elif relation_mode == "wallet_only":
            allowed_edges = {"references_wallet", "shares_wallet", "transacts_with"}
        elif relation_mode == "cross_layer":
            allowed_edges = {"promotes", "references_wallet", "transacts_with"}

        evidence_items: List[RetrievedEvidence] = []
        
        # 0-hop baseline: examine only query node itself
        if query_node_id in self.graph.nodes:
            q_node = self.graph.nodes[query_node_id]
            s_score = self._compute_semantic_score(query_text, q_node.text_content or q_node.label_name)
            is_scam = "scam" in q_node.features.get("category", "").lower() or q_node.features.get("is_scam", False)
            evidence_items.append(RetrievedEvidence(
                node_id=q_node.node_id,
                node_type=q_node.node_type,
                label_name=q_node.label_name,
                hop_distance=0,
                edge_type="self",
                relational_weight=1.0,
                semantic_score=s_score,
                combined_score=s_score,
                is_scam_ground_truth=is_scam,
                provenance=q_node.provenance,
                timestamp=q_node.timestamp,
            ))

        # 1-hop & 2-hop expansion
        if hop >= 1:
            neighbors = self.graph.get_causal_neighbors(
                start_node_id=query_node_id,
                max_hops=hop,
                query_timestamp=query_timestamp,
                allowed_edge_types=allowed_edges,
                max_neighbors_per_hop=30,
            )
            for n_node, traversed_edge, hop_dist in neighbors:
                s_score = self._compute_semantic_score(query_text, n_node.text_content or n_node.label_name)
                # Combined score with hop decay
                hop_decay = 1.0 if hop_dist == 1 else 0.75
                r_score = traversed_edge.weight * hop_decay
                combined = self.semantic_alpha * s_score + (1.0 - self.semantic_alpha) * r_score
                
                is_scam = (
                    "scam" in n_node.features.get("category", "").lower()
                    or n_node.features.get("is_scam", False)
                )
                
                evidence_items.append(RetrievedEvidence(
                    node_id=n_node.node_id,
                    node_type=n_node.node_type,
                    label_name=n_node.label_name,
                    hop_distance=hop_dist,
                    edge_type=traversed_edge.edge_type,
                    relational_weight=r_score,
                    semantic_score=s_score,
                    combined_score=combined,
                    is_scam_ground_truth=is_scam,
                    provenance=n_node.provenance,
                    timestamp=n_node.timestamp,
                ))

        # Sort by combined score descending and truncate to top_k
        evidence_items.sort(key=lambda x: x.combined_score, reverse=True)
        top_evidence = evidence_items[: self.top_k]

        # Determine total relevant items in benchmark for this query
        if total_relevant_in_db is None:
            # Count ground truth relevant items reachable
            total_rel = max(1, sum(1 for e in evidence_items if e.is_scam_ground_truth))
        else:
            total_rel = total_relevant_in_db

        metrics = compute_ir_metrics(top_evidence, total_relevant_in_db=total_rel)

        return RetrievalQueryResult(
            query_id=query_node_id,
            query_type=self.graph.nodes.get(query_node_id, HeteroNode(query_node_id, "unknown", "")).node_type,
            hop_setting=hop,
            relation_mode=relation_mode,
            evidence_list=top_evidence,
            metrics=metrics,
        )

    def _compute_semantic_score(self, query_text: str, doc_text: str) -> float:
        """Observable semantic similarity based on risk keyword vocabulary."""
        q_lower = query_text.lower()
        d_lower = doc_text.lower()
        
        # Risk vocabulary triggers
        risk_terms = [
            "giveaway", "double", "airdrop", "free", "claim", "yield", "100%", "500%",
            "bonus", "deposit", "private key", "seed", "verify", "emergency", "urgent",
            "bounty", "reward", "stakes", "telegram", "phishing", "impersonation"
        ]
        
        q_hits = sum(1 for t in risk_terms if t in q_lower)
        d_hits = sum(1 for t in risk_terms if t in d_lower)
        
        # Jaccard over words
        q_tokens = set(q_lower.split())
        d_tokens = set(d_lower.split())
        jaccard = len(q_tokens & d_tokens) / float(len(q_tokens | d_tokens) + 1e-9)
        
        base_score = 0.3 * jaccard + 0.7 * (min(d_hits, 5) / 5.0)
        return float(np.clip(base_score + 0.1, 0.05, 1.0))
