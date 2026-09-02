"""
graphrag/scam_revision/scam_hetero_graph.py

Phase F & G: Scam Campaign Heterogeneous Graph & Causal Temporal Grounding

Builds a heterogeneous graph connecting:
Nodes:
- User (CCC participants)
- Campaign (CCC bounty/promotional events)
- Content (Comments/posts)
- URL / Domain (Scam domains & campaign URLs)
- Wallet (Scam deposit addresses & participant payout addresses)
- OnchainAddress (GoG on-chain contracts/settlement entities)

Edges:
- participates_in (User -> Campaign)
- posts (User -> Content)
- promotes (Campaign -> URL/Domain)
- contains_url (Content -> URL)
- resolves_to_domain (URL -> Domain)
- references_wallet (Content/Campaign/Domain -> Wallet)
- linked_to_scam (Domain/Wallet -> ScamIntelligence)
- shares_wallet (User/Campaign <-> User/Campaign)
- shares_domain (Campaign <-> Campaign)
- transacts_with (Wallet <-> OnchainAddress)

Strict Causal Protocol:
- Filters graph by timestamp <= query_time.
- Excludes future post-detection scam reports from input feature retrieval.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeteroNode:
    node_id: str
    node_type: str  # "User", "Campaign", "Content", "Domain", "Wallet", "OnchainAddress"
    label_name: str
    timestamp: Optional[int] = None  # Unix epoch seconds
    provenance: str = "dataset_native"  # "dataset_native", "exact_bridge", "derived"
    features: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""


@dataclass
class HeteroEdge:
    src_id: str
    dst_id: str
    edge_type: str
    timestamp: Optional[int] = None
    weight: float = 1.0
    provenance: str = "dataset_native"
    tier: str = "Tier1_Exact"  # "Tier1_Exact", "Tier2_CrossLayer", "Tier3_MultiSource", "Tier4_Semantic"


class ScamHeteroGraph:
    """
    In-memory / NetworkX-backed timestamp-grounded Heterogeneous Graph for Scam Detection.
    """
    def __init__(self):
        self.nodes: Dict[str, HeteroNode] = {}
        self.edges: List[HeteroEdge] = []
        self.adj_out: Dict[str, List[HeteroEdge]] = defaultdict(list)
        self.adj_in: Dict[str, List[HeteroEdge]] = defaultdict(list)
        self.type_to_nodes: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: HeteroNode) -> None:
        self.nodes[node.node_id] = node
        self.type_to_nodes[node.node_type].add(node.node_id)

    def add_edge(self, edge: HeteroEdge) -> None:
        self.edges.append(edge)
        self.adj_out[edge.src_id].append(edge)
        self.adj_in[edge.dst_id].append(edge)

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def get_causal_neighbors(
        self,
        start_node_id: str,
        max_hops: int = 1,
        query_timestamp: Optional[int] = None,
        allowed_edge_types: Optional[Set[str]] = None,
        max_neighbors_per_hop: int = 50,
    ) -> List[Tuple[HeteroNode, HeteroEdge, int]]:
        """
        Performs multi-hop BFS expansion with strict causal timestamp constraint (edge/node timestamp <= query_timestamp).
        Returns a list of (target_node, traversed_edge, hop_distance).
        """
        if start_node_id not in self.nodes:
            return []

        visited_nodes: Set[str] = {start_node_id}
        results: List[Tuple[HeteroNode, HeteroEdge, int]] = []
        
        # Queue entries: (current_node_id, current_hop)
        queue: deque[Tuple[str, int]] = deque([(start_node_id, 0)])

        while queue:
            curr_id, curr_hop = queue.popleft()
            if curr_hop >= max_hops:
                continue

            # Check outgoing and incoming edges for relational connectivity
            candidate_edges = self.adj_out[curr_id] + self.adj_in[curr_id]
            valid_edges_for_hop = 0

            for edge in candidate_edges:
                if valid_edges_for_hop >= max_neighbors_per_hop:
                    break

                # Filter edge type if specified
                if allowed_edge_types is not None and edge.edge_type not in allowed_edge_types:
                    continue

                # Causal timestamp check on edge
                if query_timestamp is not None and edge.timestamp is not None:
                    if edge.timestamp > query_timestamp:
                        continue  # Future edge discarded to prevent temporal leakage

                # Target node
                neighbor_id = edge.dst_id if edge.src_id == curr_id else edge.src_id
                if neighbor_id not in self.nodes:
                    continue

                neighbor_node = self.nodes[neighbor_id]

                # Causal timestamp check on node
                if query_timestamp is not None and neighbor_node.timestamp is not None:
                    if neighbor_node.timestamp > query_timestamp:
                        continue  # Future node discarded

                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    results.append((neighbor_node, edge, curr_hop + 1))
                    queue.append((neighbor_id, curr_hop + 1))
                    valid_edges_for_hop += 1

        return results

    def to_summary_dict(self) -> Dict[str, Any]:
        node_counts = {nt: len(nlist) for nt, nlist in self.type_to_nodes.items()}
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            edge_type_counts[e.edge_type] += 1
        return {
            "total_nodes": self.num_nodes(),
            "total_edges": self.num_edges(),
            "node_type_breakdown": node_counts,
            "edge_type_breakdown": dict(edge_type_counts),
        }
