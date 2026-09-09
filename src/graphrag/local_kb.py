"""
graphrag/local_kb.py

Local fraud knowledge graph (in-memory, NetworkX-based).

Graph schema
────────────
Node types:
    ScamType            — top-level fraud category
    Cue                 — observable linguistic / behavioral signal
    Intent              — attacker's goal
    CounterpartyPattern — counterparty relationship pattern
    UrgencyPattern      — urgency expression pattern
    CredentialRequest   — credential / key / phrase solicitation
    InvestmentPattern   — investment / yield promise pattern
    ImpersonationPattern— identity spoofing pattern
    PhishingPattern     — link / URL based phishing pattern

Edge types:
    INDICATES           — Cue → ScamType
    RELATED_TO          — ScamType ↔ ScamType
    TARGETS             — ScamType → Intent
    PRECEDES            — Cue → Cue (temporal ordering)
    ASSOCIATED_WITH     — CounterpartyPattern → ScamType
    SIMILAR_TO          — pattern ↔ pattern (similarity)

Every node carries:
    type     : str   — one of the node types above
    keywords : list  — surface keyword triggers
    risk_weight: float — 0..1 severity weight (used in retrieval scoring)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Node / Edge dataclasses (for programmatic access)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KBNode:
    node_id: str
    node_type: str
    label: str
    keywords: List[str] = field(default_factory=list)
    risk_weight: float = 0.5
    description: str = ""


@dataclass
class KBEdge:
    src: str
    dst: str
    edge_type: str
    weight: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Graph definition
# ══════════════════════════════════════════════════════════════════════════════

def _build_default_graph() -> nx.DiGraph:
    """
    Build the default fraud knowledge graph with manually curated nodes and edges.
    This graph represents domain knowledge about financial social-engineering scams.
    """
    G = nx.DiGraph()

    # ── Scam type nodes ──────────────────────────────────────────────────────
    scam_types = [
        KBNode("ST_investment",   "ScamType", "Investment Scam",
               ["investment", "trading", "profit", "returns", "stake", "pool", "yield", "apy", "fund"],
               risk_weight=0.90),
        KBNode("ST_romance",      "ScamType", "Romance Scam",
               ["love", "relationship", "girlfriend", "boyfriend", "dearest", "miss you", "family"],
               risk_weight=0.85),
        KBNode("ST_phishing",     "ScamType", "Phishing Scam",
               ["click", "verify", "http", "link", "url", "website", "portal", "login", "account"],
               risk_weight=0.90),
        KBNode("ST_impersonation","ScamType", "Impersonation Scam",
               ["police", "bank", "official", "authority", "government", "support", "helpdesk", "team"],
               risk_weight=0.88),
        KBNode("ST_urgent",       "ScamType", "Urgent Transfer Scam",
               ["urgent", "immediately", "right now", "emergency", "asap", "quickly", "dying"],
               risk_weight=0.82),
        KBNode("ST_fake_support", "ScamType", "Fake Customer Support",
               ["support", "helpdesk", "technician", "service desk", "resolve", "fix", "stuck"],
               risk_weight=0.85),
        KBNode("ST_migration",    "ScamType", "Wallet Migration Scam",
               ["migrate", "migration", "upgrade", "hard fork", "bridge", "new contract", "old token"],
               risk_weight=0.88),
        KBNode("ST_recovery",     "ScamType", "Recovery Phrase Scam",
               ["seed", "recovery phrase", "mnemonic", "private key", "backup", "sync", "restore"],
               risk_weight=0.95),
        KBNode("ST_high_yield",   "ScamType", "High Yield Scam",
               ["guaranteed", "500%", "double", "triple", "compound", "auto", "passive"],
               risk_weight=0.92),
        KBNode("ST_grooming",     "ScamType", "Multi-stage Grooming Scam",
               ["chatting", "met online", "insider", "private", "exclusive", "group", "vip"],
               risk_weight=0.80),
    ]

    # ── Cue nodes ────────────────────────────────────────────────────────────
    cue_nodes = [
        KBNode("CUE_guaranteed_return", "Cue", "Guaranteed Return Promise",
               ["guaranteed", "certain", "100%", "fixed return", "safe profit"],
               risk_weight=0.92),
        KBNode("CUE_urgent_transfer",   "Cue", "Urgent Transfer Request",
               ["urgent", "immediately", "right now", "asap", "hurry", "deadline", "limited time"],
               risk_weight=0.80),
        KBNode("CUE_wallet_request",    "Cue", "External Wallet Address Request",
               ["send to", "transfer to", "deposit to", "wallet address", "0x"],
               risk_weight=0.75),
        KBNode("CUE_identity_verify",   "Cue", "Identity Verification Request",
               ["verify", "confirm identity", "account verification", "kyc", "authentication"],
               risk_weight=0.78),
        KBNode("CUE_escrow",            "Cue", "Escrow / Custody Request",
               ["escrow", "custody", "court order", "hold", "clearing", "settlement"],
               risk_weight=0.82),
        KBNode("CUE_seed_phrase",       "Cue", "Seed Phrase Solicitation",
               ["seed words", "recovery phrase", "12 words", "24 words", "mnemonic", "private key"],
               risk_weight=0.98),
        KBNode("CUE_url_link",          "Cue", "Suspicious URL / Link",
               ["http://", "https://", "click here", "visit", "portal", "website"],
               risk_weight=0.85),
        KBNode("CUE_authority",         "Cue", "Authority / Official Claim",
               ["official", "government", "police", "bank", "authority", "certified"],
               risk_weight=0.77),
    ]

    # ── Intent nodes ─────────────────────────────────────────────────────────
    intent_nodes = [
        KBNode("INT_steal_funds",    "Intent", "Steal Funds",        [], risk_weight=1.0),
        KBNode("INT_steal_keys",     "Intent", "Steal Private Keys", [], risk_weight=1.0),
        KBNode("INT_info_harvest",   "Intent", "Harvest Personal Info",[], risk_weight=0.9),
        KBNode("INT_false_trust",    "Intent", "Build False Trust",  [], risk_weight=0.7),
    ]

    # ── Pattern nodes ────────────────────────────────────────────────────────
    pattern_nodes = [
        KBNode("PAT_impersonation",  "ImpersonationPattern",  "Authority Impersonation",
               ["this is", "calling from", "department", "officer", "representative"],
               risk_weight=0.85),
        KBNode("PAT_urgency",        "UrgencyPattern",        "Urgency Creation",
               ["today only", "expires", "limited", "before it's too late", "last chance"],
               risk_weight=0.80),
        KBNode("PAT_credential_req", "CredentialRequest",     "Credential Solicitation",
               ["enter your", "provide your", "submit your", "send your password"],
               risk_weight=0.95),
        KBNode("PAT_investment",     "InvestmentPattern",     "High-Yield Investment Promise",
               ["risk-free", "no risk", "guaranteed profit", "join now", "exclusive opportunity"],
               risk_weight=0.90),
        KBNode("PAT_phishing",       "PhishingPattern",       "Phishing Link",
               ["click the link", "visit our website", "verify at", "confirm at"],
               risk_weight=0.88),
        KBNode("PAT_counterparty",   "CounterpartyPattern",   "Unknown Counterparty",
               ["stranger", "online contact", "new friend", "met online"],
               risk_weight=0.70),
    ]

    # ── Add all nodes ────────────────────────────────────────────────────────
    for node in scam_types + cue_nodes + intent_nodes + pattern_nodes:
        G.add_node(
            node.node_id,
            node_type=node.node_type,
            label=node.label,
            keywords=node.keywords,
            risk_weight=node.risk_weight,
            description=node.description,
        )

    # ── Edges: Cue → ScamType (INDICATES) ───────────────────────────────────
    indicates_edges = [
        ("CUE_guaranteed_return", "ST_investment"),
        ("CUE_guaranteed_return", "ST_high_yield"),
        ("CUE_urgent_transfer",   "ST_urgent"),
        ("CUE_urgent_transfer",   "ST_romance"),
        ("CUE_wallet_request",    "ST_investment"),
        ("CUE_wallet_request",    "ST_migration"),
        ("CUE_wallet_request",    "ST_romance"),
        ("CUE_identity_verify",   "ST_phishing"),
        ("CUE_identity_verify",   "ST_impersonation"),
        ("CUE_escrow",            "ST_impersonation"),
        ("CUE_escrow",            "ST_urgent"),
        ("CUE_seed_phrase",       "ST_recovery"),
        ("CUE_url_link",          "ST_phishing"),
        ("CUE_authority",         "ST_impersonation"),
        ("CUE_authority",         "ST_fake_support"),
    ]
    for src, dst in indicates_edges:
        G.add_edge(src, dst, edge_type="INDICATES", weight=1.0)

    # ── Edges: ScamType → Intent (TARGETS) ──────────────────────────────────
    targets_edges = [
        ("ST_investment",    "INT_steal_funds"),
        ("ST_romance",       "INT_steal_funds"),
        ("ST_romance",       "INT_false_trust"),
        ("ST_phishing",      "INT_steal_keys"),
        ("ST_phishing",      "INT_info_harvest"),
        ("ST_impersonation", "INT_steal_funds"),
        ("ST_impersonation", "INT_info_harvest"),
        ("ST_urgent",        "INT_steal_funds"),
        ("ST_fake_support",  "INT_steal_keys"),
        ("ST_migration",     "INT_steal_funds"),
        ("ST_recovery",      "INT_steal_keys"),
        ("ST_high_yield",    "INT_steal_funds"),
        ("ST_grooming",      "INT_false_trust"),
        ("ST_grooming",      "INT_steal_funds"),
    ]
    for src, dst in targets_edges:
        G.add_edge(src, dst, edge_type="TARGETS", weight=1.0)

    # ── Edges: ScamType ↔ ScamType (RELATED_TO) ─────────────────────────────
    related_edges = [
        ("ST_grooming",      "ST_investment"),
        ("ST_grooming",      "ST_romance"),
        ("ST_urgent",        "ST_romance"),
        ("ST_fake_support",  "ST_migration"),
        ("ST_phishing",      "ST_recovery"),
        ("ST_impersonation", "ST_phishing"),
        ("ST_high_yield",    "ST_investment"),
    ]
    for src, dst in related_edges:
        G.add_edge(src, dst, edge_type="RELATED_TO", weight=0.8)
        G.add_edge(dst, src, edge_type="RELATED_TO", weight=0.8)  # symmetric

    # ── Edges: pattern → ScamType (ASSOCIATED_WITH) ─────────────────────────
    assoc_edges = [
        ("PAT_impersonation",  "ST_impersonation"),
        ("PAT_impersonation",  "ST_fake_support"),
        ("PAT_urgency",        "ST_urgent"),
        ("PAT_urgency",        "ST_romance"),
        ("PAT_credential_req", "ST_phishing"),
        ("PAT_credential_req", "ST_recovery"),
        ("PAT_investment",     "ST_investment"),
        ("PAT_investment",     "ST_high_yield"),
        ("PAT_phishing",       "ST_phishing"),
        ("PAT_counterparty",   "ST_grooming"),
        ("PAT_counterparty",   "ST_romance"),
    ]
    for src, dst in assoc_edges:
        G.add_edge(src, dst, edge_type="ASSOCIATED_WITH", weight=0.9)

    # ── Edges: Cue → Cue (PRECEDES) ─────────────────────────────────────────
    precedes_edges = [
        ("CUE_authority",     "CUE_urgent_transfer"),
        ("CUE_authority",     "CUE_wallet_request"),
        ("CUE_url_link",      "CUE_identity_verify"),
        ("CUE_url_link",      "CUE_seed_phrase"),
        ("CUE_urgent_transfer","CUE_wallet_request"),
    ]
    for src, dst in precedes_edges:
        G.add_edge(src, dst, edge_type="PRECEDES", weight=0.7)

    logger.info(
        f"LocalKnowledgeBase: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


# ══════════════════════════════════════════════════════════════════════════════
# Public class
# ══════════════════════════════════════════════════════════════════════════════

class LocalKnowledgeBase:
    """
    In-memory fraud knowledge graph for on-device Micro-GraphRAG retrieval.

    Attributes:
        graph   : nx.DiGraph  — the full KB graph
        nodes   : List[str]   — all node IDs
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph: nx.DiGraph = graph if graph is not None else _build_default_graph()

    @property
    def nodes(self) -> List[str]:
        return list(self.graph.nodes)

    def get_node_keywords(self, node_id: str) -> List[str]:
        return self.graph.nodes[node_id].get("keywords", [])

    def get_node_risk_weight(self, node_id: str) -> float:
        return float(self.graph.nodes[node_id].get("risk_weight", 0.5))

    def get_node_type(self, node_id: str) -> str:
        return self.graph.nodes[node_id].get("node_type", "Unknown")

    def get_neighbors(self, node_id: str, hops: int = 1) -> List[str]:
        """BFS expansion from node_id up to `hops` hops."""
        visited = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                for nbr in self.graph.successors(n):
                    if nbr not in visited:
                        next_frontier.add(nbr)
                for nbr in self.graph.predecessors(n):
                    if nbr not in visited:
                        next_frontier.add(nbr)
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(node_id)
        return list(visited)

    def get_scam_type_nodes(self) -> List[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "ScamType"]

    def get_cue_nodes(self) -> List[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "Cue"]

    def summary(self) -> dict:
        type_counts: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("node_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        edge_type_counts: Dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            et = d.get("edge_type", "UNKNOWN")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }
