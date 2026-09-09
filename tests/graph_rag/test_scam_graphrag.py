"""
tests/test_scam_graphrag.py

Unit tests for Phase C-K modules of _43_GraphRAG Scam Revision:
- EntityResolver: Wallets, Domains, URLs
- ScamHeteroGraph: Node/Edge indexing, Causal temporal filtering
- ScamGraphRAGRetriever: 0-hop, 1-hop, 2-hop, IR metrics
- ScamRiskEncoder: RiskVectorV2, Neural Risk Head
- UncertaintyFusion: Adaptive uncertainty gating
"""

import pytest
import torch
import numpy as np

from dlg_gnn.graphrag.scam_revision.entity_resolver import (
    normalize_wallet,
    normalize_url_domain,
    extract_addresses_from_text,
)
from dlg_gnn.graphrag.scam_revision.scam_hetero_graph import (
    HeteroNode,
    HeteroEdge,
    ScamHeteroGraph,
)
from dlg_gnn.graphrag.scam_revision.scam_graphrag_retriever import (
    ScamGraphRAGRetriever,
)
from dlg_gnn.graphrag.scam_revision.scam_risk_encoder import (
    ScamRiskExtractor,
    ScamRiskEncoderHead,
    RiskVectorV2,
)
from dlg_gnn.fusion.uncertainty_fusion import UncertaintyFusion


def test_entity_resolver_wallet():
    # Valid ETH
    w1 = normalize_wallet("0x509963a17C9DDDC7755F34E27Dc24b57D1434367", "eth")
    assert w1 is not None
    assert w1.is_valid_format is True
    assert w1.address == "0x509963a17c9dddc7755f34e27dc24b57d1434367"
    assert w1.chain == "ethereum"

    # Valid BTC
    w2 = normalize_wallet("1MwYjNZgDJDSjXx9AZHgZMrEPZsXctHC8H", "btc")
    assert w2 is not None
    assert w2.is_valid_format is True
    assert w2.chain == "bitcoin"

    # Invalid string
    w3 = normalize_wallet("not_an_address")
    assert w3.is_valid_format is False


def test_entity_resolver_domain_url():
    # URL to eTLD+1
    d1 = normalize_url_domain("https://sub.msget.io/giveaway?ref=123")
    assert d1 is not None
    assert d1.is_valid is True
    assert d1.domain == "msget.io"
    assert d1.host == "sub.msget.io"

    # Two-part TLD
    d2 = normalize_url_domain("http://scam-site.co.uk/login")
    assert d2 is not None
    assert d2.domain == "scam-site.co.uk"


def test_scam_hetero_graph_and_causal_filtering():
    graph = ScamHeteroGraph()
    
    # Add nodes with timestamps
    graph.add_node(HeteroNode("campaign:101", "Campaign", "Scam Giveaway", timestamp=1000, text_content="Double your crypto"))
    graph.add_node(HeteroNode("domain:fake.io", "Domain", "fake.io", timestamp=900, features={"is_scam": True}))
    graph.add_node(HeteroNode("wallet:0x123", "Wallet", "0x123", timestamp=950, features={"is_scam": True}))
    graph.add_node(HeteroNode("domain:future.io", "Domain", "future.io", timestamp=1500, features={"is_scam": True}))

    # Add edges
    graph.add_edge(HeteroEdge("campaign:101", "domain:fake.io", "promotes", timestamp=1000))
    graph.add_edge(HeteroEdge("domain:fake.io", "wallet:0x123", "references_wallet", timestamp=950))
    graph.add_edge(HeteroEdge("campaign:101", "domain:future.io", "promotes", timestamp=1500))

    assert graph.num_nodes() == 4
    assert graph.num_edges() == 3

    # Causal query at t=1000: future node/edge (t=1500) must NOT be returned!
    neighbors = graph.get_causal_neighbors("campaign:101", max_hops=2, query_timestamp=1000)
    returned_node_ids = {n.node_id for n, _, _ in neighbors}
    
    assert "domain:fake.io" in returned_node_ids
    assert "wallet:0x123" in returned_node_ids
    assert "domain:future.io" not in returned_node_ids, "Causal leakage detected: future node was returned!"


def test_multihop_graphrag_retriever():
    graph = ScamHeteroGraph()
    graph.add_node(HeteroNode("c:1", "Campaign", "Bounty 1", timestamp=1000, text_content="Crypto promotion"))
    graph.add_node(HeteroNode("d:1", "Domain", "scam.com", timestamp=900, features={"category": "phishing", "is_scam": True}))
    graph.add_node(HeteroNode("w:1", "Wallet", "0xabc", timestamp=900, features={"is_scam": True}))
    
    graph.add_edge(HeteroEdge("c:1", "d:1", "promotes", timestamp=1000))
    graph.add_edge(HeteroEdge("d:1", "w:1", "references_wallet", timestamp=900))

    retriever = ScamGraphRAGRetriever(graph, top_k=5)

    # 0-hop retrieval
    res_0hop = retriever.retrieve("c:1", "crypto promotion", hop=0)
    assert len(res_0hop.evidence_list) == 1
    assert res_0hop.evidence_list[0].hop_distance == 0

    # 1-hop retrieval
    res_1hop = retriever.retrieve("c:1", "crypto promotion", hop=1)
    assert len(res_1hop.evidence_list) >= 2
    assert any(e.hop_distance == 1 for e in res_1hop.evidence_list)

    # 2-hop retrieval
    res_2hop = retriever.retrieve("c:1", "crypto promotion", hop=2)
    assert len(res_2hop.evidence_list) == 3
    assert any(e.hop_distance == 2 for e in res_2hop.evidence_list)

    # Check metrics computed
    assert "mrr" in res_2hop.metrics
    assert "hit@5" in res_2hop.metrics
    assert res_2hop.metrics["hit@5"] == 1.0


def test_risk_encoder_and_fusion():
    graph = ScamHeteroGraph()
    graph.add_node(HeteroNode("c:1", "Campaign", "Bounty", text_content="Scam crypto"))
    graph.add_node(HeteroNode("d:1", "Domain", "phishing.com", features={"category": "phishing", "is_scam": True}, provenance="Tier3_MultiSource"))
    graph.add_edge(HeteroEdge("c:1", "d:1", "promotes"))

    retriever = ScamGraphRAGRetriever(graph)
    res = retriever.retrieve("c:1", "Scam crypto", hop=1)

    extractor = ScamRiskExtractor()
    r_vec = extractor.extract(res)
    assert isinstance(r_vec, RiskVectorV2)
    assert r_vec.c_t == 1.0  # MultiSource corroboration detected

    # Neural Risk Head
    head = ScamRiskEncoderHead(in_dim=7, hidden_dim=16)
    head.eval()
    with torch.no_grad():
        p_rag = head(r_vec.to_tensor().unsqueeze(0))
    assert p_rag.shape == (1,)
    assert 0.0 <= p_rag.item() <= 1.0

    # Uncertainty Fusion
    fusion = UncertaintyFusion(lambda_u=5.0, bias=-1.0)
    p_gnn = torch.tensor([0.20])
    u_mc = torch.tensor([0.85])  # High uncertainty
    p_risk = torch.tensor([0.90])
    
    fused_prob, alpha, beta = fusion.fuse(p_gnn, u_mc, p_risk)
    assert fused_prob.item() > p_gnn.item(), "Under high uncertainty, fused probability should shift toward risk branch!"
