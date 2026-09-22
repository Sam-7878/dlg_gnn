"""
tests/scam_revision/test_future_report_masking.py

Verifies that post-query scam reports and edges (timestamp > query_timestamp)
are strictly masked during feature retrieval to prevent temporal lookahead leakage.
"""

import pytest
from dlg_gnn.graphrag.scam_revision.scam_hetero_graph import HeteroNode, HeteroEdge, ScamHeteroGraph
from dlg_gnn.graphrag.scam_revision.scam_graphrag_retriever import ScamGraphRAGRetriever


def test_future_report_masking():
    graph = ScamHeteroGraph()
    t_query = 1600000000
    
    # Query node at t_query
    graph.add_node(HeteroNode("campaign:current", "Campaign", "Bounty", timestamp=t_query, text_content="Promotion"))
    
    # Past historical evidence
    graph.add_node(HeteroNode("domain:past", "Domain", "past.com", timestamp=t_query - 10000))
    graph.add_edge(HeteroEdge("campaign:current", "domain:past", "promotes", timestamp=t_query - 10000))
    
    # Future scam report (e.g. reported 1 month after campaign)
    graph.add_node(HeteroNode("domain:future_report", "Domain", "future.com", timestamp=t_query + 86400 * 30))
    graph.add_edge(HeteroEdge("campaign:current", "domain:future_report", "promotes", timestamp=t_query + 86400 * 30))

    retriever = ScamGraphRAGRetriever(graph)
    res = retriever.retrieve("campaign:current", "Promotion", query_timestamp=t_query, hop=1)
    
    retrieved_node_ids = {e.node_id for e in res.evidence_list}
    
    assert "domain:past" in retrieved_node_ids, "Historical evidence must be retrieved"
    assert "domain:future_report" not in retrieved_node_ids, "Temporal lookahead leakage: future report was retrieved!"
