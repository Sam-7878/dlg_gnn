"""
tests/scam_revision/test_label_feature_circularity.py

Verifies that RiskVectorV2 input features do NOT contain ground-truth label indicators,
malicious registry flags, or circular feature shortcuts.
"""

import pytest
import torch
from dlg_gnn.graphrag.scam_revision.scam_hetero_graph import HeteroNode, HeteroEdge, ScamHeteroGraph
from dlg_gnn.graphrag.scam_revision.scam_graphrag_retriever import ScamGraphRAGRetriever
from dlg_gnn.graphrag.scam_revision.scam_risk_encoder import ScamRiskExtractor, RiskVectorV2


def test_no_ground_truth_in_risk_vector():
    graph = ScamHeteroGraph()
    # Node with explicit ground-truth flags that must NOT be passed into features
    graph.add_node(HeteroNode("c:1", "Campaign", "Legit Campaign", text_content="Normal discussion", features={"is_scam": True, "category": "phishing"}))
    graph.add_node(HeteroNode("d:1", "Domain", "normal.org", features={"is_scam": True}, provenance="CST+CSDB"))
    graph.add_edge(HeteroEdge("c:1", "d:1", "promotes"))

    retriever = ScamGraphRAGRetriever(graph)
    res = retriever.retrieve("c:1", "Normal discussion", hop=1)
    
    extractor = ScamRiskExtractor()
    r_vec = extractor.extract(res)
    
    # Assert features are continuous values in [0, 1] without carrying raw ground-truth booleans
    vec_list = r_vec.to_list()
    assert len(vec_list) == 7, "Risk vector must have exactly 7 dimensions"
    for val in vec_list:
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    # Since text is 'Normal discussion' without high-risk keywords (giveaway/airdrop),
    # semantic score s_t should be low even though the node has ground-truth is_scam=True in evaluation metadata!
    assert r_vec.s_t < 0.6, f"Circularity detected: semantic score {r_vec.s_t} was boosted by ground-truth flag!"
