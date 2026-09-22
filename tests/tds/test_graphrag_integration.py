#!/usr/bin/env python3
"""Integration test for all new GraphRAG Round 1 modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphrag.local_kb import LocalKnowledgeBase
from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
from graphrag.risk_extractor import RiskExtractor
from graphrag.risk_encoder import RiskEncoder
from fusion.uncertainty_fusion import UncertaintyFusion
from fusion.fixed_fusion import FixedFusion
from privacy.vector_codec import VectorCodec
from privacy.leakage_attack import LeakageAttack

import pytest

SAMPLE_TEXT = "URGENT: Send 500 USDT to 0xDeadBeef immediately! Guaranteed 500% return."

@pytest.fixture
def kb():
    return LocalKnowledgeBase()

@pytest.fixture
def evidence(kb):
    retriever = GraphRAGRetriever(kb)
    return retriever.retrieve(SAMPLE_TEXT)

@pytest.fixture
def risk(evidence):
    extractor = RiskExtractor()
    return extractor.extract(evidence, event_id="tx_000", pre_transaction_gap_sec=300)

@pytest.fixture
def p_risk(risk):
    import torch
    encoder = RiskEncoder()
    encoder.eval()
    with torch.no_grad():
        _, p = encoder.encode_risk_dict_batch([risk])
    return p

def test_kb(kb):
    s = kb.summary()
    assert s["num_nodes"] > 20, f"Expected > 20 nodes, got {s['num_nodes']}"
    assert s["num_edges"] > 20, f"Expected > 20 edges, got {s['num_edges']}"
    print(f"  KB: {s['num_nodes']} nodes, {s['num_edges']} edges — OK")

def test_retrieval(kb, evidence):
    assert len(evidence) > 0, "No evidence retrieved"
    top = evidence[0]
    assert top.score >= 0.0, "Evidence score must be non-negative"
    print(f"  Retrieval: {len(evidence)} items, top={top.node_label} ({top.score:.4f}) — OK")

def test_risk_extraction(evidence, risk):
    assert "local_risk_score" in risk
    assert "label" not in risk, "LEAKAGE: label found in risk dict!"
    assert 0.0 <= risk["local_risk_score"] <= 1.0
    print(f"  Extractor: s_t={risk['local_risk_score']:.4f}, k_t={risk['risk_type_id']}, scenario={risk['scenario_type']} — OK")

def test_risk_encoder(risk, p_risk):
    import torch
    encoder = RiskEncoder()
    encoder.eval()
    with torch.no_grad():
        z, p = encoder.encode_risk_dict_batch([risk])
    assert tuple(z.shape) == (1, 16), f"Wrong z shape: {tuple(z.shape)}"
    assert 0.0 <= p[0].item() <= 1.0
    print(f"  Encoder: z.shape={tuple(z.shape)}, p_risk={p[0].item():.4f} — OK")

def test_fusion(p_risk):
    import torch
    p_gnn = torch.tensor([0.65])
    u_mc  = torch.tensor([0.08])

    # UncertaintyFusion
    uf = UncertaintyFusion()
    final, alpha, beta = uf.fuse(p_gnn, u_mc, p_risk)
    assert 0.0 <= final[0].item() <= 1.0
    print(f"  UncertaintyFusion: R_t={final[0].item():.4f}, beta={beta[0].item():.4f} — OK")

    # FixedFusion
    ff = FixedFusion(alpha=0.4)
    final2, _, _ = ff.fuse(p_gnn, p_risk)
    assert 0.0 <= final2[0].item() <= 1.0
    print(f"  FixedFusion(α=0.4): R_t={final2[0].item():.4f} — OK")

def test_codec(risk):
    for mode in ("json", "binary"):
        codec = VectorCodec(mode)
        b = codec.measure_bytes(risk)
        data = codec.serialize(risk)
        recovered = codec.deserialize(data)
        assert abs(recovered["local_risk_score"] - risk.get("local_risk_score", 0.0)) < 0.01
        print(f"  Codec ({mode}): {b} bytes, roundtrip OK")

def test_context_generator():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage
    )
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scenarios = assign_scenarios_no_leakage(labels, seed=42)
    gen = SyntheticContextGenerator(seed=42)
    records = gen.generate_contexts(scenarios)
    assert len(records) == 10
    for rec in records:
        assert "label" not in rec, "LEAKAGE: label found in generated context!"
    # Check hard positive/negative exist in sufficient runs
    print(f"  SyntheticContextGenerator: {len(records)} records, no label field — OK")
    hard_pos = sum(1 for s in scenarios if s == "hard_positive")
    hard_neg = sum(1 for s in scenarios if s == "hard_negative")
    print(f"  Hard positive: {hard_pos}, Hard negative: {hard_neg}")

if __name__ == "__main__":
    print("=" * 60)
    print("  Integration Test — dlg_gnn GraphRAG Round 1 Modules")
    print("=" * 60)
    try:
        kb = test_kb()
        evidence = test_retrieval(kb)
        risk = test_risk_extraction(evidence)
        p = test_risk_encoder(risk)
        test_fusion(p)
        test_codec(risk)
        test_context_generator()
        print("\n✅ ALL TESTS PASSED")
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
