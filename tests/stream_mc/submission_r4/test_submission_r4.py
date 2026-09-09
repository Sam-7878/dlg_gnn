"""Unit and integration test suite for DLG-SelectiveStream SCI Submission R4."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from gog_fraud.production.submission_r4 import (
    TimestampedFrozenRelations, apply_policy, calibrate, fit_map, logits,
    metrics, read_config, select_policy
)
from validation.validate_temporal_relation_provenance import validate_temporal_relation_provenance
from validation.validate_mc_method_identity import validate_mc_method_identity


def test_temporal_relation_validator_passes():
    root = Path('results/stream_mc/sci_v3_submission_r4') if Path('results/stream_mc/sci_v3_submission_r4').exists() else Path('results/sci_v3_submission_r4')
    summary = validate_temporal_relation_provenance(root)
    assert summary['status'] == 'PASS'
    assert summary['temporal_violations'] == 0
    assert summary['max_reference_cutoff_le_target_cutoff'] is True
    assert summary['audited_targets'] == 7296


def test_mc_method_identity_validator_passes():
    root = Path('results/stream_mc/sci_v3_submission_r4') if Path('results/stream_mc/sci_v3_submission_r4').exists() else Path('results/sci_v3_submission_r4')
    predec = Path('configs/stream_mc/sci_v3_submission_r4/mc_identity_predeclaration.json') if Path('configs/stream_mc/sci_v3_submission_r4/mc_identity_predeclaration.json').exists() else Path('configs/sci_v3_submission_r4/mc_identity_predeclaration.json')
    summary = validate_mc_method_identity(root, predec)
    assert summary['status'] == 'PASS'
    assert summary['branch'] == 'MC-B'
    assert summary['primary_operating_point_T'] == 1
    assert summary['production_stochastic_dropout'] is False


def test_timestamped_relations_enforces_cutoff_safety():
    # 5 candidate references at t = 10, 20, 30, 40, 50
    cand_emb = np.array([[float(i)] * 4 for i in range(5)], dtype=np.float32)
    cand_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    cand_times = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    cand_chains = np.array(['eth', 'eth', 'bsc', 'bsc', 'eth'])
    
    rel = TimestampedFrozenRelations(cand_emb, cand_scores, cand_times, cand_chains, k=3)
    
    # Target 1 at t = 25 (eligible: t=10, 20)
    tgt_emb = np.array([[1.5] * 4], dtype=np.float32)
    tgt_score = np.array([0.25], dtype=np.float32)
    tgt_time = np.array([25], dtype=np.int64)
    
    graphs, audits = rel.graphs(tgt_emb, tgt_score, tgt_time, target_chains=['eth'])
    assert len(graphs) == 1
    assert len(audits) == 1
    audit = audits[0]
    assert audit['eligible_reference_count'] == 2
    assert audit['selected_neighbor_count'] == 2
    assert audit['max_selected_reference_cutoff'] <= 25
    assert audit['temporal_violation_count'] == 0
    
    # Star graph: 2 refs + 1 center
    assert graphs[0].x.shape[0] == 3


def test_timestamped_relations_source_chain_filtering():
    cand_emb = np.array([[float(i)] * 4 for i in range(5)], dtype=np.float32)
    cand_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    cand_times = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    cand_chains = np.array(['eth', 'eth', 'bsc', 'bsc', 'polygon'])
    
    rel = TimestampedFrozenRelations(cand_emb, cand_scores, cand_times, cand_chains, k=3)
    
    # Target at t = 100, but source chains = ['bsc']
    tgt_emb = np.array([[2.5] * 4], dtype=np.float32)
    tgt_score = np.array([0.3], dtype=np.float32)
    tgt_time = np.array([100], dtype=np.int64)
    
    graphs, audits = rel.graphs(tgt_emb, tgt_score, tgt_time, source_chains=['bsc'])
    assert audits[0]['eligible_reference_count'] == 2
    assert audits[0]['selected_neighbor_count'] == 2
    assert audits[0]['temporal_violation_count'] == 0


def test_calibration_and_logits_numerical_stability():
    scores = np.array([1e-6, 0.05, 0.5, 0.95, 1 - 1e-6])
    l = logits(scores)
    assert np.isfinite(l).all()
    mapping = {'coefficient': 1.0, 'intercept': 0.0}
    cal = calibrate(scores, mapping)
    np.testing.assert_allclose(scores, cal, atol=1e-5)


def test_policy_selection_and_application():
    np.random.seed(42)
    y = np.array([0] * 50 + [1] * 50)
    fast = np.clip(y * 0.6 + np.random.normal(0.2, 0.1, 100), 0.01, 0.99)
    deep = np.clip(y * 0.7 + np.random.normal(0.15, 0.1, 100), 0.01, 0.99)
    
    cfg = {'risk': {'budgets': [0.1, 0.2, 0.3]}}
    sel = select_policy(fast, deep, y, cfg, T=1)
    
    assert sel['mc_T'] == 1
    assert 0.0 <= sel['validation_f1'] <= 1.0
    
    final, route, pred = apply_policy(fast, deep, sel, 'primary')
    assert len(final) == 100
    assert len(route) == 100
    assert len(pred) == 100
    assert set(np.unique(pred)).issubset({0, 1})


def test_canonical_artifacts_completeness():
    root = Path('results/stream_mc/sci_v3_submission_r4') if Path('results/stream_mc/sci_v3_submission_r4').exists() else Path('results/sci_v3_submission_r4')
    
    # Check tables
    required_tables = [
        'table_main_predictive.tex',
        'table_primary_selective_frontier.tex',
        'table_statistical_evidence.tex',
        'table_calibration_metrics.tex',
        'table_cross_chain_summary.tex',
        'table_cross_chain_summary_r4.tex',
        'table_integrated_streaming.tex',
        'table_operating_point_identity.tex',
        'table_relation_temporal_audit.tex'
    ]
    for tbl in required_tables:
        path = root / 'tables' / tbl
        assert path.exists(), f"Missing table {tbl}"
        assert path.stat().st_size > 0
        
    # Check figures
    required_figures = [
        'figure_accuracy_cost_frontier.pdf',
        'figure_mc_sensitivity.pdf',
        'figure_reliability_r4.pdf',
        'figure_risk_coverage_dense.pdf',
        'figure_streaming_memory_long.pdf'
    ]
    for fig in required_figures:
        path = root / 'figures' / fig
        assert path.exists(), f"Missing figure {fig}"
        assert path.stat().st_size > 0
        
    # Check manuscripts
    ms_dir = Path('manuscript/_41_01_DLG_StreamMC') if Path('manuscript/_41_01_DLG_StreamMC').exists() else Path('manuscript')
    assert (ms_dir / 'DLG-SelectiveStream_submission.tex').exists() or (ms_dir / 'DLG-SelectiveStream_submission_r4.tex').exists()
    assert (ms_dir / 'DLG-SelectiveStream_supplementary.tex').exists() or (ms_dir / 'DLG_SelectiveStream_Supplementary_r4.tex').exists()
    assert (ms_dir / 'r4_numbers.tex').exists()
    
    # Check gate report
    gate = Path('docs/work_reports/stream_mc/112_stream_mc_submission_r4/sci_v3_submission_r4/final_scientific_gate.md')
    if not gate.exists():
        gate = Path('docs/work_reports/112_stream_mc_submission_r4/sci_v3_submission_r4/final_scientific_gate.md')
    if not gate.exists():
        gate = Path('docs/work_reports/sci_v3_submission_r4/final_scientific_gate.md')
    assert gate.exists()
    assert gate.stat().st_size > 1000
