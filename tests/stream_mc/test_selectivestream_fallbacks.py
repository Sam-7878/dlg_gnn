"""Actual trained R01 predictor exceptions; optional in source-only checkouts."""
from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from gog_fraud.streaming.selective_engine import SelectivePredictor
import gog_fraud.streaming.selective_engine as inference

MODEL=Path(__file__).resolve().parents[2]/'projects/stream_mc/results/selectivestream_r01_20261011/models/pooled_snapshot_GIN_seed11'


@pytest.fixture
def predictor():
    if not (MODEL/'policy_fit.json').is_file():pytest.skip('author-local original trained checkpoint not installed')
    return SelectivePredictor(MODEL,'cpu','full')


def graph():
    return Data(x=torch.tensor([[0.,.6931472,.6931472],[.6931472,0.,.6931472]]),edge_index=torch.tensor([[0],[1]]))


def test_empty_graph_abstains(predictor):
    result=predictor.predict(Data(x=torch.empty(0,3),edge_index=torch.empty(2,0,dtype=torch.long)),0)
    assert result['fallback']=='empty_graph' and result['final_score'] is None and result['final_label'] is None


def test_cold_start_local_at_final_threshold(predictor):
    result=predictor.predict(graph(),0)
    assert result['fallback']=='no_eligible_reference' and not result['escalated']
    assert result['final_score']==result['local_score']
    assert result['final_label']==int(result['local_score']>=result['final_threshold'])


def test_cache_miss_then_hit_loads_immutable_references(predictor):
    a=predictor.predict(graph(),2**31);miss=predictor.cache.misses
    b=predictor.predict(graph(),2**31)
    assert miss==8 and predictor.cache.misses==miss and predictor.cache.hits==8
    assert a['escalated'] and b['escalated'] and abs(a['final_score']-b['final_score'])<1e-8


def test_invalid_local_abstains(predictor,monkeypatch):
    monkeypatch.setattr(inference,'local_predict',lambda *args:(np.array([np.nan]),np.zeros((1,64)),np.array([0.])))
    result=predictor.predict(graph(),2**31)
    assert result['fallback']=='invalid_local_score' and result['final_label'] is None


def test_invalid_relational_falls_back_to_local(predictor):
    class Invalid(torch.nn.Module):
        def forward(self,data):return torch.tensor([float('nan')])
    predictor.deep=Invalid();result=predictor.predict(graph(),2**31)
    assert result['fallback']=='invalid_relational_score' and result['escalated']
    assert result['final_score']==result['local_score']


def test_soft_timeout_before_relational(predictor):
    predictor.timeout_seconds=1e-100;result=predictor.predict(graph(),2**31)
    assert result['fallback']=='soft_timeout_before_relational' and not result['escalated']
    assert result['final_score']==result['local_score']


def test_soft_timeout_after_relational(predictor,monkeypatch):
    predictor.timeout_seconds=.55
    ticks=iter([0.,.1,.2,.3,.4,.5,.6,.7,.8,.9])
    monkeypatch.setattr(inference.time,'perf_counter',lambda:next(ticks))
    result=predictor.predict(graph(),2**31)
    assert result['fallback']=='soft_timeout_after_relational' and result['escalated']
    assert result['final_score']==result['local_score']
