"""Boundary semantics independent of large provider data or a CUDA device."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from gog_fraud.data.level2.relation_builder import HistoricalReferenceIndex
from gog_fraud.streaming.selective import dropout_only,fit_threshold,binary_metrics,risk_counts
from gog_fraud.streaming.selective_state import BoundedContractState,BoundedHotCache,BoundedQueue,durable_checkpoint,load_checkpoint
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'projects/stream_mc/scripts'))
from r01_validate import identity,hashes


def event(i,contract='a',ts=None,eid=None):return {'sequence_id':i,'event_id':eid or str(i),'chain':'eth','contract_id':contract,'timestamp':i if ts is None else ts,'source':'u','target':'v'}


def test_exact_eligible_ties_and_self_exclusion():
    index=HistoricalReferenceIndex([[0.,0.],[0.,0.],[0.,0.],[1.,0.]],[.1,.2,.3,.4],['z','a','b','future'],[1,1,1,100],k=2)
    ix,n=index.query([0.,0.],5,'b');assert index.ids[ix].tolist()==['a','z'] and n==2
    graph,audit=index.target_graph(np.array([0.,0.]),.8,5,'b');assert graph.num_nodes==3 and graph.edge_index.shape==(2,4)
    assert audit['selected_reference_max_cutoff']==1
    graph,audit=index.target_graph(np.array([0.,0.]),.8,0);assert graph.num_nodes==1 and audit['fallback']=='no_eligible_reference'


def test_label_free_reference_and_invalid_fields():
    with pytest.raises(ValueError):HistoricalReferenceIndex([[0.],[1.]],[.1,.2],['a','a'],[1,2])
    with pytest.raises(ValueError):HistoricalReferenceIndex([[np.nan]],[.1],['a'],[1])
    index=HistoricalReferenceIndex([[0.]],[.1],['a'],[1])
    with pytest.raises(ValueError):index.query([np.nan],2)


def test_bn_frozen_during_dropout_only_and_mode_restored():
    model=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3),nn.Dropout(.5));model.train();before=model[1].running_mean.clone()
    with dropout_only(model,True):
        assert not model[1].training and model[2].training
        a=model(torch.ones(50,3));b=model(torch.ones(50,3));assert not torch.equal(a,b)
    assert model.training and model[1].training and torch.equal(before,model[1].running_mean)


def test_negative_only_denominators_and_threshold_tie():
    metrics=binary_metrics([0,0],[.9,.2],[1,0]);assert metrics['ap'] is None and metrics['f1'] is None and metrics['recall'] is None
    assert metrics['precision']==0 and metrics['fpr']==.5
    risk=risk_counts(np.array([0,0]),np.array([0,1]),np.array([True,True]));assert risk['selective_error'] is None and risk['direct_fraud_FNR'] is None
    threshold,f1=fit_threshold(np.array([1,0]),np.array([.5,.5]));assert threshold==.5 and f1==pytest.approx(2/3)


def test_state_capacity_ttl_dup_late_and_degree():
    state=BoundedContractState(capacity=2,max_edges=2,max_nodes=2,ttl=5)
    for i in range(4):state.update(event(i,str(i)))
    assert len(state.states)==2 and state.counters['capacity_evictions']==2
    assert not state.update(event(4,'3',ts=3,eid='3'))['accepted']
    assert state.update(event(5,'late',ts=1))['reason']=='late'
    state.update(event(6,'new',ts=20));assert len(state.states)==1 and state.counters['ttl_evictions']==2
    for i in range(7,12):state.update(event(i,'new',ts=20))
    assert state.invariants() and state.graph(('eth','new')).num_edges==2
    assert torch.allclose(state.graph(('eth','new')).x[:,2],torch.log1p(torch.tensor([2.,2.])))


def test_cache_and_queue_caps():
    cache=BoundedHotCache(capacity=2,byte_cap=16)
    for i in range(10):cache.get(i,lambda:np.ones(2,np.float32))
    assert len(cache.values)==2 and cache.bytes==16 and cache.evictions==8
    queue=BoundedQueue(2);assert queue.put(1) and queue.put(2) and not queue.put(3)
    assert queue.pop()[0]==1 and queue.high_water==2 and queue.retries==1


def test_checkpoint_identity_and_hash_failures(tmp_path):
    path=tmp_path/'state.checkpoint';durable_checkpoint(path,{'identity':{'model':'a'},'cursor':3})
    assert load_checkpoint(path,{'model':'a'})['cursor']==3
    with pytest.raises(ValueError):load_checkpoint(path,{'model':'b'})
    with pytest.raises(ValueError):hashes(tmp_path,{'state.checkpoint':'0'*64})
    with pytest.raises(ValueError):hashes(tmp_path,{'absent':'0'*64})
    with pytest.raises(ValueError):identity([{'run_id':'r','experiment_id':'a'},{'run_id':'r','experiment_id':'b'}],'r','a')
