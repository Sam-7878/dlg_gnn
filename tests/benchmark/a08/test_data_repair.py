import copy, csv, io, json, zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from gog_fraud.data.crypto_raw import transaction_features, build_observables, deterministic_knn, content_hash
from gog_fraud.data.benchmark_node_adapter import node_view, attach_labels
from gog_fraud.data.benchmark_metrics import evaluate_scores, validate_run_identity, validate_score

IDS=['polygon:0x'+format(i,'040x') for i in range(4)]

def source(): return Data(x=torch.eye(4),edge_index=torch.tensor([[0,1],[1,0]]),y=torch.tensor([1]),num_nodes=4)

def view(**kwargs): return node_view(source(),IDS,'polygon_contract_clean_v1',**kwargs)

def test_T01_scalar_node_rejected():
    with pytest.raises(ValueError):view(labels=torch.tensor([1]))

def test_T02_source_graph_target_preserved():
    raw=source();d=node_view(raw,IDS,'polygon_contract_clean_v1',labels=torch.tensor([0,1,0,1]));assert d.y.shape==(4,) and raw.y.tolist()==[1]

@pytest.mark.parametrize('labels',[torch.tensor([0,1]),torch.tensor([0.,1.,0.,1.]),torch.tensor([0,2,0,1])])
def test_T03_invalid_labels(labels):
    with pytest.raises(ValueError):view(labels=labels)

def test_T03_alias_conflict():
    raw=source();raw.labels=torch.ones(4,dtype=torch.long)
    with pytest.raises(ValueError):node_view(raw,IDS,'polygon_contract_clean_v1',labels=torch.zeros(4,dtype=torch.long))

def test_T04_duplicate_and_missing_join(tmp_path):
    with pytest.raises(ValueError):node_view(source(),[IDS[0]]*4,'polygon_contract_clean_v1',labels=torch.ones(4,dtype=torch.long))
    p=tmp_path/'labels.csv';p.write_text('Chain,Contract,Category\npolygon,'+IDS[0].split(':')[1]+',0\n')
    y,v=attach_labels(IDS,p,'polygon');assert y.tolist()==[1,-1,-1,-1] and v.tolist()==[True,False,False,False]
    p.write_text(p.read_text()+p.read_text().splitlines()[1]+'\n')
    with pytest.raises(ValueError):attach_labels(IDS,p,'polygon')

def test_T05_bad_masks():
    y=torch.tensor([0,1,0,1]);m=torch.ones(4,dtype=torch.bool)
    with pytest.raises(ValueError):view(labels=y,masks={'train_mask':m,'val_mask':m})
    with pytest.raises(ValueError):view(labels=y,masks={'test_mask':m[:1]})

def test_T06_isolate_kept():
    d=view(labels=torch.tensor([0,1,0,1]));assert d.num_nodes==4 and d.y.shape==(4,)


def rawzip(tmp_path,name):
    p=tmp_path/name
    with zipfile.ZipFile(p,'w') as z:
        for i in range(4):
            a=IDS[i].split(':')[1]; b=IDS[(i+1)%4].split(':')[1]
            z.writestr('polygon/'+a+'.csv','from,to\n'+''.join(a+','+b+'\n' for j in range(i+1)))
    return p

def test_raw_example():
    a,b=IDS[0].split(':')[1],IDS[1].split(':')[1]
    x,audit=transaction_features(io.StringIO('from,to\n'+a+','+b+'\n'+b+','+a+'\n'+a+','+b+'\n'))
    assert audit=={'rows':3,'addresses':2,'pairs':2}
    np.testing.assert_allclose(x,[np.log(4),np.log(3),np.log(3),np.log(3),np.log(3),0,1,1/3])

def test_T07_T08_T10_raw_no_label_dependency(tmp_path):
    p=rawzip(tmp_path,'raw.zip');a,_=build_observables(p,'polygon');b,_=build_observables(p,'polygon')
    assert content_hash(a)==content_hash(b)
    with pytest.raises(FileNotFoundError):attach_labels(a['node_ids'],tmp_path/'absent.csv','polygon')
    edge,_=deterministic_knn(a['x'],a['node_ids'],k=2)
    for y in [np.array([0,1,0,1]),np.array([1,0,1,0]),np.zeros(4,dtype=np.int64),np.ones(4,dtype=np.int64)]:
        d=node_view(Data(x=torch.from_numpy(a['x']),edge_index=torch.from_numpy(edge),num_nodes=4),a['node_ids'],'polygon_contract_clean_v1',labels=y)
        assert np.array_equal(d.x.numpy(),a['x']) and np.array_equal(d.edge_index.numpy(),edge)

def test_T09_T14_row_permutation_partial_tie():
    x=np.array([[0,0],[1,0],[-1,0],[0,2.]],dtype=float);e,_=deterministic_knn(x,IDS,k=1)
    p=np.array([2,0,3,1]);f,_=deterministic_knn(x[p],np.array(IDS)[p],k=1)
    assert set(map(tuple,e.T))==set(map(tuple,p[f].T))

def test_T11_T12_no_cache(tmp_path):
    from gog_fraud.data.crypto_raw import save_arrays
    p=rawzip(tmp_path,'raw.zip');(tmp_path/'polygon_hybrid_graph.pt').write_bytes(b'not executable pickle')
    a,_=build_observables(p,'polygon');target=tmp_path/'built.npz';save_arrays(target,a)
    with pytest.raises(FileExistsError):save_arrays(target,a)
    b,_=build_observables(p,'polygon');assert content_hash(a)==content_hash(b)

def test_T13_cosine_degenerate_rejected():
    with pytest.raises(ValueError):deterministic_knn(np.arange(1,5)[:,None]*np.array([[0,1.,0]]),IDS,metric='cosine')

def test_T15_invalid_values_edges_weights_scores():
    for key,value in [('x',torch.full((4,4),float('nan'))),('edge_index',torch.tensor([[0],[4]])),('edge_weight',torch.tensor([float('inf'),1.]))]:
        raw=source();setattr(raw,key,value)
        with pytest.raises(ValueError):node_view(raw,IDS,'polygon_contract_clean_v1',labels=torch.zeros(4,dtype=torch.long))
    with pytest.raises(ValueError):validate_score(np.array([np.nan]*4),4)

def test_T17_T18_lineage_no_scalar_score_fabrication():
    for r in [{}, {'input_manifest_hash':'new','split_hash':'s','model_config_hash':'c','status':'SUPPORTED_EXACT'}]:
        with pytest.raises(ValueError):validate_run_identity(r,'new','s','c')

def test_threshold_and_budget():
    y=np.array([0,1,1,0,0,1,1,0]);s=np.array([.1,.8,.8,.9,.1,.8,.8,.9]);v=np.array([True]*4+[False]*4);t=~v
    r=evaluate_scores(y,s,np.array([str(i) for i in range(8)]),v,t)
    assert r['operating_point']['threshold']==.8 and r['thresholded']['tp']==2 and r['thresholded']['fp']==1
    assert r['alert_budgets']['0.01']['k']==1
