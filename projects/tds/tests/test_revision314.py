"""Scientific invariants behind the revision's additional experiments."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Batch, Data

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/revision314'))
from statistical_audit import tied_ap, holm
from historical_models import TemporalMemoryBaseline, class_balanced_focal_loss
from train import NoChainGIN
from outputs import operating
from strict_time import make_graph


def graph(chain=0,label=0):
    return Data(x=torch.tensor([[0.,1.,1.],[1.,0.,1.]]),
                edge_index=torch.tensor([[0],[1]]),chain_id=torch.tensor(chain),
                timestamp=torch.tensor(2),y=torch.tensor(float(label)))


def test_tied_ap_matches_sklearn():
    y=np.array([1,0,1,0]);p=np.array([.5,.5,.2,.2])
    assert abs(tied_ap(y,p)-average_precision_score(y,p))<1e-12
    assert tied_ap(y,np.ones(4))==.5


def test_ap_bootstrap_duplicates_matches_threshold_grouping():
    y=np.array([1,0,1,0]);p=np.array([.9,.8,.7,.1]);draw=np.array([0,0,1,2,2,3])
    assert abs(tied_ap(y[draw],p[draw])-average_precision_score(y[draw],p[draw]))<1e-12


def test_holm_monotone_bounded_and_known_result():
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])


def test_no_chain_logits_ignore_chain_covariate():
    torch.manual_seed(7);m=NoChainGIN(3).eval()
    b=Batch.from_data_list([graph(0),graph(2)])
    with torch.no_grad():p=m(b)
    assert torch.equal(p[0],p[1])


def test_tgn_predict_then_update_is_label_independent():
    torch.manual_seed(7);m=TemporalMemoryBaseline().eval();memory=m.initial_memory(torch.device('cpu'))
    b=Batch.from_data_list([graph()]);t=torch.tensor([.5])
    with torch.no_grad():
        a,next_a=m.forward_event(b,t,memory)
        b.y.fill_(1);c,next_c=m.forward_event(b,t,memory)
    assert torch.equal(a,c) and torch.equal(next_a,next_c)
    assert torch.equal(memory,torch.zeros_like(memory))
    assert not torch.equal(next_a,memory)


def test_focal_gamma_zero_is_weighted_bce():
    x=torch.tensor([-.5,.2,1.]);y=torch.tensor([0.,1.,1.])
    expected=torch.nn.functional.binary_cross_entropy_with_logits(x,y,pos_weight=torch.tensor(2.))
    assert torch.allclose(class_balanced_focal_loss(x,y,2.,gamma=0),expected)


def test_operating_budget_ceil_stable_ties():
    y=np.array([1,0,0,1]+[0]*96);p=np.ones(100)
    out=operating(y,p)
    assert out['top_1pct_budget']==1 and out['top_1pct_captured']==1
    assert out['top_5pct_budget']==5 and out['top_5pct_captured']==2
    assert out['top_100_precision']==.02


def test_empty_strict_graph_is_explicit_zero_information():
    from types import SimpleNamespace
    from collections import Counter
    row=SimpleNamespace(label=1,chain_index=0,timestamp=2)
    empty=pd.DataFrame({'from':[],'to':[]})
    g=make_graph(empty,Counter(),Counter(),row)
    assert g.x.shape==(1,3) and torch.equal(g.x,torch.zeros_like(g.x))
    assert g.edge_index.shape==(2,0)
