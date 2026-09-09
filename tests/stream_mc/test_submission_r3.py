import copy
import numpy as np
import pytest
import torch
from scipy.special import expit,logit
from gog_fraud.production.submission_r3 import (apply_policy,calibrate,best_threshold,FrozenRelations,local_predict)
from analysis.submission_r3_statistics import (paired_bootstrap,mcnemar,holm,exact_upper,calibration_metrics)
from evidence.experiment_identity import ExperimentIdentity,require_compatible


def selection():
    return {'fast_map':{'coefficient':.5,'intercept':-2},'deep_map':{'coefficient':2,'intercept':1},
        'fast_threshold':.2,'route_cutoff':.12,'fast_weight':.2,
        'final_threshold':.6,'full_threshold':.65}


def test_calibrated_routing_and_logit_fusion_not_legacy_raw_path():
    s=selection(); raw=np.array([.5,.01]); deep=np.array([.9,np.nan])
    score,route,pred=apply_policy(raw,deep,s)
    assert route.tolist()==[True,False]  # raw-score routing would choose neither
    cf=calibrate(raw,s['fast_map']); cd=calibrate(deep[:1],s['deep_map'])
    assert score[0]==pytest.approx(expit(.2*logit(cf[0])+.8*logit(cd[0])))
    assert score[1]==cf[1] and pred.tolist()==[1,0]


def test_direct_never_reads_missing_deep_and_full_routes_everything():
    raw=np.array([.01,.5]); s=selection()
    _,route,_=apply_policy(raw,[np.nan,np.nan],s,'direct_only'); assert not route.any()
    _,route,_=apply_policy(raw,[.2,.8],s,'full_deep'); assert route.all()


def test_relation_target_independence_and_star_size():
    r=FrozenRelations(np.arange(40,dtype=np.float32).reshape(10,4),np.linspace(0,1,10),3)
    first=np.array([[3,4,5,6]],np.float32); other=np.ones((1,4),np.float32)*1000
    g=r.graphs(first,[.4])[0]; batch=r.graphs(np.r_[first,other],[.4,.8])
    assert torch.equal(g.x,batch[0].x) and torch.equal(g.edge_index,batch[0].edge_index)
    assert g.num_nodes==4 and g.num_edges==6


def test_threshold_ties_matches_brute_force():
    y=np.array([0,1,0,1,1]); p=np.array([.2,.2,.6,.6,.9])
    threshold,f1=best_threshold(y,p)
    expected=max(2*((p>=t)&(y==1)).sum()/max(1,y.sum()+(p>=t).sum()) for t in np.r_[0,p,1])
    assert f1==expected


def test_paired_bootstrap_identical_predictions_zero():
    y=np.array([0,1,0,1,1,0]); p=np.array([.1,.7,.8,.9,.3,.2]); pred=p>=.5
    result,_=paired_bootstrap(y,p,p,pred,pred,resamples=40)
    assert all(v['delta']==v['ci_low']==v['ci_high']==0 for v in result.values())
    assert mcnemar(y,pred,pred)['p_value']==1


def test_holm_and_exact_bounds():
    assert np.allclose(holm([.01,.03,.2]),[.03,.06,.2])
    assert exact_upper(0,0) is None and exact_upper(5,5)==1
    assert exact_upper(0,10)==pytest.approx(1-.05**.1)


def identity():
    return ExperimentIdentity('offline_contract_detection','contract','population','cache-v3','test','pooled',11,
        'model','primary','policy','cal','threshold','fusion',1,'none','none',0,'predictive','git','config').row()


def test_identity_mixed_population_lane_policy_fail_closed():
    first=identity(); second=copy.deepcopy(first); second['evaluation_population_id']='other'
    with pytest.raises(ValueError,match='evaluation_population'):require_compatible([first,second])
    require_compatible([first,second],group_columns=('evaluation_population_id',))
    second=copy.deepcopy(first);second['policy_family']='legacy_dual'
    with pytest.raises(ValueError,match='diagnostic'): require_compatible([second])


def test_calibration_perfect_extremes_have_zero_error():
    row=calibration_metrics(np.array([0,0,1,1]),np.array([0.,0.,1.,1.]))
    assert row['brier']==row['ece_10']==row['ece_20']==row['adaptive_ece']==0
