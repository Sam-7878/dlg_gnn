"""Recompute fixed paired tests with threshold-grouped (tie-aware) AP."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import FROZEN, OUT, SEEDS, sha, write_json
import historical_statistics as historical


def tied_ap(labels, scores):
    order=np.argsort(-scores,kind='stable')
    y=np.asarray(labels)[order];s=np.asarray(scores)[order]
    if y.sum()==0:return float('nan')
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/y.sum())


def holm(p):
    p=np.array(p,dtype=float);order=np.argsort(p,kind='stable');result=np.empty_like(p)
    result[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return result


def panels():
    result={};identity=None;files=[]
    for kind in ['proposed','mc_dropout','temperature_scaled','tgn','tgat','fraudsage']:
        result[kind]=[]
        for seed in SEEDS:
            path=FROZEN/f'raw_predictions/{kind}/seed{seed}_test.csv'
            f=pd.read_csv(path)
            if identity is None:identity=f[['event_id','timestamp','label']].copy()
            assert identity.equals(f[['event_id','timestamp','label']])
            result[kind].append(f.p_mean.to_numpy(float))
            files.append(dict(path=str(path),sha256=sha(path)))
    return identity,result,files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resamples',type=int,default=10000);args=parser.parse_args()
    identity,p,files=panels();labels=identity.label.to_numpy(int)
    deep=np.mean(p['proposed'],axis=0);mcens=np.mean(p['mc_dropout'],axis=0)
    specs=[('Proposed deterministic vs TGN',p['proposed'],p['tgn']),
           ('Proposed deterministic vs TGAT',p['proposed'],p['tgat']),
           ('Proposed deterministic vs FraudSAGE',p['proposed'],p['fraudsage']),
           ('MC Dropout vs Deterministic',p['mc_dropout'],p['proposed']),
           ('MC Dropout vs Temperature Scaling',p['mc_dropout'],p['temperature_scaled']),
           ('MC Dropout vs Deep Ensemble',p['mc_dropout'],[deep]*5),
           ('MC Dropout Ensemble vs Deep Ensemble',[mcens],[deep])]
    check=[];rng=np.random.default_rng(314)
    original_ap=historical.average_precision
    for name,left,right in specs:
        draw=rng.integers(0,len(labels),len(labels))
        for s in left+right:
            a=tied_ap(labels[draw],s[draw]);b=average_precision_score(labels[draw],s[draw])
            assert np.isclose(a,b,rtol=0,atol=1e-12)
        check.append(dict(comparison=name,bootstrap_draw_legacy_ap=original_ap(labels[draw],left[0][draw]),
                          bootstrap_draw_tie_aware_ap=tied_ap(labels[draw],left[0][draw]),
                          observed_legacy=original_ap(labels,left[0]),observed_tie_aware=tied_ap(labels,left[0])))
    pd.DataFrame(check).to_csv(OUT/'supplement/ap_tie_audit.csv',index=False)
    historical.average_precision=tied_ap
    rows=[]
    for i,(name,left,right) in enumerate(specs):
        result=historical.paired_panel_comparison(labels,left,right,n_resamples=args.resamples,seed=20260910+i)
        rows.append(dict(comparison=name,statistic='threshold-grouped average precision',**result))
        pd.DataFrame(rows).to_csv(OUT/'supplement/statistical_comparisons_corrected.csv',index=False)
        print(name,'corrected p',result['randomization_p_value_two_sided'],flush=True)
    f=pd.DataFrame(rows);f['holm_adjusted_p']=holm(f.randomization_p_value_two_sided)
    f.to_csv(OUT/'supplement/holm_adjusted_statistics.csv',index=False)
    f.to_csv(OUT/'supplement/statistical_comparisons_corrected.csv',index=False)
    write_json(OUT/'reports/statistical_implementation_audit.json',dict(
        frozen_inputs=files,identities_exact=True,resamples=args.resamples,
        finite_permutation_correction='(extreme + 1)/(R + 1)',minimum_p=1/(args.resamples+1),
        mc_vector_same_for_both_comparisons=True,
        score_randomization_null='within each seed, independent event-level score swaps; mean seed AP difference',
        observed_rank_invariance_does_not_imply_null_invariance=True,
        bug_found='Historical AP helper processed tied scores one row at a time; bootstrap duplicates induce ties. Recomputed seven protocols with threshold-grouped AP verified against sklearn.',
        historical_results_preserved=True,seed_resampling=False,
        inference_limitation='event-level resampling assumes exchangeable sampling units; it does not account for residual temporal/entity dependence'))


if __name__=='__main__':main()
