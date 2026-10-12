"""Portable public-data verification. No original addresses/private Parquet needed.

Default: standard-library archive/source checks, 280 primary raw rows, exact
paired statistics. --full-policy: 6,065 raw policy/control rows and 40 MC rows,
using disclosed CSV/NPZ and the frozen policy algorithm (scientific deps).
Neither path is retraining, independent peer review, or a timing rerun.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
from r01_public import ROOT,PROJECT,verified_archive,numeric,historical_numeric,sha,require


def full_policy(z):
    # r01_common configures optional project-local PyArrow before pandas.
    import r01_common
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from gog_fraud.streaming.selective import binary_metrics,risk_counts,calibrated
    from r01_train import apply_fitted
    expected=pd.read_csv(io.BytesIO(z.read('analysis/per_seed_metrics.csv')));expected_mc=pd.read_csv(io.BytesIO(z.read('analysis/mc_metrics.csv')));output=[];mc=[]
    directories=sorted(n.rsplit('/',1)[0] for n in z.namelist() if n.startswith('models/') and n.endswith('/complete.json'))
    require(len(directories)==25,'not all 25 original models present')
    def frame(name):return pd.read_csv(io.BytesIO(z.read(name)),float_precision='round_trip')
    def fit(name):return json.loads(z.read(name))
    def compare(row,target):
        require(len(target)==1,'ambiguous policy metric identity');truth=target.iloc[0]
        for key,value in row.items():
            if key not in truth or not (isinstance(value,(float,int,np.number)) or value is None):continue
            require((value is None and pd.isna(truth[key])) or (value is not None and pd.notna(truth[key]) and abs(float(value)-float(truth[key]))<=1e-10),'public raw metric mismatch: '+key)
    for directory in directories:
        record=fit(directory+'/complete.json');policy_fit=fit(directory+'/policy_fit.json');exp=record['experiment_id'];seed=record['seed'];backbone=record['backbone']
        require(record['run_id']==policy_fit['run_id']=='selectivestream_r01_20261011','mixed run identity')
        if exp.startswith('loco_'):require(exp[5:] not in record['train_chains'] and exp[5:] not in policy_fit['train_chains'],'source-only target entered fit')
        for split in ('validation','test'):
            raw=frame(directory+'/'+split+'_predictions.csv');require(not raw.sample_id.duplicated().any(),'duplicate aliases')
            with np.load(io.BytesIO(z.read(directory+'/'+split+'_features.npz')),allow_pickle=False) as values:benefit=values['benefit']
            y=raw.label.to_numpy();eligible=raw.reference_count.to_numpy()>0
            choices=[(control,float(control=='full'),None) for control in ('local','full','threshold_only')]+[(p['family'],p['budget'],p) for p in policy_fit['policies']]
            for method,budget,p in choices:
                # Public CSV uses pandas' shortest round-tripping float32 text.
                # Restore the executed raw neural dtype before calibration:
                # parsing these decimals directly as float64 changes threshold ties.
                score,pred,route=apply_fitted(raw.raw_local_probability.to_numpy(dtype=np.float32),raw.counterfactual_full_raw_probability.to_numpy(dtype=np.float32),benefit,policy_fit,policy=p,control=None if p else method,eligible=eligible)
                for chain in ['pooled']+sorted(raw.chain_scope.unique()):
                    mask=np.ones(len(raw),bool) if chain=='pooled' else raw.chain_scope.eq(chain).to_numpy()
                    row={'experiment_id':exp,'seed':seed,'backbone':backbone,'split':split,'chain_scope':chain,'method':method,'budget':budget,'deep_fraction':float(route[mask].mean()),**binary_metrics(y[mask],score[mask],pred[mask])}
                    if p:row.update({'random_repeat':p['random_repeat'],'budget_error':float(route[mask].mean()-budget),**risk_counts(y[mask],pred[mask],route[mask])})
                    target=expected[(expected.experiment_id==exp)&(expected.seed==seed)&(expected.backbone==backbone)&(expected.split==split)&(expected.chain_scope==chain)&(expected.method==method)&(expected.budget==budget)]
                    if p:target=target[target.random_repeat==p['random_repeat']]
                    compare(row,target);output.append(row)
        if exp=='pooled_snapshot' and backbone=='GIN':
            for method in ('degree_logistic','degree_MLP','relational_MLP','knn_mean'):
                raw=frame(directory+'/'+method+'_predictions.csv');f=fit(directory+'/'+method+'_fit.json');probability=raw.raw_probability.to_numpy(dtype=np.float64 if method=='knn_mean' else np.float32);score=calibrated(probability,f['calibration'])
                require(np.max(abs(score-raw.probability))<1e-10 and np.array_equal(score>=f['threshold'],raw.prediction),'light control calibration drift')
                row={'experiment_id':exp,'seed':seed,'backbone':backbone,'method':method,**binary_metrics(raw.label.to_numpy(),score,raw.prediction.to_numpy())}
                compare(row,expected[(expected.experiment_id==exp)&(expected.seed==seed)&(expected.backbone==backbone)&(expected.method==method)]);output.append(row)
            for T in (1,3,5,8):
                f=fit(directory+f'/mc_T{T}_fit.json')
                for split in ('validation','test'):
                    raw=frame(directory+f'/mc_T{T}_{split}.csv');score=calibrated(raw.raw_probability.to_numpy(dtype=np.float32),f['mapping'])
                    require(np.max(abs(score-raw.probability))<1e-10 and np.array_equal(score>=f['threshold'],raw.prediction),'per-T MC calibration drift')
                    truth=(raw.prediction.to_numpy()!=raw.label.to_numpy()).astype(int)
                    row={'seed':seed,'T':T,'split':split,**binary_metrics(raw.label.to_numpy(),score,raw.prediction.to_numpy()),'error_auroc':float(roc_auc_score(truth,raw.error_score)) if len(np.unique(truth))==2 else None}
                    compare(row,expected_mc[(expected_mc.seed==seed)&(expected_mc['T']==T)&(expected_mc.split==split)]);mc.append(row)
        print('PUBLIC RAW POLICY VERIFIED',directory,flush=True)
    require(len(output)==6065 and len(mc)==40,'public full policy/MC coverage mismatch')
    return output,mc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full-policy',action='store_true');parser.add_argument('--output',type=Path,default=PROJECT/'results/public_recomputed_r02');args=parser.parse_args()
    z,manifest=verified_archive()
    with z:
        for name,identity in manifest['current_source_hashes'].items():require((ROOT/name).is_file() and sha((ROOT/name).read_bytes())==identity,'original curated source drift: '+name)
        primary,paired=numeric(z);historical=historical_numeric(z)
        full,mc=full_policy(z) if args.full_policy else ([],[])
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in [('primary_metrics.csv',primary),('exact_statistics.csv',paired),('historical_exact_statistics.csv',historical),('full_policy_metrics.csv',full),('mc_metrics.csv',mc)]:
        if not data:continue
        fields=list(dict.fromkeys(key for row in data for key in row))
        with (args.output/name).open('w',newline='') as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(data)
    report={'status':'PASS','model_identities':25,'primary_raw_rows':len(primary),'full_policy_rows':len(full),'MC_rows':len(mc),'exact_seed_tests':len(paired),'separate_historical_seed_tests':len(historical),
        'scope':'public pseudonymous raw arithmetic and curated source identity; no original address mapping, retraining, new timing or independent scientific approval'}
    (args.output/'public_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
