"""All 25 ORIGINAL models and policy/split rows; no replica substitution or fitting."""
from __future__ import annotations
import hashlib
import json
import shutil
from r02_common import ROOT,R01,R02,VERIFY,RUN,read,write,sha256,csv_write,original_unchanged
from r01_common import config,digest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from gog_fraud.streaming.selective import binary_metrics,risk_counts,calibrated
from r01_train import apply_fitted


def same_metrics(expected,actual):
    for key,value in actual.items():
        if key not in expected:continue
        old=expected[key]
        if value is None or (isinstance(value,float) and np.isnan(value)):
            assert pd.isna(old),(key,old,value)
        elif isinstance(value,(float,int,np.number)):
            assert pd.notna(old) and np.isclose(float(old),float(value),atol=1e-10,rtol=0),(key,old,value)
        else:assert str(old if pd.notna(old) else '')==str(value),(key,old,value)


def main():
    VERIFY.mkdir(parents=True,exist_ok=True);cfg=config();original_unchanged()
    expected=pd.read_csv(R01/'analysis/per_seed_metrics.csv');derived=[];registries=[];mc_rows=[]
    for directory in sorted((R01/'models').iterdir()):
        if not (directory/'complete.json').is_file():continue
        record=read(directory/'complete.json');fit=read(directory/'policy_fit.json')
        exp=record['experiment_id'];seed=record['seed'];backbone=record['backbone']
        assert record['run_id']==fit['run_id']==cfg['run_id'] and record['config_sha256']==cfg['config_sha256']
        assert fit['model_hash']==digest({'local':sha256(directory/'local.pt'),'relational':sha256(directory/'relational.pt')})
        assert fit['reference_manifest_hash']==sha256(directory/'reference.npz')
        for name,identity in record['artifacts'].items():assert sha256(directory/name)==identity
        with np.load(directory/'reference.npz',allow_pickle=False) as refs:reference_chains=sorted({i.split(':')[0] for i in refs['ids']})
        assert reference_chains==sorted(record['train_chains'])
        target=exp[5:] if exp.startswith('loco_') else None
        if target:assert target not in reference_chains and target not in fit['train_chains'] and fit['test_chains']==[target]
        for p in fit['policies']:assert p['policy_hash']==digest({k:v for k,v in p.items() if k!='policy_hash'})
        context={'experiment_id':exp,'run_id':cfg['run_id'],'seed':seed,'backbone':backbone}
        for split in ('validation','test'):
            raw_path=directory/f'{split}_predictions.parquet';frame=pd.read_parquet(raw_path)
            assert not frame.sample_id.duplicated().any() and set(frame.label.unique())<=set([0,1])
            assert set(frame.chain_scope)==set(record['train_chains'] if split=='validation' else record['test_chains'])
            assert set(frame.run_id)=={cfg['run_id']} and set(frame.model_hash)=={fit['model_hash']}
            assert set(frame.experiment_id)=={exp} and set(frame.seed)=={seed}
            assert np.isfinite(frame[['raw_local_probability','counterfactual_full_raw_probability','local_probability','selective_probability']].to_numpy()).all()
            with np.load(directory/f'{split}_features.npz',allow_pickle=False) as features:benefit=features['benefit']
            y=frame.label.to_numpy();eligible=frame.reference_count.to_numpy()>0
            def emit(method,budget,prediction,score,route,policy=None):
                for chain in ['pooled']+sorted(frame.chain_scope.unique()):
                    mask=np.ones(len(frame),bool) if chain=='pooled' else frame.chain_scope.eq(chain).to_numpy()
                    row={**context,'split':split,'chain_scope':chain,'method':method,'budget':budget,'deep_fraction':float(route[mask].mean()),**binary_metrics(y[mask],score[mask],prediction[mask])}
                    if policy:row.update({'random_repeat':policy['random_repeat'],'policy_hash':policy['policy_hash'],'budget_error':float(route[mask].mean()-budget),**risk_counts(y[mask],prediction[mask],route[mask])})
                    selection=expected[(expected.experiment_id==exp)&(expected.seed==seed)&(expected.backbone==backbone)&(expected.split==split)&(expected.chain_scope==chain)&(expected.method==method)&(expected.budget==budget)]
                    if policy:selection=selection[selection.random_repeat==policy['random_repeat']]
                    assert len(selection)==1,(directory.name,split,chain,method,budget,len(selection))
                    same_metrics(selection.iloc[0],row);derived.append(row)
            for control in ('local','full','threshold_only'):
                score,pred,route=apply_fitted(frame.raw_local_probability.to_numpy(),frame.counterfactual_full_raw_probability.to_numpy(),benefit,fit,control=control,eligible=eligible)
                assert np.max(abs(score-frame[control+'_probability'].to_numpy()))<=1e-12 and np.array_equal(pred,frame[control+'_prediction'])
                emit(control,float(control=='full'),pred,score,route)
            for policy in fit['policies']:
                score,pred,route=apply_fitted(frame.raw_local_probability.to_numpy(),frame.counterfactual_full_raw_probability.to_numpy(),benefit,fit,policy=policy,eligible=eligible)
                if policy['family']=='margin' and policy['budget']==.25:
                    assert np.max(abs(score-frame.selective_probability.to_numpy()))<=1e-12 and np.array_equal(pred,frame.selective_prediction) and np.array_equal(route,frame.escalated)
                emit(policy['family'],policy['budget'],pred,score,route,policy)
        if exp=='pooled_snapshot' and backbone=='GIN':
            for method in ('degree_logistic','degree_MLP','relational_MLP','knn_mean'):
                frame=pd.read_parquet(directory/f'{method}_predictions.parquet');f=read(directory/f'{method}_fit.json')
                assert not frame.sample_id.duplicated().any() and np.isfinite(frame.probability).all()
                score=calibrated(frame.raw_probability,f['calibration']);assert np.max(abs(score-frame.probability))<=1e-12 and np.array_equal(score>=f['threshold'],frame.prediction)
                values={**context,'split':'test','chain_scope':'pooled','method':method,**binary_metrics(frame.label.to_numpy(),score,frame.prediction.to_numpy())}
                selected=expected[(expected.experiment_id==exp)&(expected.seed==seed)&(expected.backbone==backbone)&(expected.method==method)]
                assert len(selected)==1;same_metrics(selected.iloc[0],values);derived.append(values)
            original_mc=pd.read_csv(directory/'mc_metrics.csv')
            for T in (1,3,5,8):
                f=read(directory/f'mc_T{T}_fit.json')
                for split in ('validation','test'):
                    frame=pd.read_parquet(directory/f'mc_T{T}_{split}.parquet');score=calibrated(frame.raw_probability,f['mapping'])
                    assert np.max(abs(score-frame.probability))<=1e-12 and np.array_equal(score>=f['threshold'],frame.prediction)
                    values={'seed':seed,'split':split,'T':T,**binary_metrics(frame.label.to_numpy(),score,frame.prediction.to_numpy())}
                    truth=(frame.prediction.to_numpy()!=frame.label.to_numpy()).astype(int)
                    values['error_auroc']=float(roc_auc_score(truth,frame.error_score)) if len(np.unique(truth))==2 else None
                    selected=original_mc[(original_mc['T']==T)&(original_mc.split==split)]
                    assert len(selected)==1;same_metrics(selected.iloc[0],values);mc_rows.append(values)
        registries.append({**context,'original_model_dir':str(directory.relative_to(ROOT)),'original_completion_sha256':sha256(directory/'complete.json'),
            'reference_chains_checked':reference_chains,'source_only_target_excluded':bool(target and target not in reference_chains) if target else None,
            'verification_scope':'artifact identity, all raw split/policy metrics and recorded source-chain scopes; source training code separately inspected, not new independent training'})
        print('R02 RAW MODEL VERIFIED',directory.name,flush=True)
    assert len(registries)==25 and len(derived)==len(expected)
    pd.DataFrame(derived).to_csv(VERIFY/'per_seed_metrics_recomputed.csv',index=False);pd.DataFrame(mc_rows).to_csv(VERIFY/'mc_recomputed.csv',index=False)
    with (VERIFY/'model_registry_checked.jsonl').open('w') as handle:
        for row in registries:handle.write(json.dumps(row,sort_keys=True)+'\n')
    # Re-execute the recorded resampling algorithm into a new, private shadow;
    # original model/raw files are read-only links, never output destinations.
    working=R02/'working';working.mkdir(exist_ok=True)
    for name in ('models','runtime','stress'):
        path=working/name
        if not path.exists():path.symlink_to(R01/name,target_is_directory=True)
    shutil.copy2(R01/'data_audit.json',working/'data_audit.json')
    import r01_analysis as analysis
    analysis.output=lambda:working
    analysis.main()
    differences=[]
    for path in sorted((working/'analysis').glob('*.csv')):
        old=R01/'analysis'/path.name
        if not old.is_file():continue
        a=pd.read_csv(old);b=pd.read_csv(path);assert a.shape==b.shape,(path.name,a.shape,b.shape)
        numeric=a.select_dtypes(include=np.number).columns
        for col in numeric:assert np.allclose(a[col],b[col],atol=1e-10,rtol=0,equal_nan=True),(path.name,col)
        differences.append({'artifact':path.name,'rows':len(a),'old_sha256':sha256(old),'recomputed_sha256':sha256(path),'numeric_max_error_limit':1e-10,'status':'VERIFIED','scope':'CSV numeric content comparison; metadata paths/bytes may differ'})
    shutil.copy2(working/'analysis/paired_statistics.csv',VERIFY/'paired_statistics_recomputed.csv')
    csv_write(VERIFY/'table_figure_numeric_diff.csv',differences)
    write(VERIFY/'raw_numeric_validation.json',{'verification_id':RUN,'source_run_id':cfg['run_id'],'N_original_models':25,'N_raw_metric_rows_recomputed':len(derived),'N_MC_rows_recomputed':len(mc_rows),
        'all_five_seeds_checked':True,'raw_to_policy_composition_checked':True,'recorded_bootstrap_reexecuted':True,'recomputed_analysis_dir':str((working/'analysis').relative_to(ROOT)),
        'scope':'same-host implementation-side arithmetic/source validation; no new training, no external independent approval'})
    original_unchanged();print('R02 ALL ORIGINAL RAW METRICS PASS',len(derived),flush=True)


if __name__=='__main__':main()
