"""Descriptive statistics from aligned raw records, not from target table values."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from r01_common import ROOT,config,output,read,write,sha256,digest
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.streaming.selective import binary_metrics,risk_counts
from r01_train import apply_fitted


def holm(p):
    p=np.asarray(p);order=np.argsort(p);result=np.empty(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result


def paired_stats(y,p0,p1,h0,h1,seed,resamples,blocks=None):
    correct0=h0==y;correct1=h1==y;b=int((correct0&~correct1).sum());c=int((~correct0&correct1).sum())
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    groups=[np.flatnonzero(blocks==key) for key in np.unique(blocks)] if blocks is not None else None
    def measures(ix):
        yy=y[ix]
        values=[]
        for prob,pred in ((p0,h0),(p1,h1)):
            hh=pred[ix];tp=int(((hh==1)&(yy==1)).sum());den=int(yy.sum()+hh.sum())
            values.append(np.array([average_precision_score(yy,prob[ix]) if yy.sum() and (yy==0).any() else np.nan,
                2*tp/den if yy.sum() and den else np.nan,tp/yy.sum() if yy.sum() else np.nan]))
        return values[1]-values[0]
    estimates=[]
    for _ in range(resamples):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]) if groups is not None else np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        estimates.append(measures(ix))
    estimates=np.asarray(estimates);delta=measures(np.arange(len(y)))
    result={'n_local_correct_selective_wrong':b,'n_local_wrong_selective_correct':c,
        'exact_mcnemar_p':float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.,
        'bootstrap_unit':'UTC snapshot month block' if groups is not None else 'class-stratified paired contract',
        'resamples':resamples,'block_count':len(groups) if groups is not None else None,
        'scope':'conditional on the fitted model and retained test; graph/time dependence remains; not prospective inference'}
    for i,metric in enumerate(('ap','f1','recall')):
        valid=estimates[:,i][np.isfinite(estimates[:,i])]
        result.update({f'delta_{metric}':float(delta[i]),f'{metric}_ci_low':float(np.quantile(valid,.025)) if len(valid) else None,
            f'{metric}_ci_high':float(np.quantile(valid,.975)) if len(valid) else None,f'{metric}_valid_resamples':len(valid)})
    return result


def calibration_rows(y,p,context):
    rows=[];ece=0.;n=len(y)
    if not n:return [],{'N':0,'brier':None,'nll':None,'ece':None}
    bins=np.minimum(9,(p*10).astype(int))
    for i in range(10):
        mask=bins==i;count=int(mask.sum());positive=int(y[mask].sum())
        mean=float(p[mask].mean()) if count else None;rate=positive/count if count else None
        if count:
            ece+=count/n*abs(mean-rate);z=1.95996398454;center=(rate+z*z/(2*count))/(1+z*z/count)
            half=z*np.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/(1+z*z/count);low,high=center-half,center+half
        else:low=high=None
        rows.append({**context,'bin':i,'lower':i/10,'upper':(i+1)/10,'N':count,'N_positive':positive,'mean_probability':mean,'observed_rate':rate,'observed_rate_wilson_low':low,'observed_rate_wilson_high':high})
    pp=np.clip(p,1e-8,1-1e-8)
    return rows,{'N':n,'brier':float(np.mean((p-y)**2)),'nll':float(np.mean(-(y*np.log(pp)+(1-y)*np.log1p(-pp)))),'ece':ece,'binning':'10 equal-width bins; Wilson interval for observed rate'}


def historical_audit(out,cfg):
    rows=[];metrics=[]
    for seed in cfg['seeds']:
        path=ROOT/cfg['historical_results']/f'offline/seed{seed}/test_predictions.csv';frame=pd.read_csv(path)
        if frame.sample_id.duplicated().any():raise ValueError('historical repeated IDs')
        y=frame.label.to_numpy();h0=frame.direct_only_prediction.to_numpy();h1=frame.primary_prediction.to_numpy()
        row=paired_stats(y,frame.direct_only_score.to_numpy(),frame.primary_score.to_numpy(),h0,h1,seed,200)
        rows.append({'seed':seed,'raw_sha256':sha256(path),**row})
        for prefix in ('direct_only','primary','full_deep'):
            metrics.append({'seed':seed,'historical_method':prefix,**binary_metrics(y,frame[prefix+'_score'],frame[prefix+'_prediction'])})
    adjusted=holm([r['exact_mcnemar_p'] for r in rows])
    for row,p in zip(rows,adjusted):row['holm_p']=float(p)
    pd.DataFrame(rows).to_csv(out/'historical_r4_statistics_audit.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'historical_r4_metrics_recomputed.csv',index=False)
    write(out/'multiplicity_family.json',{'new_family':'five pooled GIN primary local/selective paired-correctness tests','historical_family':'five R4 seed tests recomputed separately','test':'exact two-sided binomial McNemar','correction':'Holm step-down, m=5 within each family','not_a_test_of_F1':True,'not_preregistered':True})


def main():
    cfg=config();out=output();dest=out/'analysis';dest.mkdir(exist_ok=True);historical_audit(dest,cfg)
    expected=25;model_dirs=sorted(p.parent for p in (out/'models').glob('*/complete.json'))
    if len(model_dirs)!=expected:raise ValueError(f'incomplete campaign: {len(model_dirs)}/{expected} completed model identities')
    all_metrics=[];registry=[];statistics=[];block_stats=[];risk=[];bins=[];cal=[];cases=[];prevalence=[];alerts=[];temporal=[];effects=[];mc=[]
    for model_dir in model_dirs:
        record=read(model_dir/'complete.json');fit=read(model_dir/'policy_fit.json');exp=record['experiment_id'];seed=record['seed'];backbone=record['backbone']
        if record['config_sha256']!=cfg['config_sha256'] or fit['run_id']!=cfg['run_id']:raise ValueError('mixed run/config identity')
        if exp.startswith('loco_') and exp[5:] in record['train_chains']:raise ValueError('target chain used for selection')
        for filename,expected_hash in record['artifacts'].items():
            if sha256(model_dir/filename)!=expected_hash:raise ValueError('artifact integrity failure '+filename)
        frame=pd.read_parquet(model_dir/'test_predictions.parquet');validation=pd.read_parquet(model_dir/'validation_predictions.parquet')
        if set(frame.experiment_id)!=set([exp]) or frame.sample_id.duplicated().any():raise ValueError('mixed experiment or duplicate test record')
        metrics=pd.read_csv(model_dir/'metrics.csv');all_metrics.append(metrics)
        registry.append({**record,'model_dir':str(model_dir.relative_to(ROOT)),'model_hash':fit['model_hash'],'reference_manifest_hash':fit['reference_manifest_hash'],
            'prediction_hash':sha256(model_dir/'test_predictions.parquet'),'policy_hashes':[p['policy_hash'] for p in fit['policies']]})
        context={'experiment_id':exp,'run_id':cfg['run_id'],'seed':seed,'backbone':backbone}
        y=frame.label.to_numpy();p0=frame.local_probability.to_numpy();p1=frame.selective_probability.to_numpy();h0=frame.local_prediction.to_numpy();h1=frame.selective_prediction.to_numpy();route=frame.escalated.to_numpy()
        target=frame.snapshot_cutoff.to_numpy()
        temporal.append({**context,'N':len(frame),'model_fit_after_target_count':int((record['model_fit_data_max_time']>target).sum()),
            'calibration_or_policy_after_target_count':int((record['calibration_data_max_time']>target).sum()),
            'future_selected_reference_count':int((frame.selected_reference_max_cutoff.fillna(-1)>target).sum()),
            'edge_feature_time':'unknown','label_available_time':'unknown','strict_online_eligible':False,
            'source_only_target_label_access':False if exp.startswith('loco_') else 'not_applicable'})
        # Publishing pseudonymous aligned predictions enables numeric replay, not raw source redistribution.
        public=frame.drop(columns=['sample_id','contract_id','event_start'],errors='ignore').copy()
        public.insert(0,'sample_key',[hashlib.sha256(i.encode()).hexdigest() for i in frame.sample_id])
        public.to_csv(dest/f'{model_dir.name}_paired_predictions.csv',index=False)
        if exp=='pooled_snapshot' and backbone=='GIN':
            statistics.append({**context,**paired_stats(y,p0,p1,h0,h1,seed,cfg['statistics']['bootstrap_resamples'])})
            blocks=pd.to_datetime(target,unit='s',utc=True).strftime('%Y-%m').to_numpy()
            block_stats.append({**context,**paired_stats(y,p0,p1,h0,h1,seed,500,blocks)})
            mc.append(pd.read_csv(model_dir/'mc_metrics.csv'))
        for chain in ['pooled']+sorted(frame.chain_scope.unique()):
            mask=np.ones(len(y),bool) if chain=='pooled' else frame.chain_scope.eq(chain).to_numpy()
            risk.append({**context,'chain_scope':chain,**risk_counts(y[mask],h1[mask],route[mask])})
            for path,selection in [('all',mask),('direct',mask&~route),('escalated',mask&route)]:
                bb,cc=calibration_rows(y[selection],p1[selection],{**context,'chain_scope':chain,'path':path});bins+=bb;cal.append({**context,'chain_scope':chain,'path':path,**cc})
        bsc=(frame.chain_scope=='bsc')&(frame.label==1)&(frame.selective_prediction==0)
        for _,r in frame[bsc].sort_values('local_probability').head(20).iterrows():
            cases.append({**context,'sample_key':hashlib.sha256(r.sample_id.encode()).hexdigest(),'local_probability':float(r.local_probability),'selective_probability':float(r.selective_probability),'full_probability':float(r.full_probability),
                'high_confidence_negative_p_lt_005':bool(r.local_probability<.05),'escalated':bool(r.escalated),'reference_count':int(r.reference_count),'cutoff':int(r.snapshot_cutoff),'interpretation':'descriptive false-negative case; no topology/velocity causal attribution'})
        if exp=='pooled_snapshot' and backbone=='GIN':
            with np.load(model_dir/'test_features.npz') as features:bf=features['benefit']
            raw_fast=frame.raw_local_probability.to_numpy();raw_deep=frame.counterfactual_full_raw_probability.to_numpy();eligible=frame.reference_count.to_numpy()>0
            for policy in fit['policies']:
                score,pred,rr=apply_fitted(raw_fast,raw_deep,bf,fit,policy=policy,eligible=eligible)
                correct=frame.threshold_only_prediction.to_numpy()==y if policy['family']=='margin' and policy['budget']==.25 else (frame.local_probability.to_numpy()>=policy['final_threshold'])==y
                effects.append({**context,'family':policy['family'],'budget':policy['budget'],'random_repeat':policy['random_repeat'],'N_escalated':int(rr.sum()),
                    'corrections':int((rr&~correct&(pred==y)).sum()),'harms':int((rr&correct&(pred!=y)).sum()),'denominator':int(rr.sum()),
                    'comparison':'local calibrated score at the identical final threshold; separates threshold effects'})
            for q in (.01,.05,.1):
                threshold=float(np.quantile(validation.selective_probability,1-q,method='higher'));pred=p1>=threshold
                alerts.append({**context,'validation_alert_budget':q,'validation_frozen_threshold':threshold,**binary_metrics(y,p1,pred)})
            for pi in (.01,.05,.1):
                weights=np.where(y==1,pi/max(y.mean(),1e-12),(1-pi)/max(1-y.mean(),1e-12))
                for name,prob,pred in [('local',p0,h0),('selective',p1,h1),('full',frame.full_probability.to_numpy(),frame.full_prediction.to_numpy())]:
                    tp=float(weights[(y==1)&(pred==1)].sum());fp=float(weights[(y==0)&(pred==1)].sum());fn=float(weights[(y==1)&(pred==0)].sum())
                    prevalence.append({**context,'method':name,'reweighted_prevalence':pi,'weighted_ap':float(average_precision_score(y,prob,sample_weight=weights)),'weighted_precision':tp/(tp+fp) if tp+fp else None,'weighted_f1':2*tp/(2*tp+fp+fn),'scope':'class-prior reweighting of same test, not new temporal validation'})
    for row,p in zip(statistics,holm([r['exact_mcnemar_p'] for r in statistics])):row['holm_p']=float(p)
    merged=pd.concat(all_metrics,ignore_index=True);merged.to_csv(dest/'per_seed_metrics.csv',index=False)
    test=merged[merged.split=='test'].copy();test['budget']=test.budget.fillna(-1)
    keys=['experiment_id','backbone','chain_scope','method','budget'];numeric=['ap','f1','recall','precision','mcc','deep_fraction','fp_per_1000','alert_rate','N','N_positive','tp','tn','fp','fn']
    within=test.groupby(keys+['seed'])[numeric].mean().reset_index() # random repeats averaged before seeds
    aggregate=within.groupby(keys)[numeric].agg(['mean','std','count']);aggregate.columns=['_'.join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(dest/'aggregate_metrics.csv',index=False)
    for name,rows in [('paired_statistics',statistics),('time_block_sensitivity',block_stats),('risk_counts',risk),('reliability_bins',bins),('calibration_path_metrics',cal),('bsc_error_cases',cases),('prevalence_sensitivity',prevalence),('validation_alert_budgets',alerts),('temporal_audit_rows',temporal),('relational_corrections_harms',effects)]:pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)
    pd.concat(mc,ignore_index=True).to_csv(dest/'mc_metrics.csv',index=False)
    with (dest/'experiment_registry.jsonl').open('w') as handle:
        for record in registry:handle.write(__import__('json').dumps(record,sort_keys=True)+'\n')
    write(dest/'temporal_audit.json',{'scope':'retrospective snapshots only','strict_online_validity':'claim removed','edge_time':'unknown','label_available_time':'unknown','reference_cutoff_violations':sum(t['future_selected_reference_count'] for t in temporal),
        'model_weight_future_target_count':sum(t['model_fit_after_target_count'] for t in temporal),'policy_future_target_count':sum(t['calibration_or_policy_after_target_count'] for t in temporal),
        'training_features_available_before_model_fit':'unknown (cached edge-time lineage absent)','training_labels_available_before_model_fit':'unknown',
        'local_feature_event_time_not_after_target_cutoff':'unknown','no_target_label_access_during_source_only_selection':'verified chain exclusion in fit artifacts; no target adaptation',
        'no_duplicate_contract_leakage_outside_declared_protocol':read(out/'data_audit.json')['duplicate_contracts_between_splits']==0})
    write(dest/'analysis_validation.json',{'completed_model_identities':len(registry),'expected':25,'registry_hash':sha256(dest/'experiment_registry.jsonl'),'pooled_GIN_seed_count':len(statistics),'paired_family_size':5,
        'metric':'sklearn average_precision_score, not trapezoidal PR-AUC','seed_mean_not_pooled_predictions':True,'negative_only_AP_F1_undefined':True,
        'bootstrap':'paired class-stratified contract; separate 500 replicate month-block sensitivity','analysis_scope':'descriptive after historical test inspection'})
    print('ANALYSIS COMPLETE',len(registry),flush=True)


if __name__=='__main__':main()
