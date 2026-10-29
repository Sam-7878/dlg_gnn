"""Read-only scoring, calibration and conditional uncertainty for sealed runs."""
import argparse
import json
import math
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,roc_auc_score
from common import OUT,PRIVATE,SUPP,FROZEN,SEEDS,packed,frame_from,original,sha,write

hm=original('historical_metrics')


def metric(y,p):
    m=hm.classification_metrics(y,p)
    # Ranking uses unmodified scores; the preserved probability-metric convention clips at1e-7.
    m['auc_pr']=float(average_precision_score(y,p));m['auc_roc']=float(roc_auc_score(y,p));return m


def ap(y,p):
    if not np.sum(y):return float('nan')
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
    tp=np.cumsum(y)[ends];return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/tp[-1])


def temperature(y,p):
    z=logit(np.clip(p,1e-12,1-1e-12))
    def loss(t):
        v=z/np.exp(t)
        return float(np.mean(np.logaddexp(0,v)-y*v))
    result=minimize_scalar(loss,bounds=(-4,4),method='bounded')
    assert result.success
    return float(np.exp(result.x)),float(result.fun)


def operating(y,p):
    order=np.argsort(-p,kind='stable');rows=[]
    for k in sorted(set([math.ceil(.01*len(y)),math.ceil(.05*len(y)),37,183,50,100])):
        if k>len(y):continue
        cutoff=p[order[k-1]];above=p>cutoff;tied=p==cutoff;slots=k-int(above.sum())
        captured=int(y[order[:k]].sum());n_tie=int(tied.sum());pos_tie=int(y[tied].sum())
        rows.append(dict(budget=k,captured=captured,precision=captured/k,recall=captured/y.sum(),
            boundary_tie_size=n_tie,tie_expected_captured=float(y[above].sum()+slots*pos_tie/n_tie),
            tie_min_captured=int(y[above].sum()+max(0,slots-(n_tie-pos_tie))),
            tie_max_captured=int(y[above].sum()+min(slots,pos_tie)),
            tie_policy='stable frozen sample order; fractional expectation and attainable bounds also reported'))
    return rows


def diagnostics(frame=None):
    if frame is None:frame=frame_from(packed())
    train=frame[frame.original_split=='train'];tmin=int(train.timestamp.min());scale=int(train.timestamp.max()-tmin)
    ranges=[]
    for split,g in frame.groupby('original_split',sort=False):
        t=g.timestamp.to_numpy();ideal=(t-tmin)/scale;actual=(t.astype(np.float32)-np.float32(tmin))/np.float32(scale)
        ranges.append(dict(split=split,n=len(g),raw_min=int(t.min()),raw_max=int(t.max()),train_min=tmin,train_scale_seconds=scale,
            normalized_min=float(actual.min()),normalized_max=float(actual.max()),ideal_float64_min=float(ideal.min()),ideal_float64_max=float(ideal.max()),
            beyond_train_count=int(((actual<0)|(actual>1)).sum()),float32_normalization_error_max=float(np.max(abs(actual-ideal))),
            max_initial_harmonic_phase_extrapolation=float(max(0,actual.max()-1)*1000)))
    pd.DataFrame(ranges).to_csv(OUT/'temporal_input_ranges.csv',index=False)
    intersections=[]
    for key in ['chain:contract','contract_hex_cross_chain']:
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            left=frame[frame.original_split==a];right=frame[frame.original_split==b]
            aa=set(left.chain+':'+left.contract) if key=='chain:contract' else set(left.contract)
            bb=set(right.chain+':'+right.contract) if key=='chain:contract' else set(right.contract)
            intersections.append(dict(entity=key,left=a,right=b,intersection=len(aa&bb)))
    pd.DataFrame(intersections).to_csv(OUT/'entity_overlap.csv',index=False)
    write(OUT/'state_semantics_audit.json',dict(implementation='revision314/historical_models.py:TemporalMemoryBaseline',
        display_name='chain-level recurrent-state comparator',memory_key='chain_index',shape=[3,48],
        reset='zero at each training epoch and every validation/test partition',
        prediction='current graph + previous chain state; then update GRU from current summary and time',
        label_consumed=False,heldout='state updates from unlabeled features only; weights frozen; no carry from train',
        original_TGN_node_memory=False,verification='source inspection and executable label-independence regression test'))
    life=pd.read_csv(PRIVATE/'sample_time_lineage.csv');extra=[]
    for label,g in life.groupby('label'):
        for q in [0,.25,.5,.75,1]:
            extra.append(dict(label=label,quantile=q,first_timestamp=float(g.first_valid_tx_time.quantile(q)),lifetime_days=float(g.observed_lifetime_seconds.quantile(q)/86400)))
    pd.DataFrame(extra).to_csv(OUT/'pooled_time_quantiles.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for label,g in life.groupby('label'):
        x=np.sort(g.observed_lifetime_seconds/86400)
        ax[0].plot(x,np.arange(1,len(x)+1)/len(x),label=f'Category0 positive (n={len(g):,})' if label else f'Other categories (n={len(g):,})')
    ax[0].set_xscale('symlog',linthresh=.01);ax[0].set_xlabel('Observed lifetime (days; symmetric log scale)');ax[0].set_ylabel('Empirical cumulative fraction');ax[0].legend(fontsize=8)
    d=pd.read_csv(OUT/'first_last_ordering_diagnostics.csv');x=np.arange(3)
    for i,(name,legend) in enumerate([('last_valid_tx_time','Last transaction'),('first_valid_tx_time','First transaction')]):
        vals=d[(d.ordering==name)&(d.chain=='all')].set_index('split').loc[['train','validation','test'],'prevalence']
        ax[1].bar(x+(i-.5)*.35,vals,.35,label=legend)
    ax[1].set_xticks(x,['Train','Validation','Test']);ax[1].set_ylabel('Positive contract fraction');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'figures/selection_audit.pdf');fig.savefig(OUT/'figures/selection_audit.png',dpi=180);plt.close(fig)
    # The decision tree is a publication vector figure with no generated numerical evidence.
    fig,ax=plt.subplots(figsize=(10,3.4));ax.axis('off')
    boxes=[(.02,.59,.21,.29,'Define sample and label time\nUnknown discovery time?\nRetrospective scope only'),(.28,.59,.21,.29,'Recover first/last times\nand archive provenance\nAudit sorting and lifetimes'),(.54,.59,.21,.29,'Material composition change?\nYes: separate horizon cohort\nNo: report diagnostic limits'),(.79,.59,.19,.29,'Fix primary panel\nMatch inputs and loss\nKeep archival runs separate'),(.16,.10,.30,.26,'Validate calibration and ranking\nValidation-only fitting; AP ties\nDeclare multiplicity and dependence'),(.56,.10,.30,.26,'Seal evidence, check declarations\nApprove exact files and rights\nRelease before public submission')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,s in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',fc='#eef4fa',ec='#365c7c'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=9)
    for a,b in [((.235,.735),(.275,.735)),((.495,.735),(.535,.735)),((.755,.735),(.785,.735)),((.465,.23),(.55,.23))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#365c7c'})
    ax.plot([.88,.88,.10,.10],[.57,.45,.45,.23],color='#365c7c',linewidth=1)
    ax.annotate('',xy=(.15,.23),xytext=(.10,.23),arrowprops={'arrowstyle':'->','color':'#365c7c'})
    fig.tight_layout();fig.savefig(OUT/'figures/validity_decision_tree.pdf');plt.close(fig)


def panels():
    for arm in ['gin_bce','sage_bce','summary_lr','summary_mlp','gin_focal','sage_focal','gin_no_chain']:
        yield 'matched',arm,[(s,SUPP/'runs'/arm/f'seed{s}') for s in SEEDS]
    for arm in ['gat_time','gat_no_time','lightgbm','isolation_forest']:
        yield 'additional',arm,[(s,OUT/'runs/original'/arm/f'seed{s}') for s in SEEDS]
    yield 'repeat_control','summary_lr',[(s,OUT/'runs/original/summary_lr'/f'seed{s}') for s in SEEDS]
    for arm in ['summary_lr','summary_mlp','gin_bce','gat_time']:
        yield 'horizon7',arm,[(s,OUT/'runs/horizon7'/arm/f'seed{s}') for s in SEEDS]
    for arm in ['proposed','mc_dropout','temperature_scaled','tgat','tgn','fraudsage']:
        yield 'archival',arm,[(s,None) for s in SEEDS]


def read_scores(panel,arm,seed,path,split,metadata):
    loc=FROZEN/f'raw_predictions/{arm}/seed{seed}_{split}.csv' if panel=='archival' else path/f'{split}.csv'
    f=pd.read_csv(loc);expected=metadata[metadata.split==split].reset_index(drop=True)
    assert np.array_equal(f.label.to_numpy(int),expected.label.to_numpy(int))
    assert np.array_equal(f.timestamp.to_numpy(),expected.timestamp.to_numpy())
    if 'event_id' in f:
        assert f.event_id.equals(expected.event_id)
    else:assert np.array_equal(f.sample_id.to_numpy(),expected.sample_id.to_numpy())
    return f.p_mean.to_numpy(float),loc


def outcomes(meta=None):
    if meta is None:meta=frame_from(packed()).rename(columns={'row_id':'sample_id','original_split':'split'})
    hmeta=pd.read_csv(OUT/'horizon7_population.csv')
    rows=[];cal=[];ops=[];registry=[];strata=[];comp=[];block=[];rng=np.random.default_rng(317)
    pred_dir=OUT/'predictions';pred_dir.mkdir(exist_ok=True)
    cached={}
    for panel,arm,paths in panels():
        metadata=hmeta if panel=='horizon7' else meta
        testmeta=metadata[metadata.split=='test'].reset_index(drop=True);y=testmeta.label.to_numpy(int)
        allp=[]
        for seed,path in paths:
            p,tloc=read_scores(panel,arm,seed,path,'test',metadata)
            vloc=FROZEN/f'raw_predictions/{arm}/seed{seed}_validation.csv' if panel=='archival' else path/'validation.csv'
            v=None
            if vloc.exists():v,vloc=read_scores(panel,arm,seed,path,'validation',metadata)
            is_if=arm=='isolation_forest'
            m=dict(auc_pr=float(average_precision_score(y,p)),auc_roc=float(roc_auc_score(y,p))) if is_if else metric(y,p)
            rows.append(dict(panel=panel,arm=arm,seed=seed,**m));allp.append(p)
            exported=testmeta[['sample_id','timestamp','label']].copy();exported['score']=p
            public_path=pred_dir/f'{panel}_{arm}_seed{seed}_test.csv';exported.to_csv(public_path,index=False)
            validmeta=metadata[metadata.split=='validation'].reset_index(drop=True)
            if v is not None:
                exportedv=validmeta[['sample_id','timestamp','label']].copy();exportedv['score']=v
                exportedv.to_csv(pred_dir/f'{panel}_{arm}_seed{seed}_validation.csv',index=False)
            cp=None
            if path is not None:cp=path/'checkpoint.pt'
            elif arm in ['proposed','tgat','tgn','fraudsage']:
                from common import ROOT
                cp=ROOT/f'archive/tds/data/gog_scimain_v1_preserved_panel/checkpoints/seed{seed}.pt' if arm=='proposed' else FROZEN/f'checkpoints/{arm}/seed{seed}.pt'
            registry.append(dict(run_id=f'{panel}-{arm}-s{seed}',panel=panel,arm=arm,seed=seed,
                benchmark='GoG-317-first7d-v1' if panel=='horizon7' else 'GoG-SCIMain-v1',
                label_population_sha256=sha(OUT/'horizon7_population.csv') if panel=='horizon7' else sha(meta_label_file()),
                validation_source_sha256=sha(vloc) if vloc.exists() else None,test_source_sha256=sha(tloc),numeric_test_sha256=sha(public_path),
                checkpoint_sha256=sha(cp) if cp is not None else None,comparison_policy='descriptive; matched within panel; never paired across cohorts'))
            comp.append(dict(panel=panel,arm=arm,seed=seed,benchmark='first7d' if panel=='horizon7' else 'original',
                split_ids_sha256=sha(OUT/'horizon7_population.csv') if panel=='horizon7' else sha(OUT/'original_population.csv'),
                train_selection='val AP' if panel!='archival' else 'preserved original policy; not fully trajectory-matched',
                input='24 train-scaled structural summaries+chain' if arm.startswith('summary') or arm in ['lightgbm','isolation_forest'] else 'last128 topology+full-history degrees+chain; time for GAT',
                loss='unlabeled outlier criterion' if is_if else 'class-balanced focal' if 'focal' in arm or arm=='fraudsage' else 'balanced logistic' if arm=='summary_lr' else 'weighted BCE',
                runtime='preserved archival' if panel=='archival' else 'contemporary controlled; GPU/library differences documented',
                test_n=len(y),test_positive=int(y.sum())))
            for a in operating(y,p):ops.append(dict(panel=panel,arm=arm,seed=seed,**a))
            if not is_if and v is not None:
                temp,valnll=temperature(validmeta.label.to_numpy(int),v)
                calibrated=expit(logit(np.clip(p,1e-12,1-1e-12))/temp)
                cm=metric(y,calibrated)
                cal.append(dict(panel=panel,arm=arm,seed=seed,temperature=temp,validation_nll=valnll,ap_preserved=abs(cm['auc_pr']-m['auc_pr'])<1e-12,
                                **{'raw_'+k:b for k,b in m.items()},**{'ts_'+k:b for k,b in cm.items()}))
        matrix=np.asarray(allp);cached[(panel,arm)]=matrix
        if panel not in ['matched','additional']:continue
        for chain in ['ethereum','bsc','polygon']:
            mask=testmeta.chain.to_numpy()==chain;cy=y[mask];cp=matrix[:,mask]
            vals=[ap(cy,p) for p in cp];positive=int(cy.sum());low=high=None
            if positive>=2:
                draws=[]
                for i in range(2000):
                    ids=rng.integers(0,len(cy),len(cy))
                    if cy[ids].sum():draws.append(np.mean([ap(cy[ids],p[ids]) for p in cp]))
                low,high=np.quantile(draws,[.025,.975])
            strata.append(dict(panel=panel,arm=arm,chain=chain,n=len(cy),positive=positive,prevalence=float(cy.mean()),
                ap_mean=float(np.mean(vals)),ap_seed_sd=float(np.std(vals,ddof=1)),ci95_low=low,ci95_high=high,
                status='INFERENTIAL_SUPPORT_INSUFFICIENT' if positive<2 else 'conditional ordinary event bootstrap; seed mean fixed checkpoints',n_bootstrap=0 if positive<2 else 2000))
    # Chain prior is fit only on training labels. It is a pooled-composition control.
    priors=meta[meta.split=='train'].groupby('chain').label.mean()
    test=meta[meta.split=='test'].reset_index(drop=True);y=test.label.to_numpy(int);p=test.chain.map(priors).to_numpy()
    pd.DataFrame({'sample_id':test.sample_id,'timestamp':test.timestamp,'label':y,'score':p}).to_csv(pred_dir/'chain_prior_test.csv',index=False)
    write(OUT/'chain_prior.json',dict(priors=priors.to_dict(),metrics=metric(y,p),fit='train labels only',tie_policy='threshold-grouped AP; stable frozen order for top-k with tie bounds'))
    for a in operating(y,p):ops.append(dict(panel='control',arm='chain_prior',seed=0,**a))
    # Block sensitivity: same fixed five seed pairs, resample complete UTC-hour blocks.
    l=cached[('matched','gin_bce')];r=cached[('matched','summary_mlp')]
    groups=[np.flatnonzero(test.timestamp.to_numpy()//3600==b) for b in np.unique(test.timestamp.to_numpy()//3600)]
    delta=[]
    for _ in range(2000):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if y[ids].sum():delta.append(float(np.mean([ap(y[ids],a[ids])-ap(y[ids],b[ids]) for a,b in zip(l,r)])))
    block.append(dict(comparison='matched GIN+BCE minus summary MLP',block_seconds=3600,n_blocks=len(groups),
        observed=float(np.mean([ap(y,a)-ap(y,b) for a,b in zip(l,r)])),ci95_low=float(np.quantile(delta,.025)),ci95_high=float(np.quantile(delta,.975)),
        draws_requested=2000,draws_valid=len(delta),resampled_unit='whole UTC-hour blocks; fixed checkpoints; not future-deployment inference'))
    pd.DataFrame(block).to_csv(OUT/'time_block_sensitivity.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'per_seed_metrics.csv',index=False)
    pd.DataFrame(cal).to_csv(OUT/'modelwise_calibration.csv',index=False)
    pd.DataFrame(ops).to_csv(OUT/'operational_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'chain_stratified_ap.csv',index=False)
    pd.DataFrame(registry).to_csv(OUT/'run_registry.csv',index=False)
    pd.DataFrame(comp).to_csv(OUT/'panel_comparability.csv',index=False)
    result=pd.DataFrame(rows);summary=result.groupby(['panel','arm']).agg({k:['mean','std'] for k in ['auc_pr','auc_roc','ece','adaptive_ece','nll','brier','f1']})
    summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(OUT/'model_summary.csv',index=False)
    result[result.arm.isin(['lightgbm','isolation_forest'])].to_csv(OUT/'gbdt_isoforest_results.csv',index=False)
    result[(result.panel=='additional')&result.arm.str.startswith('gat')].to_csv(OUT/'temporal_input_ablation.csv',index=False)
    old=pd.read_csv(SUPP/'supplement/holm_adjusted_statistics.csv');old['family']='preserved7 archival'
    exploratory=pd.read_csv(SUPP/'supplement/control_statistical_comparisons.csv');exploratory['family']='preserved6 matched exploratory'
    pd.concat([old,exploratory],ignore_index=True).to_csv(OUT/'statistics_and_holm.csv',index=False)
    write(OUT/'analysis_manifest.json',dict(n_registered_test_vectors=len(registry),old_vectors_preserved=65,
        optional={k:'EXECUTED' for k in ['GBDT','IsolationForest','modelwise_TS','chain_AP','chain_prior','topk','time_block_bootstrap']},
        isolation_forest_probability_losses='NOT_APPLICABLE: scores are not probabilities',
        no_cross_cohort_paired_test=True,new_family_significance_claimed=False,ap='threshold grouped, unmodified scores',
        ts_fit='validation only; positive scalar temperature, logT[-4,4]',
        seed_sd='descriptive spread; deterministic LR/repeated GBDT fits are not independent sampling replicates'))


def meta_label_file():return OUT/'original_population.csv'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics',action='store_true');args=parser.parse_args()
    source=frame_from(packed());frame=source.rename(columns={'row_id':'sample_id','original_split':'split'})
    frame[['sample_id','chain','chain_index','timestamp','label','split']].to_csv(OUT/'original_population.csv',index=False)
    diagnostics(source)
    if not args.diagnostics:outcomes(frame)


if __name__=='__main__':main()
