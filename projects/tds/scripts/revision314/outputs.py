"""Derive publication tables, operational metrics and editable vector plots."""
from __future__ import annotations

import json
import math
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import FROZEN, OUT, PROJECT, ROOT, SEEDS, load, sha, write_json
from historical_metrics import reliability_bins

DISPLAY={'gin_bce':'Local GIN + weighted BCE','gin_focal':'Local GIN + weighted focal',
         'sage_bce':'GraphSAGE + weighted BCE','sage_focal':'GraphSAGE + weighted focal',
         'summary_lr':'Degree-summary LR','summary_mlp':'Degree-summary MLP',
         'gin_no_chain':'Local GIN without chain'}


def operating(labels,scores):
    # Stable event order resolves score ties, with no label-aware tie-break.
    order=np.argsort(-scores,kind='stable'); ranked=labels[order];positive=int(labels.sum())
    out={}
    for name,k in [('top_1pct',math.ceil(.01*len(labels))),('top_5pct',math.ceil(.05*len(labels))),('top_50',50),('top_100',100)]:
        hit=int(ranked[:k].sum());out.update({f'{name}_budget':k,f'{name}_captured':hit,f'{name}_precision':hit/k,f'{name}_recall':hit/positive})
    return out


def tex_escape(v):
    return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def tex_table(path,caption,headers,rows,label):
    content='\\begin{table}[H]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\small\n'
    content+='\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n\\toprule\n'
    content+=' & '.join(headers)+r' \\'+'\n\\midrule\n'
    content+='\n'.join(' & '.join(str(v) for v in row)+r' \\' for row in rows)
    content+='\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n'
    path.write_text(content)


def save(fig,name):
    for ext in ['pdf','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)


def main():
    _,frame,_=load();labels=frame.loc[frame.split=='test','label'].to_numpy(int)
    seeds=[]
    for path in sorted((OUT/'runs').glob('*/seed*/metrics.json')):
        seeds.append(json.loads(path.read_text()))
    runs=pd.DataFrame(seeds)
    assert len(runs)==35,'Finish the frozen seven-arm matrix before publication finalization'
    runs.to_csv(OUT/'supplement/per_seed_results.csv',index=False)
    summary=runs.groupby('arm')[['auc_pr','auc_roc','brier','ece','nll']].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary=summary.reset_index();summary.to_csv(OUT/'data/supplemental_summary.csv',index=False)
    for name,arms in [('loss_fairness_results',['gin_bce','gin_focal','sage_bce','sage_focal']),
                      ('simple_baselines',['gin_bce','summary_lr','summary_mlp']),
                      ('chain_ablation',['gin_bce','gin_no_chain'])]:
        summary[summary.arm.isin(arms)].to_csv(OUT/'data'/f'{name}.csv',index=False)
    original=json.loads((FROZEN/'checkpoints_manifest.json').read_text())
    convergence=[]
    for row in original['entries']:
        best=row['best_epoch'];stop=min(20,best+5)
        convergence.append(dict(arm=row['model_key'],seed=row['seed'],scope='historical',best_epoch=best,
                                stopped_epoch=stop,max_epoch_hit=stop==20,
                                evidence='stop inferred from audited code and final best epoch; full trajectory not retained',
                                continued_rise_at_20='unknown' if stop==20 else 'not reached'))
    for row in seeds:
        convergence.append(dict(arm=row['arm'],seed=row['seed'],scope='supplemental',best_epoch=row['best_epoch'],
                                stopped_epoch=row['stopped_epoch'],max_epoch_hit=row['max_epoch_hit'],
                                evidence='complete per-epoch history or LR solver iterations',continued_rise_at_20='see history.csv'))
    pd.DataFrame(convergence).to_csv(OUT/'supplement/convergence_audit.csv',index=False)
    operations=[];binrows=[];vectors={}
    for kind in ['proposed','mc_dropout','temperature_scaled','tgat','tgn','fraudsage']:
        vectors[kind]=[]
        for seed in SEEDS:
            path=FROZEN/f'raw_predictions/{kind}/seed{seed}_test.csv'
            f=pd.read_csv(path);assert f.event_id.tolist()==frame.loc[frame.split=='test','event_id'].tolist()
            p=f.p_mean.to_numpy(float);vectors[kind].append(p)
            operations.append(dict(method=kind,seed=seed,**operating(labels,p)))
            bins=reliability_bins(labels,p);bins['method']=kind;bins['seed']=seed;binrows.append(bins)
    for key,kind in [('deep_ensemble','proposed'),('mc_ensemble','mc_dropout'),('temperature_ensemble','temperature_scaled')]:
        p=np.mean(vectors[kind],axis=0);vectors[key]=[p]
        operations.append(dict(method=key,seed='aggregate',**operating(labels,p)))
    for row in seeds:
        f=pd.read_csv(OUT/'runs'/row['arm']/f"seed{row['seed']}"/'test.csv')
        operations.append(dict(method=row['arm'],seed=row['seed'],**operating(labels,f.p_mean.to_numpy(float))))
    pd.DataFrame(operations).to_csv(OUT/'data/operational_metrics.csv',index=False)
    bins=pd.concat(binrows,ignore_index=True);bins.to_csv(OUT/'supplement/reliability_bin_support.csv',index=False)
    calibration=[]
    for (method,seed),f in bins.groupby(['method','seed']):
        f=f[f['count']>0];d=f.mean_confidence-f.empirical_positive_rate
        calibration.append(dict(method=method,seed=seed,mean_predicted_positive_probability=float((f['count']*f.mean_confidence).sum()/len(labels)),
                                true_prevalence=float(labels.mean()),bins_positive_probability_overestimated=int((d>0).sum()),
                                bins_positive_probability_underestimated=int((d<0).sum()),
                                samples_in_overestimated_bins=int(f.loc[d>0,'count'].sum()),
                                samples_in_underestimated_bins=int(f.loc[d<0,'count'].sum())))
    pd.DataFrame(calibration).to_csv(OUT/'data/calibration_direction_audit.csv',index=False)
    historical=pd.read_csv(FROZEN/'model_metrics_summary.csv')
    fmt=lambda r,m:f"{r[m+'_mean']:.4f} $\\pm$ {r[m+'_std']:.4f}"
    primary=[[tex_escape(r.method),*[fmt(r,m) if r.n_runs>1 else f"{r[m+'_mean']:.4f}" for m in ['auc_pr','auc_roc','brier','ece','nll']]] for _,r in historical.iterrows()]
    tex_table(OUT/'data/table_primary.tex','Frozen primary results: mean and sample standard deviation over five seeds; ensembles are one fixed aggregate.',
              ['Method','AP','ROC-AUC','Brier','ECE','NLL'],primary,'tab:primary')
    rows=[[DISPLAY[r.arm],*[fmt(r,m) for m in ['auc_pr','auc_roc','brier','ece','nll']]] for _,r in summary.iterrows()]
    tex_table(OUT/'data/table_controls.tex','Supplemental controls in one matched runtime, validation-selected with a 60-epoch ceiling and patience five. LR is deterministic; five repeated fits do not represent five independent solutions.',
              ['Condition','AP','ROC-AUC','Brier','ECE','NLL'],rows,'tab:controls')
    comp=pd.read_csv(OUT/'data/split_composition.csv');rows=[]
    for _,r in comp[comp.chain=='all'].iloc[:3].iterrows():
        rows.append([r.scope,f'{r.n:,}',f'{r.n_positive:,}',f'{100*r.prevalence:.2f}\\%',r.start_utc[:19].replace('T',' '),r.end_utc[:19].replace('T',' '),f'{r.duration_seconds/3600:.2f}'])
    tex_table(OUT/'data/table_split.tex','Frozen chronological partitions. Dates are UTC; elapsed hours are end minus start, not observation exposure.',
              ['Split','Samples','Positive','Prevalence','Start UTC','End UTC','Hours'],rows,'tab:split')
    stats=pd.read_csv(OUT/'supplement/statistical_comparisons_corrected.csv');rows=[]
    for _,r in stats.iterrows():
        rows.append([tex_escape(r.comparison).replace('Proposed deterministic','Local GIN'),f'{r.mean_auc_pr_difference:+.4f}',
                     f'[{r.ordinary_ci95_low:+.4f}, {r.ordinary_ci95_high:+.4f}]',f'[{r.stratified_ci95_low:+.4f}, {r.stratified_ci95_high:+.4f}]',
                     f'{r.randomization_p_value_two_sided:.4f}',f'{r.holm_adjusted_p:.4f}'])
    tex_table(OUT/'data/table_statistics.tex','Tie-aware average-precision paired statistics, 10,000 replicates per protocol. Raw and Holm-adjusted Monte Carlo p-values include the finite-resampling correction.',
              ['Comparison','$\\Delta$AP','Ordinary 95\\% CI','Stratified 95\\% CI','Raw $p$','Holm $p$'],rows,'tab:stats')
    # Framework as editable SVG/PDF; other figures also retain plotted CSVs.
    fig,ax=plt.subplots(figsize=(10,2.2));ax.axis('off')
    titles=['Official token histories\n+ contract labels','Prediction-time snapshot\nfull-history degrees + cap','Frozen temporal split\ntrain / validation / test','Ranking + calibration\nstatistics + artifact hashes']
    for i,text in enumerate(titles):
        ax.text(.125+i*.25,.5,text,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.7',facecolor='#edf4fb',edgecolor='#315d8a'),fontsize=9)
        if i<3:ax.annotate('',xy=(.25+i*.25,.5),xytext=(.22+i*.25,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    save(fig,'fig1_framework')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    d=comp[comp.chain=='all'];axes[0].bar(d.scope.iloc[:3],100*d.prevalence.iloc[:3],color='#315d8a');axes[0].set_ylabel('Fraud-category prevalence (%)')
    axes[1].bar(np.arange(1,7),100*d.prevalence.iloc[3:],color='#438d86');axes[1].set_xlabel('Consecutive test bin (608 samples)');axes[1].set_ylabel('Prevalence (%)')
    fig.tight_layout();save(fig,'fig2_temporal_shift')
    mc=pd.read_csv(ROOT/'archive/tds/results/graphrag/round_4/mc_sensitivity.csv');mc.to_csv(OUT/'data/mc_sensitivity_per_seed.csv',index=False)
    agg=mc.groupby('T').agg(ap=('auc_pr','mean'),ap_sd=('auc_pr','std'),ece=('ece','mean'),ece_sd=('ece','std'),runtime=('latency_ms','median')).reset_index()
    agg.to_csv(OUT/'data/mc_sensitivity_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    for ax,mean,sd,label in [(axes[0],'ap','ap_sd','AP'),(axes[1],'ece','ece_sd','ECE')]:
        ax.errorbar(agg.T if False else agg['T'],agg[mean],yerr=agg[sd],fmt='o-',capsize=3);ax.set_ylabel(label);ax.set_xlabel('Passes (1 = deterministic)')
    axes[2].plot(agg['T'],agg.runtime,'o-');axes[2].set_ylabel('Panel elapsed time (ms)');axes[2].set_xlabel('Passes');fig.tight_layout();save(fig,'fig3_mc_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for key,title in [('proposed','GIN seed 37'),('temperature_scaled','Temperature-scaled seed 37'),('deep_ensemble','Deep ensemble'),('mc_ensemble','MC ensemble')]:
        p=vectors[key][3] if len(vectors[key])==5 else vectors[key][0]
        b=reliability_bins(labels,p);b=b[b['count']>0]
        axes[0].plot(b.mean_confidence,b.empirical_positive_rate,'o-',label=title,markersize=3)
    axes[0].plot([0,1],[0,1],'k--',linewidth=.8);axes[0].set(xlabel='Mean predicted positive probability',ylabel='Empirical positive rate');axes[0].legend(fontsize=7)
    for key,title in [('proposed','Unscaled'),('temperature_scaled','Temperature scaled')]:
        p=vectors[key][3];axes[1].hist(p,bins=np.linspace(0,1,16),alpha=.5,label=title)
    axes[1].set(xlabel='Predicted positive probability',ylabel='Bin support');axes[1].set_yscale('log');axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'fig4_reliability')
    fig,ax=plt.subplots(figsize=(8,4))
    for arm in ['gin_bce','gin_focal','sage_bce','sage_focal','gin_no_chain','summary_mlp']:
        h=pd.read_csv(OUT/'runs'/arm/'seed7/history.csv');ax.plot(h.epoch,h.validation_ap,label=DISPLAY[arm])
    ax.set(xlabel='Epoch',ylabel='Validation AP');ax.legend(fontsize=7);fig.tight_layout();save(fig,'optional_fig5_convergence')
    write_json(OUT/'reports/runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),
                hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                original_runtime='Python 3.12; PyTorch versions in frozen manifests',
                supplemental_runtime='separate current runtime; original ranking table retained separately'))
    print(summary[['arm','auc_pr_mean','auc_pr_std','ece_mean']].to_string(index=False),flush=True)


if __name__=='__main__':main()
