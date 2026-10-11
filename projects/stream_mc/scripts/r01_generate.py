"""Numeric tables and standard scientific plots; no unpublished manuscript prose."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import json
from r01_common import ROOT,config,output,read,write,sha256
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fmt(value,digits=3):return 'NA' if pd.isna(value) else f'{value:.{digits}f}'
def pm(row,key,digits=3):
    value=lambda name:getattr(row,name) if isinstance(row,tuple) else row[name]
    return fmt(value(key+'_mean'),digits)+r' $\pm$ '+fmt(value(key+'_std'),digits)
def escape(text):return str(text).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def tex_table(headers,rows,caption,label,path):
    specs='Y'+'r'*(len(headers)-1)
    text=r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}'+'\n'
    text+=r'\caption{'+caption+r'}\label{'+label+'}\n'
    text+=r'\begin{tabularx}{\linewidth}{'+specs+r'}\toprule'+'\n'
    text+=' & '.join(headers)+r' \\\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n'
    text+=r'\bottomrule\end{tabularx}\end{table}\FloatBarrier'+'\n';path.write_text(text)


def record_provenance(generated,out,cfg):
    sources={'primary.csv':['analysis/aggregate_metrics.csv'],'budget.csv':['runtime/offline_v2/budget_runtime.csv'],
        'chains.csv':['analysis/aggregate_metrics.csv'],'loco.csv':['analysis/aggregate_metrics.csv'],
        'runtime.csv':[str(p.relative_to(out)) for p in sorted((out/'runtime').glob('*/complete.json'))],
        'mc.csv':['analysis/mc_metrics.csv'],'risk_curve_source.csv':['analysis/per_seed_metrics.csv'],
        'control_cost.csv':['runtime/control_runtime.csv'],
        'event_diagnostics.csv':[str(p.relative_to(out)) for p in sorted((out/'runtime').glob('long_*/complete.json'))]}
    records=[]
    for path in sorted(generated.glob('*.csv')):
        names=sources[path.name]
        records.append({'output':str(path.relative_to(out)),'sha256':sha256(path),'source_paths':json.dumps(names),
            'source_sha256':json.dumps([sha256(out/name) for name in names]),'run_id':cfg['run_id'],
            'unit':'serial event / OS-process repeat' if path.name in ('runtime.csv','event_diagnostics.csv') else 'held-out contract / fitted-seed',
            'runtime_measurement_run_id':cfg['run_id']+'_offline_v2' if path.name=='budget.csv' else 'declared by source artifacts',
            'uncertainty':'training-seed sample SD after within-seed averaging; event runtime repeats remain OS-process repeats'})
    for name,source in [('frontier','generated/budget.csv'),('risk','generated/risk_curve_source.csv'),('state_pressure','stress/state_pressure_trace.csv'),('reliability','analysis/reliability_bins.csv'),('mc','generated/mc.csv')]:
        path=generated/'figures'/f'{name}.png'
        records.append({'output':str(path.relative_to(out)),'sha256':sha256(path),'source_paths':json.dumps([source]),'source_sha256':json.dumps([sha256(out/source)]),
            'run_id':cfg['run_id'],'unit':'synthetic state / measured RSS' if name=='state_pressure' else 'contract',
            'uncertainty':'Wilson conditional bin rate' if name=='reliability' else 'training-seed sample SD for frontier, risk and MC; state is one measured synthetic run'})
    pd.DataFrame(records).to_csv(out/'figure_table_provenance.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paper',action='store_true');args=parser.parse_args();cfg=config();out=output();src=out/'analysis';generated=out/'generated';generated.mkdir(exist_ok=True);figs=generated/'figures';figs.mkdir(exist_ok=True)
    aggregate=pd.read_csv(src/'aggregate_metrics.csv');perseed=pd.read_csv(src/'per_seed_metrics.csv');runtime=pd.read_csv(out/'runtime/offline_v2/budget_runtime.csv')
    primary=aggregate[(aggregate.experiment_id=='pooled_snapshot')&(aggregate.chain_scope=='pooled')]
    chosen=primary[(primary.backbone=='GIN')&((primary.method.isin(['local','full','threshold_only','degree_logistic','degree_MLP','relational_MLP','knn_mean']))|((primary.method=='margin')&(primary.budget==.25)))]
    chosen=pd.concat([chosen,primary[(primary.backbone=='GATv2')&primary.method.isin(['local','full'])]])
    chosen.to_csv(generated/'primary.csv',index=False)
    budget=runtime.groupby(['seed','family','budget'])[['ap','f1','recall','deep_fraction','latency_ms_per_contract']].mean().reset_index()
    budget=budget.groupby(['family','budget'])[['ap','f1','recall','deep_fraction','latency_ms_per_contract']].agg(['mean','std']).reset_index()
    budget.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in budget.columns];budget.to_csv(generated/'budget.csv',index=False)
    chains=aggregate[(aggregate.experiment_id=='pooled_snapshot')&(aggregate.backbone=='GIN')&(aggregate.chain_scope!='pooled')&((aggregate.method.isin(['local','full']))|((aggregate.method=='margin')&(aggregate.budget==.25)))];chains.to_csv(generated/'chains.csv',index=False)
    loco=aggregate[aggregate.experiment_id.str.startswith('loco_')&(aggregate.chain_scope!='pooled')&((aggregate.method.isin(['local','full']))|((aggregate.method=='margin')&(aggregate.budget==.25)))];loco.to_csv(generated/'loco.csv',index=False)
    rr=[];event_diagnostics=[]
    for path in sorted((out/'runtime').glob('*/complete.json')):
        record=read(path);rr.append({k:v for k,v in record.items() if not isinstance(v,(dict,list))})
        if record['population']=='long':event_diagnostics.append({'policy':record['policy'],'run_id':record['run_id'],'experiment_id':'replay_systems','N':record['N_measured'],'event_fraud_ratio':record['event_fraud_ratio'],**record['metrics']})
    rr=pd.DataFrame(rr);rr.to_csv(generated/'runtime.csv',index=False)
    event_diagnostics=pd.DataFrame(event_diagnostics);event_diagnostics.to_csv(generated/'event_diagnostics.csv',index=False)
    mc=pd.read_csv(src/'mc_metrics.csv');mc=mc[mc.split=='test'].groupby('T')[['ap','f1','error_auroc','latency_ms_per_contract_batched']].agg(['mean','std']).reset_index();mc.columns=['_'.join(c).rstrip('_') for c in mc.columns];mc.to_csv(generated/'mc.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    figure,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    markers={'margin':'o','entropy':'s','random':'^','learned_benefit':'D'}
    styles={'margin':'-','entropy':'--','random':':','learned_benefit':'-.'}
    for family,group in budget.groupby('family'):
        group=group.sort_values('budget')
        axes[0].errorbar(group.latency_ms_per_contract_mean,group.ap_mean,xerr=group.latency_ms_per_contract_std,yerr=group.ap_std,marker=markers[family],linestyle=styles[family],capsize=2,label=family.replace('_',' '))
        axes[1].plot(group.budget*100,group.deep_fraction_mean*100,marker=markers[family],linestyle=styles[family],label=family.replace('_',' '))
    axes[0].set(xlabel='Actual batched ms / contract (RTX 3090)',ylabel='Average precision');axes[1].plot([0,100],[0,100],'k--',alpha=.5);axes[1].set(xlabel='Validation target budget (%)',ylabel='Test achieved deep fraction (%)');axes[0].legend(fontsize=8);figure.savefig(figs/'frontier.png');plt.close(figure)
    risk=perseed[(perseed.experiment_id=='pooled_snapshot')&(perseed.backbone=='GIN')&(perseed.split=='test')&(perseed.method=='margin')&(perseed.chain_scope!='pooled')]
    risk.to_csv(generated/'risk_curve_source.csv',index=False);figure,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True)
    for chain,g in risk.groupby('chain_scope'):
        curve=g.groupby('budget').agg(coverage=('coverage','mean'),risk=('direct_fraud_FNR','mean'),sd=('direct_fraud_FNR','std')).dropna(subset=['risk'])
        if not curve.empty:ax.errorbar(curve.coverage,curve.risk,yerr=curve.sd.fillna(0),fmt='o-',capsize=2,label=chain)
    ax.text(.02,.03,'Polygon: no positive test support',transform=ax.transAxes,fontsize=9);ax.set(xlabel='Direct coverage (contract fraction)',ylabel='Direct fraud FNR',ylim=(-.04,1.04));ax.legend();figure.savefig(figs/'risk.png');plt.close(figure)
    state=pd.read_csv(out/'stress/state_pressure_trace.csv');figure,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True);axes[0].plot(state.sequence_id,state.resident_contracts);axes[0].axhline(5000,color='k',ls='--');axes[0].set(xlabel='Synthetic input sequence',ylabel='Resident contracts');axes[1].plot(state.sequence_id,state.rss_mib);axes[1].set(xlabel='Synthetic input sequence',ylabel='Measured process RSS (MiB)');figure.savefig(figs/'state_pressure.png');plt.close(figure)
    bins=pd.read_csv(src/'reliability_bins.csv');bins=bins[(bins.experiment_id=='pooled_snapshot')&(bins.backbone=='GIN')&(bins.seed==11)&(bins.chain_scope=='pooled')&(bins.N>0)]
    figure,axes=plt.subplots(1,3,figsize=(10,3.2),constrained_layout=True)
    for ax,(path,group) in zip(axes,bins.groupby('path',sort=True)):
        ax.plot([0,1],[0,1],'k--',alpha=.5);ax.errorbar(group.mean_probability,group.observed_rate,yerr=[group.observed_rate-group.observed_rate_wilson_low,group.observed_rate_wilson_high-group.observed_rate],fmt='o',capsize=2)
        for r in group.itertuples():ax.annotate(str(r.N),(r.mean_probability,r.observed_rate),xytext=(2,4),textcoords='offset points',fontsize=6)
        ax.set(title=path,xlabel='Mean predicted probability',ylabel='Observed positive rate',xlim=(0,1),ylim=(0,1))
    figure.savefig(figs/'reliability.png');plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(9,3.3),constrained_layout=True);axes[0].errorbar(mc.T_ if 'T_' in mc else mc['T'],mc.f1_mean,yerr=mc.f1_std,fmt='o-',capsize=2);axes[0].set(xlabel='Local passes T',ylabel='Validation-calibrated test F1')
    axes[1].errorbar(mc['T'],mc.latency_ms_per_contract_batched_mean,yerr=mc.latency_ms_per_contract_batched_std,fmt='o-',capsize=2);axes[1].set(xlabel='Local passes T',ylabel='Batched ms / contract');figure.savefig(figs/'mc.png');plt.close(figure)
    record_provenance(generated,out,cfg)
    if not args.paper:return
    paper=ROOT/'projects/stream_mc/paper/current/r01';tables=paper/'generated';tables.mkdir(parents=True,exist_ok=True);(paper/'figures').mkdir(exist_ok=True)
    for path in figs.glob('*.png'):shutil.copy2(path,paper/'figures'/path.name)
    local=chosen[(chosen.backbone=='GIN')&(chosen.method=='local')].iloc[0];selective=chosen[(chosen.method=='margin')].iloc[0];full=chosen[(chosen.backbone=='GIN')&(chosen.method=='full')].iloc[0]
    summary=f"Across five seeds, primary selective minus local mean AP is {selective.ap_mean-local.ap_mean:+.4f}; mean F1 difference is {selective.f1_mean-local.f1_mean:+.4f}; mean recall difference is {selective.recall_mean-local.recall_mean:+.4f}. These are descriptive differences, not equivalence or prospective superiority claims."
    rm=chosen[chosen.method=='relational_MLP'].iloc[0];knn=chosen[chosen.method=='knn_mean'].iloc[0]
    stage_summary=f"Non-message-passing relational MLP has mean AP {rm.ap_mean:.3f} and F1 {rm.f1_mean:.3f}; simple kNN score averaging has AP {knn.ap_mean:.3f} and F1 {knn.f1_mean:.3f}. "
    if rm.f1_mean>selective.f1_mean and knn.f1_mean>selective.f1_mean:stage_summary+='Both exceed primary selective mean F1. The primary selective GATv2 path therefore does not provide an F1 advantage over these simpler controls; the measured cost and AP trade-off must be evaluated separately.'
    comparisons=[]
    margin=budget[(budget.family=='margin')&(budget.budget==.25)].iloc[0]
    for family in ('entropy','random','learned_benefit'):
        other=budget[(budget.family==family)&(budget.budget==.25)].iloc[0]
        comparisons.append(f"{escape(family)} AP {other.ap_mean:.3f} with achieved deep fraction {other.deep_fraction_mean:.3f}")
    routing=f"At the same validation target budget 0.25, margin AP is {margin.ap_mean:.3f} with achieved fraction {margin.deep_fraction_mean:.3f}; "+'; '.join(comparisons)+'. Distribution shift means these are not exact achieved-budget-matched tests; the measured cost curves, not target quota alone, govern the comparison. No router is assumed uniformly best.'
    long_margin=rr[(rr.population=='long')&(rr.policy=='margin')].iloc[0];prefix_margin=rr[(rr.population=='prefix')&(rr.policy=='margin')]
    workload=f"The primary achieved deep fraction is {selective.deep_fraction_mean*100:.1f} percent on held-out contracts, {prefix_margin.deep_fraction.mean()*100:.1f} percent on the event prefix, and {long_margin.deep_fraction*100:.2f} percent on the long replay. Thus the frozen snapshot routing quota does not transfer to these replay populations; nearly all replay events execute the relational stage. The table reports actual costs rather than inferring savings from the snapshot result."
    deterministic=mc[mc['T']==1].iloc[0];mc_summary=[]
    for _,row in mc[mc['T']>1].iterrows():
        mc_summary.append(f"T={int(row['T'])}: {row.latency_ms_per_contract_batched_mean/deterministic.latency_ms_per_contract_batched_mean:.2f} times deterministic batched cost, F1 change {100*(row.f1_mean-deterministic.f1_mean):+.2f} percentage points")
    (tables/'numbers.tex').write_text('\n'.join(r'\newcommand'+'{'+chr(92)+name+'}{'+value+'}' for name,value in [('LocalAP',fmt(local.ap_mean)),('SelectiveAP',fmt(selective.ap_mean)),('FullAP',fmt(full.ap_mean)),('SelectiveDeepPercent',fmt(selective.deep_fraction_mean*100,1)),('LongReplayDeepPercent',fmt(long_margin.deep_fraction*100,1)),('ResultSummary',summary),('RoutingSummary',routing),('WorkloadSummary',workload),('MCSummary','; '.join(mc_summary)+'.')])+'\n')
    with (tables/'numbers.tex').open('a') as handle:handle.write(r'\newcommand'+'{'+chr(92)+'StageControlSummary}{'+stage_summary+'}\n')
    support=pd.read_csv(out/'split_support.csv');tex_table(['Split/chain','$N$','Positive','Min cutoff','Max cutoff'],[[escape(r['split']+'/'+r['chain']),int(r.N),int(r.N_positive),int(r.min_cutoff),int(r.max_cutoff)] for _,r in support.iterrows()],'Retained snapshot support and UTC epoch-second cutoffs. These are chain-specific partitions, not a strict global as-of split.','tab:support',tables/'table_support.tex')
    tex_table(['Method/backbone','AP','F1','Recall','Deep'],[[escape(r.method+'/'+r.backbone),pm(r,'ap'),pm(r,'f1'),pm(r,'recall'),fmt(r.deep_fraction_mean)] for _,r in chosen.iterrows()],'Contract test performance: five-seed mean and sample SD. Primary margin budget0.25; light controls execute no GATv2 relation stage.','tab:primary',tables/'table_primary.tex')
    if (out/'runtime/control_runtime.csv').exists():
        control=pd.read_csv(out/'runtime/control_runtime.csv');within=control.groupby(['method','seed'])[['latency_ms_per_contract','pipeline_parameters','head_training_seconds']].mean().reset_index()
        cells=within.groupby('method')[['latency_ms_per_contract','pipeline_parameters','head_training_seconds']].agg(['mean','std']).reset_index();cells.columns=['_'.join(c).rstrip('_') for c in cells.columns]
        cells.to_csv(generated/'control_cost.csv',index=False)
        tex_table(['Control','Active params','ms/contract','Head train seconds'],[[escape(r.method),int(r.pipeline_parameters_mean),pm(r,'latency_ms_per_contract',2),'NA (unrecorded)' if pd.isna(r.head_training_seconds_mean) else fmt(r.head_training_seconds_mean)] for r in cells.itertuples()],
                  'Matched batch128 actual control costs, five seeds with three timing repeats. Parameters refer to executed scoring paths, not all common-harness resident weights. Retrieval controls include local encoding/search; original light-head training time was unrecorded, not zero. Separate neural-stage training costs are released.','tab:controlcost',tables/'table_control_cost.tex')
    for name,frame in [('chains',chains),('loco',loco)]:tex_table(['Chain/method','$N_+$','AP','F1','Recall','FP/1000'],[[escape(r.chain_scope+'/'+r.method),int(r.N_positive_mean),pm(r,'ap'),pm(r,'f1'),pm(r,'recall'),fmt(r.fp_per_1000_mean,2)] for _,r in frame.iterrows()],('Pooled GIN per-chain evaluation.' if name=='chains' else 'Independently retrained source-only LOCO GIN; target excluded from all fit/selection partitions.')+' Five seeds; Polygon positive-class metrics are NA.','tab:'+name,tables/f'table_{name}.tex')
    def budget_rows(frame):return [[escape(r.family)+' '+fmt(r.budget,2),fmt(r.deep_fraction_mean*100,1),pm(r,'ap'),pm(r,'f1'),pm(r,'latency_ms_per_contract',2)] for _,r in frame.iterrows()]
    tex_table(['Router/budget',r'Deep\%','AP','F1','ms/contract'],budget_rows(budget[budget.family=='margin']),'Primary margin frontier, actual batch128 local/search/selected-relational execution; three repeats per fitted seed.','tab:budget',tables/'table_budget.tex')
    tex_table(['Router/budget',r'Deep\%','AP','F1','ms/contract'],budget_rows(budget),'All fixed validation-budget routers; random replicates and timing repeats averaged within seed.','tab:budgetall',tables/'table_budget_all.tex')
    runtime_rows=[]
    for (population,policy),g in rr.groupby(['population','policy']):runtime_rows.append([escape(population+'/'+policy),len(g),int(g.N_measured.iloc[0]),fmt(g.deep_fraction.mean()*100,1),fmt(g.p50_ms.mean(),2),fmt(g.p99_ms.mean(),2),fmt(g.throughput_events_per_second.mean(),1)])
    tex_table(['Workload/policy','Runs','$N$',r'Deep\%','P50 ms','P99 ms','events/s'],runtime_rows,'Separate serial event workloads on RTX3090; prefix500+warmup25, five repetitions; long100000, one run per policy. Event labels are inherited contract labels.','tab:runtime',tables/'table_runtime.tex')
    tex_table(['Long replay policy','AP','F1','Recall','FP/1000 events'],[[escape(r.policy),fmt(r.ap),fmt(r.f1),fmt(r.recall),fmt(r.fp_per_1000,2)] for r in event_diagnostics.itertuples()],
              'Secondary event diagnostics for the same100000-event serial populations. Every event inherits its contract label; repeated contracts are not independent examples and these are not transaction-fraud scores. The inherited positive ratio is3.097 percent.','tab:eventdiagnostics',tables/'table_event_diagnostics.tex')
    stress=read(out/'stress/bounded_state_audit.json');restart=read(out/'stress/restart_audit.json')
    stress_rows=[[escape(k),str(stress[k])] for k in ['N_unique','N_additional_churn','resident_peak','capacity_evictions','edge_cap','node_cap','ttl_expiry_count','cache_entries','cache_evictions','queue_high_water','queue_retries']]
    tex_table(['Synthetic boundary observation','Observed'],stress_rows,'Actual-cap synthetic pressure; distinct from provider-labeled performance.','tab:stress',tables/'table_stress.tex')
    tex_table(['Checkpoint observation','Observed'],[[escape(k),str(restart[k])] for k in ['N_compared','score_max_abs_diff','score_bit_identical_count','label_disagreements','state_version_mismatches','replayed_input_loss_count']],'Abrupt local-file failure fixtures; bit identity, tolerance and input replay loss are distinct checks.','tab:restart',tables/'table_restart.tex')
    benefits=pd.read_csv(out/'audits/selective_benefit_denominators.csv');benefits=benefits[(benefits.family=='margin')&(benefits.budget==.25)]
    tex_table(['Seed','Escalated','Corrections/local wrong','Harms/local correct'],[[int(r.seed),int(r.N_escalated),f'{int(r.corrections)}/{int(r.N_local_wrong_escalated)}',f'{int(r.harms)}/{int(r.N_local_correct_escalated)}'] for r in benefits.itertuples()],
              'Primary actual selected-execution correction/harm counts and conditional denominators. Baseline is calibrated local at the identical final threshold; zero denominator is NA. All180 policy cells are released.','tab:benefit',tables/'table_benefit.tex')
    stats=pd.read_csv(src/'paired_statistics.csv');tex_table(['Seed','$b/c$',r'$\Delta$AP [95\% CI]',r'$\Delta$F1 [95\% CI]','Exact p','Holm p'],[[int(r.seed),f'{int(r.n_local_correct_selective_wrong)}/{int(r.n_local_wrong_selective_correct)}',f'{r.delta_ap:+.3f} [{r.ap_ci_low:+.3f},{r.ap_ci_high:+.3f}]',f'{r.delta_f1:+.3f} [{r.f1_ci_low:+.3f},{r.f1_ci_high:+.3f}]',f'{r.exact_mcnemar_p:.3g}',f'{r.holm_p:.3g}'] for r in stats.itertuples()],'New primary pooled-GIN paired descriptive statistics. McNemar discordance b/c concerns correctness, not F1.','tab:stats',tables/'table_seed_statistics.tex')
    hist=pd.read_csv(src/'historical_r4_statistics_audit.csv');tex_table(['Historical seed','$b$','$c$','Exact p','Holm p'],[[int(r.seed),int(r.n_local_correct_selective_wrong),int(r.n_local_wrong_selective_correct),f'{r.exact_mcnemar_p:.10f}',f'{r.holm_p:.10f}'] for r in hist.itertuples()],'Historical R4 raw-prediction audit only; separate family of five tests.','tab:histstats',tables/'table_historical_statistics.tex')
    temporal=pd.read_csv(src/'temporal_audit_rows.csv');tex_table(['Experiment/backbone','Seeds','Weight after target','Policy after target','Future refs'],[[escape(exp+'/'+backbone),len(g),int(g.model_fit_after_target_count.sum()),int(g.calibration_or_policy_after_target_count.sum()),int(g.future_selected_reference_count.sum())] for (exp,backbone),g in temporal.groupby(['experiment_id','backbone'])],'Counts sum per-model audit rows, not independent cases; edge/label availability remains unknown.','tab:temporal',tables/'table_temporal.tex')
    risks=pd.read_csv(src/'risk_counts.csv');risks=risks[(risks.experiment_id=='pooled_snapshot')&(risks.backbone=='GIN')&(risks.seed==11)]
    tex_table(['Seed11 chain','Coverage','Direct FNR n/d','Population miss n/d','False omission n/d'],[[escape(r.chain_scope),fmt(r.coverage),f'{int(r.direct_fraud_FNR_numerator)}/{int(r.direct_fraud_FNR_denominator)}',f'{int(r.population_direct_fraud_miss_numerator)}/{int(r.population_direct_fraud_miss_denominator)}',f'{int(r.false_omission_rate_numerator)}/{int(r.false_omission_rate_denominator)}'] for r in risks.itertuples()],'Illustrative primary seed raw denominators; all seeds and paths are released. Zero denominator means NA.','tab:riskcounts',tables/'table_risk_counts.tex')
    tex_table(['Local passes T','AP','F1','Error AUROC','ms/contract'],[[int(r['T']),pm(r,'ap'),pm(r,'f1'),pm(r,'error_auroc'),pm(r,'latency_ms_per_contract_batched',2)] for _,r in mc.iterrows()],'Local MC ablation, per-T validation maps and thresholds. T1 error score entropy; T3/5/8 population variance.','tab:mc',tables/'table_mc.tex')
    write(out/'generated_artifacts.json',{'run_id':cfg['run_id'],'macro_source':sha256(generated/'primary.csv'),'result_summary':summary,'private_tex_tables_generated':True})
    record_provenance(generated,out,cfg)
    print('FIGURES AND NUMERIC TABLES COMPLETE',flush=True)


if __name__=='__main__':main()
