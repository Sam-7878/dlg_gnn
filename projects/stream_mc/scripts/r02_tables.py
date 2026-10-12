"""Regenerate numeric-only R02 paper inputs from verified original observations."""
from __future__ import annotations
import shutil
import sys
from r02_common import ROOT,PROJECT,R01,R02,VERIFY,PAPER,RUN,read,write,sha256,csv_write,original_unchanged
import numpy as np
import pandas as pd


def main():
    assert read(VERIFY/'raw_numeric_validation.json')['N_original_models']==25
    assert len(pd.read_csv(VERIFY/'selected_execution_parity.csv'))==180
    working=R02/'working'
    shutil.copy2(R01/'split_support.csv',working/'split_support.csv')
    if not (working/'audits').exists():(working/'audits').symlink_to(R01/'audits',target_is_directory=True)
    import r01_generate as gen
    base_table=gen.tex_table;base_escape=gen.escape
    class ExplicitPaperDestination:
        def __truediv__(self,name):
            if name!='projects/stream_mc/paper/current/r01':raise ValueError('unexpected legacy writer destination')
            return PAPER
    gen.ROOT=ExplicitPaperDestination();gen.output=lambda:R02/'working'
    def pm(row,key,digits=3):
        value=lambda name:getattr(row,name) if isinstance(row,tuple) else row[name]
        mean=value(key+'_mean');sd=value(key+'_std')
        return 'NA' if pd.isna(mean) else gen.fmt(mean,digits)+(r' $\pm$ '+gen.fmt(sd,digits) if pd.notna(sd) else '')
    gen.pm=pm
    display={'degree_MLP':'degree MLP','degree_logistic':'degree logistic','relational_MLP':'relational MLP','knn_mean':'kNN mean','threshold_only':'threshold only','learned_benefit':'learned benefit','GATv2_local':'GATv2 local','GATv2_margin':'GATv2 margin'}
    def escape(value):
        text=str(value)
        for old,new in display.items():text=text.replace(old,new)
        return base_escape(text)
    gen.escape=escape
    def table(headers,rows,caption,label,path):
        for old,new in [('budget0.25','budget $q=0.25$'),('batch128','batch size 128'),('RTX3090','RTX 3090'),('prefix500+warmup25','prefix 500 + warmup 25'),('long100000','long 100,000'),('same100000','same 100,000'),('is3.097','is 3.097'),('All180','All 180')]:caption=caption.replace(old,new)
        return base_table(headers,rows,caption,label,path)
    gen.tex_table=table;sys.argv=['r01_generate.py','--paper'];gen.main()
    tables=PAPER/'generated'
    support=pd.read_csv(R01/'split_support.csv')
    support['first_UTC']=pd.to_datetime(support.min_cutoff,unit='s',utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    support['last_UTC']=pd.to_datetime(support.max_cutoff,unit='s',utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    support.to_csv(VERIFY/'snapshot_support_utc.csv',index=False)
    stack=lambda value:r'\shortstack{'+value.replace(' ',r'\\')+'}'
    table(['Split/chain','$N$','Positive','First cutoff (UTC)','Last cutoff (UTC)'],[[escape(r['split']+'/'+r['chain']),int(r.N),int(r.N_positive),stack(r.first_UTC),stack(r.last_UTC)] for _,r in support.iterrows()],
        'Retained snapshot support and human-readable UTC cutoff windows. Exact epoch seconds are supplied in the numeric manifest; these are chain-specific partitions, not a global as-of split.','tab:support',tables/'table_support.tex')
    runtime=pd.read_csv(VERIFY/'runtime_recomputed.csv');long=runtime[runtime.population=='long'].set_index('policy');margin=long.loc['margin'];full=long.loc['full']
    aggregate=pd.read_csv(working/'analysis/aggregate_metrics.csv');chosen=aggregate[(aggregate.experiment_id=='pooled_snapshot')&(aggregate.backbone=='GIN')&(aggregate.chain_scope=='pooled')]
    primary=chosen[(chosen.method=='margin')&(chosen.budget==.25)].iloc[0]
    additions={'LongMarginAP':f'{margin.ap:.3f}','LongMarginFone':f'{margin.f1:.3f}','LongMarginRecall':f'{margin.recall:.3f}','LongMarginFP':f'{margin.fp_per_1000:.2f}',
        'LongMarginThroughput':f'{margin.throughput_events_per_second:.1f}','LongFullThroughput':f'{full.throughput_events_per_second:.1f}',
        'LongThroughputReduction':f'{100*(1-margin.throughput_events_per_second/full.throughput_events_per_second):.2f}'}
    with (tables/'numbers.tex').open('a') as handle:
        for name,value in additions.items():handle.write(r'\newcommand{'+chr(92)+name+'}{'+value+'}\n')
    # Score and cost are paired from the SAME measured control run, not joined
    # to canonical scores as though they were bit-identical rankings.
    costs=pd.read_csv(R01/'runtime/control_runtime.csv');traces=pd.read_csv(R01/'runtime/control_batch_trace.csv')
    for r in costs.itertuples():
        t=traces[(traces.seed==r.seed)&(traces.method==r.method)&(traces.repeat==r.repeat)]
        assert t.N.sum()==r.N and np.isclose(t.wall_ms.sum()/r.N,r.latency_ms_per_contract,atol=1e-10)
        assert r.raw_score_max_abs_diff<=1e-6 and r.prediction_disagreements==0
    values=['ap','f1','recall','latency_ms_per_contract','pipeline_parameters']
    within=costs.groupby(['method','seed'])[values].mean().reset_index();summary=within.groupby('method')[values].agg(['mean','std']).reset_index();summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    summary.to_csv(VERIFY/'control_score_cost.csv',index=False)
    table(['Measured control','AP','F1','ms/contract','Active params'],[[escape(r.method),pm(r,'ap'),pm(r,'f1'),pm(r,'latency_ms_per_contract',2),int(r.pipeline_parameters_mean)] for r in summary.itertuples()],
        'Task-matched controls: scores and costs from the same batch-size-128 executions. Three timing repeats are averaged within each of five seeds. Retrieval controls include local encoding and search. Active parameters describe the executed path, not all resident harness weights. Original light-head fitting time is unrecorded; no training-efficiency superiority is claimed.','tab:controlcost',tables/'table_control_cost.tex')
    measured=pd.read_csv(R01/'runtime/offline_v2/budget_runtime.csv');cells=measured.groupby(['seed','family','budget'])[['deep_fraction','ap','f1','recall','latency_ms_per_contract']].mean().reset_index();matches=[]
    for p in cells[cells.family=='margin'].itertuples():
        for family in ('entropy','random','learned_benefit'):
            options=cells[(cells.seed==p.seed)&(cells.family==family)].copy();options['gap']=abs(options.deep_fraction-p.deep_fraction);c=options.sort_values(['gap','budget']).iloc[0]
            matches.append({'seed':p.seed,'primary_validation_budget':p.budget,'comparator':family,'comparator_validation_budget':c.budget,'primary_achieved_fraction':p.deep_fraction,'comparator_achieved_fraction':c.deep_fraction,'achieved_gap':c.gap,'within_005_fraction':bool(c.gap<=.05),
                'primary_ap':p.ap,'comparator_ap':c.ap,'primary_f1':p.f1,'comparator_f1':c.f1,'primary_recall':p.recall,'comparator_recall':c.recall,'primary_ms_per_contract':p.latency_ms_per_contract,'comparator_ms_per_contract':c.latency_ms_per_contract})
    matches=pd.DataFrame(matches);old=pd.read_csv(R01/'analysis/achieved_budget_matched.csv')
    for col in matches.select_dtypes(include=np.number).columns:assert np.allclose(matches[col],old[col],atol=1e-10,rtol=0)
    assert np.array_equal(matches.within_005_fraction,old.within_005_fraction)
    matches.to_csv(VERIFY/'achieved_budget_matches_recomputed.csv',index=False)
    rows=[]
    for family,g in matches[matches.primary_validation_budget==.25].groupby('comparator'):
        selected=g[g.within_005_fraction];n=len(selected)
        rows.append([escape(family),f'{n}/5',str(5-n),gen.fmt(selected.achieved_gap.mean()),gen.fmt((selected.comparator_ap-selected.primary_ap).mean()),gen.fmt(selected.primary_ms_per_contract.mean(),2),gen.fmt(selected.comparator_ms_per_contract.mean(),2)])
    table(['Comparator','Matched','Excluded','Mean gap',r'$\Delta$AP','Primary ms','Other ms'],rows,
        'Post-hoc nearest achieved-fraction comparisons for primary margin target $q=0.25$. Match tolerance is an absolute fraction gap of 0.05; ties prefer the smaller frozen validation budget. Differences and costs average only matched seed pairs. No thresholds are refitted, no cost equality is imposed, and unmatched pairs are not imputed. All 90 cells are supplied.','tab:matched',tables/'table_matched.tex')
    write(VERIFY/'generated_numeric_identity.json',{'verification_id':RUN,'source_training_run':'selectivestream_r01_20261011','source_batch_measurement':'selectivestream_r01_20261011_offline_v2',
        'aggregation_id':'r02_verified_reaggregation_20261012','source_aggregate_sha256':sha256(working/'analysis/aggregate_metrics.csv'),
        'numeric_macros_sha256':sha256(tables/'numbers.tex'),'table_sha256':{p.name:sha256(p) for p in tables.glob('*.tex')},
        'new_measurements_used_for_primary_numbers':False,'all_seeds_not_replica':True})
    original_unchanged();print('R02 VERIFIED NUMERIC PAPER INPUTS COMPLETE',flush=True)


if __name__=='__main__':main()
