"""Numeric parity/schema linkage and private package construction after execution."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path
from r01_common import ROOT,PROJECT,config,output,read,write,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--packages',action='store_true');args=parser.parse_args();out=output();cfg=config()
    import numpy as np
    import pandas as pd
    from r01_train import apply_fitted
    from r01_analysis import paired_stats
    from sklearn.metrics import average_precision_score
    engine_name='src/gog_fraud/streaming/selective_engine.py'
    with tarfile.open(out/'runtime_sources_v1.tar') as frozen:
        original_engine=ast.parse(frozen.extractfile(engine_name).read().decode())
    amended_engine=ast.parse((ROOT/engine_name).read_text())
    # The only non-constructor amendment protects final fallback metadata from
    # retrieval's empty fallback. Normalize that exact expression, not scoring.
    removed=[]
    for node in amended_engine.body:
        if isinstance(node,ast.ClassDef) and node.name=='SelectivePredictor':
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name=='predict':
                    expected=ast.dump(ast.parse("audit.pop('fallback',None)").body[0],include_attributes=False)
                    removed=[item for item in method.body if ast.dump(item,include_attributes=False)==expected]
                    method.body=[item for item in method.body if item not in removed]
    assert len(removed)==1,'unexpected fallback metadata amendment'
    def methods(tree):
        return {node.name+'/'+method.name:ast.dump(method,include_attributes=False)
                for node in tree.body if isinstance(node,ast.ClassDef)
                for method in node.body if isinstance(method,ast.FunctionDef)
                and not (node.name=='SelectivePredictor' and method.name=='__init__')}
    assert methods(original_engine)==methods(amended_engine),'identity amendment changed an inference/state method'
    identity_audit=read(out/'audits/checkpoint_identity_amendment.json')
    identity_audit['non_constructor_AST_identical_after_explicit_fallback_metadata_normalization']=True
    identity_audit['normalized_expression']="audit.pop('fallback',None); removes retrieval metadata overwrite, not numerical scoring"
    identity_audit['compared_methods']=sorted(methods(original_engine))
    write(out/'audits/checkpoint_identity_amendment.json',identity_audit)
    provider=pd.read_csv('/mnt/d/_Work/_data/gog_round7_upstream/Token Data/labels.csv')
    mapping={(r.Chain.lower(),r.Contract.lower()):int(r.Category==0) for r in provider.itertuples()}
    raw=pd.read_parquet(ROOT/cfg['raw_events']);missing=0;mismatch=0
    for r in raw.itertuples(index=False):
        expected=mapping.get((r.chain_id.lower(),r.contract_id.lower()))
        missing+=expected is None;mismatch+=expected is not None and expected!=int(r.label)
    event_audit={'N_events':len(raw),'N_unique_contracts':len(raw[['chain_id','contract_id']].drop_duplicates()),'provider_label_missing_events':missing,'provider_label_mismatch_events':mismatch,
        'label_scope':'contract research label inherited by event; not each-transfer fraud truth','raw_input_sha256':sha256(ROOT/cfg['raw_events']),
        'future_reference_violations':0,'source_event_ids_duplicate_count':int(raw.sample_id.duplicated().sum())}
    for path in sorted((out/'runtime').glob('*/event_trace.csv')):
        trace=pd.read_csv(path,low_memory=False);violation=(trace.selected_reference_max_cutoff.fillna(-1)>trace.snapshot_cutoff).sum()
        event_audit['future_reference_violations']+=int(violation)
    write(out/'audits/event_source_audit.json',event_audit)
    if missing or mismatch or event_audit['future_reference_violations']:raise ValueError('event source/reference audit failed')
    rows=[];benefits=[];day_stats=[];active=out/'runtime/offline_v2';runtime=pd.read_csv(active/'budget_runtime.csv')
    cells=runtime.groupby(['seed','family','budget'])[['deep_fraction','ap','f1','recall','latency_ms_per_contract']].mean().reset_index()
    matched=[]
    for primary in cells[cells.family=='margin'].itertuples():
        for family in ('entropy','random','learned_benefit'):
            candidates=cells[(cells.seed==primary.seed)&(cells.family==family)].copy()
            candidates['gap']=abs(candidates.deep_fraction-primary.deep_fraction);other=candidates.sort_values(['gap','budget']).iloc[0]
            matched.append({'seed':primary.seed,'primary_validation_budget':primary.budget,'comparator':family,'comparator_validation_budget':other.budget,
                'primary_achieved_fraction':primary.deep_fraction,'comparator_achieved_fraction':other.deep_fraction,'achieved_gap':other.gap,'within_005_fraction':bool(other.gap<=.05),
                'primary_ap':primary.ap,'comparator_ap':other.ap,'primary_f1':primary.f1,'comparator_f1':other.f1,'primary_recall':primary.recall,'comparator_recall':other.recall,
                'primary_ms_per_contract':primary.latency_ms_per_contract,'comparator_ms_per_contract':other.latency_ms_per_contract,
                'scope':'post-hoc nearest achieved-budget descriptive comparison among already frozen validation policies; not new test-fitted routing or exact budget equivalence'})
    pd.DataFrame(matched).to_csv(out/'analysis/achieved_budget_matched.csv',index=False)
    for model_dir in sorted((out/'models').glob('pooled_snapshot_GIN_seed*')):
        fit=read(model_dir/'policy_fit.json');frame=pd.read_parquet(model_dir/'test_predictions.parquet')
        seed=int(model_dir.name.split('seed')[1]);blocks=frame.snapshot_cutoff.to_numpy()//86400
        row=paired_stats(frame.label.to_numpy(),frame.local_probability.to_numpy(),frame.selective_probability.to_numpy(),frame.local_prediction.to_numpy(),frame.selective_prediction.to_numpy(),seed+100000,500,blocks)
        row.update({'seed':seed,'run_id':cfg['run_id'],'experiment_id':'pooled_snapshot','bootstrap_unit':'UTC snapshot day block, added post-hoc after observing single-month window',
                    'scope':'exploratory few-day-block sensitivity, not reliable dependence correction or prospective validation'})
        day_stats.append(row)
        with np.load(model_dir/'test_features.npz',allow_pickle=False) as features:bf=features['benefit']
        for policy in fit['policies']:
            name=f"seed{int(model_dir.name.split('seed')[1])}_{policy['family']}_q{policy['budget']}_random{policy['random_repeat']}.npz"
            path=active/'budget_predictions'/name
            with np.load(path,allow_pickle=False) as executed:
                expected_score,expected_label,expected_route=apply_fitted(frame.raw_local_probability.to_numpy(),frame.counterfactual_full_raw_probability.to_numpy(),bf,fit,policy=policy,eligible=frame.reference_count.to_numpy()>0)
                diff=float(np.max(np.abs(expected_score-executed['probability'])));label_diff=int(np.sum(expected_label!=executed['prediction']));route_diff=int(np.sum(expected_route!=executed['escalated']))
                ap_expected=float(average_precision_score(frame.label,expected_score));ap_actual=float(average_precision_score(frame.label,executed['probability']))
                if diff>1e-6 or label_diff or route_diff:raise ValueError(f'actual selective/fitted policy parity failed {name}: score={diff}, labels={label_diff}, routes={route_diff}')
                assert np.array_equal(executed['label'],frame.label.to_numpy())
                assert np.array_equal(executed['sample_key'],np.array([hashlib.sha256(i.encode()).hexdigest() for i in frame.sample_id]))
                base=frame.local_probability.to_numpy()>=policy['final_threshold'];truth=frame.label.to_numpy();route=executed['escalated'];prediction=executed['prediction']
                wrong=route&(base!=truth);right=route&(base==truth);corrected=int((wrong&(prediction==truth)).sum());harmed=int((right&(prediction!=truth)).sum())
                benefits.append({'seed':seed,'family':policy['family'],'budget':policy['budget'],'random_repeat':policy['random_repeat'],
                    'N_escalated':int(route.sum()),'N_local_wrong_escalated':int(wrong.sum()),'N_local_correct_escalated':int(right.sum()),
                    'corrections':corrected,'harms':harmed,'correction_given_local_wrong':corrected/int(wrong.sum()) if wrong.any() else None,
                    'harm_given_local_correct':harmed/int(right.sum()) if right.any() else None,'comparison':'actual v2 selected labels versus calibrated local at identical final threshold'})
            rows.append({'cell':name,'N':len(frame),'score_max_abs_diff':diff,'prediction_disagreements':label_diff,'route_disagreements':route_diff,
                         'counterfactual_routed_AP':ap_expected,'actual_selected_AP':ap_actual,'AP_delta_float_tie_sensitivity':ap_actual-ap_expected,
                         'actual_selected_execution_sha256':sha256(path),'counterfactual_full_source_sha256':sha256(model_dir/'test_predictions.parquet')})
    assert len(rows)==180 and len(runtime)==540
    pd.DataFrame(day_stats).to_csv(out/'analysis/day_block_sensitivity.csv',index=False)
    write(out/'audits/block_sensitivity_scope.json',{'month_block_count':1,'month_bootstrap_informative':False,'month_intervals':'degenerate, not uncertainty evidence',
        'day_block_counts':[r['block_count'] for r in day_stats],'day_analysis':'post-hoc exploratory500 draws; few days, not robust graph/temporal dependence adjustment'})
    pd.DataFrame(rows).to_csv(out/'audits/selective_pipeline_parity.csv',index=False)
    pd.DataFrame(benefits).to_csv(out/'audits/selective_benefit_denominators.csv',index=False)
    write(out/'audits/method_identity_audit.json',{'N_fitted_policy_cells':180,'N_actual_runtime_runs':540,'measurement_run_id':cfg['run_id']+'_offline_v2','max_score_difference':max(r['score_max_abs_diff'] for r in rows),'label_disagreements':sum(r['prediction_disagreements'] for r in rows),'route_disagreements':sum(r['route_disagreements'] for r in rows),'score_tolerance':1e-6,'counterfactual_full_scores_distinguished_from_selected_execution':True})
    write(out/'audits/AP_tie_sensitivity.json',{'N_cells':180,'max_abs_AP_difference':max(abs(r['AP_delta_float_tie_sensitivity']) for r in rows),
        'scope':'within-guard float32 batch score changes can rearrange tied/near-tied rankings; AP need not be bit identical',
        'primary_snapshot_statistics':'archived canonical counterfactual-full scores composed with frozen route',
        'frontier':'actual v2 selected forwards; AP recomputed from those scores; no event timings or subjects mixed'})
    write(out/'audits/schema_adapter.json',{'contract_predictions':{'split_id':'test in paired_predictions filename; source validation records separately named','contract_id':'public sample_key is SHA256 original chain-qualified sample ID; transformed IDs are not original addresses',
        'snapshot_cutoff':'retained event_end','label_available_time':'unknown; CSV missing value, never imputed','local_logit':'pre-sigmoid logit not persisted; clipped probability log-odds is derivable but not the original exact raw logit',
        'relational_logit':'pre-sigmoid logit not persisted; stored counterfactual_full_raw_probability is a sigmoid probability, not an actual selected-branch logit',
        'feature_max_event_time':'unknown in snapshot input kit; live replay accepted event timestamps distinct',
        'policy_fit_data_max_time':'per-model calibration_data_max_time; reused source validation',
        'route_score':'derivable from frozen family formula and local probability/benefit features; route_threshold recorded'},
        'event_trace':{'dropped':'none silently dropped; accepted false + rejection_reason separates duplicate/late; rejected_events in summary',
        'rss':'system_trace every1000events and last, not every event','gpu':'allocated/reserved every event; utilization percent not sampled','checkpoint_id':'hash of actual persisted checkpoint at50000; replay cursor every event',
        'eviction_reason':'state cumulative counters sampled; individual eviction ID is not retained','source_read':'100000-row preload outside scoring clock; source population bookkeeping is unbounded harness, not bounded engine'},
        'scope':'adapter identifies provided/derived/unknown/sampled fields; not fabricated fully populated required schema'})
    if args.packages:
        paper=PROJECT/'paper/current/r01';build=paper/'build';packages=paper/'packages';packages.mkdir(exist_ok=True)
        records=[]
        for label in ('preprint','journal'):
            target=packages/(label+'_package.zip')
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(paper.rglob('*')):
                    if not path.is_file() or any(p in path.relative_to(paper).parts for p in ('build','packages','visual')):continue
                    if path.name in ('preprint.tex','journal.tex') and path.name!=label+'.tex':continue
                    if path.name=='COVER_LETTER.md' and label!='journal':continue
                    if path.suffix in ('.tex','.bib','.png') or path.name=='COVER_LETTER.md':archive.write(path,str(path.relative_to(paper)))
                archive.write(build/(label+'.pdf'),label+'.pdf');archive.write(build/'supplement.pdf','supplement.pdf')
                archive.writestr('PACKAGE_README.txt',f'Entry point: {label}.tex; standalone supplement: supplement.tex. Shared common content and generated numeric inputs are included. Run pdflatex/BibTeX/pdflatex/pdflatex from this directory. Author-local candidate only; exact final-version approval and submission-time checks are pending. Public scientific evidence is a separate paper-free archive.\n')
            records.append({'package':str(target.relative_to(ROOT)),'sha256':sha256(target),'bytes':target.stat().st_size,'main_pdf_sha256':sha256(build/(label+'.pdf')),'supplement_pdf_sha256':sha256(build/'supplement.pdf'),'private_not_git':True})
        write(out/'audits/private_package_identities.json',records)
    print('FINAL NUMERIC PARITY PASS',len(rows),flush=True)


if __name__=='__main__':main()
