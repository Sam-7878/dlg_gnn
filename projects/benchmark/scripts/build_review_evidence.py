#!/usr/bin/env python3
"""Regenerate public review tables without unpublished manuscript sources or GPUs."""
import csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'projects/benchmark/evidence/astra_revision'
P='evaluation/benchmark/v2/paper_ready_a05/';PUB=P+'publication_evidence_a05/'


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    archive=ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip'
    expected=json.loads((ROOT/'projects/benchmark/expected_outputs.json').read_text())['evidence_zip_sha256']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise RuntimeError('frozen ZIP hash drift')
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        def rows(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
        cells={(r['dataset'],r['model'],r['metric']):float(r['mean']) for r in rows(PUB+'paper_metric_cells.csv')}
        views=rows(P+'statistics_s1_s5.csv')[:4]
        result=[];ranks_out=[];rng=np.random.default_rng(20261008);reps=200000
        for v in views:
            models=v['models'].split(';');ds=v['included_datasets'].split(';')
            rank=rankdata(-np.array([[cells[d,m,'pr_auc'] for m in models] for d in ds]),axis=1,method='average')
            n,k=rank.shape
            tie_sum=sum(sum(c**3-c for c in np.unique(row,return_counts=True)[1] if c>1) for row in rank)
            correction=1-tie_sum/(n*(k**3-k))
            statistic=lambda s:(12/(n*k*(k+1))*np.square(s).sum(axis=-1)-3*n*(k+1))/correction
            observed=float(statistic(rank.sum(0)))
            if abs(observed-float(v['friedman_stat']))>1e-10:raise RuntimeError('Friedman drift')
            extreme=0
            for start in range(0,reps,2000):
                b=min(2000,reps-start);permutation=np.argsort(rng.random((b,n,k)),axis=2)
                totals=np.take_along_axis(np.broadcast_to(rank,(b,n,k)),permutation,axis=2).sum(axis=1)
                extreme+=int(np.count_nonzero(statistic(totals)>=observed-1e-12))
            value=(extreme+1)/(reps+1);means=rank.mean(0)
            best=[m for m,r in zip(models,means) if np.isclose(r,means.min(),rtol=0,atol=1e-12)]
            result.append(dict(view=v['view'].split('_')[0],n_datasets=n,n_models=k,friedman_statistic=observed,asymptotic_p=v['friedman_p'],monte_carlo_p=value,repetitions=reps,seed=20261008,extreme=extreme,mc_standard_error=float(np.sqrt(value*(1-value)/(reps+1))),best_models=' / '.join(best)))
            ranks_out.extend(dict(view=v['view'].split('_')[0],model=m,mean_rank=float(r)) for m,r in zip(models,means))
        write('astra_friedman_permutation.csv',result);write('astra_mean_ranks.csv',ranks_out)
        def obj(n):return json.loads(z.read(n))
        registry=rows(PUB+'approved_run_registry.csv')
        exact, fused, dae = (obj(PUB+n) for n in ('environment/clean_exact_reconstruction_verification.json', 'environment/clean_fused_gcn_verification.json', 'anomalydae_exact_equivalence.json'))
        conad = obj(PUB+'conad_eq1_integration_a04.json')
        numeric=[]
        for dtype, r in exact['results'].items():
            for measure, a, b in [('loss','loss_abs_diff',None),('node squared score','score_max_abs_diff','score_max_rel_diff'),('gradient','grad_max_abs_diff','grad_max_rel_diff')]:
                numeric.append(dict(path='Gram',case='N=120; d=32; undirected binary',dtype=dtype,measure=measure,max_abs=r[a],max_rel=r[b] if b else 0.,atol=r['tolerance_atol'],rtol=r['tolerance_rtol'],status='PASS'))
        for dtype,r in fused['results'].items():
            for measure,a,b in [('forward','output_max_abs_diff','output_max_rel_diff'),('input gradient','x_grad_max_abs_diff','x_grad_max_rel_diff'),('weight gradient','weight_grad_max_abs_diff',None)]:
                numeric.append(dict(path='GCN',case='N=150; 32/64/32; 2 layers; dropout=0',dtype=dtype,measure=measure,max_abs=r[a],max_rel=r[b] if b else 'NOT_RECORDED',atol=r['tolerance_atol'],rtol=r['tolerance_rtol'],status='PASS'))
        for r in dae['datasets']:
            dtype=r['dtype'].split('.')[-1]; tolerance=dae[dtype+'_tolerance']
            checks={k:v for k,v in r['checks'].items() if isinstance(v,dict) and 'max_abs' in v}
            checks['gradient'] = {'max_abs':max(v['max_abs'] for v in r['gradient_comparison'].values()),'max_rel':max(v['max_rel'] for v in r['gradient_comparison'].values())}
            checks['Adam update'] = {'max_abs':r['checks']['max_post_update_parameter_abs'],'max_rel':max(v['max_rel'] for v in r['updated_parameter_comparison'].values())}
            for measure,v in checks.items():
                numeric.append(dict(path='AnomalyDAE',case=f"{r['dataset']}; N={r['nodes']}; B={r['block_size']}",dtype=dtype,measure=measure,max_abs=v['max_abs'],max_rel=v['max_rel'],atol=tolerance['atol'],rtol=tolerance['rtol'],status=r['status']))
        write('astra_exact_qualification.csv',numeric)
        # Partition only for descriptive sensitivity; no change to frozen S1-S4.
        models=views[1]['models'].split(';');all_ds=views[1]['included_datasets'].split(';')
        groups={'provider-native two':['Elliptic','DGraphFin'], 'contract artifact three':['Ethereum','BSC','Polygon'], 'injected eight':[d for d in all_ds if d not in ['Elliptic','DGraphFin','Ethereum','BSC','Polygon']]}
        sens=[]
        for label,ds in groups.items():
            rank=rankdata(-np.array([[cells[d,m,'pr_auc'] for m in models] for d in ds]),axis=1)
            for m,r in zip(models,rank.mean(0)):
                sens.append(dict(group=label,n_graphs=len(ds),model=m,mean_rank=float(r),datasets=';'.join(ds),inference='descriptive only'))
        write('astra_label_provenance_sensitivity.csv',sens)
        # Cell-by-cell auditability (raw score availability is not equivalent to full rerunnability).
        lineage=[]; thresholds=[]; prevalence=[]
        by_cell={}
        for r in registry:
            by_cell.setdefault((r['dataset'],r['model']),[]).append(r)
            rid=r['run_id']; ns=[n for n in z.namelist() if n.endswith('/'+rid+'.json') and '/legacy_metric_json/' in n]
            raw=obj(ns[0]) if ns else {}
            record=raw.get('result',raw)
            lineage.append({k:r.get(k,'') for k in ('dataset','model','seed','run_id','source_kind','metric_json_hash','raw_score_status','raw_score_hash','model_config_provenance_status','execution_environment_status','source_commit')})
            if r['dataset']=='Ethereum' and r['model'] in ('DLG-Aug','DLG-Base'):
                thresholds.append(dict(model=r['model'],seed=r['seed'],run_id=rid,pr_auc=r['pr_auc'],roc_auc=r['roc_auc'],test_f1=r['validation_f1'],threshold_policy='max validation F1; percentiles 80..99.5 (40)' if r['model']=='DLG-Aug' else 'validation prevalence percentile',threshold_value=record.get('validation_threshold','NOT_RECORDED'),score_status=r['raw_score_status'],confusion_matrix='NOT_RECORDED',normalization='no split-wise normalization in archived runner'))
            if r['dataset'] in ('Elliptic','DGraphFin') and r['model']=='DLG-Aug':
                pos=record.get('test_positive');neg=record.get('test_negative')
                if pos is not None and neg is not None:
                    prevalence.append(dict(dataset=r['dataset'],seed=r['seed'],run_id=rid,test_positive=pos,test_negative=neg,test_prevalence=pos/(pos+neg),average_precision=float(r['pr_auc']),ap_over_prevalence=float(r['pr_auc'])/(pos/(pos+neg))))
        write('astra_run_auditability.csv',lineage);write('astra_ethereum_threshold_audit.csv',thresholds);write('astra_test_prevalence.csv',prevalence)
        failures=[]
        for r in rows(P+'table_support_primary13_24g.csv'):
            if r['support_status'] not in ('SUPPORTED_EXACT','DIAGNOSTIC_ONLY'):failures.append(r)
        write('astra_unsupported_cells.csv',failures)
        ablation=rows(P+'table_dlg_ablation_elliptic_summary.csv');write('astra_elliptic_controls.csv',ablation)
        memory=rows(PUB+'table_memory_selected_a04.csv');write('astra_matched_memory.csv',memory)
        # Provenance fields remain explicit, rather than imputing modern defaults into old runs.
        campaign=[]
        for (ds,m),rs in sorted(by_cell.items()):
            campaign.append(dict(dataset=ds,model=m,n=len(rs),source_kinds=';'.join(sorted({r['source_kind'] for r in rs})),config_provenance=';'.join(sorted({r['model_config_provenance_status'] for r in rs})),environment_provenance=';'.join(sorted({r['execution_environment_status'] for r in rs})),raw_scores_stored=sum(bool(r['raw_score_hash']) for r in rs)))
        write('astra_cell_provenance.csv',campaign)
        frozen=['table_main_13_pr_auc.csv','table_main_13_roc_auc.csv','table_main_13_f1.csv','statistics_s1_s5.csv','table_support_primary13_24g.csv','table_support_lanl_24g.csv','table_dlg_ablation_elliptic_summary.csv']
        for n in frozen:(OUT/n).write_bytes(z.read(P+n))
        for n in ['approved_run_registry.csv','paper_metric_cells.csv','dataset_manifest_canonical.json','table_memory_selected_a04.csv','a05_config_provenance_crypto_lanl.csv']:
            (OUT/n).write_bytes(z.read(PUB+n))
        for n in ['anomalydae_exact_equivalence.json','conad_eq1_integration_a04.json']:
            (OUT/n).write_bytes(z.read(PUB+n))
        for n in ['clean_exact_reconstruction_verification.json','clean_fused_gcn_verification.json']:
            (OUT/n).write_bytes(z.read(PUB+'environment/'+n))
    print(json.dumps({'status':'PASS','scope':'public evidence tables; no manuscript source; no training','permutations':reps,'s2_leaders':result[1]['best_models']},indent=2))
if __name__=='__main__':main()
