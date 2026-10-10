#!/usr/bin/env python3
"""A08 current registry; recompute every new metric from persisted safe scores."""
import csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare,rankdata,wilcoxon
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
E=ROOT/'projects/benchmark/evidence/a08_data_repair';L=ROOT/'local_storage/benchmark/a08_data_repair';R=ROOT/'projects/benchmark/reports/a08_data_repair'
MODELS=['DOMINANT','AnomalyDAE','CoLA','GADNR','OCGNN','DLG-Base','DLG-Aug']
DATASETS=['Elliptic','DGraphFin','BitcoinOTC','Ethereum','BSC','Polygon','Yelp-Syn','Amazon-Syn','Reddit-Syn','Flickr-Syn','Cora-Syn','CiteSeer-Syn','PubMed-Syn']
CRYPTO={'ethereum':'Ethereum','bsc':'BSC','polygon':'Polygon'}


def sha256(path):
    # Public numeric reproduction deliberately has no training/data imports.
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def save_csv(name,rows):
    p=E/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('empty expected table '+name)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def render_public_tables(registry,legacy_stats):
    cells={};support=[]
    for d in DATASETS+['LANL-RedTeam']:
        for m in MODELS:
            rs=[r for r in registry if r['dataset']==d and r['model']==m and r['status']=='success']
            complete=len(rs)==5 and {int(r['seed']) for r in rs}==set(range(42,47))
            support.append({'dataset':d,'model':m,'support_status':'SUPPORTED_EXACT' if complete else 'NOT_FULLY_SUPPORTED','successes':len(rs),'seeds':';'.join(sorted(r['seed'] for r in rs))})
            if complete:
                for metric in ('pr_auc','roc_auc','validation_f1'):
                    values=[float(r[metric]) for r in rs if r.get(metric) not in (None,'')]
                    if len(values)==5:cells[d,m,metric]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=0)),'run_ids':[r['run_id'] for r in rs]}
    save_csv('tables/support_matrix.csv',support)
    for metric in ('pr_auc','roc_auc','validation_f1'):
        rows=[]
        for d in DATASETS:
            row={'dataset':d,'threshold_policy':'A08 validation-only' if d in CRYPTO.values() else 'historical descriptive (F1 policy varies)'}
            for m in MODELS:row[m]=f"{cells[d,m,metric]['mean']:.4f} ± {cells[d,m,metric]['sd']:.4f}" if (d,m,metric) in cells else 'N/A'
            rows.append(row)
        save_csv('tables/table_'+metric+'.csv',rows)
    # Dynamic complete-case versions of original S1-S4; S5 external descriptive.
    stats=[];ranks=[];pairs=[];rng=np.random.default_rng(20261008);reps=200000
    for prior in legacy_stats[:4]:
        models=prior['models'].split(';');base=prior['included_datasets'].split(';')+prior['excluded_datasets'].split(';')
        candidates=[d for d in DATASETS if d in base]
        included=[d for d in candidates if all((d,m,'pr_auc') in cells for m in models)]
        if len(included)<2:
            stats.append({'view':prior['view'],'status':'NOT_ESTIMABLE','included_datasets':included,'reason':'insufficient complete blocks'});continue
        matrix=np.array([[cells[d,m,'pr_auc']['mean'] for m in models] for d in included]);rank=rankdata(-matrix,axis=1,method='average');n,k=rank.shape
        tie_sum=sum(sum(c**3-c for c in np.unique(row,return_counts=True)[1] if c>1) for row in rank);corr=1-tie_sum/(n*(k**3-k))
        if corr<=0:stats.append({'view':prior['view'],'status':'NOT_ESTIMABLE','included_datasets':included,'reason':'all model ranks tied'});continue
        def stat(total):return (12/(n*k*(k+1))*np.square(total).sum(axis=-1)-3*n*(k+1))/corr
        observed=float(stat(rank.sum(0)));extreme=0
        for start in range(0,reps,2000):
            b=min(2000,reps-start);permutation=np.argsort(rng.random((b,n,k)),axis=2)
            totals=np.take_along_axis(np.broadcast_to(rank,(b,n,k)),permutation,axis=2).sum(axis=1);extreme+=int(np.count_nonzero(stat(totals)>=observed-1e-12))
        asym=friedmanchisquare(*matrix.T);means=rank.mean(0)
        stats.append({'view':prior['view'],'status':'ESTIMABLE','included_datasets':included,'excluded_datasets':[d for d in candidates if d not in included],'models':models,'n':n,'friedman_stat':observed,'asymptotic_p':float(asym.pvalue),'permutation_p':(extreme+1)/(reps+1),'extreme':extreme,'B':reps,'seed':20261008,'best_models':[m for m,v in zip(models,means) if np.isclose(v,means.min(),rtol=0,atol=1e-12)]})
        ranks.extend({'view':prior['view'],'model':m,'mean_rank':float(v),'datasets':';'.join(included)} for m,v in zip(models,means))
        # Entire declared Aug-vs-each family within view, not selected rows.
        raw=[]
        for m in models:
            if m=='DLG-Aug':continue
            diff=matrix[:,models.index('DLG-Aug')]-matrix[:,models.index(m)]
            p=1. if np.all(diff==0) else float(wilcoxon(diff,zero_method='wilcox',correction=False,alternative='two-sided',method='auto').pvalue)
            raw.append({'view':prior['view'],'comparison':'DLG-Aug vs '+m,'n':n,'raw_p':p,'wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'losses':int((diff<0).sum()),'zero_method':'wilcox','method':'scipy auto (ties/zeros use documented algorithm)'})
        order=np.argsort([r['raw_p'] for r in raw]);previous=0
        for i,index in enumerate(order):previous=max(previous,min(1.,(len(raw)-i)*raw[index]['raw_p']));raw[index]['holm_p']=previous
        pairs+=raw
    dump(E/'statistics/statistics_s1_s4.json',stats);save_csv('statistics/mean_ranks.csv',ranks);save_csv('statistics/pairwise.csv',pairs)
    sensitivity=[]
    continuity=['DOMINANT','CoLA','OCGNN','DLG-Base','DLG-Aug']
    groups={'strict_native_financial_two':['Elliptic','DGraphFin'],'clean_source_crypto_three':['Ethereum','BSC','Polygon'],
            'source_labeled_financial_five':['Elliptic','DGraphFin','Ethereum','BSC','Polygon'],
            'injected_eight':['BitcoinOTC','Yelp-Syn','Amazon-Syn','Reddit-Syn','Flickr-Syn','Cora-Syn','CiteSeer-Syn','PubMed-Syn']}
    for name,ds in groups.items():
        included=[d for d in ds if all((d,m,'pr_auc') in cells for m in continuity)]
        if not included:continue
        ranks_group=rankdata(-np.array([[cells[d,m,'pr_auc']['mean'] for m in continuity] for d in included]),axis=1)
        for m,v in zip(continuity,ranks_group.mean(0)):sensitivity.append({'group':name,'model':m,'n':len(included),'mean_rank':float(v),'datasets':';'.join(included),'scope':'descriptive; label provenance separate from topology'})
    save_csv('statistics/label_provenance_sensitivity.csv',sensitivity)
    alert=[]
    for d in CRYPTO.values():
        for m in MODELS:
            rs=[r for r in registry if r['dataset']==d and r['model']==m and r['status']=='success']
            if len(rs)!=5:continue
            for q in ('0.01','0.05'):
                for field in ('precision','recall'):
                    values=[r['alert_budgets'][q][field] for r in rs]
                    alert.append({'dataset':d,'model':m,'budget':q,'metric':field,'mean':float(np.mean(values)) if all(v is not None for v in values) else None,
                                  'sd':float(np.std(values,ddof=0)) if all(v is not None for v in values) else None,'seeds':'42;43;44;45;46'})
    save_csv('tables/alert_budget_crypto.csv',alert)
    external=[]
    for m in MODELS:
        if ('LANL-RedTeam',m,'pr_auc') in cells:external.append({'dataset':'LANL-RedTeam','model':m,'mean_ap':cells['LANL-RedTeam',m,'pr_auc']['mean'],'scope':'single graph descriptive; no omnibus inference'})
    save_csv('statistics/s5_lanl_descriptive.csv',external)
    dump(E/'number_registry.json',{'datasets':13,'models':7,'primary_supported_pairs':sum(r['support_status']=='SUPPORTED_EXACT' for r in support if r['dataset'] in DATASETS),
                                 'primary_successes':sum(r['status']=='success' and r['dataset'] in DATASETS and r['model'] in MODELS for r in registry),
                                 'external_successes':sum(r['status']=='success' and r['dataset']=='LANL-RedTeam' and r['model'] in MODELS for r in registry),
                                 'cells': [{'dataset':d,'model':m,'metric':metric,**value} for (d,m,metric),value in cells.items()]})

def main():
    from gog_fraud.data.crypto_raw import array_hash
    from gog_fraud.data.benchmark_metrics import evaluate_scores,validate_run_identity
    from gog_fraud.data.benchmark_lineage import verify_run
    config=json.loads((ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text())
    reuse=json.loads((E/'audit/historical_reuse.json').read_text())
    if reuse['status']!='PASS' or reuse['archive_sha256']!=sha256(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip'):
        raise ValueError('preserved historical evidence not qualified for reuse')
    for gate in ('G4','G5','G6'):
        if json.loads((E/'audit'/f'{gate}.json').read_text())['status']!='PASS':raise ValueError('required '+gate)
    with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip') as z:
        prefix='evaluation/benchmark/v2/paper_ready_a05/'
        legacy=list(csv.DictReader(io.StringIO(z.read(prefix+'publication_evidence_a05/approved_run_registry.csv').decode())))
        legacy_stats=list(csv.DictReader(io.StringIO(z.read(prefix+'statistics_s1_s5.csv').decode())))
    retained=[dict(r,evidence_tier='historical_preserved',policy='historical source policy; not claimed universally raw-score-reproducible') for r in legacy if r['dataset'] not in CRYPTO.values()]
    if {r['run_id'] for r in retained}!={r['run_id'] for r in reuse['preserved_records']}:raise ValueError('historical reuse membership mismatch')
    current=[];checks=[];thresholds=[];confusions=[]
    for c,label in CRYPTO.items():
        manifest=L/'frozen'/c/'input_manifest.json';input_hash=sha256(manifest)
        for model in MODELS:
            for seed in range(42,47):
                run_id=f'{c}_contract_clean_v1__{model}__seed{seed}';out=L/'runs'/run_id;rec=json.loads((out/'run_manifest.json').read_text())
                verify_run(ROOT,config,rec)
                validate_run_identity(rec,input_hash,rec['split_hash'],rec['model_config_hash'])
                if rec['status']!='SUPPORTED_EXACT':raise ValueError('unresolved support disposition: '+run_id)
                if sha256(out/'scores.npz')!=rec['raw_scores_sha256']:raise ValueError('raw score hash mismatch')
                with np.load(out/'scores.npz',allow_pickle=False) as z:arr={k:z[k] for k in z.files}
                if array_hash(arr['scores'])!=rec['score_array_hash']:raise ValueError('score array mismatch')
                metric=evaluate_scores(arr['labels'],arr['scores'],arr['node_ids'],arr['val_mask'],arr['test_mask'])
                old=json.loads((out/'metrics.json').read_text())
                if metric!=old:raise ValueError('persisted metrics mismatch '+run_id)
                if len(arr['scores'])!=json.loads(manifest.read_text())['N']:raise ValueError('population mismatch')
                view_hash=hashlib.sha256(json.dumps({'train_mask':json.loads(manifest.read_text())['arrays'][f'train_mask_{seed}'], 'val_mask':array_hash(arr['val_mask']), 'test_mask':array_hash(arr['test_mask'])},sort_keys=True).encode()).hexdigest()
                if view_hash!=rec['split_hash']:raise ValueError('split identity mismatch')
                rows={'dataset':label,'dataset_version':c+'_contract_clean_v1','model':model,'seed':str(seed),'run_id':run_id,'status':'success',
                      'pr_auc':metric['ap'],'roc_auc':metric['roc_auc'],'validation_f1':metric['thresholded']['f1'] if metric['thresholded'] else None,
                      'input_manifest_hash':input_hash,'metric_json_hash':sha256(out/'metrics.json'),'raw_score_hash':rec['raw_scores_sha256'],
                      'split_hash':rec['split_hash'],'model_config_hash':rec['model_config_hash'],'evidence_tier':'a08_raw_score_recomputable','policy':'validation-distinct-score maximum F1, largest threshold ties','test_prevalence':metric['test_prevalence'],'test_total':metric['test_total'],'test_positive':metric['test_positives'],'alert_budgets':{q:{k:v for k,v in b.items() if k!='selected_node_ids'} for q,b in metric['alert_budgets'].items()}}
                current.append(rows);checks.append({'run_id':run_id,'raw_score_sha256':rec['raw_scores_sha256'],'metric_sha256':rows['metric_json_hash'],'status':'PASS'})
                op=metric['operating_point'];thresholds.append({'dataset':label,'model':model,'seed':seed,'run_id':run_id,'threshold_kind':op['threshold_kind'],'threshold':op.get('threshold','UNDEFINED'),'source':'validation','candidate_rule':op['candidate_rule'],'tie_rule':op['tie_rule'],'validation_total':op['validation_total'],'validation_positive':op['validation_positives']})
                if metric['thresholded']:confusions.append({'dataset':label,'model':model,'seed':seed,'run_id':run_id,**metric['thresholded']})
    registry=retained+current
    dump(E/'approved_registry.json',{'schema_version':1,'campaign_id':'benchmark_a08_crypto_clean_v1','historical_archive_sha256':sha256(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip'),
                                    'records':registry,'historical_crypto_records_excluded':sum(r['dataset'] in CRYPTO.values() for r in legacy),'current_new_records':len(current)})
    save_csv('threshold_audit.csv',thresholds);save_csv('confusion_matrices.csv',confusions)
    dump(E/'audit/metric_recompute_audit.json',{'status':'PASS','runs':checks,'definition':'AP via average_precision_score; raw score direction fixed; no fabricated historical score','ddof':0})
    render_public_tables(registry,legacy_stats)
    dump(E/'audit/G7.json',{'status':'PASS','approved_registry_sha256':sha256(E/'approved_registry.json'),'number_registry_sha256':sha256(E/'number_registry.json'),
                          'recompute_audit_sha256':sha256(E/'audit/metric_recompute_audit.json'),'statistics_sha256':sha256(E/'statistics/statistics_s1_s4.json'),
                          'historical_reuse_sha256':sha256(E/'audit/historical_reuse.json'),
                          'aggregate_source_sha256':sha256(Path(__file__)),
                          'scope':'new crypto raw-score-recomputed; unaffected preserved historical scalar tier; common-policy F1 superiority across all13 is not inferred'})
    (R/'METRIC_AND_STATISTICS_REPORT.md').write_text('# A08 raw-score metric and statistics regeneration\n\nG7 PASS. New crypto metrics recomputed exactly from105 safe scores; retained historical records keep original evidence tier. All complete-case memberships, ranks,200000 block permutations and full Aug-vs-each Holm families are recalculated. Historical F1 policies remain descriptive. FINAL_PASS still requires current manuscript/PDF/reproduction audit.\n')
    print('A08 aggregate and G7 complete',flush=True)
if __name__=='__main__':
    if '--public-tables' in sys.argv:
        gate=json.loads((E/'audit/G7.json').read_text())
        if gate.get('status')!='PASS' or gate.get('approved_registry_sha256')!=sha256(E/'approved_registry.json'):
            raise ValueError('approved current numeric registry binding missing or changed')
        obj=json.loads((E/'approved_registry.json').read_text())
        if len(obj['records'])<1 or obj['current_new_records']<1:raise ValueError('empty approved registry')
        with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip') as z:
            prior=list(csv.DictReader(io.StringIO(z.read('evaluation/benchmark/v2/paper_ready_a05/statistics_s1_s5.csv').decode())))
        render_public_tables(obj['records'],prior)
        print('A08 public tables regenerated from approved numeric records; no raw-training/PDF claim')
    else:main()

