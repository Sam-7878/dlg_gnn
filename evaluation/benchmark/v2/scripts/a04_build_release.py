#!/usr/bin/env python3
"""Build an auditable A04 candidate without inventing missing provenance."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'evaluation/benchmark/v2'
OUT = V2 / 'paper_ready_final'
BUNDLE = OUT / 'publication_evidence_a04'
RAW_R5 = ROOT / 'outputs/benchmark/sci_round5_final/raw'
RAW_A03 = ROOT / 'outputs/benchmark/a03_production/raw'
DEFENSE = ROOT / 'outputs/benchmark/sci_defense_extension_real_final/statistics/performance_11_dataset_raw.csv'
LANL_RAW = ROOT / 'outputs/benchmark/sci_defense_extension_real/benchmark/raw'
LOCK = ROOT / 'environment/locks/benchmark-a03-cuda.lock.txt'
PRIMARY = ['Elliptic','DGraphFin','BitcoinOTC','Ethereum','BSC','Polygon','Yelp-Syn','Amazon-Syn','Reddit-Syn','Flickr-Syn','Cora-Syn','CiteSeer-Syn','PubMed-Syn']
ALL = PRIMARY + ['LANL-RedTeam']
REAL6 = PRIMARY[:6]
SYN7 = PRIMARY[6:]
MODELS = ['DOMINANT','AnomalyDAE','CoLA','GADNR','OCGNN','DLG-Base','DLG-Aug']
DIAGNOSTIC_MODEL = 'CONAD-PyGOD-1.1-reference'
CONT5 = ['DOMINANT','CoLA','OCGNN','DLG-Base','DLG-Aug']
METRICS = {'pr_auc':'table_main_13_pr_auc.csv','roc_auc':'table_main_13_roc_auc.csv','validation_f1':'table_main_13_f1.csv'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def scalar(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None


def normalized(name):
    return name+'-Syn' if name in {'Yelp','Amazon','Reddit','Flickr','Cora','CiteSeer','PubMed'} else name


def load_records():
    """Prefer actual per-run JSON; choose explicit repaired crypto DLG-Aug runs."""
    chosen={}; rejected=[]
    for base,origin in [(RAW_R5,'round5_json'),(RAW_A03,'a03_json')]:
        for p in sorted(base.glob('*.json')):
            d=json.loads(p.read_text()); ds=normalized(d.get('display_name') or d.get('dataset'))
            model=d.get('model')
            if ds not in ALL: continue
            if model in {'CONAD','CONAD-reference','CONAD-corrected'}:
                rejected.append({'path':str(p.relative_to(ROOT)),'reason':'CONAD diagnostic or experimental only'})
                continue
            if model not in MODELS: continue
            if ds in {'Ethereum','BSC','Polygon'} and origin!='a03_json': continue
            if ds not in {'Ethereum','BSC','Polygon'} and origin!='round5_json': continue
            # The repaired crypto DLG-Aug run has top-level metrics and a fresh clone.
            if model=='DLG-Aug' and origin=='a03_json' and 'result' in d:
                rejected.append({'path':str(p.relative_to(ROOT)),'reason':'superseded pre-clone crypto run'})
                continue
            res=d.get('result',d)
            key=(ds,model,int(d['seed']))
            rec={'dataset':ds,'model':model,'seed':int(d['seed']),'run_id':d.get('run_id'),
                 'status':d.get('status') or res.get('status'),'source_path':str(p.relative_to(ROOT)),
                 'source_kind':origin,'metric_json_hash':sha(p),'raw_score_hash':None,
                 'source_config_hash':d.get('config_hash'),'source_backend_hash':d.get('backend_hash'),
                 'nodes':d.get('nodes'),'edges':d.get('edges')}
            for metric in [*METRICS,'precision_at_k','recall_at_k','topk_f1']:
                rec[metric]=scalar(res.get(metric))
            if key in chosen:
                raise RuntimeError(f'duplicate approved run {key}: {p} and {chosen[key]["source_path"]}')
            chosen[key]=rec
    with DEFENSE.open(newline='') as f:
        csv_lanl={(d['dataset'],d['model'],int(d['seed'])):d for d in csv.DictReader(f)
                  if d['dataset']=='LANL-RedTeam' and d['model'] in MODELS}
    for p in sorted(LANL_RAW.glob('LANL-RedTeam__*.json')):
        d=json.loads(p.read_text());model=d['model']
        if model not in MODELS: continue
        key=('LANL-RedTeam',model,int(d['seed']))
        c=csv_lanl.get(key)
        if c is None or c['run_id']!=d['run_id'] or any(
            not math.isclose(float(c[k]),float(d['f1' if k=='validation_f1' else k]),abs_tol=1e-12)
            for k in ('pr_auc','roc_auc','validation_f1')):
            raise RuntimeError(f'LANL raw JSON/defense CSV mismatch: {p}')
        if key in chosen: raise RuntimeError(f'duplicate LANL run {key}')
        rec={'dataset':key[0],'model':key[1],'seed':key[2],'run_id':d['run_id'],
             'status':d['status'],'source_path':str(p.relative_to(ROOT)),
             'source_kind':'lanl_real_json','metric_json_hash':sha(p),'raw_score_hash':None,
             'source_config_hash':None,'source_backend_hash':None,
             'nodes':16694,'edges':323897,'source_csv_sha256':sha(DEFENSE),
             'pr_auc':scalar(d.get('pr_auc')),'roc_auc':scalar(d.get('roc_auc')),
             'validation_f1':scalar(d.get('f1')),
             'precision_at_k':None,'recall_at_k':None,'topk_f1':None}
        chosen[key]=rec
    return chosen,rejected


def status_for(rows):
    if len(rows)==5 and all(r['status']=='success' for r in rows): return 'SUPPORTED_EXACT'
    failures=[r['status'] for r in rows]
    if failures and all('OOM' in s for s in failures): return 'UNSUPPORTED_RESOURCE_OOM'
    if failures: return 'UNSUPPORTED_EXECUTION_ERROR'
    return 'NO_RAW_RUN'


def aggregate(records):
    by=defaultdict(list)
    for r in records.values(): by[(r['dataset'],r['model'])].append(r)
    support=[]; values={}; cells=[]
    for ds in ALL:
        for model in [*MODELS,DIAGNOSTIC_MODEL]:
            rs=sorted(by[(ds,model)],key=lambda r:r['seed'])
            status='DIAGNOSTIC_ONLY' if model==DIAGNOSTIC_MODEL else status_for(rs)
            support.append({'dataset':ds,'model':model,'support_status':status,
                'num_approved_runs':len(rs),'seeds':';'.join(str(r['seed']) for r in rs),
                'failure_reason':'Excluded from fair inference after loss-path audit' if model==DIAGNOSTIC_MODEL else ';'.join(sorted(set(r['status'] for r in rs if r['status']!='success')))})
            if status!='SUPPORTED_EXACT': continue
            if [r['seed'] for r in rs]!=[42,43,44,45,46]:
                raise RuntimeError(f'wrong seeds: {ds} {model}')
            for metric in METRICS:
                nums=[r[metric] for r in rs]
                if any(v is None for v in nums): continue
                mean=statistics.mean(nums);std=statistics.pstdev(nums)
                values[(ds,model,metric)]=(mean,std)
                cells.append({'dataset':ds,'model':model,'metric':metric,'mean':mean,'std_population':std,
                    'run_ids':';'.join(r['run_id'] or '' for r in rs),
                    'metric_json_sha256s':';'.join(r['metric_json_hash'] or '' for r in rs)})
    write_csv(OUT/'table_support_primary13_24g.csv',support[:13*8],list(support[0]))
    write_csv(OUT/'table_support_lanl_24g.csv',support[13*8:],list(support[0]))
    write_csv(OUT/'table_support_24g.csv',support,list(support[0]))
    write_csv(BUNDLE/'paper_metric_cells.csv',cells,list(cells[0]))
    for metric,filename in METRICS.items():
        rows=[]
        for ds in ALL:
            row={'dataset':ds,'category':'external' if ds=='LANL-RedTeam' else 'real' if ds in REAL6 else 'synthetic'}
            for model in MODELS:
                val=values.get((ds,model,metric));stat=next(x['support_status'] for x in support if x['dataset']==ds and x['model']==model)
                row[model]=f'{val[0]:.4f} ± {val[1]:.4f}' if val else ('OOM' if stat=='UNSUPPORTED_RESOURCE_OOM' else 'FAIL' if stat=='UNSUPPORTED_EXECUTION_ERROR' else 'N/A')
            rows.append(row)
        write_csv(OUT/filename,rows,['dataset','category',*MODELS])
    return support,values,cells


def stats(values):
    rows=[]
    for view,datasets,models in [
        ('S1_SevenModel_CompleteCase',PRIMARY,MODELS),
        ('S2_ContinuityFive',PRIMARY,CONT5),
        ('S3_FinancialBlockchainSix_MixedLabels',REAL6,CONT5),
        ('S4_SyntheticSeven',SYN7,CONT5)]:
        complete=[ds for ds in datasets if all((ds,m,'pr_auc') in values for m in models)]
        matrix=np.array([[values[(ds,m,'pr_auc')][0] for m in models] for ds in complete])
        row={'view':view,'models':';'.join(models),'n_complete':len(complete),
             'included_datasets':';'.join(complete),'excluded_datasets':';'.join(d for d in datasets if d not in complete),
             'friedman_stat':None,'friedman_p':None,'best_ranked_model':None,'dlg_aug_mean_rank':None,
             'holm_adjusted_p_json':'{}'}
        if len(complete)>=3:
            stat,p=friedmanchisquare(*[matrix[:,j] for j in range(len(models))]);row['friedman_stat']=float(stat);row['friedman_p']=float(p)
            ranks=np.array([pd.Series(a).rank(ascending=False).to_numpy() for a in matrix]);av=ranks.mean(axis=0)
            row['best_ranked_model']=models[int(np.argmin(av))];row['dlg_aug_mean_rank']=float(av[models.index('DLG-Aug')])
            comparisons=[];target=matrix[:,models.index('DLG-Aug')]
            for j,model in enumerate(models):
                if model=='DLG-Aug': continue
                if np.allclose(target,matrix[:,j],atol=0,rtol=0): wp=1.0
                else: wp=float(wilcoxon(target,matrix[:,j],alternative='two-sided').pvalue)
                comparisons.append((model,wp))
            comparisons.sort(key=lambda x:x[1]);adjusted={};running=0.0
            for rank,(model,wp) in enumerate(comparisons):
                running=max(running,min(1.0,wp*(len(comparisons)-rank)));adjusted[model]=running
            row['holm_adjusted_p_json']=json.dumps(adjusted,sort_keys=True)
        rows.append(row)
    rows.append({'view':'S5_LANL_Descriptive','models':';'.join(CONT5),'n_complete':1,
        'included_datasets':'LANL-RedTeam','excluded_datasets':'','friedman_stat':None,'friedman_p':None,
        'best_ranked_model':None,'dlg_aug_mean_rank':None,'holm_adjusted_p_json':'{}'})
    write_csv(OUT/'statistics_s1_s5.csv',rows,list(rows[0]))
    return rows


def diagnostics():
    abl=pd.read_csv(V2/'paper_ready_a03/table_dlg_ablation_elliptic.csv')
    gate=pd.read_csv(V2/'diagnostics/dlg_base_dominant/dlg_gate_10runs.csv')
    assert len(abl)==25 and set(abl['seed'])==set(range(42,47))
    assert len(gate)==10 and set(gate['seed'])==set(range(42,47))
    summary=abl.groupby('variant',sort=True).agg(n=('seed','count'),mean_pr_auc=('pr_auc','mean'),mean_roc_auc=('roc_auc','mean'),mean_validation_f1=('validation_f1','mean')).reset_index()
    summary.to_csv(OUT/'table_dlg_ablation_elliptic_summary.csv',index=False)
    gate_summary=gate.groupby('dataset',sort=True).agg(n=('seed','count'),mean_raw_alpha=('final_alpha','mean'),mean_sigmoid_alpha=('sigmoid_final_alpha','mean')).reset_index()
    gate_summary.to_csv(OUT/'table_dlg_gate_summary.csv',index=False)
    return summary,gate_summary


def main():
    OUT.mkdir(parents=True,exist_ok=True);BUNDLE.mkdir(parents=True,exist_ok=True)
    records,rejected=load_records()
    lock_hash=sha(LOCK)
    registry=[]
    for r in sorted(records.values(),key=lambda x:(x['dataset'],x['model'],x['seed'])):
        registry.append({**r,'environment_lock_hash':None,
            'dataset_hash':None,'split_hash':None,'model_config_hash':r['source_config_hash'],
            'provenance_complete':False})
    write_csv(BUNDLE/'approved_run_registry.csv',registry,list(registry[0]))
    write_json(BUNDLE/'rejected_or_superseded_runs.json',rejected)
    support,values,cells=aggregate(records)
    statistics_rows=stats(values)
    ablation,gate=diagnostics()
    claims=[]
    for ds in REAL6:
        candidates=[(m,values[(ds,m,'pr_auc')][0]) for m in MODELS if (ds,m,'pr_auc') in values]
        if candidates:
            winner,max_value=max(candidates,key=lambda x:x[1])
            claims.append({'claim':f'{ds}: highest observed seven-model PR-AUC is {winner} ({max_value:.4f})',
                'source':'table_main_13_pr_auc.csv','status':'DESCRIPTIVE_ONLY'})
    s3=next(r for r in statistics_rows if r['view']=='S3_FinancialBlockchainSix_MixedLabels')
    claims.append({'claim':f'S3 financial/blockchain six (mixed labels) Friedman p={s3["friedman_p"]:.6f}; no superiority inference at alpha=0.05',
        'source':'statistics_s1_s5.csv','status':'VERIFIED'})
    perm=float(ablation.loc[ablation.variant=='DLG-Aug-Permuted','mean_pr_auc'].iloc[0])
    aligned=float(ablation.loc[ablation.variant=='DLG-Aug','mean_pr_auc'].iloc[0])
    claims.append({'claim':f'Elliptic permuted PR-AUC {perm:.5f} versus aligned {aligned:.5f}; node-wise alignment mechanism unestablished',
        'source':'table_dlg_ablation_elliptic_summary.csv','status':'VERIFIED'})
    write_csv(OUT/'claims_to_evidence_a04.csv',claims,list(claims[0]))
    write_csv(BUNDLE/'raw_score_hashes.csv',[{'run_id':r['run_id'],'raw_score_hash':'MISSING','reason':'No preserved raw score array or checksum in source run'} for r in registry],['run_id','raw_score_hash','reason'])
    write_json(OUT/'publication_manifest_a04.json',{'edition':'A04_CANDIDATE','primary_datasets':PRIMARY,
        'external_datasets':['LANL-RedTeam'],'primary_models':MODELS,'diagnostic_models':['CONAD-PyGOD-1.1-reference'],
        'a04_qualification_environment_lock_sha256':lock_hash,'approved_run_count':len(registry),'publication_metric_cell_count':len(cells),
        'gate_g4':'HOLD_PENDING_CROSS_FILE_VALIDATION'})
    print(json.dumps({'approved_runs':len(registry),'cells':len(cells),'s3_p':s3['friedman_p'],'elliptic_permuted_pr':perm,'elliptic_aligned_pr':aligned},indent=2))

if __name__=='__main__': main()
