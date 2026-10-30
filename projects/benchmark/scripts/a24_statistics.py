from a24_common import *
import numpy as np
from scipy.stats import rankdata,friedmanchisquare,chi2,wilcoxon
rs=rows(REG);models=['DOMINANT','AnomalyDAE','CoLA','GADNR','OCGNN','DLG-Base','DLG-Aug'];datasets=list(dict.fromkeys(r['dataset'] for r in rs));agg=[]
for d in datasets:
 for model in models:
  selected=[r for r in rs if r['dataset']==d and r['model']==model];assert len(selected) in [0,5]
  if not selected:continue
  for field in ['pr_auc','roc_auc','f1']:
   x=np.array([float(r[field]) for r in selected]);agg.append({'dataset':d,'model':model,'metric':field,'n':5,'mean':float(x.mean()),'sample_SD':float(x.std(ddof=1)),'population_SD_prior':float(x.std(ddof=0)),'ddof':1,'source_registry_sha256':sha(REG),'run_ids':';'.join(r['run_id'] for r in sorted(selected,key=lambda r:int(r['seed'])))})
writecsv(PUB/'metrics_mean_sample_sd.csv',agg);writecsv(PUB/'sd_ddof_audit_a24.csv',agg)
cells={(a['dataset'],a['model'],a['metric']):a for a in agg};old=json.loads((ROOT/'projects/benchmark/analysis/statistics_a12.json').read_text());rng=np.random.default_rng(20261024);stats=[];ranks=[];pairs=[]
for i,v in enumerate(old['views'],1):
 ds=v['included_datasets'];ms=v['models'];n,k=len(ds),len(ms);a=np.array([[cells[d,m,'pr_auc']['mean'] for m in ms] for d in ds]);r=rankdata(-a,axis=1,method='average');ties=sum(sum(c**3-c for c in np.unique(row,return_counts=True)[1]) for row in r);corr=1-ties/(n*(k**3-k));q=float((12/(n*k*(k+1))*(r.sum(0)**2).sum()-3*n*(k+1))/corr);assert np.isclose(q,v['friedman_stat_Q'],atol=1e-10)
 b=0;B=1000000
 for _ in range(10):
  ix=np.argsort(rng.random((n,100000,k)),axis=2);s=np.take_along_axis(r[:,None,:],ix,axis=2).sum(0);qp=(12/(n*k*(k+1))*(s**2).sum(1)-3*n*(k+1))/corr;b+=int((qp>=q-1e-12).sum())
 assert b==v['exceedances_b'],(i,b,v['exceedances_b'])
 p=(b+1)/(B+1);srow={'view_id':'S'+str(i),'n_datasets':n,'k_models':k,'datasets':';'.join(ds),'models':';'.join(ms),'Q':q,'asymptotic_p':float(chi2.sf(q,k-1)),'permutations':B,'rng_seed':20261024,'exceedances':b,'MC_p':p,'MC_SE':float(np.sqrt(p*(1-p)/B)),'tail_upper95_if_zero':float(1-.05**(1/B)) if b==0 else '', 'Kendall_W':q/(n*(k-1))};stats.append(srow)
 for m,val in zip(ms,r.mean(0)):ranks.append({'view_id':'S'+str(i),'model':m,'mean_rank':float(val)})
 raw=[]
 for j,m in enumerate(ms):
  if m=='DLG-Aug':continue
  delta=a[:,ms.index('DLG-Aug')]-a[:,j];pval=float(wilcoxon(delta,zero_method='wilcox',alternative='two-sided').pvalue) if np.any(delta) else 1.;raw.append((m,pval))
 ix=np.argsort([p for _,p in raw]);previous=0.;adj={}
 for order,j in enumerate(ix):previous=max(previous,min(1.,(len(raw)-order)*raw[j][1]));adj[j]=previous
 for j,(m,pval) in enumerate(raw):pairs.append({'view_id':'S'+str(i),'comparison':'DLG-Aug vs '+m,'p':pval,'Holm_p':adj[j]})
 print('S'+str(i),q,'W',srow['Kendall_W'],'b',b,flush=True)
writecsv(PUB/'stats_master_a24.csv',stats);writecsv(PUB/'mean_ranks_a24.csv',ranks);writecsv(PUB/'pairwise_holm_a24.csv',pairs)
# Seed-paired differences are joined on actual seed, never array order.
paired=[]
for d in datasets:
 a={r['seed']:r for r in rs if r['dataset']==d and r['model']=='DLG-Aug'};b={r['seed']:r for r in rs if r['dataset']==d and r['model']=='DLG-Base'};assert a.keys()==b.keys();v=np.array([float(a[s]['pr_auc'])-float(b[s]['pr_auc']) for s in sorted(a)])
 paired.append({'dataset':d,'n':len(v),'paired_seeds':';'.join(sorted(a)),'mean_delta_AP':float(v.mean()),'sample_SD':float(v.std(ddof=1)),'paired_SE':float(v.std(ddof=1)/np.sqrt(len(v)))})
writecsv(PUB/'paired_deltas_a24.csv',paired)
# Historical diagnostic F1 was returned from TEST after a restricted validation quantile search.
p=ROOT/'evaluation/benchmark/v2/scripts/a03_run_elliptic_ablation.py';src=p.read_text();assert 'val_f1 = float(f1_score(y_test, pred_test' in src
perseed=ROOT/'evaluation/benchmark/v2/paper_ready_a03/table_dlg_ablation_elliptic.csv';diag=rows(perseed);assert len(diag)==25
out=[]
for v in sorted({r['variant'] for r in diag}):
 rr=[r for r in diag if r['variant']==v];assert len(rr)==5
 x=np.array([float(r['pr_auc']) for r in rr]);z=np.array([float(r['roc_auc']) for r in rr]);out.append({'variant':v,'campaign':'A03_HISTORICAL_DIAGNOSTIC','n':5,'AP_mean':float(x.mean()),'AP_sample_SD':float(x.std(ddof=1)),'ROC_mean':float(z.mean()),'ROC_sample_SD':float(z.std(ddof=1)),'precision_limit':'source CSV saved per-seed metrics rounded to4decimals;raw scores not stored'})
writecsv(PUB/'historical_diagnostic_sd_a24.csv',out)
dump(PUB/'ablation_f1_provenance_audit.json',{'status':'PASS_HISTORICAL_F1_COLUMN_OMITTED','A03_source':str(p.relative_to(ROOT)),'source_sha256':sha(p),'threshold_candidates':'40percentiles from80to99.5 of validation scores','operator':'>=','ties':'first encountered on strict F1 improvement','all_positive_included':False,'reported_validation_f1_actual_population':'TEST,not validation','diagnostic_split':'RandomState(seed) permutation60/20/20','canonical_split':'separate50/50 stratified val/test on full observable training graph','canonical_F1_copied_from_diagnostic_not_supported':'same numbers cannot establish source; canonical displayed values not supported by selected primary records','disposition':'Remove Val F1 column entirely from A3;keep AP/ROC separate campaign and sample SD from per-seed sources','why_test_prevalence_cannot_bound_validation_F1':'different evaluation population and restricted candidate set'})
dump(PUB/'statistics_recompute.json',{'status':'PASS','all_four_views_recomputed':True,'B':1000000,'seed':20261024,'all_frozen_exceedances_reproduced':True,'sample_SD_ddof':1,'canonical_registry_unchanged':sha(REG)})
