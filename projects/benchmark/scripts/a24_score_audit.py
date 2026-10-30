from a24_common import *
from a24_inputs import load_input
import numpy as np,math
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.stats import spearmanr
checks=[]
for p in sorted((LOCAL/'runs').glob('*/result.json')):
 r=json.loads(p.read_text());zpath=p.parent/'scores.npz';assert sha(zpath)==r['raw_score_sha256'];z=np.load(zpath,allow_pickle=False);y=z['labels'];s=z['scores'];te=z['test_mask'];va=z['val_mask'];assert not np.any(te&va)
 ap=float(average_precision_score(y[te],s[te]));roc=float(roc_auc_score(y[te],s[te]));assert ap==r['pr_auc'] and roc==r['roc_auc']
 for q,k in [(.01,'1'),(.05,'5')]:
  n=int(te.sum());budget=max(1,math.ceil(q*n));ids=z['node_ids'][te];order=np.lexsort((ids,-s[te]));tp=int(y[te][order[:budget]].sum());assert tp/budget==r['P_at_'+k+'pct'];assert tp/int(y[te].sum())==r['R_at_'+k+'pct'];assert tp/budget/(y[te].mean())==r['lift_'+k+'pct']
 t=r['threshold'];pred=s[va]>=t;tp=int(((y[va]==1)&pred).sum());fp=int(((y[va]==0)&pred).sum());fn=int(((y[va]==1)&~pred).sum());assert 2*tp/(2*tp+fp+fn)==r['max_validation_f1'];prev=float(y[va].mean());assert r['max_validation_f1']+1e-12>=2*prev/(1+prev)
 checks.append({'run_id':r['run_id'],'status':'PASS_INDEPENDENT_AP_ROC_TP_PRECISION_RECALL_LIFT','score_sha256':sha(zpath)})
 z.close()
writecsv(SUP/'raw_metric_recompute_checks.csv',checks)
# Exploratory degree bins, fixed here before calculating bin outcomes;no causal conclusions.
rr=rows(SUP/'real_domain_alert_budget_per_seed.csv');assert len(rr)==30
bins=[]
for d in ['Elliptic','DGraphFin','LANL-RedTeam']:
 data,_,identity=load_input(d);deg=np.bincount(data.edge_index[1].numpy(),minlength=data.num_nodes)
 for r in [r for r in rr if r['dataset']==d]:
  z=np.load(LOCAL/'runs'/r['run_id']/'scores.npz',allow_pickle=False);y=z['labels'];s=z['scores'];te=z['test_mask'];cor=float(spearmanr(s[te],deg[te]).statistic)
  for label,lo,hi in [('0',0,1),('1',1,2),('2-4',2,5),('5-15',5,16),('16+',16,10**12)]:
   mask=te&(deg>=lo)&(deg<hi);yy=y[mask];ss=s[mask];pos=int(yy.sum());n=len(yy)
   bins.append({'dataset':d,'model':r['model'],'seed':r['seed'],'in_degree_bin':label,'test_N':n,'test_positive':pos,'AP':float(average_precision_score(yy,ss)) if pos else '', 'ROC_AUC':float(roc_auc_score(yy,ss)) if 0<pos<n else '', 'mean_score':float(ss.mean()) if n else '', 'score_degree_spearman':cor,'interpretation':'exploratory association;not causal;undefined metrics omitted','score_sha256':r['raw_score_sha256']})
  z.close()
writecsv(SUP/'degree_bin_per_seed.csv',bins)
(SUP/'degree_bin_diagnostic.md').write_text('# Exploratory degree diagnostic\n\nStored directed in-degree bins0,1,2–4,5–15,16+ are applied to the30separate A24 Base/Aug vectors. AP/ROC are omitted where class support is absent. Correlation/stratum differences are hypothesis-generating associations,not evidence that degree causes below-random performance. No test-derived sign inversion,training adjustment or primary replacement was performed.\n')
dump(SUP/'supplementary_acceptance.json',{'status':'PASS','feature_only_evaluations':60,'canonical_control_runs':25,'recovery_new_GNN_runs':20,'existing_Elliptic_control_vectors_reused':10,'new_GNN_total_excluding_resource_cases':45,'full_resource_cases':4,'resource_pilots':4,'independent_score_audit_runs':len(checks),'primary_registry_sha256':sha(REG),'G9':'OPTIONAL_NOT_ATTEMPTED'})
print('Supplementary score and degree-bin audit PASS;',len(checks),'saved vector sets',flush=True)
