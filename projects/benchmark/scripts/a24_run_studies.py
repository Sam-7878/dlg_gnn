"""Finite A24 campaigns. Every fit receives graph/features without targets."""
from a24_common import *
from a24_inputs import load_input,tensor_hash
import os,sys,time,random,math,resource,gc,copy
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,torch,psutil
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(LOCAL/'deps'))
from gog_fraud.data.benchmark_metrics import evaluate_scores
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase,SharedDLGFull,SharedDOMINANT
from torch_geometric.data import Data

def metrics(y,s,ms):
 v=evaluate_scores(y,s,np.arange(len(y)),ms['val_mask'],ms['test_mask']);op=v['operating_point'];valf=op.get('validation_confusion',{}).get('f1');prev=op['validation_positives']/op['validation_total']
 if valf is not None:assert valf+1e-12>=2*prev/(1+prev)
 out={'pr_auc':v['ap'],'roc_auc':v['roc_auc'],'test_f1':v['thresholded']['f1'] if v['thresholded'] else None,'max_validation_f1':valf,'threshold':op.get('threshold'),'test_N':v['test_total'],'test_positive':v['test_positives'],'test_prevalence':v['test_prevalence']}
 for q,b in v['alert_budgets'].items():
  key='1' if q=='0.01' else '5';out.update({f'P_at_{key}pct':b['precision'],f'R_at_{key}pct':b['recall'],f'lift_{key}pct':b['precision']/v['test_prevalence'],f'K_{key}pct':b['k']})
 return out,v
class Control(SharedDLGFull):
 def __init__(self,variant,seed,**kw):super().__init__(**kw);self.variant=variant;self.row_seed=seed
 def _pretrain_level1(self,data):
  z=super()._pretrain_level1(data)
  if self.variant=='DLG-Aug-Zero':return torch.zeros_like(z)
  if self.variant=='DLG-Aug-Permuted':return z[torch.randperm(z.shape[0],generator=torch.Generator().manual_seed(self.row_seed))]
  return z

def run_gnn(dataset,variant,seed,epoch=50,kind='A24_CANONICAL_CONTROL'):
 run_id=f'{kind}__{dataset}__{variant}__seed{seed}';folder=LOCAL/'runs'/run_id
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 folder.mkdir(parents=True,exist_ok=True);data,masks,identity=load_input(dataset);dump(folder/'input_identity.json',identity)
 y=data.y.numpy().copy();d=Data(x=data.x.clone(),edge_index=data.edge_index.clone(),num_nodes=data.num_nodes)
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(4)
 common=dict(gpu=0,hid_dim=64,num_layers=4,lr=.004,dropout=0.,weight_decay=0.,batch_size=0,verbose=0,epoch=epoch)
 det=SharedDOMINANT(**common) if variant=='DOMINANT' else SharedDLGBase(**common) if variant in ['DLG-Base','DLG-Base-70'] else Control(variant,seed,l1_epochs=20,l1_hid_dim=64,l1_hops=2,**common)
 epoch_times=[];started=[None];forward=det._forward_components
 def traced(*a,**kw):torch.cuda.synchronize();started[0]=time.perf_counter();return forward(*a,**kw)
 det._forward_components=traced;step=torch.optim.Adam.step
 def timed_step(opt,*a,**kw):
  out=step(opt,*a,**kw)
  if started[0] is not None:torch.cuda.synchronize();epoch_times.append(time.perf_counter()-started[0]);started[0]=None
  return out
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();torch.optim.Adam.step=timed_step
 try:det.fit(d)
 finally:torch.optim.Adam.step=step
 torch.cuda.synchronize();fit=time.perf_counter()-t
 torch.cuda.synchronize();t=time.perf_counter();s=det.decision_function(d);torch.cuda.synchronize();score_seconds=time.perf_counter()-t;t=time.perf_counter();s=s.detach().cpu().numpy();transfer_seconds=time.perf_counter()-t
 assert len(epoch_times)==epoch
 summary,full=metrics(y,s,masks[seed]);np.savez_compressed(folder/'scores.npz',scores=s,labels=y,node_ids=np.arange(len(y)),**masks[seed]);dump(folder/'metrics.json',full)
 rec={'campaign':kind,'run_id':run_id,'dataset':dataset,'model':variant,'seed':seed,'status':'success','global_epochs':epoch,'local_epochs':20 if isinstance(det,Control) else 0,'parameter_count':sum(p.numel() for p in det.model.parameters()),'fit_seconds':fit,'score_seconds':score_seconds,'score_cpu_transfer_seconds':transfer_seconds,'mean_global_epoch_seconds':float(np.mean(epoch_times)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'process_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'raw_score_sha256':sha(folder/'scores.npz'),'input_identity_sha256':sha(folder/'input_identity.json'),'split_sha256':hashlib.sha256(json.dumps(identity['split_hashes'][str(seed)],sort_keys=True).encode()).hexdigest(),**summary}
 dump(folder/'epoch_times.json',epoch_times);dump(folder/'result.json',rec);print(run_id,'SUCCESS',round(fit,2),'s AP',summary['pr_auc'],flush=True)
 del det,d,data;gc.collect();torch.cuda.empty_cache();return rec

def controls():
 rs=[]
 for s in range(42,47):
  for v in ['DLG-Base','DLG-Aug','DLG-Aug-Permuted','DLG-Aug-Zero','DLG-Base-70']:
   rs.append(run_gnn('Elliptic',v,s,70 if v=='DLG-Base-70' else 50));writecsv(SUP/'elliptic_ablation_25_runs.csv',rs)
 paired=[]
 for v in ['DLG-Aug','DLG-Aug-Permuted','DLG-Aug-Zero','DLG-Base-70']:
  delta=np.array([next(r['pr_auc'] for r in rs if r['seed']==s and r['model']==v)-next(r['pr_auc'] for r in rs if r['seed']==s and r['model']=='DLG-Base') for s in range(42,47)])
  paired.append({'variant':v,'reference':'DLG-Base','n_paired_seeds':5,'paired_seeds':'42;43;44;45;46','delta_AP_mean':float(delta.mean()),'delta_AP_sample_SD':float(delta.std(ddof=1)),'delta_AP_SE':float(delta.std(ddof=1)/np.sqrt(5)),'comparison_campaign':'A24_CANONICAL_CONTROL'})
 writecsv(SUP/'elliptic_paired_deltas.csv',paired)

def features():
 from sklearn.ensemble import IsolationForest
 from pyod.models.ecod import ECOD
 rs=[];manifest=[]
 for dataset in ['Elliptic','DGraphFin','Ethereum','BSC','Polygon','LANL-RedTeam']:
  data,masks,identity=load_input(dataset);x=data.x.numpy();y=data.y.numpy();assert np.isfinite(x).all();dump(LOCAL/(dataset+'_feature_input.json'),identity)
  cached=None
  for model in ['IForest','ECOD']:
   for seed in range(42,47):
    rid=f'A24_FEATURE_ONLY__{dataset}__{model}__seed{seed}';folder=LOCAL/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():r=json.loads((folder/'result.json').read_text());rs.append(r);manifest.append({'run_id':rid,'raw_score_sha256':r['raw_score_sha256'],'feature_hash':identity['feature_hash'],'split_hash':r['split_hash']});continue
    fit=0.;scoretime=0.;reused=model=='ECOD' and cached is not None
    if reused:s=cached
    else:
     det=IsolationForest(n_estimators=100,max_samples=256,max_features=1.,bootstrap=False,contamination='auto',random_state=seed,n_jobs=4) if model=='IForest' else ECOD(contamination=.1,n_jobs=1)
     t=time.perf_counter();det.fit(x);fit=time.perf_counter()-t;t=time.perf_counter();s=-det.score_samples(x) if model=='IForest' else np.asarray(det.decision_scores_);scoretime=time.perf_counter()-t
     if model=='ECOD':cached=s
    summary,full=metrics(y,s,masks[seed]);np.savez_compressed(folder/'scores.npz',scores=s,labels=y,node_ids=np.arange(len(y)),**masks[seed]);dump(folder/'metrics.json',full)
    r={'campaign':'A24_SUPPLEMENTARY_FEATURE_ONLY','run_id':rid,'dataset':dataset,'model':model,'seed':seed,'status':'success','fit_seconds':fit,'score_seconds':scoretime,'process_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'deterministic_fit_reused':reused,'feature_hash':identity['feature_hash'],'split_hash':hashlib.sha256(json.dumps(identity['split_hashes'][str(seed)],sort_keys=True).encode()).hexdigest(),'raw_score_sha256':sha(folder/'scores.npz'),**summary}
    dump(folder/'result.json',r);rs.append(r);manifest.append({'run_id':rid,'raw_score_sha256':r['raw_score_sha256'],'feature_hash':identity['feature_hash'],'split_hash':r['split_hash']});writecsv(SUP/'feature_only_per_seed.csv',rs);writecsv(SUP/'feature_only_raw_scores_manifest.csv',manifest);print(rid,'SUCCESS AP',r['pr_auc'],flush=True)
  del data,x,y,cached;gc.collect()

def resources():
 rs=[]
 for dataset,model in [('Cora-Syn','DOMINANT'),('DGraphFin','DOMINANT'),('Ethereum','DLG-Aug'),('Reddit-Syn','DLG-Base')]:
  # One-epoch pilot is a separately labelled measurement, never a production score replacement.
  try:r=run_gnn(dataset,model,42,1,'A24_RESOURCE_ONE_EPOCH_PILOT');r['projection_50_epochs_seconds']=r['mean_global_epoch_seconds']*50;r['status']='MEASURED_PILOT_NOT_PRIMARY';rs.append(r)
  except Exception as e:
   rs.append({'campaign':'A24_RESOURCE_ONE_EPOCH_PILOT','dataset':dataset,'model':model,'status':'FAILED_RUNTIME','error':repr(e)});dump(LOCAL/(dataset+'_resource_failure.json'),rs[-1]);print('RESOURCE',dataset,repr(e),flush=True)
  keys=list(dict.fromkeys(k for r in rs for k in r));writecsv(SUP/'measured_resource_cases.csv',[{k:r.get(k,'') for k in keys} for r in rs])
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['controls','features','resources']);args=ap.parse_args();{'controls':controls,'features':features,'resources':resources}[args.stage]()
