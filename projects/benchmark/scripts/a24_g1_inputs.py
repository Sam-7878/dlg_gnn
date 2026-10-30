from a24_common import *
from a24_inputs import load_input,tensor_hash
import sys,numpy as np
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.crypto_raw import build_observables,deterministic_knn,array_hash,save_arrays
cfg=json.loads((ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text());records=rows(REG);identities=[];bindings=[]
for name in ['Elliptic','DGraphFin','Ethereum','BSC','Polygon','LANL-RedTeam','Cora-Syn']:
 data,masks,ident=load_input(name);identities.append(ident);print(name,'identity verified',ident['N'],ident['E'],flush=True)
 if name in ['Ethereum','BSC','Polygon']:
  c=name.lower();dest=LOCAL/'raw_rebuild'/c;dest.mkdir(parents=True,exist_ok=True);p=Path(cfg['raw_root'])/'transactions'/(c+'.zip')
  a,members=build_observables(p,c,lambda i,n:print('RAW',c,i,n,flush=True) if i%3000==0 else None);edges,rel=deterministic_knn(a['x'],a['node_ids'],k=5);a['edge_index']=edges
  assert np.array_equal(a['x'],data.x.numpy()) and np.array_equal(edges,data.edge_index.numpy()),name
  save_arrays(dest/'observables.npz',a)
  matches=[r for r in records if r['dataset']==name];assert len(matches)==35
  for r in matches:
   folder=ROOT/'local_storage/benchmark/a08_data_repair/runs'/r['run_id'];scores=folder/'scores.npz';metric=folder/'metrics.json';manifest=folder/'run_manifest.json'
   assert sha(scores)==r['raw_score_hash'] and sha(metric)==r['metric_json_hash'];rm=json.loads(manifest.read_text());assert rm['input_manifest_hash']==ident['input_manifest_sha256']
   with np.load(scores,allow_pickle=False) as z:
    assert np.array_equal(z['labels'],data.y.numpy())
    for k in ['val_mask','test_mask']:assert np.array_equal(z[k],masks[int(r['seed'])][k])
   bindings.append({'run_id':r['run_id'],'dataset':name,'seed':r['seed'],'scores_hash':sha(scores),'metrics_hash':sha(metric),'input_manifest_hash':ident['input_manifest_sha256'],'status':'MATCHED_ACTUAL_A08_INPUT'})
  ident.update(raw_rebuild_matches_frozen=True,raw_zip_sha256=sha(p),raw_members_count=len(members),reconstruction_command='python projects/benchmark/scripts/a24_g1_inputs.py',relation_audit=rel)
 dump(PUB/'a24_repo_manuscript_consistency.json',{'status':'IN_PROGRESS','datasets':identities,'bindings':bindings})
 del data
writecsv(PUB/'score_input_binding_105.csv',bindings)
old=rows(ROOT/'projects/benchmark/publication/a23/dataset_graph_identity_canonical.csv');byname={r['dataset']:r for r in identities}
for r in old:
 n=r['dataset_id']
 if n in ['Ethereum','BSC','Polygon']:
  i=byname[n]
  for k,v in [('model_input_N',i['N']),('constructed_graph_N',i['N']),('model_input_E',i['E']),('constructed_graph_E',i['E']),('feature_hash',i['feature_hash']),('edge_hash',i['edge_hash']),('label_hash',i['label_hash'])]:r[k]=v
  r['node_count_semantics']='A08 score-bound full raw contract population;exact binary Euclidean k=5 feature similarity;no transfer-edge union'
writecsv(PUB/'dataset_graph_identity_a24.csv',old)
dump(PUB/'a24_repo_manuscript_consistency.json',{'status':'PASS_INPUT_RECONSTRUCTION_AND_SCORE_BINDING','datasets':identities,'bindings_count':len(bindings),'finding':'A23 graph table accidentally reused pre-A08 hybrid metadata. A24 corrects descriptions and N/E/hash,not frozen455 metrics.','new_primary_runs':0})
print('G1 input proof PASS;105 actual score bindings;three independent raw reconstructions',flush=True)

subprocess.run([sys.executable,str(ROOT/'projects/benchmark/scripts/a24_metadata.py')],check=True,cwd=ROOT)
