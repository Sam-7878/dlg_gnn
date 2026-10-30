from a24_common import *
import sys,numpy as np,torch
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(ROOT/'src'))
def tensor_hash(a):
 a=np.asarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def load_input(name):
 from torch_geometric.data import Data
 old={r['dataset_id']:r for r in json.loads((ROOT/'projects/benchmark/evidence/astra_revision/dataset_manifest_canonical.json').read_text())}[name]
 if name in ['Ethereum','BSC','Polygon']:
  p=ROOT/'local_storage/benchmark/a08_data_repair/frozen'/name.lower()/'contract_graph.npz';a=np.load(p,allow_pickle=False)
  data=Data(x=torch.from_numpy(a['x'].copy()),edge_index=torch.from_numpy(a['edge_index'].copy()),y=torch.from_numpy(a['labels'].copy()))
  masks={s:{k:a[k+'_'+str(s)].copy() for k in ['train_mask','val_mask','test_mask']} for s in range(42,47)}
  im=p.parent/'input_manifest.json';m=json.loads(im.read_text());assert sha(p)==m['serialized_file_sha256']
  identity={'source_path':str(p.relative_to(ROOT)),'source_sha256':sha(p),'input_manifest_sha256':sha(im),'source_kind':'A08_SCORE_BOUND_FROZEN','old_A23_graph_metadata_mismatch':True,'relation':'exact standardized feature Euclidean k=5 symmetrized binary union; no transfer-edge union'}
 else:
  p=located(old['constructed_artifact_path']);o=torch.load(p,map_location='cpu',weights_only=False)
  data=Data(**{k:v for k,v in o.items() if k in ['x','edge_index','y','num_nodes','train_mask','val_mask','test_mask','eval_mask']}) if isinstance(o,dict) else o
  identity={'source_path':str(p.relative_to(ROOT)),'source_sha256':sha(p),'source_kind':'A04_FROZEN_CONSTRUCTED_TENSOR'}
  for k,v in [('feature_hash',data.x),('edge_hash',data.edge_index),('label_hash',data.y.view(-1))]:assert tensor_hash(v.cpu().numpy())==old[k],(name,k)
  n=data.num_nodes;y=data.y.numpy().reshape(-1);masks={}
  for s in range(42,47):
   if name=='DGraphFin':ms={k:getattr(data,k).cpu().numpy().copy() for k in ['train_mask','val_mask','test_mask']}
   elif name=='LANL-RedTeam':
    rng=np.random.RandomState(s);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng.shuffle(pos);rng.shuffle(neg);a,b=int(.2*len(pos)),int(.2*len(neg));ms={}
    for k,ind in [('val_mask',np.r_[pos[:a],neg[:b]]),('test_mask',np.r_[pos[a:2*a],neg[b:2*b]]),('train_mask',np.r_[pos[2*a:],neg[2*b:]])]:
     z=np.zeros(n,dtype=bool);z[ind]=True;ms[k]=z
   else:
    va,te=train_test_split(np.arange(n),test_size=.5,random_state=s,stratify=y);ms={'train_mask':np.ones(n,dtype=bool),'val_mask':np.zeros(n,dtype=bool),'test_mask':np.zeros(n,dtype=bool)};ms['val_mask'][va]=True;ms['test_mask'][te]=True
   for k,a in ms.items():
    val=old[k+'_hash'];expected=json.loads(val)[str(s)] if val.startswith('{') else val
    assert tensor_hash(a)==expected,(name,k,s)
   masks[s]=ms
 identity.update(dataset=name,N=int(data.num_nodes),E=int(data.edge_index.shape[1]),F=int(data.x.shape[1]),feature_hash=tensor_hash(data.x.numpy()),edge_hash=tensor_hash(data.edge_index.numpy()),label_hash=tensor_hash(data.y.numpy()),split_hashes={str(s):{k:tensor_hash(a) for k,a in m.items()} for s,m in masks.items()})
 return data,masks,identity
