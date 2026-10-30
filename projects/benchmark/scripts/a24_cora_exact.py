from a24_common import *
from a24_inputs import load_input
import importlib.util,torch,numpy as np,copy
spec=importlib.util.spec_from_file_location('a22_fixture',ROOT/'projects/benchmark/scripts/a22_g1_dense_exact_training_equivalence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data,_,identity=load_input('Cora-Syn');m.N=data.num_nodes;m.D=data.x.shape[1];m.HID=64;m.L1DIM=64;m.make_graph=lambda seed:(data.clone(),{'N':data.num_nodes,'E':data.edge_index.shape[1],'F':data.x.shape[1],'input_hash':identity['source_sha256']})
torch.set_num_threads(4);rs=[]
for family in ['DOMINANT','DLG-Base','DLG-Aug']:
 torch.manual_seed(42);d=m.aug_data(data.clone(),42) if family=='DLG-Aug' else data.clone();ref=m.make_reference(family,42);model=m.init_ref_model(ref,d);state=copy.deepcopy(model.state_dict());model.train();opt=torch.optim.Adam(model.parameters(),lr=m.LR,weight_decay=0.);loss,score=ref.forward_model(m.sampled(d));opt.zero_grad();loss.backward();gr={n:p.grad.clone() for n,p in model.named_parameters()};opt.step();after={n:p.detach().clone() for n,p in model.named_parameters()}
 prod=m.make_production(family);pd=d.clone()
 if family=='DLG-Aug':pd._dlg_full_augmented=True
 pm=m.init_prod_model(prod,pd,state);prod.fit(pd);checks={}
 for name,a,b in [('loss',torch.tensor(float(loss)),torch.tensor(prod.loss_history_[0])),('node_score',score.detach(),prod.decision_score_),*[(f'gradient/{n}',gr[n],p.grad) for n,p in pm.named_parameters()],*[(f'oneAdam/{n}',after[n],p.detach()) for n,p in pm.named_parameters()]]:
  diff=(a-b).abs();tol=1e-5+1e-4*a.abs();checks[name]={'max_abs':float(diff.max()),'max_tolerance_fraction':float((diff/tol).max()),'pass':bool(torch.all(diff<=tol))}
 rs.append({'family':family,'N':2708,'dtype':'float32','atol':1e-5,'rtol':1e-4,'checks':checks,'pass':all(v['pass'] for v in checks.values())});print(family,rs[-1]['pass'],flush=True)
dump(SUP/'cora_exact_loss_equivalence.json',{'status':'PASS' if all(r['pass'] for r in rs) else 'FAIL_NUMERICAL_EQUIVALENCE','source_reference':'installed PyGOD DOMINANT and repository dense DLG branches;same weights;full N2708Cora','input_identity':identity,'results':rs,'not_a_trajectory_claim':True})
