#!/usr/bin/env python3
"""Small CPU qualification of production Gram and fused GCN APIs, not a training campaign."""
import copy, csv, hashlib, json, sys
from pathlib import Path
import torch
from torch_geometric.nn import GCN
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_squared_error
from gog_fraud.models.pygod.sparse_message import SparseFusedGCN,normalized_sparse_adjt
OUT=ROOT/'projects/benchmark/reports/astra_revision'
torch.set_num_threads(1)


def err(a,b):
    a=a.detach();b=b.detach();delta=(a-b).abs()
    return {'max_abs':float(delta.max()),'max_rel':float((delta/(a.abs()+1e-12)).max())}


def main():
    results=[]
    for dtype in (torch.float64,torch.float32):
        atol,rtol=(1e-10,1e-8) if dtype==torch.float64 else (1e-5,1e-4)
        for case in ('binary directed','weighted duplicates and self-loops'):
            torch.manual_seed(20261008)
            edge=torch.tensor([[0,1,2,3,4,5,6,7,2,2,5,5],[1,2,3,4,5,6,7,0,2,3,5,5]])
            weights=torch.tensor([.2,.4,.7,1.,1.3,.8,.5,1.1,.6,.3,.2,.4],dtype=dtype)
            if case=='binary directed':
                edge=edge[:,:8];weights=torch.ones(8,dtype=dtype)
            n=9;d=5
            target=torch.sparse_coo_tensor(edge,weights,(n,n)).coalesce().to_dense()
            a=torch.nn.Parameter(torch.randn(n,d,dtype=dtype));b=torch.nn.Parameter(a.detach().clone())
            oa=torch.optim.Adam([a],lr=.01);ob=torch.optim.Adam([b],lr=.01)
            dense=(target-a@a.T).square().sum(1)
            sparse=exact_dot_product_row_squared_error(b,edge,edge_weight=weights)
            loss_a=dense.mean();loss_b=sparse.mean();loss_a.backward();loss_b.backward()
            checks={'loss':err(loss_a,loss_b),'node_score':err(dense,sparse),'gradient':err(a.grad,b.grad)}
            for aa,bb in ((dense,sparse),(loss_a,loss_b),(a.grad,b.grad)):
                if not torch.allclose(aa,bb,atol=atol,rtol=rtol):raise RuntimeError(('Gram',case,dtype))
            oa.step();ob.step();checks['Adam_update']=err(a,b)
            if not torch.allclose(a,b,atol=atol,rtol=rtol):raise RuntimeError('Gram Adam')
            results.append(dict(path='Gram',case=case,dtype=str(dtype),n=n,d=d,seed=20261008,atol=atol,rtol=rtol,checks=checks,status='PASS'))
            for layers in (2,4):
                torch.manual_seed(20261008)
                ref=GCN(5,7,num_layers=layers,out_channels=5,dropout=0.,normalize=True,add_self_loops=True).to(dtype=dtype)
                fused=SparseFusedGCN(5,7,num_layers=layers,out_channels=5,dropout=0.,normalize=True,add_self_loops=True).to(dtype=dtype)
                fused.load_state_dict(ref.state_dict())
                x=torch.randn(n,5,dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_(True)
                oa=torch.optim.Adam(ref.parameters(),lr=.01);ob=torch.optim.Adam(fused.parameters(),lr=.01)
                forward=ref(x,edge,edge_weight=weights)
                operator=normalized_sparse_adjt(edge,n,edge_weight=weights,dtype=dtype,device=torch.device('cpu'))
                candidate=fused(y,operator)
                forward.square().mean().backward();candidate.square().mean().backward()
                checks={'forward':err(forward,candidate),'input_gradient':err(x.grad,y.grad)}
                grads=[(a.grad,b.grad) for a,b in zip(ref.parameters(),fused.parameters())]
                checks['parameter_gradient']={k:max(err(a,b)[k] for a,b in grads) for k in ('max_abs','max_rel')}
                for a,b in [(forward,candidate),(x.grad,y.grad)]+grads:
                    if not torch.allclose(a,b,atol=atol,rtol=rtol):raise RuntimeError(('GCN',case,layers,dtype))
                oa.step();ob.step();params=list(zip(ref.parameters(),fused.parameters()))
                checks['Adam_update']={k:max(err(a,b)[k] for a,b in params) for k in ('max_abs','max_rel')}
                for a,b in params:
                    if not torch.allclose(a,b,atol=atol,rtol=rtol):raise RuntimeError(('GCN Adam',dtype))
                results.append(dict(path='GCN',case=case,dtype=str(dtype),n=n,layers=layers,dropout=0,seed=20261008,atol=atol,rtol=rtol,checks=checks,status='PASS'))
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['projects/benchmark/scripts/astra_exact_cpu.py','src/gog_fraud/models/pygod/exact_reconstruction.py','src/gog_fraud/models/pygod/sparse_message.py']}
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',scope='post-review numerical qualification; CPU; no benchmark metrics replaced',torch_version=torch.__version__,source_hashes=sources,results=results)
    (OUT/'exact_cpu_qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=[dict(path=r['path'],case=r['case'],dtype=r['dtype'],layers=r.get('layers','NA'),measure=m,**v,atol=r['atol'],rtol=r['rtol'],status=r['status']) for r in results for m,v in r['checks'].items()]
    with (OUT/'exact_cpu_qualification.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({'status':'PASS','cases':len(results),'max_adam_absolute_error':max(r['checks']['Adam_update']['max_abs'] for r in results)},indent=2))
if __name__=='__main__':main()
