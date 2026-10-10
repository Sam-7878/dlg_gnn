#!/usr/bin/env python3
"""Actual RTX3090 float32 production primitives; no predictive metrics."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.nn import GCN
from pygod.nn import AnomalyDAEBase
from pygod.nn.functional import double_recon_loss
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_squared_error,exact_double_reconstruction_score
from gog_fraud.models.pygod.sparse_message import SparseFusedGCN,normalized_sparse_adjt
from gog_fraud.models.pygod.shared_reconstruction import ExactAnomalyDAEBase
from gog_fraud.data.crypto_raw import sha256
OUT=ROOT/'projects/benchmark/evidence/a08_data_repair/audit/cuda_exactness.json'
ATOL,RTOL=1e-5,1e-4

def compare(a,b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('nonfinite qualification tensor')
    passed=torch.allclose(a,b,atol=ATOL,rtol=RTOL)
    if not passed:raise AssertionError(f'elementwise tolerance failed: max_abs={(a-b).abs().max().item()}')
    return {'max_abs':(a-b).abs().max().item(),'atol':ATOL,'rtol':RTOL,'pass':True}


def main():
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX3090 required')
    torch.set_num_threads(2);device=torch.device('cuda:0');results=[]
    for case in ['binary undirected','weighted duplicates self_loops']:
        torch.manual_seed(20261008)
        edge=torch.tensor([[0,1,2,3,4,5,6,7,2,2,5,5],[1,2,3,4,5,6,7,0,2,3,5,5]],device=device)
        weights=torch.tensor([.2,.4,.7,1.,1.3,.8,.5,1.1,.6,.3,.2,.4],device=device)
        if case=='binary undirected':edge=edge[:,:8];edge=torch.cat([edge,edge.flip(0)],1);weights=torch.ones(edge.shape[1],device=device)
        n=9;target=torch.sparse_coo_tensor(edge,weights,(n,n)).coalesce().to_dense()
        a=torch.nn.Parameter(torch.randn(n,5,device=device));b=torch.nn.Parameter(a.detach().clone())
        oa=torch.optim.Adam([a],lr=.01);ob=torch.optim.Adam([b],lr=.01)
        dense=(target-a@a.T).square().sum(1);sparse=exact_dot_product_row_squared_error(b,edge,edge_weight=weights)
        la,lb=dense.mean(),sparse.mean();la.backward();lb.backward();checks={'score':compare(dense,sparse),'loss':compare(la,lb),'gradient':compare(a.grad,b.grad)}
        oa.step();ob.step();checks['update']=compare(a,b);results.append({'path':'Gram','case':case,'checks':checks})
        for layers in [2,4]:
            torch.manual_seed(20261008)
            ref=GCN(5,7,num_layers=layers,out_channels=5,dropout=0,normalize=True,add_self_loops=True).to(device)
            candidate=SparseFusedGCN(5,7,num_layers=layers,out_channels=5,dropout=0,normalize=True,add_self_loops=True).to(device)
            candidate.load_state_dict(ref.state_dict())
            x=torch.randn(n,5,device=device,requires_grad=True);y=x.detach().clone().requires_grad_()
            oa=torch.optim.Adam(ref.parameters(),lr=.01);ob=torch.optim.Adam(candidate.parameters(),lr=.01)
            aa=ref(x,edge,edge_weight=weights);bb=candidate(y,normalized_sparse_adjt(edge,n,edge_weight=weights,dtype=x.dtype,device=device))
            aa.square().mean().backward();bb.square().mean().backward()
            checks={'forward':compare(aa,bb),'input_grad':compare(x.grad,y.grad)}
            checks['parameter_gradients']=[compare(a.grad,b.grad) for a,b in zip(ref.parameters(),candidate.parameters())]
            oa.step();ob.step();checks['update']=[compare(a,b) for a,b in zip(ref.parameters(),candidate.parameters())]
            results.append({'path':'fused_GCN','case':case,'layers':layers,'dropout':0,'checks':checks})
    # Dense AnomalyDAE vs exact row blocks, accumulated before one Adam update.
    torch.manual_seed(20261008);n,f=17,8;x=torch.randn(n,f,device=device)
    edge=torch.tensor([[i for i in range(n)]+[(i+1)%n for i in range(n)],[(i+1)%n for i in range(n)]+[i for i in range(n)]],device=device)
    dense=AnomalyDAEBase(in_dim=f,num_nodes=n,emb_dim=64,hid_dim=64,dropout=0).to(device)
    exact=ExactAnomalyDAEBase(copy.deepcopy(dense)).to(device)
    target=torch.sparse_coo_tensor(edge,torch.ones(edge.shape[1],device=device),(n,n)).coalesce().to_dense()
    dx,ds=dense(x,edge,n);ex,ez=exact(x,edge)
    reference=double_recon_loss(x,dx,target,ds,.5,.5,.5);reference.mean().backward()
    blocks=[]
    for start in range(0,n,4):
        rows=torch.arange(start,min(start+4,n),device=device)
        score=exact_double_reconstruction_score(x,ex,ez,edge,weight=.5,positive_weight_attribute=.5,positive_weight_structure=.5,sigmoid_structure=True,rows=rows,backend='chunked_exact',chunk_size=4)
        blocks.append(score.detach());(score.sum()/n).backward(retain_graph=start+4<n)
    checks={'forward':compare(dx,ex),'score':compare(reference,torch.cat(blocks))}
    checks['gradient']=[compare(a.grad,b.grad) for a,b in zip(dense.parameters(),exact.parameters())]
    oa=torch.optim.Adam(dense.parameters(),lr=.004);ob=torch.optim.Adam(exact.parameters(),lr=.004);oa.step();ob.step()
    checks['update']=[compare(a,b) for a,b in zip(dense.parameters(),exact.parameters())]
    results.append({'path':'AnomalyDAE_row_block','case':'binary undirected','n':n,'f':f,'chunk_size':4,'dropout':0,'one_step_after_all_blocks':True,'checks':checks})
    report={'status':'PASS','dtype':'float32','device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'seed':20261008,
            'atol':ATOL,'rtol':RTOL,'scope':'small CUDA forward/loss/score/gradient/one-update; primary defaults dropout=0; not full trajectory equivalence',
            'results':results,'source_hashes':{p:sha256(ROOT/p) for p in ['projects/benchmark/scripts/a08_cuda_qualification.py','src/gog_fraud/models/pygod/exact_reconstruction.py','src/gog_fraud/models/pygod/sparse_message.py','src/gog_fraud/models/pygod/shared_reconstruction.py']}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2)+'\n');print('CUDA exactness PASS',len(results),flush=True)
if __name__=='__main__':main()
