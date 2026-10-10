#!/usr/bin/env python3
"""Actual sampled GADNR forward/score/loss/gradient/Adam CPU and RTX3090 checks."""
import copy,hashlib,json,random,sys,xml.etree.ElementTree as ET
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.models.pygod.gadnr import GADNRBase
from gog_fraud.models.a08_gadnr_numerics import stable_kl_neighbor_loss
E=ROOT/'projects/benchmark/evidence/a08_data_repair';L=ROOT/'local_storage/benchmark/a08_data_repair'
ATOL,RTOL=1e-5,1e-4


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a,b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a,b,atol=ATOL,rtol=RTOL):
        raise AssertionError('GADNR numerical equivalence failed: '+str((a-b).abs().max().item()))
    return float((a-b).detach().abs().max())


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX3090 required')
    results=[]
    for device in ('cpu','cuda:0'):
        torch.manual_seed(20261010);random.seed(20261010)
        n=17;x=torch.randn(n,8,device=device)*.1
        edge=torch.tensor([[i for i in range(n)]+[(i+1)%n for i in range(n)],[(i+1)%n for i in range(n)]+[i for i in range(n)]],device=device)
        reference=GADNRBase(in_dim=8,hid_dim=64,encoder_layers=1,deg_dec_layers=4,fea_dec_layers=3,
                            sample_size=2,sample_time=3,full_batch=False,neigh_loss='KL',device=device,
                            lambda_loss1=.01,lambda_loss2=.1,lambda_loss3=.8).to(device)
        stable=copy.deepcopy(reference);stable.neighbor_loss=stable_kl_neighbor_loss
        optim1=torch.optim.Adam(reference.parameters(),lr=.01,weight_decay=.0003)
        optim2=torch.optim.Adam(stable.parameters(),lr=.01,weight_decay=.0003)
        ids=list(range(n));neighbors={i:[i,(i+1)%n,(i-1)%n] for i in ids};mapping={i:i for i in ids};degree=torch.full((n,),3,device=device)
        losses=[];outputs=[]
        for model in (reference,stable):
            torch.manual_seed(46);torch.cuda.manual_seed_all(46);random.seed(46)
            values=model(x,edge,ids,neighbors,mapping)
            result=model.loss_func(*values,degree);result[0].backward();losses.append(result);outputs.append(values)
        checks={'center_embedding':compare(outputs[0][0],outputs[1][0]),'degree_output':compare(outputs[0][1],outputs[1][1]),
                'loss_and_scores':[compare(a,b) for a,b in zip(losses[0],losses[1])]}
        checks['parameter_gradients']=[]
        for a,b in zip(reference.parameters(),stable.parameters()):
            if a.grad is None or b.grad is None:
                if (a.grad is None)!=(b.grad is None):raise AssertionError('gradient availability changed')
            else:checks['parameter_gradients'].append(compare(a.grad,b.grad))
        optim1.step();optim2.step()
        checks['one_Adam_update']=[]
        for (name,a),b in zip(reference.named_parameters(),stable.parameters()):
            try:checks['one_Adam_update'].append(compare(a,b))
            except AssertionError:
                print('UPDATE_MISMATCH',device,name,'gradient_max_delta',float((a.grad-b.grad).abs().max()) if a.grad is not None else None,flush=True)
                raise
        results.append({'device':device,'nodes':n,'features':8,'hidden':64,'sample_size':2,'sample_time':3,'status':'PASS','checks':checks})
        print('QUALIFIED',device,flush=True)
    suites=list(ET.parse(L/'gadnr_numerical_tests.xml').getroot().iter('testsuite'))
    if sum(int(s.attrib['tests']) for s in suites)!=16 or any(int(s.attrib[k]) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('16 required arithmetic/boundary fixtures incomplete')
    report={'status':'PASS','source_sha256':sha(ROOT/'src/gog_fraud/models/a08_gadnr_numerics.py'),
            'qualification_source_sha256':sha(Path(__file__)),'test_source_sha256':sha(ROOT/'tests/benchmark/a08/test_gadnr_numerics.py'),
            'junit_sha256':sha(L/'gadnr_numerical_tests.xml'),'arithmetic_tests':16,'actual_sampled_model_cases':results,
            'atol':ATOL,'rtol':RTOL,'scope':'same implemented detached KL term; CPU/RTX3090 sampled forward/loss/score/parameter gradient/one Adam update; no full trajectory equality claim'}
    (E/'audit/gadnr_numerical_qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ACTUAL_GADNR_CPU_RTX3090_EQUIVALENCE_PASS')


if __name__=='__main__':main()
