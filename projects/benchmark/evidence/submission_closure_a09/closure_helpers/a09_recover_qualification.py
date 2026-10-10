#!/usr/bin/env python3
"""Recover missing error observations from identical A08 tiny fixtures.
No production source, tolerance, training run or original qualification is edited.
"""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3];E=ROOT/'projects/benchmark/evidence/a08_data_repair';O=ROOT/'projects/benchmark/evidence/submission_closure_a09';L=ROOT/'local_storage/benchmark/a09_submission_closure'
def load(name):
 p=ROOT/'projects/benchmark/scripts'/f'a08_{name}.py';s=importlib.util.spec_from_file_location('frozen_'+name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m,p
import torch
# Definition has an absolute floor matching the frozen atol/rtol transition.
# allclose uses the candidate as reference, exactly as the original compare.
measurements=[]
def observe(a,b,atol=1e-5,rtol=1e-4):
 delta=(a-b).detach().abs();reference=b.detach().abs();floor=atol/rtol
 return {'max_abs':float(delta.max()),'max_relative_with_floor':float((delta/reference.clamp_min(floor)).max()),'relative_floor':floor,'max_tolerance_fraction':float((delta/(atol+rtol*reference)).max()),'atol':atol,'rtol':rtol,'pass':bool(torch.allclose(a,b,atol=atol,rtol=rtol))}
def main():
 original={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'audit/cuda_exactness.json',E/'audit/gadnr_numerical_qualification.json']}
 c,cp=load('cuda_qualification');base=c.compare
 def compare(a,b):
  passed=base(a,b);obs=observe(a,b);measurements.append({'fixture':'CUDA','shape':list(a.shape),**obs})
  # The original row-block fixture compared its 17-vector scores but omitted
  # the scalar mean-loss observation. Recover that mean from the same tensors.
  if list(a.shape)==[17]:measurements.append({'fixture':'AnomalyDAE mean loss recovered from same score vectors','shape':[],**observe(a.mean(),b.mean())})
  return {**passed,**obs}
 c.compare=compare;c.OUT=L/'qualification_recovery/cuda_fixture.json';c.main()
 g,gp=load('gadnr_qualification');baseg=g.compare
 def gadnr_compare(a,b):
  value=baseg(a,b);measurements.append({'fixture':'GADNR','shape':list(a.shape),**observe(a,b)});return value
 g.compare=gadnr_compare;g.E=L/'qualification_recovery';g.L=ROOT/'local_storage/benchmark/a08_data_repair';g.E.joinpath('audit').mkdir(parents=True,exist_ok=True);g.main()
 # Recover arithmetic observations using the same original seeded fixtures.
 tp=ROOT/'tests/benchmark/a08/test_gadnr_numerics.py';s=importlib.util.spec_from_file_location('frozen_scalar_tests',tp);test=importlib.util.module_from_spec(s);s.loader.exec_module(test)
 scalar=[]
 for valid in [1,2]:
  for f in [8,64]:
   for scale in [.1,1.,100.]:
    generator=torch.Generator().manual_seed(42)
    p=torch.randn(1,2,f,generator=generator,dtype=torch.float64)*scale
    t=torch.randn(1,2,f,generator=generator,dtype=torch.float64)*scale
    reference=test.dense_reference(p,t,valid)
    actual=test.stable_kl_neighbor_loss(p,t,valid,'cpu')
    scalar.append({'hidden':f,'valid_samples':valid,'scale':scale,**observe(reference,actual,1e-7,1e-8)})
 assert all(x['pass'] for x in measurements+scalar)
 report={'status':'PASS','purpose':'original reports lack relative tensor observations and row-block mean-loss measurement; same tiny fixtures recover only those missing observations','production_training_runs':0,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,gp,tp,Path(__file__)]},'relative_definition':'max |a-b| / max(|b|, atol/rtol); b is allclose reference; near-zero floor=0.1 at frozen 1e-5/1e-4; tolerance_fraction=max |a-b|/(atol+rtol*|b|)','measurements':measurements,'scalar_recovery':scalar,'scope':'same implemented detached scalar term; sampled model forward/loss/scores/gradients/one Adam update, not full training trajectory'}
 (O/'qualification_recovery.json').write_text(json.dumps(report,indent=2)+'\n')
 assert original=={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in original}
 print('A09 missing qualification observations recovered without modifying A08')
if __name__=='__main__':main()
