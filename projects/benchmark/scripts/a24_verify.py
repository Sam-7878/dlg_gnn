#!/usr/bin/env python3
"""Public A24 integrity and scalar replay. Does not imply raw-score reproduction."""
from a24_common import *
import math,statistics,argparse

def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def replay():
 grouped={}
 for r in rows(REG):
  require(r['status']=='success','selected status')
  for metric in ['pr_auc','roc_auc','f1']:
   grouped.setdefault((r['dataset'],r['model'],metric),[]).append(float(r[metric]))
 expected=rows(PUB/'metrics_mean_sample_sd.csv');require(len(expected)==len(grouped),'metric group count')
 for r in expected:
  a=grouped[(r['dataset'],r['model'],r['metric'])];require(len(a)==5,'five seeds')
  require(math.isclose(statistics.mean(a),float(r['mean']),rel_tol=1e-12,abs_tol=1e-14),'mean '+str(r))
  require(math.isclose(statistics.stdev(a),float(r['sample_SD']),rel_tol=1e-12,abs_tol=1e-14),'SD '+str(r))
 groups={}
 for file,study in [('feature_only_per_seed.csv','feature_only'),('elliptic_ablation_25_runs.csv','elliptic_controls'),('real_domain_alert_budget_per_seed.csv','real_domain_alert')]:
  for r in rows(SUP/file):
   for metric in ['pr_auc','roc_auc','test_f1','max_validation_f1','P_at_1pct','P_at_5pct','R_at_1pct','R_at_5pct','lift_1pct','lift_5pct']:groups.setdefault((study,r['dataset'],r['model'],metric),[]).append(float(r[metric]))
 for r in rows(SUP/'supplementary_summary.csv'):
  a=groups[(r['study'],r['dataset'],r['model'],r['metric'])];require(len(a)==5,'supplementary five seeds')
  require(math.isclose(statistics.mean(a),float(r['mean']),rel_tol=1e-12,abs_tol=1e-14),'supplementary mean')
  require(math.isclose(statistics.stdev(a),float(r['sample_SD']),rel_tol=1e-12,abs_tol=1e-14),'supplementary SD')
 return {'primary_metric_groups':len(grouped),'supplementary_metric_groups':len(groups),'ddof':1}
def verify():
 manifest=json.loads((PUB/'public_payload_manifest.json').read_text())
 for p,d in manifest['files'].items():require((ROOT/p).is_file() and sha(ROOT/p)==d,'public payload drift '+p)
 rr=rows(REG);primary=[r for r in rr if r['dataset']!='LANL-RedTeam'];require(len(rr)==455 and len(primary)==420,'455=420+35')
 require(len({(r['dataset'],r['model']) for r in primary})==84,'84 supported primary pairs')
 require(len({(r['dataset'],r['model'],r['seed']) for r in rr})==455,'run uniqueness')
 for s,n,k in zip(rows(PUB/'stats_master_a24.csv'),[8,13,6,7],[7,5,5,5]):
  require(int(s['n_datasets'])==n and int(s['k_models'])==k,'stats view membership')
  require(math.isclose(float(s['Q'])/(n*(k-1)),float(s['Kendall_W']),rel_tol=1e-12),'Kendall W')
  require(math.isclose((int(s['exceedances'])+1)/(int(s['permutations'])+1),float(s['MC_p']),rel_tol=1e-12),'MC add-one')
 require(len(rows(SUP/'feature_only_per_seed.csv'))==60,'feature 60');require(len(rows(SUP/'elliptic_ablation_25_runs.csv'))==25,'controls 25')
 require(len(rows(SUP/'real_domain_alert_budget_per_seed.csv'))==30,'alert 30');require(len(rows(SUP/'measured_full_resource_cases.csv'))==4,'full resources 4')
 require(json.loads((SUP/'cora_exact_loss_equivalence.json').read_text())['status']=='PASS','Cora exact')
 require(json.loads((SUP/'supplementary_acceptance.json').read_text())['independent_score_audit_runs']==113,'113 raw-score audits')
 return {'status':'PASS_PUBLIC_INTEGRITY_AND_SCALAR_REPLAY','frozen_runs':455,'primary_support':'84/91','payload_files':len(manifest['files']),**replay(),'limitation':'Raw third-party inputs/new per-node scores and A24 manuscript remain private; no neural retraining is asserted.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['verify','tables','paper'],default='verify');a=ap.parse_args();result=verify()
 if a.mode=='paper':
  writer=ROOT/'projects/benchmark/scripts/a24_build_manuscript.py'
  require(writer.is_file() and (LOCAL/'a23_snapshot').is_dir(),'A24 PRIVATE_INPUT_REQUIRED: unpublished manuscript writer/snapshot intentionally withheld; public --mode verify and --mode tables are available. No historical PDF fallback.')
  subprocess.run([__import__('sys').executable,str(writer)],check=True,cwd=ROOT)
  import os
  for v,stem in [('preprints','DLG-Benchmark_A24_Preprints'),('applsci','DLG-Benchmark_A24_AppliedSciences')]:subprocess.run(['latexmk','-g','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=PAPER/v,env=dict(os.environ,SOURCE_DATE_EPOCH='1793318400',FORCE_SOURCE_DATE='1'),stdout=subprocess.DEVNULL,check=True)
  result['paper']='two author-local A24 PDFs compiled; semantic and visual review separate'
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
