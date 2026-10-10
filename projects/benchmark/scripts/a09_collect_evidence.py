#!/usr/bin/env python3
"""A09 read-only scientific audit; new scalar companion, never training."""
import csv, hashlib, importlib.util, io, json, shutil, sys, zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.benchmark_metrics import evaluate_scores
E=ROOT/'projects/benchmark/evidence/a08_data_repair'; L=ROOT/'local_storage/benchmark/a08_data_repair'
O=ROOT/'projects/benchmark/evidence/submission_closure_a09'; P=ROOT/'projects/benchmark/reports/submission_closure_a09'; T=ROOT/'local_storage/benchmark/a09_submission_closure'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='selected_node_ids'}
 if isinstance(v,list):return [clean(x) for x in v]
 return v

def main():
 for p in [O,P,T]:p.mkdir(parents=True,exist_ok=True)
 protected=[p for p in E.rglob('*') if p.is_file()]+list((ROOT/'projects/benchmark/reports/a08_data_repair').glob('*'))
 protected += [ROOT/'projects/benchmark/paper/current'/n for n in ['DLG-Benchmark_A08.tex','DLG-Benchmark_A08.pdf','mdpi/DLG-Benchmark_A08_MDPI.tex','mdpi/DLG-Benchmark_A08_MDPI.pdf']]
 protected += [ROOT/'projects/benchmark/evidence'/n for n in ['public_numeric_evidence.zip','a08_public_numeric_evidence.zip','a08_public_release.json']]
 before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
 dump(T/'a08_preservation_before.json',before)
 registry=json.loads((E/'approved_registry.json').read_text());records=registry['records'];fresh=[r for r in records if r['evidence_tier']=='a08_raw_score_recomputable']
 assert len(fresh)==105 and all(r['status']=='success' for r in fresh)
 checks=[];flat=[]
 for r in fresh:
  folder=L/'runs'/r['run_id'];scores=folder/'scores.npz';metrics=folder/'metrics.json';run=json.loads((folder/'run_manifest.json').read_text());a=np.load(scores,allow_pickle=False)
  chain=r['dataset'].lower();g=np.load(L/'frozen'/chain/'contract_graph.npz',allow_pickle=False);seed=str(r['seed'])
  for key in ['node_ids','labels']:assert np.array_equal(a[key],g[key]),(r['run_id'],key)
  for key in ['val_mask','test_mask']:assert np.array_equal(a[key],g[key+'_'+seed])
  assert sha(scores)==r['raw_score_hash'] and sha(metrics)==r['metric_json_hash']
  original=json.loads(metrics.read_text());recomputed=evaluate_scores(a['labels'],a['scores'],a['node_ids'],a['val_mask'],a['test_mask'])
  assert recomputed==original,r['run_id']
  for k,m in [('pr_auc','ap'),('roc_auc','roc_auc')]:assert r[k]==recomputed[m]
  assert r['validation_f1']==recomputed['thresholded']['f1']
  safe={'run_id':r['run_id'],'dataset':r['dataset'],'model':r['model'],'seed':r['seed'],'original_private_metric_sha256':sha(metrics),'metrics':clean(original),'redaction':'selected_node_ids omitted; scalar budgets retained; original hash identifies private original'}
  dump(O/'scalar_metrics'/f"{r['run_id']}.json",safe)
  row={k:r[k] for k in ['dataset','model','seed','run_id']};row.update(ap=original['ap'],roc_auc=original['roc_auc'],test_total=original['test_total'],test_positives=original['test_positives'],test_negatives=original['test_total']-original['test_positives'],test_prevalence=original['test_prevalence'],threshold=original['operating_point']['threshold'],validation_total=original['operating_point']['validation_total'],validation_positives=original['operating_point']['validation_positives'])
  row.update(original['thresholded'])
  for q in ['0.01','0.05']:
   for k in ['k','precision','recall','score_boundary_ties']:row['alert_'+q+'_'+k]=original['alert_budgets'][q][k]
  flat.append(row);checks.append({'run_id':r['run_id'],'status':'EXACT_JSON_EQUAL','scores_sha256':sha(scores),'metrics_sha256':sha(metrics),'alignment':'node_ids/labels/val/test equal frozen arrays','input_manifest_hash':r['input_manifest_hash'],'split_hash':r['split_hash'],'model_config_hash':r['model_config_hash'],'environment_lock_hash':run.get('environment_lock_hash'),'execution_manifest_hash':run.get('execution_manifest_hash')})
 csvout(O/'scalar_metrics_105.csv',flat);dump(O/'cpu_metric_recheck.json',{'status':'PASS','runs':105,'exact_json_equal':105,'checks':checks,'score_policy':'original code: validation grouped distinct scores, largest threshold tie; test >=; ceil budgets; stable-ID alert ties','source_sha256':sha(ROOT/'src/gog_fraud/data/benchmark_metrics.py')})
 # No raw identifiers or arrays are copied.
 dump(O/'approved_registry.json',clean(registry))
 support=[];impact=[]
 inputs=json.loads((E/'input_manifest.json').read_text())['datasets']
 for chain in ['ethereum','bsc','polygon']:
  g=np.load(L/'frozen'/chain/'contract_graph.npz',allow_pickle=False);im=next(x for x in inputs if x['dataset_id'].startswith(chain+'_'));selected=[r for r in fresh if r['dataset'].lower()==chain]
  assert len(selected)==35
  inputfile=L/'frozen'/chain/'input_manifest.json';selection_hash=hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  impact.append({'dataset':chain.capitalize() if chain!='bsc' else 'BSC','dataset_version':im['dataset_id'],'input_manifest_sha256':sha(inputfile),'scientific_content_hash':im['scientific_content_hash'],'x_hash':im['arrays']['x'],'edge_hash':im['arrays']['edge_index'],'target_hash':im['arrays']['labels'],'split_selection_hash':hashlib.sha256(json.dumps({k:v for k,v in im['arrays'].items() if '_mask_' in k},sort_keys=True,separators=(',',':')).encode()).hexdigest(),'approved_run_count':35,'approved_selection_sha256':selection_hash,'run_ids':';'.join(r['run_id'] for r in selected),'status':'A08_REVISION2_COMPLETE','planning_snapshot':'projects/benchmark/reports/a08_data_repair/impact_map.csv'})
  for seed in range(42,47):
   y=g['labels'][g['test_mask_'+str(seed)]];pi=float(y.mean());support.append({'dataset':selected[0]['dataset'],'seed':seed,'test_N':len(y),'test_positive':int(y.sum()),'test_negative':int(len(y)-y.sum()),'pi_test_AP_reference':pi,'always_positive_F1':2*pi/(1+pi),'test_mask_hash':im['arrays']['test_mask_'+str(seed)],'target_hash':im['arrays']['labels'],'interpretation':'prevalence reference, not exact finite-sample random AP expectation'})
 csvout(O/'test_support_by_seed.csv',support);csvout(P/'impact_map_final.csv',impact);shutil.copy2(P/'impact_map_final.csv',O/'impact_map_final.csv')
 # Exact original RNG stream, unrounded means, all 18 Holm rows.
 spec=importlib.util.spec_from_file_location('a08_aggregate_readonly',ROOT/'projects/benchmark/scripts/a08_aggregate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.E=T/'numeric_replay'
 with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip') as z:
  prior=list(csv.DictReader(io.StringIO(z.read('evaluation/benchmark/v2/paper_ready_a05/statistics_s1_s5.csv').decode())))
  canonical=json.loads(z.read('evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/dataset_manifest_canonical.json'));dump(O/'historical_dataset_contracts.json',canonical)
 m.render_public_tables(records,prior);replays=[]
 for p in sorted(m.E.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(m.E);source=E/rel;assert source.is_file() and p.read_bytes()==source.read_bytes(),str(rel)
  replays.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p),'byte_identical_to_A08':True})
  dest=O/'replayed'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 dump(O/'statistics_replay_check.json',{'status':'PASS','source_sha256':sha(ROOT/'projects/benchmark/scripts/a08_aggregate.py'),'RNG':'numpy.default_rng(20261008), single stream S1-S4','B':200000,'batch':2000,'tie_rank':'average','extreme_rule':'>= statistic - 1e-12','p_rule':'(extreme+1)/(200000+1)','Holm':'full Aug-vs-each family separately per S1-S4','artifacts':replays})
 # Public companion only adds what the 226-file package omits.
 with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/a08_public_numeric_evidence.zip') as z:
  names=z.namelist();dump(O/'existing_226_payload_inventory.json',{'zip_sha256':sha(ROOT/'projects/benchmark/evidence/a08_public_numeric_evidence.zip'),'payload_count':len([n for n in names if not n.endswith('/')]),'payload':[{'path':n,'bytes':z.getinfo(n).file_size,'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in names if not n.endswith('/')]})
 for p in (E/'audit').glob('*.json'):
  dst=O/'original_gates_and_qualification'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
 for name in ['source_manifest.json','input_manifest.json','split_manifest.json','execution_manifest.json','model_settings_frozen.json']:
  shutil.copy2(E/name,O/name)
 after={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()};assert before==after
 dump(O/'a08_preservation_check.json',{'status':'PASS','checked_files':len(before),'identity_manifest_sha256':sha(T/'a08_preservation_before.json'),'identities':before})
 print('A09 CPU recheck PASS: 105 exact metric JSONs; 15 actual test supports; byte-identical full precision permutation/Holm/tables; A08 files unchanged',flush=True)
if __name__=='__main__':main()
