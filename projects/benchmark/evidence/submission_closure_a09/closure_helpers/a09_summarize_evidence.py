#!/usr/bin/env python3
"""Expose provenance, missing-run reasons, and measured qualification limits."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'projects/benchmark/evidence/submission_closure_a09';P=ROOT/'projects/benchmark/reports/submission_closure_a09';E=ROOT/'projects/benchmark/evidence/a08_data_repair'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def out(name,rows):
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 shutil.copy2(O/name,P/name)
def main():
 registry=json.loads((E/'approved_registry.json').read_text())['records'];unsupported=[]
 for d,m in [('Yelp-Syn','AnomalyDAE'),('Reddit-Syn','AnomalyDAE'),('Elliptic','GADNR'),('DGraphFin','GADNR'),('Yelp-Syn','GADNR'),('Reddit-Syn','GADNR'),('Flickr-Syn','GADNR')]:
  row={'dataset':d,'model':m,'actual_support':'NOT_FULLY_SUPPORTED','evidence_campaign':'historical final support panel','event_class':'NO_APPROVED_RAW_RUN','device_envelope':'no run-bound device record for absent run','reason':'No approved raw run; execution-failure cause is not established by absence','source_record':'evaluation/benchmark/v2/paper_ready_a05/table_support_primary13_24g.csv','source_sha256':'','source_config_hash':'not available for absent run','source_backend_hash':'not available for absent run','current_A08_retested':False}
  if m=='AnomalyDAE':
   p=ROOT/f'evaluation/benchmark/v2/diagnostics/anomalydae_chunked/preflight_{d}.json';j=json.loads(p.read_text());assert j['error_type']=='OutOfMemoryError';row.update(evidence_campaign='A05 exact row-block seed42 preflight',event_class='OBSERVED_CUDA_OOM',device_envelope='RTX3090 24 GiB; block=256; full encoder before first block',reason='Observed CUDA OutOfMemoryError in full encoder forward; not projected timeout',source_record=str(p.relative_to(ROOT)),source_sha256=sha(p))
   target=O/'resource_records'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
  elif d=='Flickr-Syn':
   r=next(r for r in registry if r['dataset']==d and r['model']==m)
   with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip') as z:
    member='evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/legacy_metric_json/'+r['run_id']+'.json';b=z.read(member);j=json.loads(b)
   assert j['failure_type']=='OutOfMemoryError';row.update(evidence_campaign='Round5 seed42',event_class='OBSERVED_HISTORICAL_CUDA_OOM',device_envelope='CUDA reported 8 GiB; legacy environment not run-bound',reason='Observed CUDA OOM; actual_epochs=0; no current RTX3090 impossibility claim',source_record='public_numeric_evidence.zip:'+member,source_sha256=hashlib.sha256(b).hexdigest(),source_config_hash=r['source_config_hash'],source_backend_hash=r['source_backend_hash'])
   (O/'resource_records/flickr_gadnr_seed42.json').write_bytes(b)
  else:
   with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip') as z:row['source_sha256']=hashlib.sha256(z.read(row['source_record'])).hexdigest()
  unsupported.append(row)
 out('unsupported_cells.csv',unsupported)
 q=json.loads((E/'audit/cuda_exactness.json').read_text());rows=[]
 def flatten(v):
  if isinstance(v,dict) and 'max_abs' in v:return [v]
  if isinstance(v,dict):return sum((flatten(x) for x in v.values()),[])
  if isinstance(v,list):return sum((flatten(x) for x in v),[])
  return []
 for r in q['results']:
  for field,v in r['checks'].items():
   vv=flatten(v);rows.append({'family':r['path'],'case':r['case'],'quantity':field,'N':r.get('n',9),'E_stored':34 if r['path']=='AnomalyDAE_row_block' else (16 if r['case']=='binary undirected' else 12),'H':64 if r['path']=='AnomalyDAE_row_block' else (7 if r['path']=='fused_GCN' else 5),'layers':r.get('layers','not applicable'),'dtype':'float32','device':q['device'],'max_abs':max(x['max_abs'] for x in vv),'max_relative':'NOT_STORED_IN_ORIGINAL_CUDA_RECORD','relative_definition':'combined allclose |a-b| <= atol+rtol*|b|; supplemental measured floor relative if device available','atol':q['atol'],'rtol':q['rtol'],'source_record':'a08_data_repair/audit/cuda_exactness.json','source_sha256':sha(E/'audit/cuda_exactness.json'),'implementation_hashes':json.dumps(q['source_hashes'],sort_keys=True),'conditions':'dropout0; 2/4 GCN layers with normalization self-loops; Gram input edges no loops for binary, weighted duplicates with loops otherwise; Anomaly block4, all blocks before one Adam'})
 g=json.loads((E/'audit/gadnr_numerical_qualification.json').read_text())
 for case in g['actual_sampled_model_cases']:
  for field,v in case['checks'].items():
   values=v if isinstance(v,list) else [v];rows.append({'family':'GADNR sampled model','case':case['device'],'quantity':field,'N':17,'E_stored':34,'H':64,'layers':1,'dtype':'float32 with float64 detached neighbor arithmetic','device':case['device'],'max_abs':max(values),'max_relative':'NOT_STORED_IN_ORIGINAL_RECORD','relative_definition':'same combined allclose; original detached term does not prove whole training equivalence','atol':g['atol'],'rtol':g['rtol'],'source_record':'a08_data_repair/audit/gadnr_numerical_qualification.json','source_sha256':sha(E/'audit/gadnr_numerical_qualification.json'),'implementation_hashes':json.dumps({k:v for k,v in g.items() if k.endswith('sha256')},sort_keys=True),'conditions':'sample2/time3; identical seed46; deterministic algorithms/CUBLAS :4096:8; one Adam; initial nondeterministic update mismatch retained'})
 # Historical CPU relative values are available, but are not relabeled CUDA.
 h=ROOT/'projects/benchmark/reports/astra_revision/exact_cpu_qualification.json';j=json.loads(h.read_text())
 for r in j['results']:
  for field,v in r['checks'].items():
   rows.append({'family':'historical CPU '+r['path'],'case':r['case'],'quantity':field,'N':r['n'],'E_stored':'see source fixture','H':r.get('d','see source fixture'),'layers':r.get('layers','not applicable'),'dtype':r['dtype'],'device':'CPU','max_abs':v['max_abs'],'max_relative':v['max_rel'],'relative_definition':'max |a-b|/(|a|+1e-12), original astra_exact_cpu.py; absolute floor avoids division by zero','atol':r['atol'],'rtol':r['rtol'],'source_record':str(h.relative_to(ROOT)),'source_sha256':sha(h),'implementation_hashes':json.dumps(j['source_hashes'],sort_keys=True),'conditions':'historical post-review fixture; not CUDA production or full trajectory'})
 out('numerical_qualification_summary.csv',rows)
 # Author-local source README hash matches the frozen provider anchor.
 provider=ROOT.parent/'gog/README.md';j=json.loads((E/'source_manifest.json').read_text());assert sha(provider)==j['provider_readme_sha256']
 record={'source_project':'Multi-Chain Graphs of Graphs','authors':'Bingqiao Luo; Zhen Zhang; Qian Wang; Bingsheng He','paper_year':2024,'doi':'10.52202/079017-0894','repository':'https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs','download':'https://drive.google.com/drive/folders/1VV5ht9Eh8WGtKfkS0ipIk0FNI7g-WJfJ','local_provider_README_sha256':sha(provider),'frozen_provider_README_sha256':j['provider_readme_sha256'],'mapping':'README Category0=fraud; derived target=1 for Category0, 0 for other observed categories, -1 missing','source_population':'original per-token transfer CSVs; derived feature-similarity relation is not provider global graph','license_observation':'upstream LICENSE downloaded 2026-10-10 says CC BY-NC-SA4.0; this is a source observation, not a new redistribution or reviewer-sharing authorization','source_manifest_sha256':sha(E/'source_manifest.json'),'author_source_confirmation':'pending coauthor review; technical README hash equality established'}
 (O/'provider_identity.json').write_text(json.dumps(record,indent=2)+'\n')
 (P/'HISTORICAL_STATUS_NOTE.md').write_text('# Final dependency status\n\nThe A08 impact_map.csv and NODE_LABEL_CONTRACT_REPORT.md heading are preserved planning/history snapshots. The authoritative completed crypto selections are impact_map_final.csv (35 actual approved runs per chain). A08 FINAL_ACCEPTANCE remains historical and unchanged. Shared Stream/TDS API fixtures passed, but their empirical paper campaigns are not newly approved by this closure.\n')
 print('A09 support, qualification and provider evidence summaries generated')
if __name__=='__main__':main()
