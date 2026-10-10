#!/usr/bin/env python3
"""Verify preserved numerical identity without manufacturing old raw evidence."""
import csv,hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.crypto_raw import sha256
E=ROOT/'projects/benchmark/evidence/a08_data_repair'


def main():
    archive=ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip'
    checks=[]
    with zipfile.ZipFile(archive) as z:
        prefix='evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/'
        rows=list(csv.DictReader(io.StringIO(z.read(prefix+'approved_run_registry.csv').decode())))
        sources={}
        for name in z.namelist():
            if name.endswith('.json'):sources.setdefault(hashlib.sha256(z.read(name)).hexdigest(),[]).append(name)
        for row in rows:
            if row['dataset'] in ('Ethereum','BSC','Polygon'):continue
            matches=sources.get(row['metric_json_hash'],[])
            if not matches:raise ValueError('preserved run JSON hash unavailable: '+row['run_id'])
            raw=json.loads(z.read(matches[0]))
            for field in ('pr_auc','roc_auc','validation_f1'):
                if field in raw and row.get(field) not in ('',None):
                    if float(raw[field])!=float(row[field]):raise ValueError('preserved numeric mismatch: '+row['run_id']+'/'+field)
            checks.append({'run_id':row['run_id'],'dataset':row['dataset'],'model':row['model'],'seed':int(row['seed']),
                           'metric_json_sha256':row['metric_json_hash'],'verified_archive_member':matches[0],
                           'dataset_hash':row['dataset_hash'],'split_hash':row['split_hash'],'model_config_hash':row['model_config_hash'],
                           'source_backend_hash':row['source_backend_hash'],'provenance_tier':row.get('provenance_tier',''),
                           'raw_score_status':row.get('raw_score_status','UNKNOWN'),'status':'PRESERVED_NUMERIC_IDENTITY_VERIFIED'})
        # Source snapshots and original version bytes remain the historical identity.
        original_sources={name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()
                          if ('/source_snapshot/' in name or '/source_versions/' in name) and name.endswith('.py')}
    shared={}
    for path in sorted((ROOT/'src/gog_fraud/models/pygod').glob('*.py')):
        relative=str(path.relative_to(ROOT));head=subprocess.run(['git','show','HEAD:'+relative],cwd=ROOT,capture_output=True)
        if head.returncode!=0 or hashlib.sha256(head.stdout).hexdigest()!=sha256(path):raise ValueError('A08 shared detector change expands rerun scope: '+relative)
        shared[relative]=sha256(path)
    obj={'status':'PASS','scope':'A08 dependency-impact and unchanged preserved numeric identity; original historical evidence tier is retained, not upgraded to a current raw-score rerun',
         'archive_sha256':sha256(archive),'historical_crypto_excluded':sum(r['dataset'] in ('Ethereum','BSC','Polygon') for r in rows),
         'preserved_records':checks,'original_archived_source_hashes':original_sources,'a08_shared_detector_hashes':shared,
         'a08_shared_semantics_changed':False,'source_comparison_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'limitations':['Historical source snapshots differ from current code and remain explicitly versioned; no current code is attributed to old predictions.',
                        'Legacy source metrics do not universally include raw scores, run-bound environment or threshold loss curves; gaps remain disclosed.',
                        'New adapter/raw builder/metric path is A08 contract-only and is not injected into preserved noncrypto or LANL runners.']}
    (E/'audit/historical_reuse.json').write_text(json.dumps(obj,indent=2)+'\n')
    print('Verified historical JSON bytes for',len(checks),'unaffected runs; no tier upgrade')
if __name__=='__main__':main()
