"""Post-hosting lineage metadata, not a science/measurement/asset change.
Run after r02_finalize.py, which emits summary-level bindings.
"""
from __future__ import annotations
import csv
import json
import zipfile
from r02_common import ROOT,PROJECT,R01,VERIFY,RUN,read,write,sha256,csv_write,original_unchanged


def main():
    original_unchanged();manifest=read(R01/'release_manifest.json')
    with (VERIFY/'claim_to_raw_r02.csv').open() as handle:rows=list(csv.DictReader(handle))
    rows=[r for r in rows if not r['claim_id'].startswith('raw_model_')]
    for r in rows:
        r['public_derivative_member']='';r['public_derivative_sha256']=''
        if r['claim_id']=='snapshot_primary':r['limitation']+='; summary input, not raw predictions: explicit110 raw-model rows below'
    with zipfile.ZipFile(PROJECT/'evidence/r01_numeric_evidence.zip') as z:public=json.loads(z.read('release_manifest.json'))['files']
    added=[]
    for directory in sorted((R01/'models').iterdir()):
        if not (directory/'complete.json').is_file():continue
        model=read(directory/'complete.json')
        for path in sorted(directory.glob('*.parquet')):
            if not (path.name in ('test_predictions.parquet','validation_predictions.parquet') or path.name.endswith('_predictions.parquet') or path.name.startswith('mc_T')):continue
            relative=str(path.relative_to(R01));assert sha256(path)==manifest['files'][relative]
            member='models/'+directory.name+'/'+path.stem+'.csv';assert member in public
            mc=path.name.startswith('mc_T');output=VERIFY/('mc_recomputed.csv' if mc else 'per_seed_metrics_recomputed.csv')
            added.append({'claim_id':'raw_model_'+directory.name+'_'+path.stem,'manuscript_location':'Main/Supplement original policy, control or MC raw metrics','experiment_id':model['experiment_id'],
                'seed_scope':str(model['seed']),'source_version':'training_sources.zip plus frozen policy composition','raw_path':str(path.relative_to(ROOT)),'raw_sha256':sha256(path),
                'aggregation_command':'python projects/stream_mc/scripts/r02_run.py numeric','output_path':str(output.relative_to(ROOT)),'output_sha256':sha256(output),'verified_status':'VERIFIED',
                'limitation':'original-ID Parquet versus pseudonymous CSV are distinct bytes; restore float32 raw dtype except kNN float64; public full-policy verifies composition, not original address identity',
                'public_derivative_member':member,'public_derivative_sha256':public[member]['sha256']})
    assert len(added)==110,len(added);csv_write(VERIFY/'claim_to_raw_r02.csv',rows+added)
    write(VERIFY/'raw_lineage_binding_identity.json',{'verification_id':RUN,'N_raw_model_inputs':len(added),'original_raw_manifest':sha256(R01/'release_manifest.json'),
        'output_sha256':sha256(VERIFY/'claim_to_raw_r02.csv'),'binding_source_sha256':sha256(__file__),
        'scope':'post-hosting metadata extension; original/published code and numeric assets unchanged, no new training/timing'})
    print('R02 RAW/PUBLIC LINEAGE BINDINGS PASS',len(added),flush=True)


if __name__=='__main__':main()
