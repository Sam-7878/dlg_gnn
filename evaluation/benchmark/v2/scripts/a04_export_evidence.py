#!/usr/bin/env python3
"""Export an honest compact A04 per-run evidence chain."""
import csv,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'evaluation/benchmark/v2/paper_ready_final'
B=OUT/'publication_evidence_a04'
REG=B/'approved_run_registry.csv'
MAN=OUT/'dataset_manifest_canonical.json'
LOCK=ROOT/'environment/locks/benchmark-a04-cuda.lock.txt'
LEGACY_ENV=ROOT/'outputs/benchmark/sci_round5_final/manifests/environment_freeze.json'
LEGACY_LOCK=ROOT/'evaluation/benchmark/v2/environment/legacy/20261002T145148Z/requirements-v1-freeze.txt'
SESSION_PROBE=ROOT/'evaluation/benchmark/v2/environment/journal_cuda/environment_probe.json'
CFG=ROOT/'outputs/benchmark/sci_round5_final/manifests/config_snapshot.yaml'
CRYPTO=ROOT/'evaluation/benchmark/v2/scripts/a03_run_crypto_production.py'
CRYPTO_REPAIR=ROOT/'evaluation/benchmark/v2/scripts/a03_rerun_crypto_dlg_aug.py'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def encode(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def mask_value(row,key,seed):
    value=row.get(key)
    if not value:return None
    if isinstance(value,str) and value.startswith('{'):return json.loads(value).get(str(seed))
    return value


def main():
    manifest={x['dataset_id']:x for x in json.loads(MAN.read_text())}
    registry=list(csv.DictReader(REG.open()))
    for sub in ['run_manifests','metric_json','model_configs']:(B/sub).mkdir(parents=True,exist_ok=True)
    config_hash=sha(CFG);lock_hash=sha(LOCK)
    shutil.copyfile(LEGACY_ENV,B/'environment'/'round5_execution_environment_freeze.json')
    shutil.copyfile(LEGACY_LOCK,B/'environment'/'round5_archived_legacy_venv_inventory.txt')
    for rec in registry:
        source=ROOT/rec['source_path'];d=json.loads(source.read_text())
        ds=rec['dataset'];model=rec['model'];seed=int(rec['seed']);m=manifest[ds]
        tensor_hash=m.get('constructed_tensor_sha256')
        rec['dataset_hash']=tensor_hash or ''
        masks={k:mask_value(m,k,seed) for k in ('train_mask_hash','val_mask_hash','test_mask_hash')}
        rec['split_hash']=hashlib.sha256(encode(masks)).hexdigest() if all(masks.values()) else ''
        if rec['source_kind']=='round5_json':
            config={'source_kind':'round5_json','source_config_hash':d.get('config_hash'),
                    'config_snapshot_sha256':config_hash,'model':model,
                    'python_class':d.get('python_class'),'configured_epochs':d.get('configured_epochs'),
                    'message_backend':d.get('message_backend'),'reconstruction_backend':d.get('reconstruction_backend')}
            env_source=LEGACY_ENV
            env_status='ARCHIVED_LEGACY_VENV_INVENTORY_MATCHES_FREEZE_NOT_RUN_BOUND'
            candidate_lock_hash=sha(LEGACY_LOCK)
            config_status='ORIGINAL_RUN_CONFIG_HASH'
        elif rec['source_kind']=='a03_json':
            script=CRYPTO_REPAIR if model=='DLG-Aug' else CRYPTO
            config={'source_kind':'a03_json','source_script_sha256':sha(script),
                    'model':model,'dataset':ds,'epochs':30 if ds=='Ethereum' else 40,
                    'details':'Runner defaults and PyGOD version apply; no original per-run config hash was stored.'}
            env_source=SESSION_PROBE
            env_status='A03_SESSION_PROBE_NOT_BOUND_TO_RUN'
            candidate_lock_hash=lock_hash
            config_status='SCRIPT_DEFAULTS_RECONSTRUCTED_NOT_RUN_BOUND'
        else:
            config={'source_kind':'lanl_real_json','source_csv_sha256':rec.get('source_csv_sha256'),
                    'benchmark_origin':d.get('benchmark_origin'),'model':model,
                    'configured_epochs':d.get('configured_epochs'),
                    'source_config_status':'NOT_STORED_IN_ORIGINAL_RUN'}
            env_source=None
            env_status='LANL_EXECUTION_ENVIRONMENT_NOT_STORED'
            candidate_lock_hash=''
            config_status='ORIGINAL_RUN_CONFIG_NOT_STORED'
        cfg_hash=hashlib.sha256(encode(config)).hexdigest()
        rec['model_config_hash']=cfg_hash
        rec['model_config_provenance_status']=config_status
        cfg_path=B/'model_configs'/f'{cfg_hash}.json'
        if not cfg_path.exists():write_json(cfg_path,config)
        id=rec['run_id'];metric_path=B/'metric_json'/f'{id}.json'
        shutil.copyfile(source,metric_path)
        rec['metric_json_path']=str(metric_path.relative_to(ROOT));rec['metric_json_hash']=sha(metric_path)
        rec['raw_score_hash']=''
        rec['raw_score_status']='MISSING_NOT_STORED_BY_SOURCE_RUN'
        rec['dataset_verification_status']=m['verification_status']
        rec['environment_lock_hash']=''
        rec['execution_environment_evidence_path']=str(env_source.relative_to(ROOT)) if env_source else ''
        rec['execution_environment_evidence_sha256']=sha(env_source) if env_source else ''
        rec['execution_environment_status']=env_status
        rec['execution_environment_candidate_lock_hash']=candidate_lock_hash
        rec['a04_qualification_environment_lock_hash']=lock_hash
        rec['provenance_complete']='false'
        run_manifest={key:rec.get(key) for key in rec}
        run_manifest['masks']=masks
        run_manifest['source_metric_sha256']=sha(source)
        write_json(B/'run_manifests'/f'{id}.json',run_manifest)
    fields=list(registry[0])
    with REG.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(registry)
    score_rows=[{'run_id':r['run_id'],'raw_score_hash':'MISSING','reason':r['raw_score_status']} for r in registry]
    with (B/'raw_score_hashes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(score_rows[0]));w.writeheader();w.writerows(score_rows)
    shutil.copyfile(OUT/'dataset_manifest_canonical.csv',B/'dataset_manifest_canonical.csv')
    shutil.copyfile(OUT/'dataset_manifest_canonical.json',B/'dataset_manifest_canonical.json')
    print(json.dumps({'run_manifests':len(registry),'source_json':len(registry),
                      'lanl_original_json':sum(r['source_kind']=='lanl_real_json' for r in registry),
                      'split_hash_present':sum(bool(r['split_hash']) for r in registry),
                      'raw_score_hash_present':sum(bool(r['raw_score_hash']) for r in registry)},indent=2))

if __name__=='__main__':main()
