#!/usr/bin/env python3
"""A08 stage facade. No fallback to old graphs/results, and no force-pass."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from gog_fraud.data.crypto_raw import (FEATURE_NAMES, array_hash, build_observables, content_hash,
                                       deterministic_knn, save_arrays, sha256)
from gog_fraud.data.benchmark_node_adapter import attach_labels, node_view
EVIDENCE = ROOT/'projects/benchmark/evidence/a08_data_repair'
REPORTS = ROOT/'projects/benchmark/reports/a08_data_repair'
LOCAL = ROOT/'local_storage/benchmark/a08_data_repair'
CONFIG = ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml'
CHAINS = ['polygon', 'bsc', 'ethereum']


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def utc(): return datetime.now(timezone.utc).isoformat()


def code_identity():
    paths = ['src/gog_fraud/data/crypto_raw.py', 'src/gog_fraud/data/benchmark_node_adapter.py',
             'projects/benchmark/scripts/a08_data_repair.py', 'src/gog_fraud/data/level2/relation_builder.py']
    return {p: sha256(ROOT/p) for p in paths}


def inventory(cfg):
    for path in (EVIDENCE, REPORTS, LOCAL): path.mkdir(parents=True, exist_ok=True)
    patch = subprocess.check_output(['git','diff','--binary','HEAD'], cwd=ROOT)
    (LOCAL/'source_at_inventory.patch').write_bytes(patch)
    files = []
    candidates = list((ROOT/'projects/benchmark/evidence').glob('*.zip'))
    candidates += list((ROOT/'archive').rglob('*A05*.zip'))+list((ROOT/'release').rglob('*.zip'))
    candidates += [ROOT/'projects/benchmark/DATASET_CONSTRUCTION_AUDIT.md',ROOT/'projects/benchmark/evidence/astra_revision/MANIFEST.json',
                   ROOT/'projects/benchmark/evidence/astra_revision/relation_builder_migration_audit.json', ROOT/'environment/locks/benchmark-a03-cuda.lock.txt']
    candidates += list((ROOT/'projects/benchmark/paper/current').rglob('*.pdf')) + list((ROOT/'projects/benchmark/paper/current').rglob('*.tex'))
    candidates += [Path(cfg['raw_root'])/'labels.csv']
    for c in CHAINS:
        candidates += [Path(cfg['raw_root'])/'transactions'/f'{c}.zip', Path(cfg['legacy_provider_root'])/c/f'{c}_level2_graph.pt']
    for p in dict.fromkeys(candidates):
        exists = p.is_file()
        files.append({'path': str(p), 'exists': exists, 'bytes': p.stat().st_size if exists else 0,
                      'sha256': sha256(p) if exists else 'MISSING',
                      'visibility': 'author-local' if p.suffix in ('.tex','.pdf','.pt') or str(p).startswith(cfg['raw_root']) else 'public-evidence-or-historical-local',
                      'role': 'unqualified_migration' if p.suffix == '.pt' else 'source_or_preserved_evidence'})
    migration = json.loads((ROOT/'projects/benchmark/evidence/astra_revision/relation_builder_migration_audit.json').read_text())
    denylist = []
    for r in migration['results']:
        denylist += [{'dataset': r['dataset'], 'class': 'UNQUALIFIED_MIGRATION', 'sha256': r['new_sha256']},
                     {'dataset': r['dataset'], 'class': 'HISTORICAL_LABEL_INFORMED', 'sha256': r['old_canonical_raw_sha256']}]
    write_json(EVIDENCE/'old_input_denylist.json', denylist)
    write_json(EVIDENCE/'inventory.json', {'time':utc(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'dirty_diff_sha256': hashlib.sha256(patch).hexdigest(), 'interpreter':sys.executable,'os':platform.platform(),'files':files,
               'status':'PASS' if all((Path(cfg['raw_root'])/'transactions'/f'{c}.zip').is_file() for c in CHAINS) else 'BLOCKED'})
    print('G0 inventory recorded', flush=True)


def build(cfg, chain, workspace):
    out = LOCAL/workspace/chain
    if (out/'observables.npz').exists():
        meta = json.loads((out/'observables_manifest.json').read_text())
        if sha256(out/'observables.npz') != meta['serialized_file_sha256']: raise ValueError('existing build corrupted')
        print(chain,workspace,'NO_CHANGE', flush=True); return
    if not (EVIDENCE/'feature_spec.yaml').exists() or not (EVIDENCE/'relation_spec.yaml').exists(): raise ValueError('G1 specifications missing')
    transaction_zip = Path(cfg['raw_root'])/'transactions'/f'{chain}.zip'
    arrays, raw_members = build_observables(transaction_zip, chain, lambda i,n: print(f'{workspace} {chain} {i}/{n}',flush=True))
    edges, relation_audit = deterministic_knn(arrays['x'], arrays['node_ids'], k=cfg['relation']['k'])
    arrays['edge_index'] = edges
    save_arrays(out/'observables.npz', arrays)
    write_json(out/'raw_members.json', raw_members)
    manifest = {'schema_version':1,'source':str(transaction_zip),'source_zip_sha256':sha256(transaction_zip),
                'builder_code_hashes':code_identity(),'feature_names':FEATURE_NAMES,'labels_accessed':False,
                'scientific_content_hash':content_hash(arrays),'serialized_file_sha256':sha256(out/'observables.npz'),
                'arrays':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':array_hash(v)} for k,v in arrays.items()},
                'raw_member_manifest_sha256':sha256(out/'raw_members.json'),'relation_audit':relation_audit,'build_time':utc()}
    write_json(out/'observables_manifest.json',manifest)
    print(chain,workspace,'BUILT',len(arrays['node_ids']),edges.shape[1],flush=True)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as f: return {k:f[k] for k in f.files}


def audit(cfg):
    import torch
    from torch_geometric.data import Data
    results = []
    for c in CHAINS:
        a = read_arrays(LOCAL/'build_a'/c/'observables.npz'); b = read_arrays(LOCAL/'build_b'/c/'observables.npz')
        if content_hash(a) != content_hash(b): raise ValueError('raw-only A/B differs')
        labels,valid = attach_labels(a['node_ids'],Path(cfg['raw_root'])/'labels.csv',c)
        n = len(labels); rng = np.random.RandomState(20261009)
        attached = []
        for condition, target in [('original',labels),('permuted',rng.permutation(labels)),('all_zero',np.zeros(n,dtype=np.int64)),('all_one',np.ones(n,dtype=np.int64))]:
            source = Data(x=torch.from_numpy(a['x']), edge_index=torch.from_numpy(a['edge_index']),y=torch.tensor([1]),num_nodes=n)
            view = node_view(source,a['node_ids'],c+'_contract_clean_v1',labels=target,label_valid_mask=valid)
            if view.y.shape != (n,) or source.y.shape != (1,): raise ValueError('target contract')
            if not np.array_equal(view.x.numpy(),a['x']) or not np.array_equal(view.edge_index.numpy(),a['edge_index']): raise ValueError('label dependence')
            attached.append({'condition':condition,'x_hash':array_hash(view.x.numpy()),'edge_hash':array_hash(view.edge_index.numpy()),'label_hash':array_hash(view.y.numpy())})
        # Labels-unavailable construction is the actual completed build_a/build_b
        # process: its only data argument is the CSV ZIP and no label path exists
        # in the feature/relation function interfaces. Attachment separately fails.
        try: attach_labels(a['node_ids'],LOCAL/'NO_LABEL_FILE.csv',c)
        except FileNotFoundError: pass
        else: raise ValueError('missing labels unexpectedly accepted')
        order=rng.permutation(n); shuffled_edge,_=deterministic_knn(a['x'][order],a['node_ids'][order],k=cfg['relation']['k'])
        canonical=np.sort(order[shuffled_edge][0]*n+order[shuffled_edge][1])
        if not np.array_equal(canonical,a['edge_index'][0]*n+a['edge_index'][1]): raise ValueError('row-order dependent relation')
        norms=np.linalg.norm(a['x'],axis=1); positive=norms>0
        directions=a['x'][positive]/norms[positive,None]
        degree=np.bincount(a['edge_index'][0],minlength=n)
        results.append({'chain':c,'status':'PASS','N':n,'E':a['edge_index'].shape[1],'F':a['x'].shape[1],
                        'positive_count':int(labels[valid].sum()),'unknown_count':int((~valid).sum()),
                        'raw_only_two_builds_equal':True,'raw_build_without_label_argument':True,
                        'counterfactual_scope':'Entire raw observable construction A/B has no labels interface; four separate whole-population target attachments plus unavailable-label rejection; no learned encoder',
                        'label_conditions':attached,'row_shuffle_invariant':True,
                        'unique_feature_rows':np.unique(a['x'],axis=0).shape[0], 'unique_normalized_directions':np.unique(directions,axis=0).shape[0],
                        'zero_norm_rows':int((~positive).sum()),'column_variance':a['x'].var(0).tolist(),
                        'column_min':a['x'].min(0).tolist(),'column_max':a['x'].max(0).tolist(),
                        'degree_min':int(degree.min()),'degree_max':int(degree.max()),'isolates':int((degree==0).sum()),
                        'self_loops':int((a['edge_index'][0]==a['edge_index'][1]).sum()),
                        'content_hash':content_hash(a)})
    write_json(EVIDENCE/'audit/counterfactual_and_rebuild.json',{'status':'PASS','time':utc(),'results':results,'code_hashes':code_identity()})
    print('G3 actual three-chain raw rebuild and target audit PASS',flush=True)


def freeze(cfg):
    report=json.loads((EVIDENCE/'audit/counterfactual_and_rebuild.json').read_text())
    if report['status'] != 'PASS' or len(report['results']) != 3: raise ValueError('G3 required')
    if report['code_hashes'] != code_identity(): raise ValueError('G3 source changed')
    unit=json.loads((EVIDENCE/'audit/regression_tests.json').read_text())
    if unit['status']!='PASS': raise ValueError('regressions required')
    manifests=[]; splits=[]
    for c in CHAINS:
        arrays=read_arrays(LOCAL/'build_a'/c/'observables.npz')
        arrays['labels'],arrays['label_valid_mask']=attach_labels(arrays['node_ids'],Path(cfg['raw_root'])/'labels.csv',c)
        n=len(arrays['node_ids']); ids=np.flatnonzero(arrays['label_valid_mask'])
        for seed in cfg['seeds']:
            perm=np.random.RandomState(seed).permutation(ids)
            cut1,cut2=int(.6*len(ids)),int(.8*len(ids))
            masks={}
            for name,idx in [('train_mask',perm[:cut1]),('val_mask',perm[cut1:cut2]),('test_mask',perm[cut2:])]:
                mask=np.zeros(n,dtype=bool);mask[idx]=True;arrays[f'{name}_{seed}']=mask;masks[name]=array_hash(mask)
            splits.append({'dataset_id':c+'_contract_clean_v1','seed':seed,'masks':masks,'policy':'RandomState(seed) permutation of canonical valid stable IDs; 60/20/20, unstratified as actual historical code',
                           'prior_membership_preserved':False,'reason':'raw population includes contracts excluded by old converted JSON availability; canonical address order replaces unverified old positional mapping'})
        out=LOCAL/'frozen'/c; artifact=out/'contract_graph.npz'
        save_arrays(artifact,arrays)
        m={'schema_version':1,'dataset_id':c+'_contract_clean_v1','dataset_version':1,'prediction_unit':'contract_node',
           'artifact_relative_path':'contract_graph.npz','serialized_file_sha256':sha256(artifact),'scientific_content_hash':content_hash(arrays),
           'N':n,'E':arrays['edge_index'].shape[1],'F':arrays['x'].shape[1],'positive_count':int(arrays['labels'][arrays['label_valid_mask']].sum()),'unknown_count':int((~arrays['label_valid_mask']).sum()),
           'feature_spec_hash':sha256(EVIDENCE/'feature_spec.yaml'),'relation_spec_hash':sha256(EVIDENCE/'relation_spec.yaml'),
           'source_manifest_hash':sha256(LOCAL/'build_a'/c/'observables_manifest.json'),'raw_zip_sha256':sha256(Path(cfg['raw_root'])/'transactions'/f'{c}.zip'),
           'label_source_hash':sha256(Path(cfg['raw_root'])/'labels.csv'),'builder_code_hashes':code_identity(),'config_hash':sha256(CONFIG),
           'environment_lock_hash':sha256(ROOT/'environment/locks/benchmark-a03-cuda.lock.txt'),
           'arrays':{k:array_hash(v) for k,v in arrays.items()},'split_records':[s for s in splits if s['dataset_id']==c+'_contract_clean_v1'],
           'scientific_status':'FROZEN_APPROVED_INPUT','freeze_time_utc':utc(), 'supersedes':'historical label-informed hybrid and unqualified migration, retained in denylist'}
        write_json(out/'input_manifest.json',m);manifests.append(m)
    write_json(EVIDENCE/'input_manifest.json',{'schema_version':1,'campaign_id':cfg['campaign_id'],'datasets':manifests})
    write_json(EVIDENCE/'split_manifest.json',splits)
    with (EVIDENCE/'planned_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset_id','model','seed','global_epochs','local_epochs','guard_seconds','guard_scope']);w.writeheader()
        for c in CHAINS:
            for model in cfg['models']:
                for seed in cfg['seeds']: w.writerow(dict(dataset_id=c+'_contract_clean_v1',model=model,seed=seed,global_epochs=cfg['epochs'][c],local_epochs=20 if model=='DLG-Aug' else 0,guard_seconds=86400,guard_scope='per_model_dataset_seed'))
    print('G4 input frozen; maximum 105 cells',flush=True)


def final_check():
    # Construction source version is preserved; only the acceptance dispatcher
    # changes. The scientific builder functions are byte-for-byte unchanged.
    import subprocess
    return subprocess.run([sys.executable, str(ROOT/'projects/benchmark/scripts/a08_final_check.py')], cwd=ROOT).returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['inventory','build','audit','freeze','final-check']);p.add_argument('--config',default=str(CONFIG));p.add_argument('--chain',choices=CHAINS);p.add_argument('--workspace',default='build_a',choices=['build_a','build_b']);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    if a.stage=='inventory':inventory(cfg)
    elif a.stage=='build':
        for c in ([a.chain] if a.chain else CHAINS): build(cfg,c,a.workspace)
    elif a.stage=='audit':audit(cfg)
    elif a.stage=='freeze':freeze(cfg)
    else:return final_check()
    return 0
if __name__=='__main__':raise SystemExit(main())
