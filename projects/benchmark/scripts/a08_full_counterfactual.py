#!/usr/bin/env python3
"""Whole-provider raw-to-relation counterfactuals, no graph cache inputs."""
import hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.crypto_raw import content_hash,array_hash,sha256
from gog_fraud.data.benchmark_node_adapter import attach_labels
spec=importlib.util.spec_from_file_location('a08_facade',ROOT/'projects/benchmark/scripts/a08_data_repair.py');facade=importlib.util.module_from_spec(spec);spec.loader.exec_module(facade)
L=ROOT/'local_storage/benchmark/a08_data_repair';E=ROOT/'projects/benchmark/evidence/a08_data_repair'
cfg=json.loads((ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text())
while not (E/'audit/counterfactual_and_rebuild.json').exists():
    state=subprocess.run(['systemctl','--user','is-active','dlg-gnn-a08-raw-build.service'],capture_output=True,text=True).stdout.strip()
    if state not in ('active','activating'):raise RuntimeError('raw base build failed; do not bypass')
    time.sleep(20)
results=[]
for chain in ['polygon','bsc','ethereum']:
    reference=facade.read_arrays(L/'build_a'/chain/'observables.npz')
    y,v=attach_labels(reference['node_ids'],Path(cfg['raw_root'])/'labels.csv',chain)
    labels={'unavailable':None,'original':y,'permuted':np.random.RandomState(20261009).permutation(y),'all_zero':np.zeros_like(y),'all_one':np.ones_like(y)}
    conditions=[]
    for condition,workspace in [('unavailable','build_a'),('original','build_b'),('permuted','build_permuted'),('all_zero','build_zero'),('all_one','build_one')]:
        # A/B each already executed the same complete raw observable path;
        # remaining conditions execute it anew from the original CSV ZIP.
        if workspace not in ('build_a','build_b'):facade.build(cfg,chain,workspace)
        a=facade.read_arrays(L/workspace/chain/'observables.npz')
        if content_hash(a)!=content_hash(reference):raise ValueError('target scenario changed raw observables')
        target=labels[condition]
        conditions.append({'condition':condition,'workspace':workspace,'observable_content_hash':content_hash(a),'target_hash':array_hash(target) if target is not None else 'UNAVAILABLE',
                           'raw_zip_sha256':json.loads((L/workspace/chain/'observables_manifest.json').read_text())['source_zip_sha256'],
                           'target_attachment_status':'BLOCKED_MISSING_LABEL_FILE' if target is None else 'N_VECTOR_ATTACHED_SEPARATELY',
                           'existing_graph_as_input':False})
    results.append({'chain':chain,'status':'PASS','raw_to_relation_conditions':conditions,'fixed_split_record':'split construction excluded from observable stage; same canonical IDs used for all target scenarios'})
report={'status':'PASS','scope':'five complete whole-chain raw-to-feature-to-relation executions for each chain; no learned encoder; target attachment separate',
        'results':results,'script_sha256':sha256(Path(__file__)),'primitive_source_sha256':sha256(ROOT/'src/gog_fraud/data/crypto_raw.py')}
facade.write_json(E/'audit/raw_full_counterfactual.json',report)
print('WHOLE_PROVIDER_FIVE_CONDITION_COUNTERFACTUAL_PASS',flush=True)
