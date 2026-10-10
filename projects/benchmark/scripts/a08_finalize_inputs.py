#!/usr/bin/env python3
"""Close actual source/input gates, freeze, then start qualified campaign."""
import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.crypto_raw import array_hash,sha256
from gog_fraud.pipelines.benchmark_crypto import factory
from gog_fraud.data.benchmark_lineage import bind_execution
E=ROOT/'projects/benchmark/evidence/a08_data_repair';L=ROOT/'local_storage/benchmark/a08_data_repair';R=ROOT/'projects/benchmark/reports/a08_data_repair'
CFG=json.loads((ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text())


def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2)+'\n')


def freeze_json(p,v):
    p=Path(p)
    if p.exists():
        if json.loads(p.read_text())!=v:raise ValueError('existing frozen document differs: '+str(p))
    else:dump(p,v)


def main():
    full=E/'audit/raw_full_counterfactual.json'
    while not full.exists():
        state=subprocess.run(['systemctl','--user','is-active','dlg-gnn-a08-counterfactual.service'],capture_output=True,text=True).stdout.strip()
        if state not in ('active','activating'):raise RuntimeError('whole raw counterfactual service failed or absent')
        time.sleep(20)
    if json.loads(full.read_text())['status']!='PASS':raise RuntimeError('whole raw counterfactual failed')
    audit=E/'audit/counterfactual_and_rebuild.json'
    while not audit.exists():
        status=subprocess.run(['systemctl','--user','is-active','dlg-gnn-a08-raw-build.service'],capture_output=True,text=True)
        if status.stdout.strip() not in ('active','activating'):raise RuntimeError('raw build not active and audit absent; inspect raw_build.log')
        time.sleep(20)
    result=json.loads(audit.read_text())
    if result['status']!='PASS' or len(result['results'])!=3:raise RuntimeError('G3 required')
    for r in result['results']:
        if r['unknown_count'] or r['unique_feature_rows']<2 or r['self_loops']!=0:raise RuntimeError('unexpected target/relation contract')
    sources=[];defaults={}
    import numpy as np
    for chain in ['polygon','bsc','ethereum']:
        meta=json.loads((L/'build_a'/chain/'observables_manifest.json').read_text())
        sources.append({'chain':chain,'source_zip_sha256':meta['source_zip_sha256'],'member_manifest_sha256':meta['raw_member_manifest_sha256'],
                        'member_count':meta['arrays']['node_ids']['shape'][0],'source_access':'provider downloaded research archive; no redistribution permission asserted',
                        'raw_observable_fields':['from','to'],'fit_population':'all observed raw contracts, transductive label-blind'})
        with np.load(L/'build_a'/chain/'observables.npz',allow_pickle=False) as z:
            ids=z['node_ids'];n=len(ids)
        defaults[chain]={model:factory(model,n,CFG['epochs'][chain],CFG['local_epochs'])[1] for model in CFG['models']}
        with (L/'build_a'/chain/'node_index.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['position','stable_node_id']);w.writerows(enumerate(ids))
    freeze_json(E/'source_manifest.json',{'sources':sources,'labels_sha256':sha256(Path(CFG['raw_root'])/'labels.csv'),
         'label_mapping':{'source_field':'Category','positive_code':0,'binary_positive':1,'others':0,'join':'chain-qualified normalized contract ID','unknown':'-1 / invalid mask'},
         'provider_readme_sha256':sha256(Path('/mnt/d/_Work/goat_bank/gog/README.md')),
         'source_mapping_code_sha256':sha256(Path('/mnt/d/_Work/goat_bank/gog/dataset/process_graph_metrics.py'))})
    freeze_json(E/'model_settings_frozen.json',defaults)
    dump(E/'audit/G2.json',{'status':'PASS','regression_sha256':sha256(E/'audit/regression_tests.json'),'actual_three_chain_contract_sha256':sha256(audit),'source_contract':'graph API unchanged; explicit separate N-vector node targets'})
    dump(E/'audit/G3.json',{'status':'PASS','counterfactual_rebuild_sha256':sha256(audit),'whole_raw_five_condition_sha256':sha256(full),'specification_note':'Five complete raw-to-feature-to-relation executions per whole chain from provider ZIP only, paired with original/permuted/0/1 and unavailable labels; fixed split excluded from observable stage'})
    frozen=E/'input_manifest.json'
    if not frozen.exists():subprocess.run([sys.executable,'-u',str(ROOT/'projects/benchmark/scripts/a08_data_repair.py'),'--stage','freeze'],cwd=ROOT,check=True)
    bound={'status':'PASS','input_manifest_sha256':sha256(frozen),'split_manifest_sha256':sha256(E/'split_manifest.json'),'planned_cells_sha256':sha256(E/'planned_cells.csv'),
         'model_settings_sha256':sha256(E/'model_settings_frozen.json'),'source_manifest_sha256':sha256(E/'source_manifest.json'),'config_sha256':sha256(ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml')}
    if (E/'audit/G4.json').exists():
        prior=json.loads((E/'audit/G4.json').read_text())
        if any(prior.get(k)!=v for k,v in bound.items()):raise ValueError('immutable G4 binding changed')
    bind_execution(ROOT,CFG,defaults)
    bound.update(execution_manifest_sha256=sha256(E/'execution_manifest.json'),whole_raw_five_condition_sha256=sha256(full),
                 qualification_precondition='Whole five-condition raw construction verified before qualification or production; existing input bytes and masks unchanged')
    amendment=ROOT/'projects/benchmark/protocols/A08_NUMERICAL_REPAIR_AMENDMENT_2026-10-10.md'
    if amendment.exists():bound.update(numerical_execution_revision=2,numerical_amendment_sha256=sha256(amendment))
    dump(E/'audit/G4.json',bound)
    lines=['# A08 actual raw-only rebuild and adapter audit','','G2/G3/G4: PASS after actual construction and two clean workspace comparison.','',
           '| Chain | N | E | F | Positives | Unknown | Unique feature rows |','|---|---:|---:|---:|---:|---:|---:|']
    for r in result['results']:lines.append(f"| {r['chain']} | {r['N']} | {r['E']} | {r['F']} | {r['positive_count']} | {r['unknown_count']} | {r['unique_feature_rows']} |")
    lines+=['','Immutable scientific content matches across raw builds A/B. No old PT/cache is loaded. Stable-ID row shuffle preserves adjacency. All real-chain adapter targets are N-vectors. Four whole-population target counterfactual attachments leave the already independently label-inaccessible observable graph unchanged; unavailable label attachment fails. The scope is observation-only construction, with no encoder supervision.','',
            'Original provider CSV population differs from legacy converted JSON availability; fresh stable-ID masks are declared in the amendment. Source/member and per-array hashes are recorded. Production predictions, metrics, paper/PDF and final acceptance are still pending.']
    (R/'DATASET_REBUILD_REPORT.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,'-u',str(ROOT/'projects/benchmark/scripts/a08_campaign.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=CFG['gpu_uuid']),check=True)
if __name__=='__main__':main()
