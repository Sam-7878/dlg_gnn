"""R03 identity lock, CPU-only protocol diagnosis and read-only public access."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from r03_common import ROOT,PROJECT,R01,R02,OLD,PAPER,OUT,RUN,read,write,sha256,artifact,sources,unchanged,now,command_record


def baseline():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'r02_final_identity_lock.json').exists():print('EXISTING BASELINE RECHECK',unchanged());return
    receipt=read(R02/'verification/execution_paper_20261012T083831Z.json')
    assert receipt['actual_exit_code']==0 and sha256(ROOT/receipt['log_path'])==receipt['log_sha256']
    for name,h in receipt['source_hashes'].items():assert sha256(ROOT/name)==h
    build=read(R02/'verification/paper_build_audit.json');visual=read(R02/'verification/visual_inspection_r02.json')
    for r in build['builds']:
        stem=Path(r['entry']).stem;pdf=OLD/'build_r02'/(stem+'.pdf')
        assert sha256(pdf)==r['pdf_sha256']==visual['pdf_sha256'][stem]
        assert r['pages']==visual['pages_inspected'][stem]
        for a in r['rendered_pages']+r['contact_sheets']:assert sha256(ROOT/a['path'])==a['sha256']
    source=sources(OLD);assert len(source)==33,len(source)
    packages=read(R02/'verification/private_package_identities_r02.json')
    for a in packages['packages']:
        assert sha256(ROOT/a['path'])==a['sha256']
        with zipfile.ZipFile(ROOT/a['path']) as z:
            members={n for n in z.namelist() if Path(n).suffix in {'.tex','.bib','.png'}}
            assert members=={str(p.relative_to(OLD)) for p in source}
            for p in source:assert hashlib.sha256(z.read(str(p.relative_to(OLD)))).hexdigest()==sha256(p)
    numeric=read(R02/'verification/generated_numeric_identity.json')
    assert len(numeric['table_sha256'])==19
    for name,h in numeric['table_sha256'].items():assert sha256(OLD/'generated'/name)==h
    original=read(R01/'release_manifest.json');preserved={str((R01/name).relative_to(ROOT)):h for name,h in original['files'].items()}
    for p in [*OLD.rglob('*'),*R02.rglob('*')]:
        if p.is_file() and not p.is_symlink() and '/working/' not in str(p):preserved[str(p.relative_to(ROOT))]=sha256(p)
    for name,h in preserved.items():assert sha256(ROOT/name)==h
    write(OUT/'r02_final_identity_lock.json',{'run_id':RUN,'created_utc':now(),'baseline_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'R01_original_manifest_targets':939,'R02_paper_source_inputs':33,'R02_generated_inputs':{'total':19,'tables':18,'numeric_macro_file':1},'R02_final_receipt':artifact(R02/'verification/execution_paper_20261012T083831Z.json'),
        'R02_final_build_audit':artifact(R02/'verification/paper_build_audit.json'),'R02_final_visual':artifact(R02/'verification/visual_inspection_r02.json'),
        'R02_final_source_inputs':[artifact(p) for p in source],'preserved_file_hashes':preserved,
        'scope':'same-operator local byte recheck of late R02 receipt and original inputs, not a third party receiving/retraining the original939'})
    write(OUT/'environment_r03.json',{'run_id':RUN,'recorded_utc':now(),'python':sys.version,'executable':sys.executable,'os_release':Path('/etc/os-release').read_text(),
        'package_versions':{k:__import__('importlib.metadata',fromlist=['version']).version(k) for k in ('torch','torch-geometric','pygod','numpy','pandas','scikit-learn')},
        'GPU_training_or_replay_requested':False})
    assert not PAPER.exists(),'R03 candidate exists; never reset author edits'
    for p in source:
        dest=PAPER/p.relative_to(OLD);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(OUT/'r02_final_source_and_receipt_binding.json',{'R02_final_receipt_utc':receipt['utc_execution_stamp'],'actual_exit_code':receipt['actual_exit_code'],
        'PDFs':{Path(r['entry']).stem:r['pdf_sha256'] for r in build['builds']},'pages':[r['pages'] for r in build['builds']],
        'source_binding':'33 current R02 source files byte-identical to both preserved final private packages; final PDF/render hashes match late083831Z receipt, not080130Z',
        'R02_numeric_inputs_all_hash_match':19,'scope':'post-host Git receipts genuinely available locally; not members of pre-hosting R02 publicZIP'})
    print('R03 BASELINE/FINAL R02 BINDING PASS',len(preserved),flush=True)


def diagnose():
    """No fit, neural forward, GPU execution or historical-timing replacement."""
    import numpy as np
    import pandas as pd
    import torch
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from r01_train import load_cache,subset,degree_features
    from r01_common import config
    from gog_fraud.streaming.selective import calibrated,binary_metrics
    unchanged();cfg=config();torch.set_num_threads(1)
    graphs,metadata=subset(load_cache(cfg),cfg['chains'],'test')
    x=np.stack([degree_features(g) for g in graphs]);records=[]
    timing=pd.read_csv(R01/'runtime/control_runtime.csv');baseline_metrics=pd.read_csv(R01/'analysis/per_seed_metrics.csv')
    for seed in cfg['seeds']:
        directory=R01/'models'/f'pooled_snapshot_GIN_seed{seed}'
        weights=torch.load(directory/'controls.pt',map_location='cpu',weights_only=False)
        raw=pd.read_parquet(directory/'degree_logistic_predictions.parquet');fit=read(directory/'degree_logistic_fit.json')
        assert np.array_equal(raw.sample_id,[m['sample_id'] for m in metadata]);y=raw.label.to_numpy()
        scaler=StandardScaler();scaler.mean_=np.asarray(weights['degree_feature_scaler_mean']);scaler.scale_=np.asarray(weights['degree_feature_scaler_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=11;scaler.n_samples_seen_=17020
        xx=scaler.transform(x).astype(np.float32)
        lr=LogisticRegression(C=1,class_weight='balanced',max_iter=1000);lr.coef_=weights['logistic_coef'];lr.intercept_=weights['logistic_intercept'];lr.classes_=np.array([0,1]);lr.n_features_in_=11
        canonical_raw=lr.predict_proba(xx)[:,1]
        variants={'canonical_api':canonical_raw,'vector_expit':expit(xx@lr.coef_[0]+lr.intercept_[0]),'manual_full_exp':1/(1+np.exp(-(xx@lr.coef_[0]+lr.intercept_[0]))),
            'manual_batch128_exp':np.concatenate([1/(1+np.exp(-(xx[i:i+128]@lr.coef_[0]+lr.intercept_[0]))) for i in range(0,len(xx),128)])}
        canonical=calibrated(raw.raw_probability.to_numpy(),fit['calibration'])
        target=timing[(timing.seed==seed)&(timing.method=='degree_logistic')]
        scores={}
        for name,p in variants.items():
            score=calibrated(p,fit['calibration']);pred=score>=fit['threshold'];scores[name]=score
            values=binary_metrics(y,score,pred);records.append({'seed':seed,'variant':name,'raw_dtype':str(p.dtype),'raw_max_abs_diff':float(np.max(abs(p-raw.raw_probability.to_numpy()))),
                'calibrated_max_abs_diff':float(np.max(abs(score-canonical))),'prediction_disagreements':int((pred!=raw.prediction.to_numpy()).sum()),
                'unique_raw_scores':int(len(np.unique(p))),'unique_calibrated_scores':int(len(np.unique(score))),'ap':values['ap'],'f1':values['f1'],
                'canonical_recorded_ap':float(baseline_metrics[(baseline_metrics.seed==seed)&(baseline_metrics.method=='degree_logistic')].ap.iloc[0]),
                'timing_recorded_ap_mean':float(target.ap.mean()),'timing_recorded_guard_max':float(target.raw_score_max_abs_diff.max()),
                'matches_recorded_timing_ap':bool(np.allclose(values['ap'],target.ap,atol=1e-12,rtol=0))})
        # Pseudonymous keys only; these are NEW CPU diagnostic arrays, not
        # restored historical per-repeat probability vectors.
        np.savez_compressed(OUT/f'degree_logistic_cpu_seed{seed}.npz',label=y,sample_key=np.array([hashlib.sha256(str(v).encode()).hexdigest() for v in raw.sample_id]),**scores)
    from r02_common import csv_write
    csv_write(OUT/'degree_logistic_cpu_diagnostic.csv',records)
    matches=pd.read_csv(R02/'verification/achieved_budget_matches_recomputed.csv')
    assert len(matches)==90 and np.allclose(abs(matches.comparator_achieved_fraction-matches.primary_achieved_fraction),matches.achieved_gap,atol=1e-12)
    assert np.array_equal(matches.within_005_fraction,matches.achieved_gap<=.05)
    summaries=[]
    for family,g in matches[matches.primary_validation_budget==.25].groupby('comparator'):
        selected=g[g.within_005_fraction];summaries.append({'comparator':family,'matched':len(selected),'excluded':5-len(selected),
            'delta_ap_comparator_minus_margin':None if not len(selected) else float((selected.comparator_ap-selected.primary_ap).mean())})
    write(OUT/'table_protocol_verification.json',{'run_id':RUN+'_CPU_protocol_diagnostic','recorded_utc':now(),'rows':records,'budget_matching':summaries,'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (PROJECT/'scripts/r01_train.py',PROJECT/'scripts/r01_controls_timing.py',Path(__file__))},
        'N_test_contracts':len(metadata),'dataset_same':True,'GPU_calls':0,'model_fits':0,'historical_timing_vectors_available':False,
        'scope':'new CPU formula diagnostic using saved coefficients/features; explains supportable implementation differences, not recovery of unrecorded original timing vectors or new runtime measurement'})
    unchanged();print('R03 CPU PROTOCOL DIAGNOSTIC PASS',len(records),flush=True)


def access():
    from r01_public import verified_archive
    manifest=read(PROJECT/'evidence/release_manifest_r02.json');previous=read(PROJECT/'evidence/access_verification_r02.json');receipts=[];start=now()
    # Explicit public endpoints; no credential helper, token or API writes.
    tag='stream-mc-r02-20261012';prefix='https://github.com/Sam-7878/dlg_gnn/releases/download/'+tag+'/'
    with tempfile.TemporaryDirectory(prefix='streammc_r03_access_') as directory:
        dest=Path(directory)
        for a in manifest['assets']:
            name=Path(a['path']).name;url=a['download_url'];assert url==prefix+name
            h=hashlib.sha256();count=0
            request=urllib.request.Request(url,headers={'User-Agent':'StreamMC-R03-read-only-access-audit','Accept-Encoding':'identity'})
            with urllib.request.urlopen(request,timeout=30) as response,(dest/name).open('wb') as handle:
                status=response.status
                for block in iter(lambda:response.read(2**20),b''):handle.write(block);h.update(block);count+=len(block)
            assert h.hexdigest()==a['sha256'] and count==a['bytes']
            receipts.append({'name':name,'versioned_url':url,'http_status':status,'bytes':count,'sha256':h.hexdigest(),'authentication':False})
            print('R03 PUBLIC ASSET PASS',name,flush=True)
        z,identity=verified_archive(dest/'r01_numeric_evidence.zip',dest/'r01_release.json')
        with z:count_original=len(z.namelist())
        with zipfile.ZipFile(dest/'r02_verification_evidence.zip') as z:
            record=json.loads(z.read('release_manifest.json'));assert set(record['files'])==set(z.namelist())-{'release_manifest.json'}
            for name,i in record['files'].items():data=z.read(name);assert hashlib.sha256(data).hexdigest()==i['sha256'] and len(data)==i['bytes']
            payload_count=len(record['files'])
        stage=dest/'source';stage.mkdir()
        with zipfile.ZipFile(dest/'r02_scientific_source.zip') as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n and '/paper/' not in n and not n.endswith(('.tex','.bib','.pdf')) for n in z.namelist());z.extractall(stage)
        evidence=stage/'projects/stream_mc/evidence';evidence.mkdir(parents=True,exist_ok=True)
        for name in ['r01_release.json']+[f'r01_numeric_evidence.zip.part{i:03d}' for i in range(1,4)]:shutil.copy2(dest/name,evidence/name)
        commands=[command_record([sys.executable,'-m','venv',dest/'stdlib_venv'],stage,OUT/'access_fresh_venv.log')]
        assert commands[-1]['actual_exit_code']==0
        commands.append(command_record([dest/'stdlib_venv/bin/python','projects/stream_mc/scripts/r02_public.py'],stage,OUT/'access_stdlib_numeric.log'));assert commands[-1]['actual_exit_code']==0
        numeric=json.loads((stage/'projects/stream_mc/results/public_recomputed_r02/public_verification.json').read_text())
    from r02_common import csv_write
    csv_write(OUT/'release_asset_identities.csv',receipts)
    write(OUT/'public_access_final.json',{'status':'VERIFIED','run_id':RUN,'started_utc':start,'finished_utc':now(),'downloads':receipts,'R01_assembled_bytes':258850451,
        'R01_assembled_sha256':read(PROJECT/'evidence/r01_release.json')['archive_sha256'],
        'R01_public_members':count_original,'R02_verified_payload_members':payload_count,'public_numeric':numeric,'new_commands':commands,
        'previous_full_policy6065_MC40_receipt':artifact(PROJECT/'evidence/access_verification_r02.json'),'previous_full_policy_reused_not_rerun':True,
        'release_source_commit':manifest['target_source_commit'],'post_host_receipts_not_originalZIP_members':True,'external_writes':0,
        'scope':'real new unauthenticated full downloads/payload hashes and fresh stdlib source-path execution; prior full-policy6065 verification explicitly reused, no external human peer review'})
    unchanged();print('R03 PUBLIC ACCESS PASS',len(receipts),count_original,payload_count,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['baseline','diagnose','access']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    begin=now()
    try:globals()[args.stage]()
    except Exception as exc:
        write(OUT/(args.stage+'_failure_'+begin.replace(':','-')+'.json'),{'stage':args.stage,'status':'FAILED','recorded_utc':now(),'error_type':type(exc).__name__,'message':str(exc)[:300],'script_sha256':sha256(__file__)});raise
    write(OUT/(args.stage+'_execution.json'),{'stage':args.stage,'status':'PASS','started_utc':begin,'finished_utc':now(),'actual_exit_code':0,'script':artifact(__file__),'run_id':RUN})


if __name__=='__main__':main()
