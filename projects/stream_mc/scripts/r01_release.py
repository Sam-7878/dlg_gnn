"""Build a curated, address-free numeric/input archive; never package paper prose."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from r01_common import ROOT,PROJECT,CONFIG,config,output,read,write,sha256,git_sha


def archive_write(archive,name,data):
    item=zipfile.ZipInfo(name,(2026,10,11,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
    item.external_attr=0o100644<<16;archive.writestr(item,data)


def input_kit(cfg,out):
    import numpy as np
    import torch
    import pandas as pd
    cache=torch.load(ROOT/cfg['graph_cache'],map_location='cpu',weights_only=False)
    ids=sorted(m['sample_id'] for values in cache['metadata'].values() for m in values)
    ranks={value:i for i,value in enumerate(ids)};aliases={};meta=[];edges=[];offsets=[0];nodes=[]
    for split in ('train','validation','test'):
        for graph,m in zip(cache['graphs'][split],cache['metadata'][split]):
            contract_alias=f"contract_{ranks[m['sample_id']]:08d}"
            alias=f"{m['chain']}:{contract_alias}:{m['event_end']}";aliases[m['sample_id']]=alias
            meta.append({'split_id':split,'chain':m['chain'],'sample_key':hashlib.sha256(m['sample_id'].encode()).hexdigest(),
                'reference_alias':alias,'contract_alias':contract_alias,'event_start':m['event_start'],'snapshot_cutoff':m['event_end'],'label':m['label'],
                'label_source':'retained snapshot, cross-checked against provider labels.csv','label_available_time':'unknown','feature_max_event_time':'unknown'})
            edges.append(graph.edge_index.numpy().astype(np.int16));offsets.append(offsets[-1]+graph.num_edges);nodes.append(graph.num_nodes)
    dest=out/'public_inputs';dest.mkdir(exist_ok=True)
    pd.DataFrame(meta).to_csv(dest/'snapshot_manifest.csv',index=False)
    np.savez_compressed(dest/'bounded_edges.npz',edges=np.concatenate(edges,axis=1),edge_offsets=np.array(offsets,np.int64),node_counts=np.array(nodes,np.int16))
    contract_aliases={':'.join(key.split(':')[:2]):':'.join(value.split(':')[:2]) for key,value in aliases.items()}
    events=pd.read_parquet(ROOT/cfg['raw_events']).sort_values(['event_time','block_number','transaction_index','sample_id'],kind='stable').reset_index(drop=True)
    event_contracts=sorted({f'{r.chain_id}:{r.contract_id}' for r in events.itertuples(index=False)})
    for i,key in enumerate(event_contracts):
        if key not in contract_aliases:contract_aliases[key]=key.split(':')[0]+f':event_contract_{i:08d}'
    event_rows=[]
    for i,r in enumerate(events.itertuples(index=False)):
        event_rows.append({'sequence_id':i,'event_id':hashlib.sha256(r.sample_id.encode()).hexdigest(),'chain':r.chain_id,
            'contract_id':contract_aliases[f'{r.chain_id}:{r.contract_id}'].split(':')[1],
            'timestamp':int(r.event_time),'source':hashlib.sha256(r.src.encode()).hexdigest(),'target':hashlib.sha256(r.dst.encode()).hexdigest(),'label':int(r.label)})
    pd.DataFrame(event_rows).to_csv(dest/'events.csv',index=False)
    # The input export is checked elementwise, not declared equivalent by construction alone.
    with np.load(dest/'bounded_edges.npz',allow_pickle=False) as recovered:
        i=0
        for split in ('train','validation','test'):
            for graph in cache['graphs'][split]:
                assert np.array_equal(graph.edge_index.numpy(),recovered['edges'][:,recovered['edge_offsets'][i]:recovered['edge_offsets'][i+1]])
                assert graph.num_nodes==recovered['node_counts'][i];i+=1
    assert [aliases[i] for i in ids]==sorted(aliases.values()),'alias tie ordering changed'
    write(dest/'transform.json',{'source_cache_sha256':sha256(ROOT/cfg['graph_cache']),'N_snapshots':len(meta),'edge_and_node_roundtrip':'exact elementwise',
        'split_and_graph_order':'preserved','reference_tie_order':'global lexicographic original sample_id rank, zero-padded within each chain; verified order preserved',
        'sample_key':'SHA-256 of original sample_id, linkable pseudonym, not a guarantee of anonymity','removed':'contract/account addresses, original graph IDs and stale source paths',
        'feature_reconstruction':'log1p directed in/out/total degree, identical float32 computation to frozen loader','unknown':'edge event times and label-availability times',
        'event_input':'100000 retained events in original chronological/tie order, normalized event IDs, stable contract aliases and SHA256 account aliases; equality/order preserved, not anonymous',
        'event_source_sha256':sha256(ROOT/cfg['raw_events']),
        'scope':'retained bounded snapshots, not a newly verified raw-event-to-snapshot construction',
        'dataset_credit':'Bingqiao Luo, Zhen Zhang, Qian Wang, Bingsheng He; Multi-Chain Graphs of Graphs, NeurIPS 2024',
        'provider':'https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs','license':'CC BY-NC-SA 4.0',
        'license_url':'https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs/blob/main/LICENSE','access_date':'2026-10-11'})
    return dest,aliases


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs-only',action='store_true');args=parser.parse_args()
    cfg=config();out=output();inputs,aliases=input_kit(cfg,out)
    if args.inputs_only:return
    import numpy as np
    import pandas as pd
    import torch
    evidence=PROJECT/'evidence';evidence.mkdir(exist_ok=True);payload={}
    def include(path,name):payload[name]=Path(path).read_bytes()
    for path in inputs.iterdir():include(path,'inputs/'+path.name)
    for folder in ('analysis','generated','stress','audits','runtime'):
        for path in sorted((out/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.csv','.json','.jsonl','.npz','.png','.xml'):
                include(path,str(path.relative_to(out)))
    for name in ('environment.json','environment.lock','execution_plan.json','evidence_inventory.csv','data_audit.json','split_overlap.json','split_support.csv','training_source_manifest.json','figure_table_provenance.csv','generated_artifacts.json'):
        include(out/name,name)
    include(out/'training_sources.zip','training_sources.zip')
    include(out/'runtime_sources_v1.tar','runtime_sources_v1.tar')
    include(out/'runtime_sources_v2.tar','runtime_sources_v2.tar')
    for path in sorted(out.glob('execution_*.json')):include(path,path.name)
    for path in sorted((out/'execution_logs').glob('*.log')):
        # Campaign console contains commands/numeric progress, not paper source.
        include(path,'execution_logs/'+path.name)
    include(CONFIG,'configs/r01.json')
    for path in sorted((PROJECT/'configs').glob('*.yaml')):include(path,'configs/'+path.name)
    include(PROJECT/'configs/author_declarations_r01.json','configs/author_declarations_r01.json')
    include(PROJECT/'requirements-parquet.txt','environment/requirements-parquet.txt')
    tex_version=subprocess.check_output(['pdflatex','--version'],text=True)
    bib_version=subprocess.check_output(['bibtex','--version'],text=True)
    payload['environment/tex_versions.txt']=(tex_version+'\n'+bib_version).encode()
    parameter_rows=[]
    for model_dir in sorted(p.parent for p in (out/'models').glob('*/complete.json')):
        record=read(model_dir/'complete.json');name=model_dir.name
        for stage,key in [('local','local_training'),('relational','relational_training')]:
            parameter_rows.append({'model_id':name,'stage':stage,'parameters':record[key]['parameter_count'],'training_seconds':record[key]['training_seconds'],'training_time_scope':'frozen actual training loop, includes host iteration'})
        for path in model_dir.glob('*.json'):include(path,'models/'+name+'/'+path.name)
        for path in model_dir.glob('*.pt'):include(path,'models/'+name+'/'+path.name)
        # Numeric reference tensors are unchanged; aliases intentionally change the file identity.
        with np.load(model_dir/'reference.npz',allow_pickle=False) as refs:
            buffer=io.BytesIO();np.savez_compressed(buffer,embedding=refs['embedding'],score=refs['score'],cutoffs=refs['cutoffs'],ids=np.array([aliases[i] for i in refs['ids']]))
        payload['models/'+name+'/reference_alias.npz']=buffer.getvalue()
        payload['models/'+name+'/reference_transform.json']=json.dumps({'original_reference_sha256':sha256(model_dir/'reference.npz'),
            'alias_reference_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),'numeric_arrays_unchanged':True,'identity':'derivative pseudonymous reference artifact; not the original file hash'},sort_keys=True).encode()
        for pattern in ('*features.npz','*_predictions.parquet','mc_T*_test.parquet','mc_T*_validation.parquet'):
            for path in sorted(model_dir.glob(pattern)):
                if path.suffix=='.npz':include(path,'models/'+name+'/'+path.name)
                else:
                    frame=pd.read_parquet(path)
                    for column in ('sample_id','contract_id'):
                        if column in frame:
                            frame[column]=frame[column].map(lambda i:hashlib.sha256(str(i).encode()).hexdigest())
                    payload['models/'+name+'/'+path.stem+'.csv']=frame.to_csv(index=False).encode()
        if (model_dir/'controls.pt').exists():
            control=torch.load(model_dir/'controls.pt',map_location='cpu',weights_only=False)
            for key in ('degree_mlp','relational_mlp'):
                parameter_rows.append({'model_id':name,'stage':key,'parameters':sum(t.numel() for t in control[key].values()),'training_seconds':None,'training_time_scope':'not recorded separately; no fabricated training cost'})
            parameter_rows.append({'model_id':name,'stage':'degree_logistic','parameters':control['logistic_coef'].size+control['logistic_intercept'].size,'training_seconds':None,'training_time_scope':'not recorded separately'})
    pd.DataFrame(parameter_rows).to_csv(out/'model_training_cost.csv',index=False);include(out/'model_training_cost.csv','model_training_cost.csv')
    # Historical discordance remains independently recomputable from aligned pseudonymous records.
    for seed in cfg['seeds']:
        source=ROOT/cfg['historical_results']/f'offline/seed{seed}/test_predictions.csv';frame=pd.read_csv(source)
        frame['sample_id']=frame['sample_id'].map(lambda s:hashlib.sha256(s.encode()).hexdigest())
        payload[f'historical_r4/seed{seed}_predictions.csv']=frame.to_csv(index=False).encode()
    runners=[p for p in (PROJECT/'scripts').glob('r01_*.py') if p.name!='r01_build_paper.py']
    code_paths=runners+list((ROOT/'src/gog_fraud/streaming').glob('*.py'))+list((ROOT/'src/gog_fraud/data/io').glob('*.py'))+list((ROOT/'src/gog_fraud/data/level2').glob('*.py'))+list((ROOT/'tests/stream_mc').rglob('*.py'))+[ROOT/'src/gog_fraud/__init__.py']
    code_hashes={str(path.relative_to(ROOT)):sha256(path) for path in code_paths}
    for path in code_paths:include(path,'code/'+str(path.relative_to(ROOT)))
    for path in PROJECT.glob('*.md'):include(path,'documentation/'+path.name)
    for path in (PROJECT/'protocols').glob('*'):
        if path.is_file():include(path,'protocols/'+path.name)
    manifest={'run_id':cfg['run_id'],'revision_id':cfg['revision_id'],'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in sorted(payload.items())},
        'source_commit':{'baseline':read(out/'execution_plan.json')['scientific_baseline_commit'],'at_packaging':git_sha(),'executed_training_source':'training_source_manifest.json; four exact source hashes, not an invented clean commit'},
        'config_sha256':sha256(CONFIG),'input_snapshot_sha256':sha256(out/'input_snapshot.zip'),'environment_sha256':sha256(out/'environment.lock'),
        'current_source_hashes':code_hashes,'manifest_excludes_itself':True,
        'withheld':'unsubmitted manuscript sources/PDFs and original address mappings; original provider archives accessible upstream, not mirrored',
        'public_release_status':'local versioned release candidate; no public hosted URL or submission asserted',
        'gate_P':'HOLD: exact-version author approval and independent external verification not provided','gate_J':'HOLD: same approval plus submission-time official checks',
        'limitations':['retrospective snapshots; edge and label times unknown','public alias reference hashes differ from original identities','light-control training times not separately recorded','full-process input preload is not bounded by resident state caps']}
    archive=evidence/'r01_numeric_evidence.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        for name,data in sorted(payload.items()):archive_write(bundle,name,data)
        archive_write(bundle,'release_manifest.json',(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
    write(evidence/'r01_release.json',{'run_id':cfg['run_id'],'archive_sha256':sha256(archive),'archive_bytes':archive.stat().st_size,'file_count':len(payload),'source_hashes':code_hashes,'hosted_public_release':False})
    # Local manifest checks actual original artifacts, separately from the curated derivative archive.
    files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('release_manifest.json','checksums.sha256') and '.checkpoint' not in p.name and p.suffix!='.tmp'}
    write(out/'release_manifest.json',{'run_id':cfg['run_id'],'config_sha256':sha256(CONFIG),'files':files,'public_archive_sha256':sha256(archive),'exclusions':'manifest itself, checksums sidecar, transient/opaque checkpoints; paper bytes are linked through separate build audit'})
    (out/'checksums.sha256').write_text(''.join(f'{value}  {name}\n' for name,value in files.items()))
    print('R01 RELEASE CANDIDATE',sha256(archive),len(payload),archive.stat().st_size,flush=True)


if __name__=='__main__':main()
