#!/usr/bin/env python3
"""Fail-closed end-to-end A08 checker; public integrity is a separate scope."""
import csv,json,sys,re,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.crypto_raw import sha256,content_hash,array_hash
from gog_fraud.data.benchmark_metrics import evaluate_scores
from gog_fraud.data.benchmark_lineage import verify_run
E=ROOT/'projects/benchmark/evidence/a08_data_repair';L=ROOT/'local_storage/benchmark/a08_data_repair';R=ROOT/'projects/benchmark/reports/a08_data_repair'


def require(value,message):
    if not value:raise ValueError(message)


def bound_document(root,gate,field,relative):
    path=Path(root)/relative
    require(field in gate and path.is_file(), 'missing required binding: '+field)
    require(sha256(path)==gate[field], 'gate artifact binding mismatch: '+field)


def check_gate_bindings(root,objects):
    base='projects/benchmark/evidence/a08_data_repair/'
    report='projects/benchmark/reports/a08_data_repair/'
    entries={
        0:{'inventory_sha256':base+'inventory.json','impact_map_sha256':report+'impact_map.csv'},
        1:{'feature_spec_sha256':base+'feature_spec.yaml','relation_spec_sha256':base+'relation_spec.yaml','amendment_sha256':'projects/benchmark/protocols/A08_DATA_REPAIR_AMENDMENT.md'},
        2:{'regression_sha256':base+'audit/regression_tests.json','actual_three_chain_contract_sha256':base+'audit/counterfactual_and_rebuild.json'},
        3:{'counterfactual_rebuild_sha256':base+'audit/counterfactual_and_rebuild.json','whole_raw_five_condition_sha256':base+'audit/raw_full_counterfactual.json'},
        4:{'input_manifest_sha256':base+'input_manifest.json','split_manifest_sha256':base+'split_manifest.json','planned_cells_sha256':base+'planned_cells.csv','model_settings_sha256':base+'model_settings_frozen.json','source_manifest_sha256':base+'source_manifest.json','config_sha256':'configs/benchmark/a08_crypto_clean_v1.yaml','execution_manifest_sha256':base+'execution_manifest.json'},
        5:{'cuda_exactness_sha256':base+'audit/cuda_exactness.json'},
        7:{'approved_registry_sha256':base+'approved_registry.json','number_registry_sha256':base+'number_registry.json','recompute_audit_sha256':base+'audit/metric_recompute_audit.json','statistics_sha256':base+'statistics/statistics_s1_s4.json','historical_reuse_sha256':base+'audit/historical_reuse.json','aggregate_source_sha256':'projects/benchmark/scripts/a08_aggregate.py'},
        8:{'paper_build_manifest_sha256':'local_storage/benchmark/a08_data_repair/paper_build_manifest.json'},
        9:{'pdf_review_sha256':base+'audit/pdf_review.json','paper_number_crosswalk_sha256':base+'paper_number_crosswalk.csv','paper_number_registry_sha256':base+'paper_number_registry.csv','claim_evidence_map_sha256':base+'claim_evidence_map.csv'},
        10:{'clean_export_verify_sha256':base+'audit/clean_export_verify_A08.json','public_private_boundary_sha256':base+'audit/public_private_boundary.json','final_checker_tests_sha256':base+'audit/final_checker_tests.json','public_payload_manifest_sha256':base+'public_payload_manifest.json','public_release_identity_sha256':'projects/benchmark/evidence/a08_public_release.json'}
    }
    for number,bindings in entries.items():
        for field,relative in bindings.items():bound_document(root,objects[number],field,relative)
    for number,key in [(5,'actual_chain_integration'),(6,'records')]:
        entries=objects[number].get(key,[])
        require(len(entries)==(7 if number==5 else 105),'gate lacks actual model/run coverage')
        for row in entries:bound_document(root,row,'sha256',row['path'])
    amendment=Path(root)/'projects/benchmark/protocols/A08_NUMERICAL_REPAIR_AMENDMENT_2026-10-10.md'
    if amendment.is_file():
        bound_document(root,objects[4],'numerical_amendment_sha256',str(amendment.relative_to(root)))
        bound_document(root,objects[5],'gadnr_numerical_qualification_sha256',base+'audit/gadnr_numerical_qualification.json')
        row=objects[5]['failed_chain_integration'];bound_document(root,row,'sha256',row['path'])


def check_pdf(root,p):
    file=Path(root)/p['path']
    require(file.is_file() and sha256(file)==p['sha256'],'old/missing PDF')
    info=subprocess.check_output(['pdfinfo',str(file)],text=True)
    match=re.search(r'^Pages:\s+(\d+)',info,re.M)
    require(match is not None and int(match.group(1))==p['pages'],'actual PDF page count mismatch')
    pages=p.get('page_reviews',[])
    require(p['pages']>0 and len(pages)==p['pages'],'empty/incomplete visual review')
    require({int(x['page']) for x in pages}==set(range(1,p['pages']+1)) and all(x['status']=='PASS' for x in pages),'page review not complete')
    for page in pages:
        bound_document(root,page,'render_sha256',page['render_path'])
        require(bool(page.get('observations')),'page has no manual review observations')
    return subprocess.check_output(['pdftotext',str(file),'-'],text=True)


def check_claim_map(root,current):
    root=Path(root);e=root/'projects/benchmark/evidence/a08_data_repair'
    rows=list(csv.DictReader((e/'claim_evidence_map.csv').open()))
    categories={'input_provenance','threshold_and_metric_policy','support_and_run_counts','statistical_scope','resource_claims','old_new_population_comparison','public_private_availability','historical_diagnostics','citations_and_equations'}
    require(rows and {r['artifact_path'] for r in rows}==set(current),'claim mapping missing reviewed artifact')
    for artifact,pdf_hash in current.items():
        subset=[r for r in rows if r['artifact_path']==artifact]
        require({r['claim_id'] for r in subset}==categories,'claim mapping missing required semantic category')
        for row in subset:
            require(row['pdf_hash']==pdf_hash and row['check_status']=='PASS','claim mapping has old/unreviewed PDF')
            require(bool(row['page_or_table']) and bool(row['rendered_value_or_claim']),'claim mapping missing actual location/observation')
            name=Path(row['source_metric_or_table_key'])
            require(not name.is_absolute() and '..' not in name.parts,'unsafe claim source path')
            bound_document(root,row,'source_sha256',name)


def check_raw_rebuilds(root):
    """Check actual fifteen local observable artifacts, beyond report PASS strings."""
    root=Path(root);e=root/'projects/benchmark/evidence/a08_data_repair';l=root/'local_storage/benchmark/a08_data_repair'
    proof=json.loads((e/'audit/raw_full_counterfactual.json').read_text())
    require(proof['status']=='PASS' and len(proof['results'])==3,'raw counterfactual chain coverage missing')
    require(proof['script_sha256']==sha256(root/'projects/benchmark/scripts/a08_full_counterfactual.py')
            and proof['primitive_source_sha256']==sha256(root/'src/gog_fraud/data/crypto_raw.py'),'raw counterfactual source changed')
    require({r['chain'] for r in proof['results']}=={'polygon','bsc','ethereum'},'raw proof has wrong chains')
    for result in proof['results']:
        conditions=result['raw_to_relation_conditions']
        require(len(conditions)==5 and {c['condition'] for c in conditions}=={'original','permuted','all_zero','all_one','unavailable'},'five whole raw conditions missing')
        chain=result['chain'];reference=None
        with np.load(l/'frozen'/chain/'contract_graph.npz',allow_pickle=False) as f:
            frozen={k:f[k] for k in f.files}
        for condition in conditions:
            path=l/condition['workspace']/chain
            manifest=json.loads((path/'observables_manifest.json').read_text())
            require(manifest['labels_accessed'] is False and not condition['existing_graph_as_input'],'label/cache entered observable builder')
            require(sha256(path/'observables.npz')==manifest['serialized_file_sha256'],'actual raw observable bytes changed')
            require(sha256(path/'raw_members.json')==manifest['raw_member_manifest_sha256'],'raw member lineage changed')
            with np.load(path/'observables.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
            h=content_hash(a)
            require(h==manifest['scientific_content_hash']==condition['observable_content_hash'],'raw observable proof does not match actual arrays')
            require(reference is None or h==reference,'counterfactual changed actual observable content');reference=h
            for name,value in a.items():
                require({'shape':list(value.shape),'dtype':value.dtype.str,'sha256':array_hash(value)}==manifest['arrays'][name],'raw array manifest mismatch')
            for name in ('x','edge_index','node_ids'):
                require(np.array_equal(a[name],frozen[name]),'frozen training input differs from raw rebuild')
            require(condition['raw_zip_sha256']==manifest['source_zip_sha256'],'raw ZIP lineage mismatch')
            labels={'original':frozen['labels'],'permuted':np.random.RandomState(20261009).permutation(frozen['labels']),
                    'all_zero':np.zeros_like(frozen['labels']),'all_one':np.ones_like(frozen['labels'])}
            expected='UNAVAILABLE' if condition['condition']=='unavailable' else array_hash(labels[condition['condition']])
            require(condition['target_hash']==expected,'counterfactual target hash mismatch')


def resolve_crosswalk_number(root,row):
    """Resolve an actual display back to approved source values; arbitrary keys fail."""
    e=Path(root)/'projects/benchmark/evidence/a08_data_repair'
    source=row['source'];key=row['source_key'];parts=key.split('/')
    if source=='number_registry.json':
        obj=json.loads((e/source).read_text())
        if len(parts)==1:
            value=str(obj[key])
        elif len(parts)==3:
            d,m,metric=parts
            if m=='DLG-Aug-minus-DLG-Base':
                cells={(c['dataset'],c['model'],c['metric']):c for c in obj['cells']}
                value=f"{cells[d,'DLG-Aug',metric]['mean']-cells[d,'DLG-Base',metric]['mean']:+.4f}"
            else:
                cell=next(c for c in obj['cells'] if (c['dataset'],c['model'],c['metric'])==(d,m,metric))
                if row.get('allowed_rounding')=='4 decimals mean only':value=f"{cell['mean']:.4f}"
                else:value=f"{cell['mean']:.4f} ± {cell['sd']:.4f}"
                require(set(filter(None,row.get('run_ids','').split(';')))==set(cell['run_ids']),'crosswalk seed/run membership differs')
        else:raise ValueError('invalid numeric cell source key')
    elif source=='input_manifest.json':
        obj=next(m for m in json.loads((e/source).read_text())['datasets'] if m['dataset_id']==parts[0])
        value=f"{100*obj['positive_count']/obj['N']:.2f}%" if parts[1]=='positive_count divided by N' else str(obj[parts[1]])
    elif source=='statistics/statistics_s1_s4.json':
        obj=next(m for m in json.loads((e/source).read_text()) if m['view']==parts[0]);number=obj[parts[1]]
        value=str(number) if parts[1]=='n' else f"{number:.6f}" if parts[1]=='permutation_p' else f"{number:.4f}"
    elif source=='tables/alert_budget_crypto.csv':
        obj=next(m for m in csv.DictReader((e/source).open()) if [m['dataset'],m['model'],m['budget'],m['metric']]==parts)
        value=f"{float(obj['mean']):.4f} ± {float(obj['sd']):.4f}"
    elif source=='statistics/pairwise.csv':
        obj=next(m for m in csv.DictReader((e/source).open()) if [m['view'],m['comparison']]==parts[:2])
        require(parts[2] in {'raw_p','holm_p','wins','ties','losses'},'unknown pairwise field')
        value=str(int(obj[parts[2]])) if parts[2] in {'wins','ties','losses'} else f"{float(obj[parts[2]]):.5f}"
    elif source in {'statistics/mean_ranks.csv','statistics/label_provenance_sensitivity.csv'}:
        require(len(parts)==3 and parts[2]=='mean_rank','invalid mean-rank key')
        identity='view' if source=='statistics/mean_ranks.csv' else 'group'
        obj=next(m for m in csv.DictReader((e/source).open()) if [m[identity],m['model']]==parts[:2])
        value=f"{float(obj['mean_rank']):.4f}"
    elif source=='run_manifests':
        chain,model,metric=parts
        records=[json.loads((e/source/f'{chain}_contract_clean_v1__{model}__seed{s}.json').read_text()) for s in range(42,47)]
        require(set(filter(None,row.get('run_ids','').split(';')))=={r['run_id'] for r in records},'resource crosswalk run membership differs')
        value=f"{sum(r['wall_time_seconds'] for r in records)/5:.1f}" if metric=='mean_seconds' else f"{max(r['peak_allocated'] for r in records)/2**30:.3f}" if metric=='max_allocated_gib' else None
        require(value is not None,'unknown resource metric')
    else:raise ValueError('unknown numeric crosswalk source: '+source)
    require(value==row['rendered'],'PDF number differs from approved source value: '+row['claim_id'])
    return value


def check_loss_budget(record,curve,n):
    samples=curve.get('samples',[])
    require(samples and all(np.isfinite(sample['loss']) for sample in samples),'missing/nonfinite actual loss observations')
    count=sum(sample['phase']=='training' for sample in samples)
    epochs=record['planned_global_epochs']
    if curve['granularity']=='epoch':expected=epochs
    elif curve['granularity']=='forward_model call (minibatch)':
        batch=record['resolved_model_config']['scalar_attributes']['batch_size']
        require(batch>0,'invalid sampled batch budget')
        expected=epochs*((n+batch-1)//batch)
    else:raise ValueError('unknown actual loss observation granularity')
    require(count==expected,'actual training loss calls do not cover declared epoch budget')


def validate_final(root=ROOT):
    e=Path(root)/'projects/benchmark/evidence/a08_data_repair';l=Path(root)/'local_storage/benchmark/a08_data_repair';r=Path(root)/'projects/benchmark/reports/a08_data_repair'
    gates=[];objects=[]
    for i in range(11):
        p=e/'audit'/f'G{i}.json';require(p.is_file(),f'G{i} missing')
        obj=json.loads(p.read_text());require(obj.get('status')=='PASS' and not obj.get('is_template'),f'G{i} failed/template/skipped')
        gates.append({'id':f'G{i}','sha256':sha256(p)})
        objects.append(obj)
    check_gate_bindings(root,objects)
    check_raw_rebuilds(root)
    config=json.loads((Path(root)/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text())
    inputs=json.loads((e/'input_manifest.json').read_text());require(len(inputs['datasets'])==3,'three current inputs required')
    denied={x['sha256'] for x in json.loads((e/'old_input_denylist.json').read_text())}
    records=json.loads((e/'approved_registry.json').read_text())['records'];require(records,'empty registry')
    plan=list(csv.DictReader((e/'planned_cells.csv').open()));require(len(plan)==105,'planned scope must account for105 combinations')
    expected={(p['dataset_id'],p['model'],int(p['seed'])) for p in plan};require(len(expected)==105,'duplicate planned cell')
    seen=set();hashes={}
    for m in inputs['datasets']:
        chain=m['dataset_id'].split('_')[0];path=l/'frozen'/chain/'contract_graph.npz'
        require(m['source_manifest_hash']==sha256(l/'build_a'/chain/'observables_manifest.json'),'frozen construction manifest binding differs')
        require(m['feature_spec_hash']==sha256(e/'feature_spec.yaml') and m['relation_spec_hash']==sha256(e/'relation_spec.yaml'),'frozen feature/relation specification differs')
        require(m['serialized_file_sha256'] not in denied and m['scientific_content_hash'] not in denied,'old input relabeled')
        require(sha256(path)==m['serialized_file_sha256'],'input bytes mismatch')
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        require(content_hash(a)==m['scientific_content_hash'],'input content mismatch')
        require(a['labels'].shape==(m['N'],) and len(set(a['node_ids'].tolist()))==m['N'],'node target/ID mismatch')
        require(a['x'].shape==(m['N'],m['F']) and a['edge_index'].shape==(2,m['E']),'new E/F mismatch')
        hashes[m['dataset_id']]=sha256(l/'frozen'/chain/'input_manifest.json')
    for record in records:
        if record.get('evidence_tier')!='a08_raw_score_recomputable':continue
        key=(record['dataset_version'],record['model'],int(record['seed']))
        require(key in expected and key not in seen,'missing/duplicate/unknown seed identity');seen.add(key)
        require(record['input_manifest_hash']==hashes[key[0]],'new metadata old score identity')
        out=l/'runs'/record['run_id'];run=json.loads((out/'run_manifest.json').read_text())
        require(run['status']=='SUPPORTED_EXACT','unresolved runtime/data/numerical failure')
        verify_run(root,config,run)
        require(run['scope']=='primary' and run['input_manifest_hash']==record['input_manifest_hash'] and run['split_hash']==record['split_hash'],'qualification/old input mixed into primary')
        require(sha256(out/'scores.npz')==record['raw_score_hash']==run['raw_scores_sha256'],'raw scores absent or corrupted')
        require(sha256(out/'final_model_state.pt')==run['checkpoint_sha256'],'checkpoint binding invalid')
        with np.load(out/'scores.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        check_loss_budget(run,json.loads((out/'loss_curve.json').read_text()),len(a['scores']))
        metrics=evaluate_scores(a['labels'],a['scores'],a['node_ids'],a['val_mask'],a['test_mask'])
        require(metrics==json.loads((out/'metrics.json').read_text()),'raw metric recomputation mismatch')
        for field,value in [('pr_auc',metrics['ap']),('roc_auc',metrics['roc_auc']),('validation_f1',metrics['thresholded']['f1'] if metrics['thresholded'] else None)]:
            require(record.get(field)==value,'registry metric differs from recomputed raw score: '+field)
        execution=json.loads((e/'execution_manifest.json').read_text())
        require(run['code_hashes']==execution['source_hashes'],'run source map differs from frozen execution')
        chain=key[0].split('_')[0]
        with np.load(l/'frozen'/chain/'contract_graph.npz',allow_pickle=False) as frozen:
            require(run['original_model_input']=={name:array_hash(frozen[name]) for name in ('x','edge_index')},'model input signature differs from frozen arrays')
        if record['model']=='DLG-Aug':
            with np.load(out/'augmented_input.npz',allow_pickle=False) as aug:
                require(array_hash(aug['x'])==run['actual_augmented_input_hash'],'augmentation input hash mismatch')
                require(aug['local_score'].shape==(len(a['scores']),) and np.isfinite(aug['local_score']).all(),'invalid local stage diagnostic')
    require(seen==expected,'current five-seed coverage missing')
    numbers=json.loads((e/'number_registry.json').read_text())
    for cell in numbers['cells']:
        subset=[q for q in records if q['dataset']==cell['dataset'] and q['model']==cell['model'] and q['status']=='success']
        require(len(subset)==5 and {int(q['seed']) for q in subset}==set(range(42,47)),'numeric table has incomplete/duplicate seed set')
        values=[float(q[cell['metric']]) for q in subset]
        require(cell['mean']==float(np.mean(values)) and cell['sd']==float(np.std(values,ddof=0)),'numeric table mean/SD mismatch')
        require(set(cell['run_ids'])=={q['run_id'] for q in subset},'numeric table run membership mismatch')
    review=json.loads((e/'audit/pdf_review.json').read_text());require(review.get('semantic_status')=='PASS' and review.get('layout_status')=='PASS','PDF semantic/layout checks not passed')
    pdfs=review.get('artifacts',[]);require(len(pdfs)>=2,'actual neutral and MDPI required')
    current={};texts={}
    for p in pdfs:
        texts[p['path']]=check_pdf(root,p)
        current[p['path']]=p['sha256']
    cross=list(csv.DictReader((e/'paper_number_crosswalk.csv').open()));require(len(cross)>20,'empty number crosswalk')
    require(all(row['artifact_path'] in current and row['pdf_hash']==current[row['artifact_path']] and row['check_status']=='PASS' for row in cross),'PDF registry crosswalk mismatch')
    for row in cross:
        require(bool(row.get('claim_id')) and bool(row.get('source_key')),'crosswalk has no numeric source identity')
        resolve_crosswalk_number(root,row)
        rendered=''.join(row['rendered'].split());text=''.join(texts[row['artifact_path']].split())
        require(rendered in text,'crosswalk number absent from actual PDF: '+row['claim_id'])
    declared=list(csv.DictReader((e/'paper_number_registry.csv').open()))
    declared_keys={(row['claim_id'],row['source'],row['source_key'],row['rendered']) for row in declared}
    require(len(declared_keys)==len(declared)>20,'empty/duplicate manuscript numeric expected list')
    for artifact in current:
        actual={(row['claim_id'],row['source'],row['source_key'],row['rendered']) for row in cross if row['artifact_path']==artifact}
        require(actual==declared_keys,'PDF numeric audit does not cover all current registered displays')
    check_claim_map(root, current)
    export=json.loads((e/'audit/clean_export_verify_A08.json').read_text())
    require(export['status']=='PASS' and export['numeric_replay_identical'] and export['reviewed_build_identity_unchanged'],'public clean export/local paper failed/skipped')
    payload=json.loads((e/'public_payload_manifest.json').read_text());require(payload['file_count']==len(payload['files'])>0,'empty public export expected list')
    checkout=Path(export['checkout']);require(checkout.is_dir(),'actual clean export absent')
    for entry in payload['files']:
        path=checkout/entry['path']
        require(path.is_file() and path.stat().st_size==entry['bytes'] and sha256(path)==entry['sha256'],'actual clean export bytes differ: '+entry['path'])
    require(export['public_payload_manifest_sha256']==sha256(e/'public_payload_manifest.json'),'clean export payload manifest changed')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('a08_public_checker',Path(root)/'projects/benchmark/scripts/a08_public_evidence.py')
    public=module_from_spec(spec);spec.loader.exec_module(public)
    public.ROOT=Path(root);public.E=e;public.BUNDLE=Path(root)/'projects/benchmark/evidence/a08_public_numeric_evidence.zip';public.OUTER=Path(root)/'projects/benchmark/evidence/a08_public_release.json'
    require(public.verify()['status']=='PASS','current public numeric bundle failed')
    privacy=json.loads((e/'audit/public_private_boundary.json').read_text());require(privacy['status']=='PASS' and privacy['private_payload_matches']==[],'private manuscript exposed')
    return {'gates':gates,'current_pdf_hashes':current,'primary_planned_runs':len(expected),'scope':'current A08 scientific/package verification; author approval/publication separate'}


def main():
    try:detail=validate_final();status='FINAL_PASS';blockers=[]
    except Exception as err:
        detail={'checker_error_type':type(err).__name__};status='NOT_PASSED';blockers=[str(err)]
    R.mkdir(parents=True,exist_ok=True)
    (R/'FINAL_ACCEPTANCE_A08.json').write_text(json.dumps({'schema_version':1,'required_gate_ids':[f'G{i}' for i in range(11)],'final_scientific_status':status,'submission_readiness':'NOT_CLEARED',
        'public_release_status':'AWAITING_AUTHOR_AUTHORIZATION','blockers':blockers,'detail':detail},indent=2)+'\n')
    print(status,blockers);return 0 if status=='FINAL_PASS' else 1
if __name__=='__main__':raise SystemExit(main())
