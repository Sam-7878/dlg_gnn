#!/usr/bin/env python3
"""Cross-file A04 publication validator. Missing evidence is a hard failure."""
from __future__ import annotations
import csv,hashlib,json,math,re,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'evaluation/benchmark/v2/paper_ready_final'
B=OUT/'publication_evidence_a04'
PRIMARY=['Elliptic','DGraphFin','BitcoinOTC','Ethereum','BSC','Polygon','Yelp-Syn','Amazon-Syn','Reddit-Syn','Flickr-Syn','Cora-Syn','CiteSeer-Syn','PubMed-Syn']
MODELS=['DOMINANT','AnomalyDAE','CoLA','GADNR','OCGNN','DLG-Base','DLG-Aug']
CONT5=['DOMINANT','CoLA','OCGNN','DLG-Base','DLG-Aug']


def rows(path):return list(csv.DictReader(path.open(newline='')))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def value(s):
    try:return float(s)
    except (ValueError,TypeError):return None


def main():
    checks=[]
    def check(id,ok,detail):checks.append({'check_id':id,'status':'PASS' if ok else 'FAIL','detail':detail})
    manifest=json.loads((OUT/'dataset_manifest_canonical.json').read_text())
    dmap={x['dataset_id']:x for x in manifest}
    check('dataset_set',len(manifest)==14 and set(dmap)==set(PRIMARY+['LANL-RedTeam']),f'{len(manifest)} rows')
    old={x['dataset']+'-Syn' if x['dataset'] in {'Yelp','Amazon','Reddit','Flickr','Cora','CiteSeer','PubMed'} else x['dataset']:x for x in json.loads((ROOT/'outputs/benchmark/sci_round5_final/manifests/data_freeze.json').read_text())['datasets']}
    mismatch=[];missing=[];inconsistent=[]
    for d,x in dmap.items():
        if x['verification_status']!='LOADER_OBSERVED':missing.append(d+':'+x['verification_status'])
        if d in old and any(x.get(k)!=old[d][k] for k in ('feature_hash','edge_hash','label_hash')):mismatch.append(d)
        n=x.get('evaluation_nodes');p=x.get('positive_evaluation_nodes');rate=x.get('evaluation_positive_rate')
        if not (isinstance(n,int) and isinstance(p,int) and 0<=p<=n and abs(p/n-rate)<1e-12):inconsistent.append(d)
        if not x.get('feature_dimension') or not x.get('constructed_tensor_sha256'):inconsistent.append(d)
    check('loader_identity',not missing and not mismatch,f'unverified={missing}; round5_hash_mismatch={mismatch}')
    artifacts_bad=[];builder_hash=sha(ROOT/'evaluation/benchmark/v2/scripts/a04_build_dataset_manifest.py')
    for d,x in dmap.items():
        path=ROOT/x['constructed_artifact_path'] if x.get('constructed_artifact_path') else None
        if (not path or not path.is_file() or sha(path)!=x['constructed_artifact_sha256']
            or not x['builder_commit'] or x['builder_source_sha256']!=builder_hash):
            artifacts_bad.append(d)
    check('constructed_artifacts',not artifacts_bad,f'missing_or_hash_bad={artifacts_bad}')
    check('population_prevalence',not inconsistent,f'inconsistent={sorted(set(inconsistent))}')
    check('native_feature_dimension',all(dmap[d]['feature_dimension']==8 for d in ('Ethereum','BSC','Polygon')),str({d:dmap[d]['feature_dimension'] for d in ('Ethereum','BSC','Polygon')}))
    check('elliptic_population',dmap['Elliptic']['graph_nodes']==46564 and dmap['Elliptic']['source_graph_nodes']==203769 and dmap['Elliptic']['positive_evaluation_nodes']==4545,str({k:dmap['Elliptic'][k] for k in ('graph_nodes','source_graph_nodes','evaluation_nodes','positive_evaluation_nodes')}))
    check('lanl_prediction_unit',dmap['LANL-RedTeam']['prediction_unit']=='destination_computer_node' and dmap['LANL-RedTeam']['positive_evaluation_nodes']==301,str({k:dmap['LANL-RedTeam'][k] for k in ('prediction_unit','positive_evaluation_nodes')}))
    registry=rows(B/'approved_run_registry.csv');keys=[(r['dataset'],r['model'],r['seed']) for r in registry]
    check('run_key_uniqueness',len(keys)==len(set(keys)),f'{len(keys)} approved rows')
    bad_hash=[];missing_fields=[];csv_only=[];no_scores=[]
    for r in registry:
        key='/'.join((r['dataset'],r['model'],r['seed']))
        metric=ROOT/r['metric_json_path'];run=B/'run_manifests'/f'{r["run_id"]}.json'
        if not metric.is_file() or sha(metric)!=r['metric_json_hash'] or not run.is_file():bad_hash.append(key)
        if r['dataset_hash']!=dmap[r['dataset']].get('constructed_tensor_sha256'):bad_hash.append(key+':dataset')
        if not r['split_hash'] or not r['model_config_hash']:missing_fields.append(key)
        if r['source_kind']=='defense_csv_row_only':csv_only.append(key)
        if r['status']=='success' and not r['raw_score_hash']:no_scores.append(key)
    check('run_identity_hashes',not bad_hash and not missing_fields,f'bad_hash={len(bad_hash)}; missing_split_or_config={len(missing_fields)}')
    round5_config=json.loads((ROOT/'outputs/benchmark/sci_round5_final/manifests/execution_hashes.json').read_text())['config_hash']
    config_unbound=[];config_mismatch=[]
    for r in registry:
        if r['status']!='success':continue
        if r['source_kind']=='round5_json' and r['source_config_hash']!=round5_config:
            config_mismatch.append(r['run_id'])
        if r['model_config_provenance_status']!='ORIGINAL_RUN_CONFIG_HASH':
            config_unbound.append(r['run_id'])
    check('model_config_originals',not config_unbound and not config_mismatch,
          f'original config absent/unbound={len(config_unbound)}; Round5 config mismatch={len(config_mismatch)}')
    a04_lock=sha(ROOT/'environment/locks/benchmark-a04-cuda.lock.txt')
    legacy_env=ROOT/'outputs/benchmark/sci_round5_final/manifests/environment_freeze.json'
    legacy_lock=ROOT/'evaluation/benchmark/v2/environment/legacy/20261002T145148Z/requirements-v1-freeze.txt'
    legacy=json.loads(legacy_env.read_text())
    env_misattributed=[];env_missing=[]
    for r in registry:
        if r['a04_qualification_environment_lock_hash']!=a04_lock:env_misattributed.append(r['run_id']+':qualification')
        if r['source_kind']=='round5_json':
            if (r['execution_environment_evidence_sha256']!=sha(legacy_env)
                or r['execution_environment_candidate_lock_hash']!=sha(legacy_lock)
                or r['environment_lock_hash']==a04_lock):
                env_misattributed.append(r['run_id']+':legacy')
        if r['status']=='success' and not r['environment_lock_hash']:env_missing.append(r['run_id'])
    check('environment_lineage_not_misattributed',not env_misattributed and legacy['pytorch']=='2.5.1+cu121'
          and 'torch==2.5.1+cu121' in legacy_lock.read_text().splitlines(),
          f'Round5 original torch={legacy["pytorch"]}; falsely attributed={len(env_misattributed)}')
    success_count=sum(r['status']=='success' for r in registry)
    check('execution_environment_lock',not env_missing,f'exact historical execution locks missing={len(env_missing)} of {success_count} successful runs')
    check('metric_json_originals',not csv_only,f'CSV-only LANL metrics={len(csv_only)}; extracted JSON is not original raw run')
    check('raw_score_hashes',not no_scores,f'missing={len(no_scores)} of {success_count} successful runs')
    support=rows(OUT/'table_support_24g.csv')
    expected_support=set(MODELS)|{'CONAD-PyGOD-1.1-reference'}
    diagnostic_rows=[x for x in support if x['model']=='CONAD-PyGOD-1.1-reference']
    check('seven_model_support',len(support)==112 and set(x['model'] for x in support)==expected_support and len(diagnostic_rows)==14 and all(x['support_status']=='DIAGNOSTIC_ONLY' for x in diagnostic_rows),f'{len(support)} rows; diagnostic={len(diagnostic_rows)}')
    cells=rows(B/'paper_metric_cells.csv');by=defaultdict(list)
    for r in registry:by[(r['dataset'],r['model'])].append(r)
    bad_cells=[]
    for cell in cells:
        ds,model,metric=cell['dataset'],cell['model'],cell['metric']
        run_values=[value(r[metric]) for r in by[(ds,model)] if r['status']=='success']
        if len(run_values)!=5 or any(v is None for v in run_values) or not math.isclose(float(cell['mean']),np.mean(run_values),abs_tol=1e-12):bad_cells.append(ds+'/'+model+'/'+metric)
        table={'pr_auc':'table_main_13_pr_auc.csv','roc_auc':'table_main_13_roc_auc.csv','validation_f1':'table_main_13_f1.csv'}[metric]
        display=next(x[model] for x in rows(OUT/table) if x['dataset']==ds)
        if not display.startswith(f'{float(cell["mean"]):.4f} ± '):bad_cells.append(ds+'/'+model+'/'+metric+':display')
    check('paper_cells_from_runs',not bad_cells,f'{len(cells)} cells; bad={bad_cells[:8]}')
    table=rows(OUT/'table_main_13_pr_auc.csv')
    check('conad_diagnostic_only',not any('CONAD' in name for name in table[0]) and all('CONAD' not in x['view'] for x in rows(OUT/'statistics_s1_s5.csv')),'no CONAD column or inferential view')
    ablation=rows(OUT/'table_dlg_ablation_elliptic_summary.csv');raw=rows(ROOT/'evaluation/benchmark/v2/paper_ready_a03/table_dlg_ablation_elliptic.csv')
    abbad=[]
    for a in ablation:
        group=[x for x in raw if x['variant']==a['variant']]
        if len(group)!=5 or abs(np.mean([float(x['pr_auc']) for x in group])-float(a['mean_pr_auc']))>1e-12:abbad.append(a['variant'])
    check('elliptic_25_row_aggregate',len(raw)==25 and not abbad,f'rows={len(raw)}; bad={abbad}')
    gate=rows(OUT/'table_dlg_gate_summary.csv');graw=rows(ROOT/'evaluation/benchmark/v2/diagnostics/dlg_base_dominant/dlg_gate_10runs.csv')
    gbad=[]
    for g in gate:
        group=[x for x in graw if x['dataset']==g['dataset']]
        if len(group)!=5 or abs(np.mean([float(x['final_alpha']) for x in group])-float(g['mean_raw_alpha']))>1e-12 or abs(np.mean([float(x['sigmoid_final_alpha']) for x in group])-float(g['mean_sigmoid_alpha']))>1e-12:gbad.append(g['dataset'])
    check('gate_alpha_units',len(graw)==10 and not gbad,f'raw rows={len(graw)}; bad={gbad}')
    stats=rows(OUT/'statistics_s1_s5.csv');s1=next(x for x in stats if x['view'].startswith('S1_'));s3=next(x for x in stats if x['view']=='S3_FinancialBlockchainSix_MixedLabels')
    check('statistics_policy',len(stats)==5 and len(s1['models'].split(';'))==7 and 'CONAD' not in s1['models'] and float(s3['friedman_p'])>0.05,f'S1 complete={s1["n_complete"]}; S3 p={s3["friedman_p"]}')
    claims=rows(OUT/'claims_to_evidence_a04.csv')
    bad_claims=[]
    for ds in PRIMARY[:6]:
        crow=next((x for x in claims if x['claim'].startswith(ds+':')),None)
        table_row=next(x for x in table if x['dataset']==ds)
        candidates={m:value(table_row[m].split(' ± ')[0]) for m in MODELS if ' ± ' in table_row[m]}
        winner=max(candidates,key=candidates.get)
        if not crow or winner not in crow['claim']:bad_claims.append(ds)
    check('claims_match_table',not bad_claims and not any('DLG-Aug achieves highest' in x['claim'] for x in claims),f'bad={bad_claims}')
    memory=OUT/'table_memory_selected_a04.csv'
    memory_rows=rows(memory) if memory.is_file() else []
    memory_pairs=defaultdict(set);memory_bad=[]
    for r in memory_rows:
        key=(r['dataset'],r['model']);memory_pairs[key].add(r['envelope'])
        path=ROOT/r['raw_log_path']
        if not path.is_file() or sha(path)!=r['raw_log_sha256']:
            memory_bad.append(str(key)+':raw_log')
            continue
        raw=json.loads(path.read_text())
        if raw.get('dataset_tensor_sha256')!=dmap[r['dataset']].get('constructed_tensor_sha256') or '3090' not in raw.get('device',''):
            memory_bad.append(str(key)+':identity')
        if r['envelope']=='cap8g' and raw.get('allocator_cap_bytes')!=8*2**30:
            memory_bad.append(str(key)+':cap')
    expected_memory={('Cora-Syn','DOMINANT'),('DGraphFin','DOMINANT'),('Ethereum','DLG-Aug'),('Reddit-Syn','DLG-Base')}
    complete_memory={key for key,envelopes in memory_pairs.items() if envelopes=={'full','cap8g'}}
    check('measured_memory_only',len(memory_rows)==8 and complete_memory==expected_memory and not memory_bad,
          f'{len(memory_rows)} raw-backed rows; complete={len(complete_memory)}; bad={memory_bad}')
    env=B/'environment';wheels=rows(env/'wheel_artifacts_a04.csv')
    check('wheel_artifacts',all(x['status']=='HASH_VERIFIED' for x in wheels),f'{sum(x["status"]!="HASH_VERIFIED" for x in wheels)} missing wheel artifacts')
    recreation=(env/'environment_recreation_report.md').read_text()
    check('clean_recreation','**Status:** PASS' in recreation,'clean recreation and checks not PASS')
    status='PASS' if all(c['status']=='PASS' for c in checks) else 'HOLD'
    result={'gate_g4':status,'passed':sum(c['status']=='PASS' for c in checks),'total':len(checks),'checks':checks}
    (OUT/'validation_summary_a04.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# A04 cross-file validation',f'\n**Gate G4: {status}** — {result["passed"]}/{result["total"]} checks pass.\n','| Check | Status | Detail |','|---|---|---|']
    lines += [f'| {c["check_id"]} | {c["status"]} | {c["detail"].replace("|","/")} |' for c in checks]
    (OUT/'validation_summary_a04.md').write_text('\n'.join(lines)+'\n')
    pub=json.loads((OUT/'publication_manifest_a04.json').read_text());pub['gate_g4']=status;pub['validation_summary_sha256']=sha(OUT/'validation_summary_a04.json')
    (OUT/'publication_manifest_a04.json').write_text(json.dumps(pub,indent=2)+'\n')
    print(json.dumps({'gate_g4':status,'passed':result['passed'],'total':len(checks),'failures':[c['check_id'] for c in checks if c['status']=='FAIL']},indent=2))
    return status=='PASS'

if __name__=='__main__':sys.exit(0 if main() else 1)
