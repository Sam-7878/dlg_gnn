"""Bind actual R02 receipts/claims/gates without inventing author approval."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from r02_common import ROOT,PROJECT,R01,R02,VERIFY,RUN,read,write,sha256,csv_write,artifact,original_unchanged


def main():
    original_unchanged();baseline=read(VERIFY/'handoff_baseline.json')
    assert all(sha256(ROOT/p)==h for p,h in baseline['private_paper_files'].items())
    numeric=read(VERIFY/'raw_numeric_validation.json');assert numeric['N_original_models']==25 and numeric['N_raw_metric_rows_recomputed']==6065
    xml=ET.parse(VERIFY/'test_results.xml').getroot();cases=xml.findall('.//testcase');assert len(cases)==33 and not xml.findall('.//failure') and not xml.findall('.//error')
    negative=read(VERIFY/'negative_fixtures/negative_fixtures.json');assert len(negative)==5 and all(r['detected'] and r['actual_exit_code']!=0 for r in negative)
    build=read(VERIFY/'paper_build_audit.json');visual=read(VERIFY/'visual_inspection_r02.json')
    assert all(visual['pdf_sha256'][r['entry'][:-4]]==r['pdf_sha256'] for r in build['builds'])
    access=read(PROJECT/'evidence/access_verification_r02.json');assert access['status']=='VERIFIED' and access['public_numeric']['full_policy_rows']==6065
    release=read(PROJECT/'evidence/release_manifest_r02.json');author=read(PROJECT/'configs/author_declarations_r02.json')
    commands=[read(p) for p in sorted(VERIFY.glob('execution_*.json'))]
    definitions=[
        ('SS-R02-00','VERIFIED','handoff_only','Exact original 939 local targets/private full handoff and separately transformed public derivative; actual accessible source/evidence release','Original IDs/old private paper/checkpoints restricted; exact prior partial-packing action unavailable',[R02/'handoff_manifest_verification.json',R02/'source_identity_r02.json']),
        ('SS-R02-01','VERIFIED','reaggregation','25 original models, 6065 raw policy/control rows,40 MC rows,bootstrap/Holm and numeric table/figure identity','Same-host implementer verification, descriptive inference, unknown time/label truth; no new training',[VERIFY/'raw_numeric_validation.json',VERIFY/'table_figure_numeric_diff.csv',VERIFY/'public_local_numeric/public_verification.json']),
        ('SS-R02-02','VERIFIED','new_measurement','18 original replay traces,540 active v2 timings,180 selected pipeline cells,5000-cap workload,109 restart pairs,33 tests/5 negative exits; distinct cheap new state probe','Recorded wall duration lacks independent raw start/end timestamps; single long run per policy; original crash exits checked, not rerun; memory/local-failure scope only',[VERIFY/'runtime_recomputed.csv',VERIFY/'state_boundary_verification.json',VERIFY/'restart_failure_model_verification.json',VERIFY/'test_results.xml',VERIFY/'negative_fixtures/negative_fixtures.json']),
        ('SS-R02-03','VERIFIED','editorial','RQ1 snapshot tradeoff, RQ2 replay budget failure, RQ3 state/failure boundary; main poor event diagnostics and UTC test support','No prospective transaction-fraud safety, broad chain/generalized speedup or formal risk guarantee',[PROJECT/'docs/r02/paper_claim_revision_map.md',VERIFY/'snapshot_support_utc.csv']),
        ('SS-R02-04','ACCEPTED_LIMITATION','reaggregation','90 post-hoc nearest achieved-budget pairs with exclusions;90 same-run control score/cost records','Original head fitting time NA (unrecorded); no training-efficiency claim; nearest quota not exact equal latency',[VERIFY/'achieved_budget_matches_recomputed.csv',VERIFY/'control_score_cost.csv',PROJECT/'docs/r02/control_training_time_resolution.md']),
        ('SS-R02-05','VERIFIED','editorial','20 main references, primary methods, DOI/version and URL printing, NA/spacing/label-free semantics; exact final build/render inspected','Platt mirror access limited; no unverified DOI/book pagination; temporal/anomaly references are context, not run baselines',[PROJECT/'docs/r02/reference_verification.md',VERIFY/'paper_build_audit.json',VERIFY/'visual_inspection_r02.json']),
        ('SS-R02-06','VERIFIED','handoff_only','Actual public downloads/SHA checks, fresh stdlib venv and isolated-path6065/40 verification; large evidence external to small private submission package','Full scientific dependencies explicitly reused; not independent external-human audit; publisher429 requires final rule/deadline recheck',[PROJECT/'evidence/access_verification_r02.json',PROJECT/'evidence/release_manifest_r02.json',VERIFY/'private_package_identities_r02.json']),
        ('SS-R02-07','OPEN','editorial','Real final build/visual/package hashes; user funding/conflicts/no duplicate confirmations recorded','Final AI wording, CRediT and exact-version all-author source/PDF/evidence approval missing; no approval file or paper submission',[VERIFY/'paper_build_audit.json',VERIFY/'private_package_identities_r02.json',PROJECT/'configs/author_declarations_r02.json'])]
    work=[];ledger=[]
    for item,status,change,retained,limits,outputs in definitions:
        identities=[artifact(p) for p in outputs]
        work.append({'work_item_id':item,'status':status,'original_R01_artifact_identity':artifact(R01/'release_manifest.json'),'R02_change_type':change,
            'executed_commands':[{'stage':r['stage'],'actual_exit_code':r['actual_exit_code'],'command':r['command'],'utc_execution_stamp':r['utc_execution_stamp']} for r in commands],
            'execution_environment':artifact(VERIFY/'verification_environment.json'),'inputs_and_hashes':[artifact(R01/'release_manifest.json'),artifact(PROJECT/'evidence/r01_release.json')],'outputs_and_hashes':identities,
            'independent_verification_scope':'same implementer/host; author-local original raw audit plus real public downloads/isolated path; no external human scientific approval',
            'retained_claims':retained,'removed_or_narrowed_claims':limits,'known_limitations':limits,'responsible_author_or_operator':'Codex implementation operator; author approval reserved to actual authors','verification_date':'2026-10-12'})
        ledger.append({'claim_id':item,'status':status,'retained_claim':retained,'limitation':limits,'evidence_paths':';'.join(i['path'] for i in identities),'evidence_sha256':';'.join(i['sha256'] for i in identities)})
    gate=[{'gate':g,'status':'OPEN','science_and_evidence':'VERIFIED within declared scope','build_and_visual':'VERIFIED exact final PDF hashes','external_code_evidence_access':'VERIFIED actual unauthenticated downloads','exact_all_author_version_approval':'OPEN','ai_wording':'OPEN','credit_confirmation':'OPEN','official_submission_policy_recheck':'OPEN' if g=='J' else 'not journal-specific'} for g in ('P','J')]
    result={'revision_id':RUN,'original_manifest_targets_unchanged':939,'original_private_paper_unchanged':True,'work_items':work,'gates':gate,
        'public_release_url':access['release_url'],'target_source_commit':release['target_source_commit'],'author_confirmations':author,
        'all_author_approval_file_created':False,'paper_publication_or_submission':False,'status':'scientific/source/evidence closure verified within scope; paper publication/submission gates OPEN'}
    write(PROJECT/'evidence/closure_r02.json',result);csv_write(PROJECT/'evidence/claim_ledger_r02.csv',ledger)
    bindings=[('snapshot_primary','Main Results/Table primary',R01/'analysis/per_seed_metrics.csv','25models/five seeds','frozen training_sources.zip',VERIFY/'per_seed_metrics_recomputed.csv'),
        ('statistics','Main Statistics/Supplement paired',R01/'analysis/paired_statistics.csv','GIN11/22/33/44/55','frozen analysis algorithm',VERIFY/'paired_statistics_recomputed.csv'),
        ('budget_selected','Main budget frontier',R01/'runtime/offline_v2/budget_runtime.csv','GINfive seeds×36policies×3repeats','runtime_sources_v2.tar',VERIFY/'selected_execution_parity.csv'),
        ('events','Main replay/diagnostics',R01/'runtime/long_margin_repeat0/event_trace.csv','single long run/policy','runtime_sources_v1.tar',VERIFY/'runtime_recomputed.csv'),
        ('state','Main state/Supplement',R01/'stress/state_pressure_trace.csv','synthetic75000 keys','original r01_stress.py plus distinct R02probe',VERIFY/'state_boundary_verification.json'),
        ('restart','Supplement checkpoint',R01/'stress/checkpoint_diff.csv','109pairs/three recorded exits','amended schema2 identity source',VERIFY/'restart_failure_model_verification.json'),
        ('MC','Supplement MC',R01/'analysis/mc_metrics.csv','five seeds×T1/3/5/8×val/test','training_sources.zip',VERIFY/'mc_recomputed.csv')]
    csv_write(VERIFY/'claim_to_raw_r02.csv',[{'claim_id':claim,'manuscript_location':loc,'experiment_id':'selectivestream_r01_20261011','seed_scope':seeds,'source_version':version,'raw_path':str(raw.relative_to(ROOT)),'raw_sha256':sha256(raw),
        'aggregation_command':'python projects/stream_mc/scripts/r02_run.py '+('systems' if claim in ('budget_selected','events','state','restart') else 'numeric'),'output_path':str(out.relative_to(ROOT)),'output_sha256':sha256(out),'verified_status':'VERIFIED','limitation':next(r['known_limitations'] for r in work if r['work_item_id']==('SS-R02-02' if claim in ('budget_selected','events','state','restart') else 'SS-R02-01'))} for claim,loc,raw,seeds,version,out in bindings])
    text='# DLG-SelectiveStream R02 closure — 12 October 2026\n\n'
    text+='Scientific/evidence revision implemented; paper publication/submission remains **OPEN**, not falsely complete.\n\nPublic release: '+access['release_url']+'\n\n'
    for row in work:text+='## '+row['work_item_id']+' — '+row['status']+'\n\n'+row['retained_claims']+'\n\nLimitations: '+row['known_limitations']+'\n\nEvidence: '+', '.join('`'+i['path']+'`' for i in row['outputs_and_hashes'])+'\n\n'
    text+='## Gate P / Gate J\n\n- [x] Original939 hashes and private paper unchanged; all25/five-seed raw numerics and declared systems scope verified.\n- [x] Actual code/evidence release and unauthenticated downloads; fresh stdlib/isolated scientific-input execution.\n- [x] Science-centered claims, adverse results, matched-pair exclusions and temporal/label limitations.\n- [x] Exact final clean builds, references, all-page inspection and private package byte/hash records.\n- [ ] All authors approve exact final source/PDF/evidence version and CRediT.\n- [ ] Author supplies final AI-assistance wording.\n- [ ] Gate J: immediately recheck official deadline/portal total-file policy (prior429 access limitation).\n\nNo author approval record, preprint posting or journal submission is fabricated. Original head fitting time is an accepted limitation, not a reason to invent timing.\n'
    (PROJECT/'CLOSURE_R02.md').write_text(text)
    print('R02 CLOSURE WRITTEN; GATE P/J OPEN FOR ACTUAL AUTHOR/POLICY ITEMS',flush=True)


if __name__=='__main__':main()
