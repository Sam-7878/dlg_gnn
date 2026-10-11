"""Render structured closure/claim evidence from existing, hash-identified artifacts."""
from __future__ import annotations
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from r01_common import ROOT,PROJECT,config,output,read,write,sha256,git_sha


def main():
    cfg=config();out=output();analysis=out/'analysis';audits=out/'audits'
    registry=[json.loads(s) for s in (analysis/'experiment_registry.jsonl').read_text().splitlines()]
    temporal=read(analysis/'temporal_audit.json');state=read(out/'stress/bounded_state_audit.json');restart=read(out/'stress/restart_audit.json')
    builds=read(audits/'build_audit.json');visual=read(audits/'visual_review.json');parity=read(audits/'method_identity_audit.json')
    cleanroom=read(PROJECT/'evidence/cleanroom_r01.json');release=read(PROJECT/'evidence/r01_release.json')
    assert len(registry)==25 and all(b['actual_clean_build'] and b['unresolved_count']==b['overflow_count']==0 for b in builds)
    assert visual['all_pages_reviewed'] and visual['total_pages_reviewed']==32 and visual['unresolved_clipping_or_overlap_count']==0
    assert {v['entry']:v['pdf_sha256'] for v in visual['pdfs']}=={b['entry']:b['pdf_sha256'] for b in builds}
    for b in builds:
        assert sha256(PROJECT/'paper/current/r01/build'/Path(b['entry']).with_suffix('.pdf'))==b['pdf_sha256']
    assert cleanroom['archive_sha256']==release['archive_sha256']
    assert cleanroom['stdlib_only_numeric_verification']=='PASS'
    assert len(cleanroom['author_local_private_source_builds'])==4
    suites=list(ET.parse(audits/'pytest_distribution.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==33 and all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
    package_receipts=read(audits/'private_package_identities.json')
    assert len(package_receipts)==2 and all(sha256(ROOT/p['package'])==p['sha256'] for p in package_receipts)
    boundary=read(PROJECT/'evidence/archive_boundary_r01.json')
    assert boundary['archive_sha256']==release['archive_sha256'] and boundary['literal_address_member_count']==0 and not boundary['private_manuscript_members']
    assert all(p['abstract_words_pdftotext']<=200 and p['pdf_sha256']==next(b['pdf_sha256'] for b in builds if b['entry']==p['entry']+'.tex') for p in boundary['private_rendered_abstracts'])
    replica=read(audits/'alias_replication_all.json')
    assert replica['retraining']['N_completed_models']==5 and replica['retraining']['local_and_relational_parameter_tensors_bit_identical']
    assert replica['replay']['N_input']==525 and replica['replay']['decision_route_state_event_order_disagreements']==0
    runtime=[read(p) for p in sorted((out/'runtime').glob('*/complete.json'))]
    assert len(runtime)==18 and sum(r['population']=='long' and r['N_measured']==100000 for r in runtime)==3
    assert parity['N_fitted_policy_cells']==180 and parity['label_disagreements']==parity['route_disagreements']==0 and parity['max_score_difference']<=parity['score_tolerance']
    declarations=read(PROJECT/'configs/author_declarations_r01.json');author=declarations['exact_final_r01_version_approved_by_all_authors']
    abstract_words=len((PROJECT/'paper/current/r01/common/abstract.tex').read_text().split())
    assert abstract_words<=200,'source abstract exceeds200 words before numeric-only macro expansion'
    def artifact(path):
        path=Path(path)
        if not path.is_file():raise ValueError('closure missing actual artifact '+str(path))
        return {'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size}
    specs=[
        ('00','PASS','Inventory and new identity', ['execution_plan.json','evidence_inventory.csv','training_source_manifest.json'],'r01_inventory.py; r01_data_audit.py; r01_source_freeze.py',
         f"25 complete independent models; original directory snapshot retained; reviewed original ZIP unavailable, byte identity not asserted.", 'Original raw-edge cache construction remains unverified.'),
        ('01','PASS','Actual build, hash and negative tests',['audits/build_audit.json','audits/negative_fixtures.json','audits/visual_review.json','audits/pytest_distribution.xml'],'r01_validate.py --mode fixtures; --mode paper; r01_render.py; explicit visual inspection; pytest -q tests/stream_mc --junitxml=projects/stream_mc/results/selectivestream_r01_20261011/audits/pytest_distribution.xml',
         f"Three actual clean builds, zero unresolved/overfull; all {visual['total_pages_reviewed']} rendered pages reviewed; negative CLI fixtures nonzero.",'Compiler/text/visual inspection are distinct from peer review.'),
        ('02','PASS','Pooled/source-only separation',['analysis/experiment_registry.jsonl','generated/chains.csv','generated/loco.csv'],'r01_train.py --campaign all; r01_analysis.py',
         '10 pooled backbone fits and15 independent source-only fits; target chains excluded from all fit/selection paths.','Source-only exclusion is not complete temporal as-of validity.'),
        ('03','CLAIM REMOVED','Strict online temporal validity',['analysis/temporal_audit.json','analysis/temporal_audit_rows.csv','split_overlap.json'],'r01_data_audit.py; r01_analysis.py',
         f"Future retrieved refs {temporal['reference_cutoff_violations']}; model-fit-after-target counts {temporal['model_weight_future_target_count']}; policy-after-target {temporal['policy_future_target_count']}; edge/label times unknown.",'Counts sum model rows, not independent contracts; retained scope retrospective only.'),
        ('04','PASS','Frozen specification and exceptions',['audits/method_identity_audit.json','audits/schema_adapter.json','audits/pytest_final.xml'],'r01_final_audits.py; pytest -q tests/stream_mc',
         f"180 actual/fitted policy-cell comparisons; label/route disagreements0; max score delta {parity['max_score_difference']:.3g}; actual trained fallback tests.",'Soft timeout is not GPU cancellation; invalid input abstention differs from local fallback.'),
        ('05','PASS','Raw paired statistics',['analysis/paired_statistics.csv','analysis/historical_r4_statistics_audit.csv','analysis/time_block_sensitivity.csv','analysis/day_block_sensitivity.csv'],'r01_analysis.py; r01_final_audits.py; r01_public.py --mode verify',
         '2000 paired class-stratified resamples per primary seed; month-block500 is degenerate with one month, additional day-block500 is post-hoc; exact McNemar/Holm families distinct.','Descriptive after test inspection; few-day blocks do not reliably correct graph/time dependence; repeated test subjects remain.'),
        ('06','PASS','Numbers, plots and workload separation',['figure_table_provenance.csv','runtime/offline_v2/budget_runtime_complete.json','audits/floating_batch_audit.json','generated/runtime.csv'],'r01_replay.py; r01_generate.py --paper',
         '540 measured batched-policy runs;15 separate-process prefix repeats;3 distinct100k-event workloads.','Deep fractions, contracts/events and timing replicates/training seeds are not pooled.'),
        ('07','PARTIAL' if not author else 'PASS','Method title and author declarations',['generated_artifacts.json'],'shared R01 wrappers; author-provided Benchmark declarations',
         f'Deterministic/retrospective title,{abstract_words}-word source abstract with generated numeric macros and replay quota-transfer failure; supplied funding/correspondence/ORCIDs adopted.','Exact final StreamMC approval, roles/conflicts and final declarations require all authors, not inferred from Benchmark.'),
        ('08','PASS','Budget and routing benefit comparisons',['runtime/offline_v2/budget_runtime.csv','analysis/achieved_budget_matched.csv','analysis/relational_corrections_harms.csv'],'r01_replay.py --mode offline; r01_final_audits.py',
         'Six frozen budgets, margin/entropy/random3/learned-benefit; actual selected execution and same-threshold correction/harm counts.','Nearest achieved-budget comparisons are explicitly post-hoc, some unmatched; no uniformly best router asserted.'),
        ('09','PARTIAL','Task-matched GNN/light controls',['runtime/control_runtime.csv','audits/control_parity.json','model_training_cost.csv'],'r01_controls_timing.py; r01_release.py',
         'Same-input controls, real parameter counts and actual inference timings; independent local/relational train times recorded.','Original light-head training wall times were not separately recorded; separate alias-input cost probe is not substituted for those original times.'),
        ('10','PASS','Actual boundary and recovery tests',['stress/bounded_state_audit.json','stress/restart_audit.json','stress/checkpoint_diff.csv','audits/checkpoint_identity_amendment.json'],'r01_stress.py --mode all; --mode restart; amended identity verification',
         f"25k unique+50k churn, resident cap {state['resident_peak']}, cache {state['cache_entries']}, queue {state['queue_high_water']}; restart score max {restart['score_max_abs_diff']:.3g}, labels/loss0.",'Checkpoint metadata amendment rerun covers full-path local-file abrupt failures, not every policy, power loss or distributed exactly-once.'),
        ('11','PASS','Chain/path operational risk',['analysis/risk_counts.csv','analysis/reliability_bins.csv','analysis/bsc_error_cases.csv','analysis/prevalence_sensitivity.csv','analysis/validation_alert_budgets.csv'],'r01_analysis.py',
         'Raw denominator-aware risks, calibration counts/Wilson intervals, BSC cases, unchanged validation alert thresholds and same-test class-prior sensitivity.','Polygon has0 test positives; no future-positive cohort, no causal explanation or safety bound.'),
        ('12','PASS','MC fairness and independent contribution',['analysis/mc_metrics.csv','generated/mc.csv'],'r01_train.py; frozen dropout-only tests; supplied Benchmark inspected; primary literature verified',
         'T1 deterministic entropy; T3/5/8 dropout-only variance with independent per-T calibration/costs; predecessor and Benchmark overlap separated.','Task/architecture-conditional; not a general indictment of MC dropout or first invention of early exit/calibration/LRU.'),
        ('13','PARTIAL','Packages and clean-environment reproduction',['audits/private_package_identities.json','audits/alias_replication_all.json'],'r01_release.py; r01_public.py --mode verify/tables; r01_full_reproduce.py --confirm-full --seed 11; r01_public_replay.py --confirm-full --population prefix --policy margin; r01_replication_audit.py --mode all; r01_archive_audit.py --paper; r01_cleanroom.py --paper',
         'Private preprint/journal source/PDF packages plus paper-free numeric/input/model ZIP; five actual seed11 alias-input retrainings with bit-identical local/relational parameters and prediction agreement,525 actual replayed events with decision/route/state agreement; fresh stdlib-only venv raw arithmetic and four fresh source builds.','Local copy only, same agent/host; alias-input retraining checks one seed, not all25 reruns; no public re-download/external reviewer/hosted publication/submission; exact-version author approval pending.')]
    changed={
        '00': ['r01_inventory.py','r01_data_audit.py','r01_source_freeze.py'],
        '01': ['r01_validate.py','r01_render.py','r01_archive_audit.py'],
        '02': ['r01_train.py','r01_analysis.py'],
        '03': ['r01_data_audit.py','r01_analysis.py'],
        '04': ['r01_train.py','r01_final_audits.py','r01_identity_audit.py'],
        '05': ['r01_analysis.py','r01_public.py','r01_final_audits.py'],
        '06': ['r01_replay.py','r01_offline_v2.py','r01_generate.py','r01_float_audit.py'],
        '07': ['r01_generate.py'],
        '08': ['r01_train.py','r01_replay.py','r01_final_audits.py'],
        '09': ['r01_train.py','r01_controls_timing.py','r01_release.py','r01_replication_audit.py'],
        '10': ['r01_stress.py','r01_identity_audit.py'],
        '11': ['r01_analysis.py','r01_generate.py'],
        '12': ['r01_train.py','r01_analysis.py','r01_generate.py'],
        '13': ['r01_release.py','r01_public.py','r01_full_reproduce.py','r01_public_replay.py','r01_replication_audit.py','r01_archive_audit.py','r01_cleanroom.py','r01_closure.py']}
    affected={
        '00':'Evidence identity/provenance; unavailable original review ZIP not verified.',
        '01':'All manuscript references/figures/tables and build/layout checks, not peer review.',
        '02':'Main3.3/6.3 pooled versus source-only identities and independent fit claims.',
        '03':'Main3.2/limitations; strict online/as-of assertions removed, retrospective scope retained.',
        '04':'Main4 specification, frozen routing, retrieval and fallback semantics; supplement2/8.',
        '05':'Main5.1 and supplement4 paired statistics; descriptive not preregistered inference.',
        '06':'Main6.1/6.2/6.4 actual workload-specific numbers and figures; no deep-fraction latency proxy.',
        '07':'Title/abstract and author declarations; no unprovided exact-version approval.',
        '08':'Main6.2 budget frontier; same-threshold corrections/harms, no uniformly best router.',
        '09':'Main6.1 and control-cost table; light controls have higher F1, original light-head training times unknown.',
        '10':'Main6.4 and supplement8 bounded core state/local restart, not whole-process O(1) or distributed exactly-once.',
        '11':'Main6.3 and risk/calibration plots; distinct denominators, Polygon support undefined, BSC failures retained.',
        '12':'Related-work contribution scope and supplement7 MC sensitivity; deterministic method primary.',
        '13':'Data availability/private submission packages; local reproducibility not publicly hosted release or independent external approval.'}
    records=[]
    for suffix,status,title,names,command,observed,limitations in specs:
        extra=[artifact(PROJECT/'evidence'/n) for n in ('r01_release.json','archive_boundary_r01.json','cleanroom_r01.json')] if suffix=='13' else []
        sources=[artifact(PROJECT/'scripts'/name) for name in changed[suffix]]
        if suffix in ('04','08','10'):
            sources += [artifact(ROOT/'src/gog_fraud/streaming'/name) for name in ('selective.py','selective_engine.py','selective_state.py')]
        if suffix in ('02','04','08'):sources.append(artifact(ROOT/'src/gog_fraud/data/level2/relation_builder.py'))
        if suffix=='07':sources += [artifact(PROJECT/'paper/current/r01/common'/name) for name in ('abstract.tex','authors.tex','declarations.tex')]
        records.append({'task_id':'SS-R01-'+suffix,'status':status,'title':title,'changed_source_commit':git_sha(),
            'changed_files':sources,'affected_manuscript_claims':affected[suffix],
            'evidence':[artifact(out/name) for name in names]+extra,'procedure':f"From dlg_gnn, Python={sys.executable}, PYTHONPATH=src, CUBLAS_WORKSPACE_CONFIG=:4096:8; "+command,
            'observed':observed,'limitations':limitations,'author_approval':author,'reviewer':'Codex implementation-side audit, 2026-10-11',
            'independent_verification':'No external independent approval. Fresh same-agent venv/workspace scope in evidence/cleanroom_r01.json.'})
    report={'run_id':cfg['run_id'],'tasks':records,'gate_P':'HOLD','gate_J':'HOLD','release_archive_sha256':release['archive_sha256'],
        'release_scope':'local candidate, not uploaded','author_declaration_source':artifact(ROOT/declarations['source_path']),
        'cleanroom':artifact(PROJECT/'evidence/cleanroom_r01.json'),'remaining_blockers':['Exact-version author approval','Public hosting/access verification if claimed','Submission-time official instructions/deadline recheck'],
        'scientific_limitations':['strict temporal validity removed','original light-head training wall time absent','independent external verification not performed']}
    write(PROJECT/'evidence/closure_r01.json',report)
    lines=['# DLG-SelectiveStream R01 closure','', 'Gate P / Gate J: **HOLD**. Scientific implementation checks are separate from author approval and external publication.','',
        'Exact per-task artifact SHA-256, procedure, observation, limitation, source commit and reviewer scope: `evidence/closure_r01.json`. No placeholder validator or public URL is asserted.','',
        '## Required four-way summary','',
        '- Existing values retained: inherited split support/provider mapping; R4 remains historical with corrected raw paired statistics. Benchmark contributes author facts and overlap context, not StreamMC scores.',
        '- Changed by re-execution:25 new pooled/LOCO fits, policies/predictions, descriptive statistics, matched workload costs, calibration/risk/MC figures and common numeric manuscript inputs.',
        '- Claims removed: strict online/as-of safety, production readiness, universal predictive/tail gains, Polygon positive detection, universal MC statements, fabricated URL/DOI and preregistration.',
        '- New experiments:6budgets/4routers, GATv2/light/threshold controls, raw paired/block tests, path risk/alerts/prior sensitivity, cap/cache/queue/TTL/churn and abrupt restart, private builds and fresh local verification.','',
        '## Task closure','']
    for r in records:
        lines += [f"### {r['task_id']} — {r['status']}: {r['title']}",'',r['observed'],'',f"Procedure: `{r['procedure']}`",'', 'Evidence:', '']
        lines += [f"- `{a['path']}` — SHA-256 `{a['sha256']}`" for a in r['evidence']]
        lines += ['',f"Changed-source commit: `{r['changed_source_commit']}`; private paper sources are excluded from this commit. Exact changed-file hashes are in the machine-readable closure record.",'', 'Changed files:', '']
        lines += [f"- `{a['path']}` — SHA-256 `{a['sha256']}`" for a in r['changed_files']]
        lines += ['',f"Affected manuscript claims: {r['affected_manuscript_claims']}"]
        lines += ['',f"Limitations: {r['limitations']}",'', 'Reviewer: Codex implementation-side audit; no external independent approval. Exact final-version author approval: pending.','']
    lines += ['## Handoff','', 'Active private scientific source: `paper/current/r01`. Source/PDF packages: `paper/current/r01/packages`. Public candidate: `evidence/r01_numeric_evidence.zip`, exact identity in `r01_release.json`. Paper source is not in Git or the public ZIP. No push, submission, publication or email was performed.','',
        'SS-R01-09 remains PARTIAL for the unrecorded ORIGINAL light-head training wall time; the inference/control evidence is verified. This missing timing is not repaired by fabricating old logs or relabeling a new probe. Strict-time claims are removed under SS-R01-03, not treated as zero violations.','']
    (PROJECT/'CLOSURE_R01.md').write_text('\n'.join(lines))
    claims=[]
    mapping=[('snapshot_scope','Materials/temporal','Retrospective snapshots, not strict online detection','temporal_audit.json','verified','retained'),
             ('primary','Results/primary','Five-seed means from paired contract predictions','aggregate_metrics.csv','verified','changed_by_new_run'),
             ('source_only','Results/LOCO','Target excluded from source-only fit and selection','experiment_registry.jsonl','verified','new'),
             ('statistics','Supplement/statistics','Exact paired correctness and descriptive paired intervals','paired_statistics.csv','verified','changed_by_new_run'),
             ('risk','Results/risk','Denominator-aware direct risk, no formal safety','risk_counts.csv','verified','new'),
             ('polygon','Results/chains','Zero positive support; positive metrics undefined','per_seed_metrics.csv','verified','retained'),
             ('bsc_cases','Supplement/errors','Descriptive false negatives without causal attribution','bsc_error_cases.csv','verified','new'),
             ('mc','Supplement/MC','Task-limited dropout ablation, primary deterministic','mc_metrics.csv','verified','new')]
    for claim,location,text,name,status,disposition in mapping:
        path=analysis/name;claims.append({'claim_id':claim,'manuscript_location':location,'exact_claim_text':text,'claim_type':'scope/result','experiment_id':'explicit per-artifact registry rows',
            'source_artifact':str(path.relative_to(ROOT)),'source_sha256':sha256(path),'aggregation_command':'r01_analysis.py; r01_generate.py --paper',
            'verification_status':status,'known_limitations':'retrospective shared test; unknown edge/label time; seed SD != conditional bootstrap','disposition':disposition,'reviewer':'Codex implementation-side','review_date':'2026-10-11'})
    for claim,text,name in [('budget','Actual selected execution, not a deep-fraction latency proxy','runtime/offline_v2/budget_runtime.csv'),('boundaries','Declared resident/cache/queue bounds tested, not whole-process O(1)','stress/bounded_state_audit.json'),('restart','Local abrupt-file failure scope only','stress/restart_audit.json'),('build','Actual compiler/visual checks, not author approval','audits/build_audit.json')]:
        path=out/name;claims.append({**claims[0],'claim_id':claim,'manuscript_location':'Results/systems or supplement','exact_claim_text':text,'source_artifact':str(path.relative_to(ROOT)),'source_sha256':sha256(path),'aggregation_command':'r01_campaign.py; actual separate verification','disposition':'new'})
    with (PROJECT/'evidence/claim_ledger_r01.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=claims[0]);writer.writeheader();writer.writerows(claims)
    (PROJECT/'CLAIM_LEDGER.md').write_text('# R01 claim ledger\n\nMachine-readable exact claim/scope, artifact hash, command, disposition and reviewer: `evidence/claim_ledger_r01.csv`.\n\n'+ '\n'.join(f"- {r['claim_id']}: {r['exact_claim_text']} ({r['disposition']}); `{r['source_artifact']}`." for r in claims)+'\n\nRemoved: strict as-of/live safety, production readiness, uniform superiority, Polygon positive detection and generalized MC claims. Approval/publication remain HOLD; see CLOSURE_R01.md.\n')
    print('STRUCTURED CLOSURE COMPLETE',len(records),'Gate P/J HOLD',flush=True)


if __name__=='__main__':main()
