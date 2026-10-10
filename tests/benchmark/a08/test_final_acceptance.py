"""Negative acceptance checks use isolated fixtures, never campaign evidence."""
import importlib.util
import json
from pathlib import Path
import pytest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('a08_checker',ROOT/'projects/benchmark/scripts/a08_final_check.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


@pytest.mark.parametrize('change',['missing','template','skipped','failed','status_only'])
def test_fake_gate_cannot_produce_pass(tmp_path,change):
    audit=tmp_path/'projects/benchmark/evidence/a08_data_repair/audit';audit.mkdir(parents=True)
    for number in range(11):(audit/f'G{number}.json').write_text(json.dumps({'status':'PASS'}))
    if change=='missing':(audit/'G2.json').unlink()
    if change=='template':(audit/'G2.json').write_text(json.dumps({'status':'PASS','is_template':True}))
    if change in ('skipped','failed'):(audit/'G2.json').write_text(json.dumps({'status':change.upper()}))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):checker.validate_final(tmp_path)


def test_changed_artifact_rejects_previous_gate_hash(tmp_path):
    p=tmp_path/'artifact.json';p.write_text('{}')
    good={'identity':checker.sha256(p)}
    checker.bound_document(tmp_path,good,'identity','artifact.json')
    p.write_text('{"status":"PASS"}')
    with pytest.raises(ValueError,match='binding mismatch'):checker.bound_document(tmp_path,good,'identity','artifact.json')


def test_missing_or_mislabeled_pdf_rejected(tmp_path):
    pdf={'path':'paper.pdf','sha256':'historical','pages':22,'page_reviews':[]}
    with pytest.raises(ValueError):checker.check_pdf(tmp_path,pdf)
    (tmp_path/'paper.pdf').write_bytes(b'not a PDF')
    with pytest.raises(ValueError):checker.check_pdf(tmp_path,pdf)


def test_fake_page_count_or_incomplete_review_rejected(tmp_path,monkeypatch):
    (tmp_path/'paper.pdf').write_bytes(b'fixture mocked parser; not evidence')
    pdf={'path':'paper.pdf','sha256':checker.sha256(tmp_path/'paper.pdf'),'pages':22,'page_reviews':[]}
    monkeypatch.setattr(checker.subprocess,'check_output',lambda *a,**k:'Pages: 15\n')
    with pytest.raises(ValueError,match='page count'):checker.check_pdf(tmp_path,pdf)
    pdf['pages']=15
    with pytest.raises(ValueError,match='incomplete'):checker.check_pdf(tmp_path,pdf)


def test_page_pass_without_render_or_observation_rejected(tmp_path,monkeypatch):
    p=tmp_path/'paper.pdf';p.write_bytes(b'mocked fixture only')
    monkeypatch.setattr(checker.subprocess,'check_output',lambda *a,**k:'Pages: 1\n')
    pdf={'path':'paper.pdf','sha256':checker.sha256(p),'pages':1,'page_reviews':[{'page':1,'status':'PASS','render_path':'absent.png','render_sha256':'bad'}]}
    with pytest.raises(ValueError):checker.check_pdf(tmp_path,pdf)


def test_execution_cannot_be_bound_after_scores(tmp_path,monkeypatch):
    from gog_fraud.data import benchmark_lineage as lineage
    evidence=tmp_path/'projects/benchmark/evidence/a08_data_repair';(evidence/'run_manifests').mkdir(parents=True)
    (evidence/'run_manifests/old.json').write_text('{}')
    monkeypatch.setattr(lineage,'execution_sources',lambda root:{'model.py':'sha'})
    with pytest.raises(ValueError,match='after predictions'):lineage.bind_execution(tmp_path,{}, {})


def test_execution_binding_does_not_rewrite_changed_configuration(tmp_path,monkeypatch):
    from gog_fraud.data import benchmark_lineage as lineage
    evidence=tmp_path/'projects/benchmark/evidence/a08_data_repair';evidence.mkdir(parents=True)
    monkeypatch.setattr(lineage,'execution_sources',lambda root:{'model.py':'sha'})
    lineage.bind_execution(tmp_path,{'epoch':40},{})
    before=(evidence/'execution_manifest.json').read_bytes()
    with pytest.raises(ValueError,match='changed after binding'):lineage.bind_execution(tmp_path,{'epoch':30},{})
    assert (evidence/'execution_manifest.json').read_bytes()==before


@pytest.mark.parametrize('defect',['wrong_input','wrong_split','wrong_config','qualification','short_epoch','missing_score','wrong_score_hash','mixed_labels','selected_seed_mask'])
def test_invalid_run_lineage_rejected(tmp_path,monkeypatch,defect):
    from gog_fraud.data import benchmark_lineage as lineage
    identity={'input_manifest_hash':'new','split_hash':'split42','model_config_hash':'fixed','execution_manifest_hash':'exec','scope':'primary','resolved_model_config':{}}
    monkeypatch.setattr(lineage,'expected_run_identity',lambda *a,**k:identity)
    root=tmp_path/'local_storage/benchmark/a08_data_repair';output=root/'runs'/'fixture';output.mkdir(parents=True)
    frozen=root/'frozen/polygon';frozen.mkdir(parents=True)
    arrays={'node_ids':np.array(['a','b','c']),'labels':np.array([0,1,0]),'val_mask':np.array([True,False,False]),'test_mask':np.array([False,True,True]),'scores':np.array([.2,.8,.4])}
    np.savez_compressed(frozen/'contract_graph.npz',node_ids=arrays['node_ids'],labels=arrays['labels'],val_mask_42=arrays['val_mask'],test_mask_42=arrays['test_mask'])
    if defect=='mixed_labels':arrays['labels']=np.array([1,0,0])
    if defect=='selected_seed_mask':arrays['test_mask']=np.array([True,False,True])
    np.savez_compressed(output/'scores.npz',**arrays)
    record={**identity,'dataset_id':'polygon_contract_clean_v1','model':'DOMINANT','seed':42,'run_id':'fixture','status':'SUPPORTED_EXACT','planned_global_epochs':40,'completed_epochs':40,'score_array_hash':lineage.array_hash(arrays['scores'])}
    for name,field in [('scores.npz','raw_scores_sha256'),('final_model_state.pt','checkpoint_sha256'),('metrics.json','metrics_sha256'),('loss_curve.json','loss_curve_sha256')]:
        if name!='scores.npz':(output/name).write_bytes(b'test fixture only, not trained')
        record[field]=lineage.sha256(output/name)
    if defect.startswith('wrong_') and defect!='wrong_score_hash':record[{'wrong_input':'input_manifest_hash','wrong_split':'split_hash','wrong_config':'model_config_hash'}[defect]]='old'
    if defect=='qualification':record['scope']='qualification'
    if defect=='short_epoch':record['completed_epochs']=1
    if defect=='missing_score':(output/'scores.npz').unlink()
    if defect=='wrong_score_hash':record['raw_scores_sha256']='oldscore'
    with pytest.raises((ValueError,FileNotFoundError)):lineage.verify_run(tmp_path,{'epochs':{'polygon':40}},record)


def test_unexpected_checker_error_clears_previous_pass(tmp_path,monkeypatch):
    report=tmp_path/'FINAL_ACCEPTANCE_A08.json'
    report.write_text(json.dumps({'final_scientific_status':'FINAL_PASS'}))
    monkeypatch.setattr(checker,'R',tmp_path)
    def broken():raise RuntimeError('corrupt artifact parser')
    monkeypatch.setattr(checker,'validate_final',broken)
    assert checker.main()==1
    result=json.loads(report.read_text())
    assert result['final_scientific_status']=='NOT_PASSED' and result['detail']['checker_error_type']=='RuntimeError'


@pytest.mark.parametrize('defect',['fabricated_number','arbitrary_source','wrong_seed_membership'])
def test_pdf_crosswalk_must_resolve_to_actual_approved_number(tmp_path,defect):
    evidence=tmp_path/'projects/benchmark/evidence/a08_data_repair';evidence.mkdir(parents=True)
    (evidence/'number_registry.json').write_text(json.dumps({'cells':[{'dataset':'Polygon','model':'DLG-Aug','metric':'pr_auc','mean':.12345,'sd':.00456,'run_ids':['seed42','seed43','seed44','seed45','seed46']}]}))
    row={'claim_id':'Polygon/AP','rendered':'0.1235 ± 0.0046','source':'number_registry.json','source_key':'Polygon/DLG-Aug/pr_auc','run_ids':'seed42;seed43;seed44;seed45;seed46','allowed_rounding':'4 decimals'}
    assert checker.resolve_crosswalk_number(tmp_path,row)==row['rendered']
    if defect=='fabricated_number':row['rendered']='0.9999 ± 0.0046'
    if defect=='arbitrary_source':row['source']='invented_PASS.json'
    if defect=='wrong_seed_membership':row['run_ids']='seed42'
    with pytest.raises(ValueError):checker.resolve_crosswalk_number(tmp_path,row)


def test_raw_counterfactual_pass_without_five_actual_builds_rejected(tmp_path):
    evidence=tmp_path/'projects/benchmark/evidence/a08_data_repair/audit';evidence.mkdir(parents=True)
    (evidence/'raw_full_counterfactual.json').write_text(json.dumps({'status':'PASS','results':[]}))
    with pytest.raises(ValueError,match='coverage'):checker.check_raw_rebuilds(tmp_path)


@pytest.mark.parametrize('defect',['empty','short','nonfinite','inference_only'])
def test_declared_epoch_metadata_without_actual_training_observations_rejected(defect):
    record={'planned_global_epochs':2,'resolved_model_config':{'scalar_attributes':{'batch_size':3}}}
    curve={'granularity':'forward_model call (minibatch)','samples':[{'phase':'training','loss':1.} for _ in range(4)]}
    checker.check_loss_budget(record,curve,5)
    if defect=='empty':curve['samples']=[]
    if defect=='short':curve['samples']=curve['samples'][:2]
    if defect=='nonfinite':curve['samples'][0]['loss']=float('nan')
    if defect=='inference_only':
        for sample in curve['samples']:sample['phase']='inference'
    with pytest.raises(ValueError):checker.check_loss_budget(record,curve,5)


@pytest.mark.parametrize('source,identity,field,number,display',[
    ('statistics/mean_ranks.csv','view','mean_rank','2.375','2.3750'),
    ('statistics/label_provenance_sensitivity.csv','group','mean_rank','2.0','2.0000'),
    ('statistics/pairwise.csv','view','wins','4','4'),
])
@pytest.mark.parametrize('defect',['wrong_value','wrong_field'])
def test_statistics_display_rejects_fabricated_values_or_fields(tmp_path,source,identity,field,number,display,defect):
    evidence=tmp_path/'projects/benchmark/evidence/a08_data_repair'
    file=evidence/source;file.parent.mkdir(parents=True)
    model='DLG-Aug vs DOMINANT' if source.endswith('pairwise.csv') else 'DLG-Aug'
    column='comparison' if source.endswith('pairwise.csv') else 'model'
    file.write_text(f'{identity},{column},{field}\nS1,{model},{number}\n')
    row={'claim_id':'statistics/actual','source':source,'source_key':f'S1/{model}/{field}','rendered':display}
    assert checker.resolve_crosswalk_number(tmp_path,row)==display
    if defect=='wrong_value':row['rendered']='99.9999'
    else:row['source_key']=f'S1/{model}/unapproved_field'
    with pytest.raises((ValueError,KeyError)):checker.resolve_crosswalk_number(tmp_path,row)

@pytest.mark.parametrize('defect',['missing_category','changed_source','old_pdf','empty_observation'])
def test_claim_mapping_rejects_missing_or_stale_actual_evidence(tmp_path,defect):
    import csv
    e=tmp_path/'projects/benchmark/evidence/a08_data_repair';e.mkdir(parents=True)
    source=tmp_path/'actual_evidence.json';source.write_text('{}')
    categories=['input_provenance','threshold_and_metric_policy','support_and_run_counts','statistical_scope','resource_claims','old_new_population_comparison','public_private_availability','historical_diagnostics','citations_and_equations']
    rows=[{'artifact_path':'reviewed.pdf','pdf_hash':'current','claim_id':c,'check_status':'PASS','page_or_table':'1','rendered_value_or_claim':'fixture observation only','source_metric_or_table_key':'actual_evidence.json','source_sha256':checker.sha256(source)} for c in categories]
    def save():
        with (e/'claim_evidence_map.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save();checker.check_claim_map(tmp_path,{'reviewed.pdf':'current'})
    if defect=='missing_category':rows.pop()
    if defect=='changed_source':source.write_text('{"altered":true}')
    if defect=='old_pdf':rows[0]['pdf_hash']='old'
    if defect=='empty_observation':rows[0]['rendered_value_or_claim']=''
    save()
    with pytest.raises(ValueError):checker.check_claim_map(tmp_path,{'reviewed.pdf':'current'})
