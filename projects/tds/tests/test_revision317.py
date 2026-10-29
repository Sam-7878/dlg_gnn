"""Scientific and release invariants for the sampling-validity revision."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'results/317'
spec=importlib.util.spec_from_file_location('tds317_replay',PROJECT/'scripts/revision317/replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def test_complete_lineage_uses_contract_unit_and_unknown_label_time():
    f=pd.read_csv(OUT/'sample_time_lineage.csv')
    assert len(f)==24316 and f.sample_id.is_unique
    assert (f.first_valid_tx_time<=f.last_valid_tx_time).all()
    assert (f.original_sample_cutoff==f.last_valid_tx_time).all()
    assert f.label_time_unknown.all() and f.label_asof_time_if_available.isna().all()
    assert ((f.category==0).astype(int)==f.label).all()


def test_selection_support_does_not_manufacture_censoring_causality():
    d=json.loads((OUT/'selection_decision.json').read_text())
    assert d['decision']=='SELECTION_EFFECT_SUPPORTED'
    assert d['original_prevalence'][0]>d['original_prevalence'][2]
    assert d['first_time_prevalence'][0]<d['first_time_prevalence'][2]
    assert d['right_censoring_causal_mechanism_established'] is False
    assert d['exact_collection_cutoff_policy_verified'] is False


def test_horizon_eligibility_does_not_require_survival_and_rebuilds_features():
    f=pd.read_csv(OUT/'sample_time_lineage.csv').set_index('sample_id')
    h=pd.read_csv(OUT/'horizon7_population.csv')
    expected=f.index[f.first_valid_tx_time+7*86400<=1709251199]
    assert set(h.sample_id)==set(expected) and len(h)==24179
    assert np.array_equal(h.timestamp.to_numpy(),f.loc[h.sample_id,'first_valid_tx_time'].to_numpy()+7*86400)
    assert (f.loc[h.sample_id,'last_valid_tx_time'].to_numpy()<h.timestamp.to_numpy()).any()
    a=pd.read_csv(OUT/'horizon7_feature_audit.csv')
    assert len(a)==len(h) and (a.max_input_timestamp<=a.cutoff).all()
    assert (a.valid_rows_through_cutoff>=1).all() and (a.edges<=128).all()


def test_new_cohort_is_never_marked_pairwise_comparable_to_original():
    m=json.loads((OUT/'horizon7_dataset_manifest.json').read_text())
    assert m['test_comparable_to_original'] is False and m['paired_cross_cohort_inference'] is False
    a=pd.read_csv(OUT/'original_population.csv');b=pd.read_csv(OUT/'horizon7_population.csv')
    assert set(a[a.split=='test'].sample_id)!=set(b[b.split=='test'].sample_id)


def test_all_45_new_fits_are_protocol_bound_and_score_once():
    files=list((OUT/'runs').glob('*/*/seed*/metrics.json'));assert len(files)==45
    digest=replay.sha(OUT/'experiment_preregistration.json')
    for p in files:
        d=json.loads(p.read_text());assert d['test_access_count']==1 and d['protocol_sha256']==digest
        assert replay.sha(p.parent/'test.csv')==d['test_predictions_sha256']
        assert replay.sha(p.parent/'checkpoint.pt')==d['checkpoint_sha256']


def test_tie_grouped_ap_matches_sklearn_with_bootstrap_duplicates():
    y=np.array([1,0,1,0]);p=np.array([.5,.5,.2,.2]);ids=np.array([0,0,1,2,2,3])
    assert abs(replay.tied_ap(y,p)-average_precision_score(y,p))<1e-12
    assert abs(replay.tied_ap(y[ids],p[ids])-average_precision_score(y[ids],p[ids]))<1e-12


def test_calibration_is_positive_and_rank_preserving_not_an_ap_upgrade():
    c=pd.read_csv(OUT/'modelwise_calibration.csv')
    assert len(c)==95 and (c.temperature>0).all() and c.ap_preserved.all()
    assert np.allclose(c.raw_auc_pr,c.ts_auc_pr,atol=1e-12,rtol=0)


def test_outlier_scores_have_no_probability_losses_and_polygon_no_fake_ci():
    m=pd.read_csv(OUT/'per_seed_metrics.csv');f=m[m.arm=='isolation_forest']
    assert len(f)==5 and f[['ece','nll','brier','f1']].isna().all().all()
    c=pd.read_csv(OUT/'chain_stratified_ap.csv');p=c[c.chain=='polygon']
    assert (p.positive==1).all() and p.ci95_low.isna().all() and p.ap_mean.notna().all()


def test_registered_65_original_and_45_new_arrays_are_distinct():
    r=pd.read_csv(OUT/'run_registry.csv');assert len(r)==110 and r.run_id.is_unique
    assert len(r[r.panel.isin(['matched','archival'])])==65
    s=pd.read_csv(OUT/'statistics_and_holm.csv')
    assert sorted(s.groupby('family').size())==[6,7]
    assert (s.randomization_p_value_two_sided>=1/10001-1e-15).all()


def test_science_cannot_authorize_publication_without_real_proofs():
    s=json.loads((OUT/'submission_readiness_317.json').read_text())
    assert s['READY_PREPRINT'] is False and s['READY_MDPI'] is False
    assert s['AUTHOR_APPROVAL_COMPLETE'] is False and s['FUNDING_VERIFIED'] is False
    assert s['PUBLIC_RELEASE_VERIFIED'] is False and s['PREPRINT_DOI'] is None
    assert all(s[k] is False for k in ['SUBMITTED_PREPRINT','POSTED_PREPRINT','SUBMITTED_MDPI'])
