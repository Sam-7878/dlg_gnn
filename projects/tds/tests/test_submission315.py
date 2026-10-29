"""Prevent a scientific PASS or a truthy placeholder from authorizing submission."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/revision315'))
from gates import REQUIRED, evaluate


def test_science_without_author_and_release_fails_closed():
    out=evaluate({'ARTIFACT_INTEGRITY_PASS':True,'SCIENTIFIC_EVIDENCE_COMPLETE':True,
                  'MDPI_SOURCE_COMPILE_PASS':True,'CLAIM_CONSISTENCY_PASS':True})
    assert out['decision']=='HOLD_AUTHOR_ACTION' and out['SUBMISSION_READY'] is False
    assert 'PUBLIC_RELEASE_VERIFIED' in out['pending_requirements']


def test_unverified_strings_cannot_pass_a_gate():
    flags=dict.fromkeys(REQUIRED,True);flags['PUBLIC_RELEASE_VERIFIED']='pending'
    assert evaluate(flags)['SUBMISSION_READY'] is False
    assert evaluate({})['decision']=='HOLD_EVIDENCE_INCONSISTENCY'


def test_readiness_never_claims_submission_or_overrides_identity_failure():
    flags=dict.fromkeys(REQUIRED,True)
    assert evaluate(flags)['SUBMISSION_READY'] is True
    assert evaluate(flags)['SUBMITTED'] is False
    flags['ARTIFACT_INTEGRITY_PASS']=False
    assert evaluate(flags)['decision']=='HOLD_EVIDENCE_INCONSISTENCY'
