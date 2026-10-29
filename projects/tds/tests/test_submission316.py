"""Prevent readiness, DOI, or approval claims unsupported by venue evidence."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('tds316_gates',Path(__file__).resolve().parents[1]/'scripts/revision316/gates.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def test_science_and_builds_do_not_imply_author_approval():
    out=g.evaluate({'P0':True,'P4':True,'P5':True,'P7':True},{'J1':True,'J2':True})
    assert out['READY_PREPRINT'] is False and out['READY_MDPI'] is False
    assert out['PREPRINT_DOI'] is None and out['SUBMITTED_PREPRINT'] is False


def test_every_preprint_gate_is_required_and_strings_fail_closed():
    flags=dict.fromkeys(g.P_KEYS,True)
    assert g.evaluate(flags,{})['READY_PREPRINT'] is True
    for key in g.P_KEYS:
        f=flags.copy();f[key]='pending'
        assert g.evaluate(f,{})['READY_PREPRINT'] is False


def test_ready_preprint_never_claims_submitted_or_posted():
    out=g.evaluate(dict.fromkeys(g.P_KEYS,True),{})
    assert out['READY_PREPRINT'] is True
    assert out['SUBMITTED_PREPRINT'] is False and out['POSTED_PREPRINT'] is False
    assert out['PREPRINT_DOI'] is None


def test_journal_waits_for_actual_preprint_posting():
    j=dict.fromkeys(g.J_KEYS,True)
    assert g.evaluate({},j)['READY_MDPI'] is False
    platform=g.platform_status({'proof_verified':True,'submission_id':'recorded'},
                               {'proof_verified':True,'doi':'10.12345/example','url':'https://example.org/preprint','version':'1'})
    assert g.evaluate({},j,platform)['READY_MDPI'] is True
    assert g.evaluate({},j,platform)['SUBMITTED_MDPI'] is False


def test_doi_without_verified_submission_and_posting_is_not_reported():
    p={'proof_verified':True,'doi':'10.12345/example','url':'https://example.org/preprint','version':'1'}
    assert g.platform_status(posting=p)['PREPRINT_DOI'] is None
    p['proof_verified']='true'
    assert g.platform_status({'proof_verified':True,'submission_id':'recorded'},p)['POSTED_PREPRINT'] is False


def test_signoff_is_hash_scope_and_author_specific():
    records=[dict(author_id=i,scope='preprints',proof_verified=True,article_sha256='a',supplement_sha256='s',
                  negative_finding_accepted=True,public_posting_irreversibility_accepted=True,date='2026-10-29')
             for i in ['S.-S.P.','K.-H.K.']]
    assert g.author_signoffs_valid(records,'a','s','preprints')
    assert not g.author_signoffs_valid(records,'changed','s','preprints')
    assert not g.author_signoffs_valid(records,'a','s','mdpi')
    assert not g.author_signoffs_valid([records[0],records[0]],'a','s','preprints')


def test_null_platform_identifiers_fail_closed_without_making_a_doi():
    state=g.platform_status({'proof_verified':True,'submission_id':'recorded'},
                            {'proof_verified':True,'doi':None,'url':None,'version':None})
    assert state['POSTED_PREPRINT'] is False and state['PREPRINT_DOI'] is None
