"""Publication gates and exact-file evidence: no model/runtime initialization."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
spec = importlib.util.spec_from_file_location('gates318', PROJECT / 'scripts/revision318/gates.py')
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


def approved_facts():
    return dict(exact_file_author_approvals=True, official_funding_evidence=True,
                funder_role_confirmed=True, declarations_confirmed=True, rights_cleared=True,
                public_release_verified=True, anonymous_replay_pass=True,
                both_author_preprint_consent=True, both_author_journal_consent=True,
                special_issue_selection_verified=True, preprint_posted=True,
                preprint_doi='10.example/test-fixture-only', preprint_disclosed_to_journal=True)


@pytest.mark.parametrize('missing', ['exact_file_author_approvals', 'rights_cleared',
                                    'anonymous_replay_pass', 'official_funding_evidence'])
def test_supplied_author_facts_or_scientific_success_cannot_replace_a_required_gate(missing):
    f = approved_facts()
    f[missing] = False
    result = gates.evaluate(True, True, f)
    assert not result['READY_PREPRINT'] and not result['READY_MDPI']
    f[missing] = 'VERIFIED'  # Only explicit boolean evidence is accepted here.
    assert not gates.evaluate(True, True, f)['READY_PREPRINT']


def test_publication_posting_and_journal_are_distinct():
    f = approved_facts()
    f['preprint_doi'] = None
    result = gates.evaluate(True, True, f)
    assert result['READY_PREPRINT'] and not result['READY_MDPI']
    f = approved_facts()
    f['special_issue_selection_verified'] = False
    assert not gates.evaluate(True, True, f)['READY_MDPI']
    assert not gates.evaluate(False, True, approved_facts())['READY_PREPRINT']
    assert not gates.evaluate(True, False, approved_facts())['READY_PREPRINT']


def test_current_artifacts_and_unsigned_approval_forms_bind_same_exact_bytes():
    state = json.loads((PROJECT / 'results/318/candidate_identity.json').read_text())
    assert state['current_final_candidate_count'] == 1
    approval = json.loads((PROJECT / 'submission/author_records/318/approval_identity.json').read_text())
    assert approval['candidate_id'] == state['current_final_candidate']
    forms = list((PROJECT / 'submission/author_records/318').glob('author_signoff_*.md'))
    assert len(forms) == 2
    artifacts = [e for v in state['venues'] for e in v['current'].values()]
    for e in artifacts:
        assert hashlib.sha256((ROOT / e['path']).read_bytes()).hexdigest() == e['sha256']
        assert e in approval['artifacts']
        assert all(e['sha256'] in f.read_text() for f in forms)
    assert approval['author_details_and_declarations_confirmed']
    assert not approval['permission_to_publish_or_submit']
    supplied = PROJECT / 'submission/author_records/317/author_confirmation_from_a16.json'
    assert hashlib.sha256(supplied.read_bytes()).hexdigest() == approval['supplied_author_record_sha256']


def test_table_reflow_preserves_scientific_rows_and_current_authority():
    base = PROJECT / 'paper/submission_317/shared'
    current = PROJECT / 'paper/submission_318/shared'
    for n in ['manuscript_body.tex','references.bib']:
        assert (base / n).read_bytes() == (current / n).read_bytes()
    before = (base / 'data/reusable_validity_checklist.tex').read_text()
    after = (current / 'data/reusable_validity_checklist.tex').read_text()
    assert before[before.index('\\caption'):] == after[after.index('\\caption'):]
    readiness = json.loads((PROJECT / 'results/318/submission_readiness_318.json').read_text())
    assert readiness['science_gate'] == readiness['candidate_integrity_gate'] == 'PASS'
    assert readiness['author_details_and_declarations_confirmed']
    assert readiness['original_317_science_changed'] is False
    assert readiness['READY_PREPRINT'] is readiness['READY_MDPI'] is False
