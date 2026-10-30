"""Author attestation must persist without turning into invented external proof."""
import importlib.util
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'results/317'
FOLLOW=OUT/'author_confirmation_20261030'
spec=importlib.util.spec_from_file_location('author317_common',PROJECT/'scripts/revision317/common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


def test_author_confirmation_is_current_but_not_a_fabricated_external_receipt():
    s=json.loads((OUT/'submission_readiness_317.json').read_text())
    assert s['AUTHOR_DETAILS_AND_DECLARATIONS_CONFIRMED'] is True
    assert s['IDENTITY_AND_CORRESPONDENCE_CONFIRMED'] is True
    assert s['CREDIT_CONFIRMED'] is True and s['COI_DECLARATION_CONFIRMED'] is True
    assert s['FUNDING_STATEMENT_CONFIRMED'] is True
    assert s['OFFICIAL_AWARD_DOCUMENT_VERIFIED'] is False
    assert s['EXACT_FILE_COAUTHOR_SIGNATURES_VERIFIED'] is False
    assert s['PUBLIC_RELEASE_VERIFIED'] is False and s['PREPRINT_DOI'] is None
    assert all(s[k] is False for k in ['SUBMITTED_PREPRINT','POSTED_PREPRINT','SUBMITTED_MDPI'])


def test_author_update_preserves_scientific_body_and_sealed_numeric_package():
    s=json.loads((FOLLOW/'confirmation_status.json').read_text())
    assert s['scientific_body_unchanged'] is True and s['numerical_archive_unchanged'] is True
    assert common.sha(PROJECT/'publication/317/TDS_317_numeric_release_candidate.zip')==s['numeric_candidate_sha256']
    assert s['numerical_csvs_verified']>200
    build=json.loads((PROJECT/'reports/317/author_confirmation_20261030/build_verification.json').read_text())
    base=json.loads((PROJECT/'private/317/before_author_confirmation_20261030/baseline.json').read_text())
    assert build['shared_body_sha256']==base['body_sha256']
    assert build['shared_bib_sha256']==base['bib_sha256']


def test_current_wrappers_and_declared_facts_match_a16_user_confirmation():
    record=json.loads((PROJECT/'submission/author_records/317/author_confirmation_from_a16.json').read_text())
    for venue,entry in [('preprints','preprint_main.tex'),('mdpi_applsci_anomaly','applsci_article.tex')]:
        text=(PROJECT/'paper/submission_317'/venue/entry).read_text()
        for a in record['authors']:
            assert a['name'] in text and a['affiliation'] in text
            if a.get('orcid'):assert a['orcid'] in text
            else:assert a['orcid_provided_in_A16'] not in text
        assert 'requires confirmation' not in text
    declarations=(PROJECT/'paper/submission_317/shared/declarations.tex').read_text()
    assert record['coi'] in declarations and record['funding_identifier'] in declarations
    assert 'outstanding' not in declarations and 'await author verification' not in declarations
