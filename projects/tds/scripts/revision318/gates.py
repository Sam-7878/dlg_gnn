"""Evidence gates: supplied author facts never imply permission to publish."""
def evaluate(science_pass, candidate_pass, facts):
    common = all(facts.get(k) is True for k in (
        'exact_file_author_approvals', 'official_funding_evidence',
        'funder_role_confirmed', 'declarations_confirmed', 'rights_cleared'))
    evidence = (facts.get('public_release_verified') is True and facts.get('anonymous_replay_pass') is True)
    evidence = evidence or facts.get('restricted_plan_accepted_by_platform') is True
    preprint = science_pass and candidate_pass and common and evidence and facts.get('both_author_preprint_consent') is True
    # Follow the requested Preprints-first sequence; posting requires a real DOI/disclosure.
    posting_ok = (facts.get('preprint_posted') is not True or
                  (bool(facts.get('preprint_doi')) and facts.get('preprint_disclosed_to_journal') is True))
    journal = (preprint and facts.get('both_author_journal_consent') is True and
               facts.get('special_issue_selection_verified') is True and posting_ok)
    return dict(science_gate='PASS' if science_pass else 'FAIL',
                candidate_integrity_gate='PASS' if candidate_pass else 'FAIL',
                preprint_release_gate='PASS' if preprint else 'HOLD_AUTHOR_ACTION',
                journal_submission_gate='PASS' if journal else 'HOLD_AUTHOR_ACTION',
                READY_PREPRINT=bool(preprint), READY_MDPI=bool(journal))
