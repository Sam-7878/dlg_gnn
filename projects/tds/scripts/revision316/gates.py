"""Pure fail-closed publication gates; readiness never performs submission."""
import re

P_KEYS=tuple(f'P{i}' for i in range(8))
J_KEYS=tuple(f'J{i}' for i in range(8))


def author_signoffs_valid(records,article_sha256,supplement_sha256,scope):
    accepted=set()
    for e in records:
        if (e.get('scope')==scope and e.get('proof_verified') is True
            and e.get('article_sha256')==article_sha256
            and e.get('supplement_sha256')==supplement_sha256
            and e.get('negative_finding_accepted') is True
            and (scope!='preprints' or e.get('public_posting_irreversibility_accepted') is True)
            and e.get('date')):
            accepted.add(e.get('author_id'))
    return accepted=={'S.-S.P.','K.-H.K.'}


def platform_status(submission=None,posting=None,journal_submission=None):
    submission=submission or {};posting=posting or {};journal_submission=journal_submission or {}
    submitted=(submission.get('proof_verified') is True and bool(submission.get('submission_id')))
    doi=posting.get('doi');url=posting.get('url')
    posted=(submitted and posting.get('proof_verified') is True
            and isinstance(doi,str) and bool(re.fullmatch(r'10\.\d{4,9}/\S+',doi))
            and isinstance(url,str) and url.startswith('https://') and bool(posting.get('version')))
    return dict(SUBMITTED_PREPRINT=submitted,POSTED_PREPRINT=posted,
                PREPRINT_DOI=posting['doi'] if posted else None,
                PREPRINT_URL=posting['url'] if posted else None,
                PREPRINT_VERSION=posting['version'] if posted else None,
                SUBMITTED_MDPI=journal_submission.get('proof_verified') is True and bool(journal_submission.get('submission_id')))


def evaluate(p,j,platform=None):
    platform=platform or platform_status()
    p_out={k:p.get(k) is True for k in P_KEYS};j_out={k:j.get(k) is True for k in J_KEYS}
    ready_p=all(p_out.values())
    # The author's requested sequence is preprint-first. Readiness of the
    # journal package does not manufacture a posted preprint or DOI.
    ready_j=all(j_out.values()) and platform.get('POSTED_PREPRINT') is True
    return dict(P=p_out,J=j_out,READY_PREPRINT=ready_p,READY_MDPI=ready_j,
                preprint_decision='READY_FOR_AUTHOR_ACTION' if ready_p else 'HOLD_AUTHOR_ACTION',
                journal_decision='READY_FOR_AUTHOR_ACTION' if ready_j else 'HOLD_AUTHOR_ACTION_AND_VENUE_PREFLIGHT',
                missing_preprint=[k for k,v in p_out.items() if not v],
                missing_mdpi=[k for k,v in j_out.items() if not v]+([] if platform.get('POSTED_PREPRINT') is True else ['PREPRINT_FIRST_WORKFLOW']),
                **platform)
