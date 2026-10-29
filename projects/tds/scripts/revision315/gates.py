"""Fail-closed submission readiness, independent of scientific runtime."""

REQUIRED = (
    'ARTIFACT_INTEGRITY_PASS', 'SCIENTIFIC_EVIDENCE_COMPLETE',
    'AUTHOR_APPROVAL_COMPLETE', 'FUNDING_VERIFIED', 'DECLARATIONS_COMPLETE',
    'PUBLIC_RELEASE_VERIFIED', 'MDPI_SOURCE_COMPILE_PASS',
    'CLAIM_CONSISTENCY_PASS', 'PUBLISHER_PREFLIGHT_COMPLETE', 'DELIVERY_COMPLETE',
)


def evaluate(flags):
    missing = [key for key in REQUIRED if flags.get(key) is not True]
    ready = not missing
    if flags.get('ARTIFACT_INTEGRITY_PASS') is not True:
        decision = 'HOLD_EVIDENCE_INCONSISTENCY'
    else:
        decision = 'SUBMISSION_READY' if ready else 'HOLD_AUTHOR_ACTION'
    return dict(flags={k: flags.get(k) is True for k in REQUIRED},
                pending_requirements=missing, decision=decision,
                SUBMISSION_READY=ready, SUBMITTED=False)
