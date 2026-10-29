"""Separate supported scientific scope from human/publication authorization."""
S_KEYS=('S0','S1','S2','S3','S4','S5','S6')


def evaluate(science,technical,proofs=None):
    proofs=proofs or {}
    s=all(science.get(k) is True for k in S_KEYS)
    author=proofs.get('author_exact_hash_signoffs') is True
    funding=proofs.get('official_funding') is True
    identity=proofs.get('identity_correspondence') is True
    declarations=proofs.get('coi_funder_originality_rights') is True
    license_ok=proofs.get('preprint_and_numeric_rights') is True
    public=proofs.get('immutable_anonymous_TDS_release') is True
    availability=public or proofs.get('publisher_accepted_restriction') is True
    ready_p=s and author and funding and identity and declarations and license_ok and availability and technical.get('preprint_clean_build') is True and technical.get('numeric_cold_replay') is True
    # A DOI string alone is not authenticated platform evidence.
    posted=proofs.get('preprint_posting_receipt') is True
    ready_j=ready_p and posted and proofs.get('journal_exact_hash_signoffs') is True and proofs.get('live_SI_portal') is True and technical.get('mdpi_clean_build') is True
    return dict(GATE_S='PASS' if s else 'HOLD_SCIENTIFIC_EVIDENCE',SCIENCE_CHECKS=science,
        SCIENTIFIC_SCOPE='sampling-associated composition; retrospective archive evaluation; no identified censoring mechanism',
        COLLECTION_EXACT_CUTOFF_VERIFIED=False,LABEL_DISCOVERY_TIME_AVAILABLE=False,
        AUTHOR_APPROVAL_COMPLETE=author,FUNDING_VERIFIED=funding,PUBLIC_RELEASE_VERIFIED=public,
        READY_PREPRINT=ready_p,READY_MDPI=ready_j,PREPRINT_DOI=None,
        SUBMITTED_PREPRINT=False,POSTED_PREPRINT=False,SUBMITTED_MDPI=False,
        decision='HOLD_AUTHOR_ACTION_AND_PUBLIC_RELEASE' if s and not ready_p else 'HOLD_SCIENTIFIC_EVIDENCE' if not s else 'READY_FOR_AUTHORIZED_PLATFORM_ACTION',
        technical_checks=technical,proofs_verified=proofs)
