"""Assemble private handoff and compute readiness from evidence, never submit."""
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from paths import ROOT, PAPER, INPUT, RESULTS, REPORTS, PUBLICATION
from freeze import sha, write
from gates import evaluate


def load(path):
    return json.loads(path.read_text())


def evidence_exists(record):
    if not isinstance(record,dict) or record.get('verified') is not True:
        return False
    p=ROOT/record.get('path','')
    return p.is_file() and record.get('sha256')==sha(p)


def main():
    audit=load(REPORTS/'evidence_audit.json')
    build=load(REPORTS/'build_verification.json')
    claims=load(REPORTS/'claim_consistency.json')
    local=load(REPORTS/'candidate_local_verification.json')
    local['test_output']=re.sub(r'/tmp/tds315-numeric-[^/\s]+','<fresh-extraction>',
                               local['test_output'].replace(str(ROOT.parent),'<workspace>'))
    write(REPORTS/'candidate_local_verification.json',local)
    assert local['archive_sha256']==sha(PUBLICATION/'TDS_315_numeric_candidate.zip')
    assert local['invariant_tests']=='8 passed'
    # Verify the complete original seal again at handoff, accepting only the
    # two path repairs with the exact post-repair hashes already audited.
    repairs={e['path']:e['after'] for e in audit['source_path_repairs']}
    for e in load(RESULTS/'input_hash_manifest.json')['files']:
        assert sha(ROOT/e['path'])==repairs.get(e['path'],e['sha256']),e['path']
    final=load(PAPER/'final_artifact_manifest.json')
    for e in final['files']:
        assert sha(PAPER/e['path'])==e['sha256'],e['path']
    article=sha(PAPER/'applsci_article.pdf');supplement=sha(PAPER/'supplement/supplementary.pdf')
    author=load(PAPER/'author_closure_status.json')
    signoffs=author['final_signoffs']
    signed=set()
    for e in signoffs:
        if (e.get('article_sha256')==article and e.get('supplement_sha256')==supplement
            and e.get('negative_finding_accepted') is True and e.get('date')
            and evidence_exists(e.get('confirmation_record'))):
            signed.add(e.get('author_id'))
    manuscript=(PAPER/'applsci_article.tex').read_text()
    approval=(author['author_identity_confirmed'] is True and signed=={'S.-S.P.','K.-H.K.'}
              and 'Final approval of this TDS manuscript version remains pending' not in manuscript
              and 'Corresponding-author designation pending' not in manuscript)
    release=author.get('public_release') or {}
    released=(author['numeric_redistribution_approved'] is True
              and evidence_exists(release.get('anonymous_access_proof'))
              and bool(release.get('url')) and bool(release.get('tag'))
              and bool(re.fullmatch('[0-9a-f]{40}',release.get('commit','')))
              and release.get('archive_sha256')==local['archive_sha256']
              and 'immutable public tag and anonymous download verification remain pending' not in manuscript)
    needed=['final_cover_letter.md','mdpi_submission_metadata.md','data_code_availability.md',
            'author_approval_record.md','credit_confirmed.json','funding_evidence_register.md',
            'declarations_confirmed.md','submission_source.zip']
    flags=dict(ARTIFACT_INTEGRITY_PASS=audit['status']=='PASS',
        SCIENTIFIC_EVIDENCE_COMPLETE=audit['status']=='PASS' and claims['status']=='PASS',
        AUTHOR_APPROVAL_COMPLETE=approval,
        FUNDING_VERIFIED=evidence_exists(author['official_funding_evidence']),
        DECLARATIONS_COMPLETE=(author['declarations_complete'] is True and evidence_exists(author['declarations_confirmation_record'])
                               and 'TDS-specific conflict-of-interest declaration is pending' not in manuscript),
        PUBLIC_RELEASE_VERIFIED=released,
        MDPI_SOURCE_COMPILE_PASS=build['status']=='PASS' and build['all_clean_pdf_text_and_rendered_pixels_identical'] is True,
        CLAIM_CONSISTENCY_PASS=claims['status']=='PASS',
        PUBLISHER_PREFLIGHT_COMPLETE=(author['publisher_preflight_complete'] is True and evidence_exists(author['publisher_preflight_evidence'])),
        DELIVERY_COMPLETE=all((PAPER/n).is_file() for n in needed))
    result=evaluate(flags)
    result.update(work_order='TDS-WO-315-SUBMISSION-CLOSURE',date='2026-10-29',
        article_sha256=article,supplement_sha256=supplement,
        source_zip_sha256=sha(PAPER/'submission_source.zip'),
        numeric_candidate_sha256=local['archive_sha256'],
        user_authorized_a16_sections_applied=True,
        new_training_performed=False,release_url=release.get('url'),release_commit=release.get('commit'),
        gates={'G0':'PASS','G1':'PARTIAL_AUTHOR_FACTS; FINAL_APPROVAL_PENDING',
               'G2':'LOCAL_CANDIDATE_PASS; PUBLIC_RELEASE_PENDING','G3':'PASS',
               'G4':'SOURCE_BUILD_PASS; CURRENT_TEMPLATE_AND_PORTAL_PENDING',
               'G5':'LOCAL_DELIVERY_PASS','G6':result['decision']})
    write(REPORTS/'submission_readiness_315.json',result)
    report=f'''# TDS 315 submission closure

**Scientific evidence: COMPLETE. Engineering delivery: COMPLETE with documented
publisher access limits. Submission: {result['decision']}. SUBMISSION_READY=false;
SUBMITTED=false.** No new training, Git push, public release or journal submission.

The canonical preserved R314 input is `projects/tds/paper/applsci/`.
The revised private delivery is `projects/tds/paper/applsci/submission_315/`.
Task numerics are in `projects/results/tds/315_applsci/`; public audit, code,
publication metadata and submission index are separated under `projects/tds/`.
The user's existing old-path deletions were preserved. No Benchmark files changed.

## Completed evidence and manuscript work

- Sealed 405 inputs; original PDF/ZIP identities and CRC retained; 88 primary
  artifacts and 20 primary + 35 supplemental checkpoint hashes verified.
- All 35 supplemental and 30 primary test arrays align at 3648 samples /107
  positives. Seven supplemental AP mean/sample-SD pairs match sealed results.
- Corrected seven-family and six-family Holm values checked separately; one
  complete original comparison replayed with all three 10000-replicate protocols.
- Existing eight scientific tests pass in the repository and manuscript-free
  candidate, including a fresh extracted ZIP. Three new fail-closed gate tests
  guard submission state. PyTorch/PyG produce 32 upstream deprecation warnings;
  there are no failed scientific tests.
- Abstract (180 words) and Conclusions disclose competitive summary MLP AP0.4751.
  Primary and supplemental panels remain distinct. Corrected Holm0.0920,
  ensemble p0.1279 and NLL0.0986 are retained. Five bibliography booktitle prefixes
  corrected without key/metadata changes; 19 references cited.
- The human explicitly authorized A16 CRediT, funding, IRB, consent and AI-use
  sections for TDS. Those sections were applied and their provenance recorded.
  The Benchmark final-publication approval sentence and public tag were not
  treated as TDS approval or release evidence.
- Article13 + supplement4 pages compile with zero final layout/reference
  warnings. Source ZIP CRC passes; embedded final PDFs match. Fresh clean builds
  match text and rendered pixels on all17 pages. Priority visual review completed;
  the orphan Abbreviations heading was corrected.
- Local numeric candidate: 118 manifested payloads, 65 identifier-reduced test
  arrays, scientific sources/tests, licenses and replay environment. Privacy
  inventory passes. No manuscript, raw transaction ZIP, contract address or
  checkpoint is included. It has not been published.

## Remaining author/publisher closure

1. Confirm the two authors' identity, exact affiliations and corresponding author;
   approve this final version's hashes and explicitly accept the negative finding.
2. Supply the official award agreement or support-agency record. The requested
   IITP acknowledgment is applied, but official award verification is pending.
3. Confirm actual COI, originality/concurrent submission, third-party permissions
   and final coauthor submission consent. Authorized IRB/consent/AI text is applied.
4. Approve numeric redistribution and the exact local candidate before any public
   posting. Record the real immutable TDS commit/tag/URL and anonymous download
   replay, or obtain author/editor approval for a documented availability alternative.
5. Recheck current official MDPI template and SuSy upload requirements. Official
   pages were partly HTTP429 and current template download HTTP403. The retained
   official R314 class compiles; equivalence to the advertised September2026 update
   is not verified. The official listing advertises20 November2026; live portal
   acceptance remains unverified.

No additional scientific development is required by this audit. Pending author
and release statements remain visible; technical completion alone cannot clear G1/G2/G6.

## Approval identities

Article SHA-256: `{article}`

Supplement SHA-256: `{supplement}`

Private source ZIP SHA-256: `{result['source_zip_sha256']}`

Numeric candidate SHA-256: `{result['numeric_candidate_sha256']}`

Full approval manifest: private `final_artifact_manifest.json`. Public gate state:
`submission_readiness_315.json`. See publisher_preflight.md, release_verification.md,
reproduction_levels.md, build_verification.json and claim_to_evidence_matrix.csv
for the evidence and limits of each PASS.
'''
    (REPORTS/'submission_readiness_315.md').write_text(report)
    for name in ['license_and_redistribution_review.md','release_verification.md','reproduction_levels.md']:
        shutil.copy2(PUBLICATION/name,PAPER/name)
    for p in REPORTS.iterdir():
        if p.is_file() and p.name!='delivery_archive.json':shutil.copy2(p,PAPER/p.name)
    for name in ['candidate_archive.json','candidate_manifest.json']:
        shutil.copy2(PUBLICATION/name,PAPER/name)
    (PAPER/'evidence').mkdir(exist_ok=True)
    shutil.copy2(PUBLICATION/'TDS_315_numeric_candidate.zip',PAPER/'evidence/TDS_315_numeric_candidate.zip')
    # Bind final consent to the exact artifact hashes without inventing signatures.
    with (PAPER/'author_approval_record.md').open('a') as f:
        text=(PAPER/'author_approval_record.md').read_text()
        if '## Frozen 315 approval identities' not in text:
            f.write(f'\n## Frozen 315 approval identities\n\nArticle: `{article}`\n\nSupplement: `{supplement}`\n\nNo signature or final consent is recorded.\n')
    # Full handoff excludes transient compiler files and rendered QA bitmaps;
    # their hashes and the full private QA directory remain locally available.
    delivery=PAPER/'TDS_315_submission_candidate.zip'
    members=[]
    for p in sorted(PAPER.rglob('*')):
        if not p.is_file() or p==delivery or 'qa' in p.relative_to(PAPER).parts:continue
        if p.name=='delivery_archive.json':continue
        if p.suffix in {'.aux','.log','.bbl','.blg','.fls','.out','.fdb_latexmk','.pyc'}:continue
        members.append(p)
    with zipfile.ZipFile(delivery,'w',zipfile.ZIP_DEFLATED) as z:
        for p in members:z.write(p,p.relative_to(PAPER))
    with zipfile.ZipFile(delivery) as z:
        assert z.testzip() is None
        for rel in ['applsci_article.pdf','supplement/supplementary.pdf','submission_source.zip']:
            import hashlib
            assert hashlib.sha256(z.read(rel)).hexdigest()==sha(PAPER/rel)
    write(REPORTS/'delivery_archive.json',dict(path=str(delivery.relative_to(ROOT)),sha256=sha(delivery),
        bytes=delivery.stat().st_size,files=len(members),crc='PASS',embedded_final_pdfs_and_source_zip_match=True,
        public=False,submitted=False))
    print(json.dumps(dict(decision=result['decision'],submission_ready=result['SUBMISSION_READY'],
                          pending=result['pending_requirements'],delivery_sha256=sha(delivery)),indent=2))


if __name__=='__main__':main()
