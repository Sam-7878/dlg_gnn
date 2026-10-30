"""Apply a recorded user confirmation without fabricating external evidence."""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from common import ROOT,PROJECT,PAPER,OUT,REPORTS,PRIVATE,sha,write

AUTHORS=PROJECT/'submission/author_records/317'
UPDATE='author_confirmation_20261030'
FOLLOW_REPORTS=REPORTS/UPDATE
FOLLOW_RESULTS=OUT/UPDATE


def load(path):return json.loads(path.read_text())


def verify_numerical_seal(baseline):
    archive=PROJECT/'publication/317/TDS_317_numeric_release_candidate.zip'
    assert sha(archive)==baseline['numeric_zip_sha256']
    checked=0
    with zipfile.ZipFile(archive) as z:
        original=json.loads(z.read('manifest.json'))
        for e in original['files']:
            if e['path'].endswith('.csv'):
                assert sha(OUT/e['path'])==e['sha256'],e['path'];checked+=1
    return checked


def main(rebuild=False):
    FOLLOW_REPORTS.mkdir(exist_ok=True);FOLLOW_RESULTS.mkdir(exist_ok=True)
    record_path=AUTHORS/'author_confirmation_from_a16.json';record=load(record_path)
    assert record['confirmed_by']=='user instruction in this conversation'
    assert sha(AUTHORS/'A16_author_reference_20261030.tex')==record['source_sha256']
    baseline=load(PRIVATE/'before_author_confirmation_20261030/baseline.json')
    assert sha(PAPER/'shared/manuscript_body.tex')==baseline['body_sha256']
    assert sha(PAPER/'shared/references.bib')==baseline['bib_sha256']
    checked=verify_numerical_seal(baseline)
    marker=FOLLOW_REPORTS/'build_verification.json'
    if rebuild or not marker.exists():
        subprocess.run([sys.executable,str(PAPER/'shared/prepare.py')],cwd=ROOT,check=True)
        import build
        build.REPORTS=FOLLOW_REPORTS
        build.main()
    build_state=load(marker);assert build_state['status']=='PASS'
    assert build_state['shared_body_sha256']==baseline['body_sha256']
    assert build_state['shared_bib_sha256']==baseline['bib_sha256']
    identities={}
    for v in build_state['venues']:
        folder=PAPER/v['venue'];entry='preprint_main.pdf' if v['venue']=='preprints' else 'applsci_article.pdf'
        assert sha(folder/entry)==v['local']['pdf_sha256']
        assert sha(folder/'declarations.tex')==build_state['shared_declarations_sha256']
        identities[v['venue']]=dict(article_sha256=v['local']['pdf_sha256'],supplement_sha256=v['supplement']['pdf_sha256'],
            source_zip_sha256=v['source_zip_sha256'],article_pages=v['local']['pages'],supplement_pages=v['supplement']['pages'])
    for venue,entry in [('preprints','preprint_main.tex'),('mdpi_applsci_anomaly','applsci_article.tex')]:
        text=(PAPER/venue/entry).read_text()
        for a in record['authors']:
            assert a['name'] in text and a['affiliation'] in text
            if a.get('orcid'):assert a['orcid'] in text
            elif a.get('orcid_provided_in_A16'):assert a['orcid_provided_in_A16'] not in text
        assert 'requires confirmation' not in text and 'designation pending' not in text
    declarations=(PAPER/'shared/declarations.tex').read_text()
    assert record['funding_identifier'] in declarations and record['coi'] in declarations
    assert 'outstanding' not in declarations and 'await author verification' not in declarations
    # User attestation confirms declarations; no award document, signature, or receipt is invented.
    facts=dict(date=record['date'],author_details_and_declarations_confirmed=True,
        basis='explicit user instruction to apply A16 author-approval items to TDS',
        reference_source_path=record['source_path'],reference_source_sha256=record['source_sha256'],
        confirmation_record_sha256=sha(record_path),funding_statement_confirmed=True,
        official_award_document_verified=False,independent_coauthor_signatures_verified=False,
        author_orcid_status={a['name']:('user correction; checksum verified' if a.get('orcid_verification_basis')=='explicit user correction' else 'source checksum verified' if a.get('orcid') else 'source identifier invalid; omitted') for a in record['authors']},
        tds_release_verified=False,files=identities,scientific_body_unchanged=True,
        numerical_archive_unchanged=True,numerical_csvs_verified=checked,numeric_candidate_sha256=baseline['numeric_zip_sha256'])
    write(FOLLOW_RESULTS/'confirmation_status.json',facts)
    old=load(OUT/'submission_readiness_317.json')
    old.update(date=record['date'],identity=identities,
        AUTHOR_DETAILS_AND_DECLARATIONS_CONFIRMED=True,IDENTITY_AND_CORRESPONDENCE_CONFIRMED=True,
        CREDIT_CONFIRMED=True,FUNDING_STATEMENT_CONFIRMED=True,IRB_CONSENT_DECLARATIONS_CONFIRMED=True,
        COI_DECLARATION_CONFIRMED=True,AI_DISCLOSURE_CONFIRMED=True,
        AUTHOR_CONFIRMATION_BASIS='explicit user confirmation + cited A16 source',
        EXACT_FILE_COAUTHOR_SIGNATURES_VERIFIED=False,
        AUTHOR_APPROVAL_COMPLETE=False,FUNDING_VERIFIED=False,OFFICIAL_AWARD_DOCUMENT_VERIFIED=False,
        PUBLIC_RELEASE_VERIFIED=False,READY_PREPRINT=False,READY_MDPI=False,
        SUBMITTED_PREPRINT=False,POSTED_PREPRINT=False,SUBMITTED_MDPI=False,PREPRINT_DOI=None,
        decision='AUTHOR_DECLARATIONS_CONFIRMED_PUBLICATION_CLOSURE_PENDING',
        pending=['TDS-specific approved release/rights and anonymous verification or accepted restriction',
                 'official award-document verification is distinct from the confirmed funding statement',
                 'exact-file coauthor signoff is not an independently verified signature record',
                 'actual preprint posting/DOI, live SI portal and journal receipt'],
        author_confirmation_report=str((FOLLOW_REPORTS/'confirmation_report.md').relative_to(ROOT)))
    write(OUT/'submission_readiness_317.json',old)
    write(FOLLOW_RESULTS/'submission_readiness_317.json',old)
    write(AUTHORS/'approval_identity.json',dict(date=record['date'],files=identities,
        numeric_candidate=baseline['numeric_zip_sha256'],a16_workflow_wording_authorized_for_TDS=True,
        author_details_and_declarations_confirmed=True,confirmation_record_sha256=sha(record_path),
        author_signatures=[],public_release_approved=False))
    rows=[]
    for a in record['authors']:
        rows.append('|'+a['name']+'|'+a['affiliation']+'|'+a['email']+'|'+(a.get('orcid') or '미기재: A16 번호 체크섬 오류')+'|'+('예' if a['corresponding'] else '아니오')+'|')
    kim=next(a for a in record['authors'] if a['name']=='Ki-Hyung Kim')
    kim_note=('Kim의 ORCID '+kim['orcid']+'와 최신 Graduate School 소속은 사용자의 직접 정정에 따라 반영했습니다.' if kim.get('orcid_verification_basis')=='explicit user correction' else 'Kim의 A16 ORCID는 체크섬이 맞지 않아 PDF에서 생략했습니다. 임의의 대체번호는 쓰지 않습니다.')
    (AUTHORS/'author_approval_317.md').write_text('''# 317 저자 확인 반영 기록

**상태: 사용자 확인에 따라 A16 저자정보와 선언문 적용 완료 (2026-10-30).**
사용자는 이 확인서의 사항을 A16 소스에 따라 진행하고 동일한 작업 방식을
적용하라고 확인했습니다. 아래 저자정보와 역할·연구비 문구·IRB/동의·COI·AI
사용 선언은 이 확인을 근거로 확정했으며, 같은 사항을 다시 미확인으로 취급하지 않습니다.

|저자|소속|이메일|ORCID|교신저자|
|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

교신 연락처: +82-31-219-2433. CRediT는 A16의 역할 분담을 동일 적용합니다.
'''+kim_note+'''
연구비 문구: IITP–ITRC / Korea MSIT / IITP-2026-RS-2021-II211835.
IRB와 동의: Not applicable. 이해상충: no conflict of interest.
AI 고지는 A16의 도구명을 따르되 TDS의 실제 코드·분석 지원 내용도 유지합니다.

이 기록은 사용자가 확인한 저자 진술이며, 공식 award 원본이나 별도로 서명된
공동저자 문서를 새로 만들어낸 것이 아닙니다. 그러한 외부 증빙, TDS 공개
릴리스·원시/파생 자료 권리, 실제 게시·투고 영수증은 별도 사실로 기록합니다.
현재 TDS의 tag/DOI/플랫폼 접수 기록은 생성하지 않았습니다.

아래는 정보를 반영하여 독립 빌드한 현재 원고·부록·source ZIP 해시입니다.
수정 전 버전은 private/317/before_author_confirmation_20261030에 보존했습니다.

```json
'''+json.dumps(identities,indent=2)+'\n```\n')
    report='''# A16 author confirmation applied to TDS317

Date:2026-10-30. **Author information and declarations: CONFIRMED_BY_USER.**
The user explicitly instructed applying the A16 author-approval items to TDS.
Names/order, the two supplied affiliations, email contacts, corresponding author and
telephone, the current valid ORCIDs, CRediT, funding wording, IRB/consent and COI have been applied.
Kim's ORCID and affiliation follow the user's explicit correction where provided;
the private source audit retains the original A16 value without copying it into PDFs.
The shared declarations and Methods retain truthful AI coding/analysis disclosure.
The corresponding-author and author-declaration placeholder text is removed.

This is an author-provided attestation, not a fabricated official award document,
independent signature or publication receipt. AUTHOR_APPROVAL_COMPLETE retains its
legacy exact-file-signoff meaning; it does not mean the supplied author identities
and declarations remain unconfirmed. Those are now separately and explicitly true.
No verified TDS release, DOI, anonymous release replay, or submitted/posted event is inferred.

The unchanged scientific body, bibliography and numerical ZIP are hash-checked.
Both articles, Supplements and venue source ZIPs were rebuilt independently.
All score arrays and experiment results retain their existing identities. The
immutable pre-update state is retained in an author-local rollback snapshot.

Current machine record: `results/317/author_confirmation_20261030/confirmation_status.json`.
Private factual record and current file hashes: `submission/author_records/317/`.
'''
    (FOLLOW_REPORTS/'confirmation_report.md').write_text(report)
    write(FOLLOW_REPORTS/'scientific_preservation.json',dict(status='PASS',body_sha256=baseline['body_sha256'],
        bibliography_sha256=baseline['bib_sha256'],numeric_zip_sha256=baseline['numeric_zip_sha256'],numeric_csvs_checked=checked))
    write(FOLLOW_RESULTS/'source_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('finalize.py')]],
        scope='author-confirmation extension only; original numerical package retains its sealed source snapshot'))
    # Make the current status easy to find while retaining earlier evidence as history.
    readme=REPORTS/'README.md';text=readme.read_text()
    link='[Current author confirmation](author_confirmation_20261030/confirmation_report.md)'
    if link not in text:readme.write_text(text+'\n'+link+' — user-confirmed author facts supersede the earlier pending declarations.\n')
    work=REPORTS/'work_report_317.md';text=work.read_text()
    note='> 2026-10-30 author update: author identities and declarations are confirmed by the user and applied. See [current confirmation](author_confirmation_20261030/confirmation_report.md). Earlier pending-author descriptions below are a historical snapshot.\n\n'
    if note not in text:work.write_text(note+text)
    for name in ['A16_to_TDS_declarations_audit.md','preprints_mdpi_policy_audit.md']:
        path=REPORTS/name;text=path.read_text()
        if note not in text:path.write_text(note+text)
    for delivery in ['preprints','applsci']:
        write(PROJECT/f'submission/{delivery}/317/approval_identity.json',facts)
    print('A16 author details and declarations applied; scientific inputs/results unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rebuild',action='store_true');args=parser.parse_args()
    main(rebuild=args.rebuild)
