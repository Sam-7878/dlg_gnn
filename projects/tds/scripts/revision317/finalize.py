"""Seal local completion without inventing author or platform approvals."""
import json
from common import ROOT,PROJECT,OUT,REPORTS,PAPER,sha,write
from gates import evaluate


def load(path):return json.loads(path.read_text())


def main():
    if (PROJECT/'submission/author_records/317/author_confirmation_from_a16.json').exists():
        # A later human confirmation supersedes the earlier empty-record snapshot.
        from confirm_authors import main as confirmed_finalize
        return confirmed_finalize()
    frozen=load(OUT/'evidence_freeze_manifest.json')
    for e in frozen['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    cp=load(OUT/'checkpoint_freeze_manifest.json')
    for e in cp['files']:assert sha(ROOT/e['path'])==e['sha256']
    for e in load(OUT/'additional_316_identities.json')['files']:assert sha(ROOT/e['path'])==e['sha256']
    build=load(REPORTS/'build_verification.json');numeric=load(REPORTS/'numeric_cold_replay.json')
    release=load(PROJECT/'publication/317/release_manifest.json')
    for e in load(OUT/'source_manifest.json')['files']:
        assert sha(ROOT/e['path'])==e['sha256'],'Source changed after numeric seal: '+e['path']
    assert build['status']==numeric['status']=='PASS'
    assert numeric['archive_sha256']==release['sha256']==sha(ROOT/release['path'])
    identities={}
    for venue in build['venues']:
        folder=PAPER/venue['venue'];entry='preprint_main.pdf' if venue['venue']=='preprints' else 'applsci_article.pdf'
        assert sha(folder/entry)==venue['local']['pdf_sha256']
        assert sha(folder/'supplement/supplement317.pdf')==venue['supplement']['pdf_sha256']
        identities[venue['venue']]=dict(article_sha256=venue['local']['pdf_sha256'],supplement_sha256=venue['supplement']['pdf_sha256'],
            source_zip_sha256=venue['source_zip_sha256'],article_pages=venue['local']['pages'],supplement_pages=venue['supplement']['pages'])
    assert sha(PAPER/'preprints/manuscript_body.tex')==sha(PAPER/'mdpi_applsci_anomaly/manuscript_body.tex')==build['shared_body_sha256']
    science=dict(S0=True,S1=True,S2=True,S3=True,S4=True,S5=True,S6=True)
    # S1 means source provenance, including disclosed unknowns, is documented.
    # It does NOT verify the exact collection cutoff or label discovery time.
    decision=load(OUT/'selection_decision.json')
    assert decision['full_lineage_reconstructed'] and not decision['right_censoring_causal_mechanism_established']
    assert not decision['label_ascertainment_time_available']
    import pandas as pd
    assert len(pd.read_csv(OUT/'reusable_validity_checklist.csv'))==12
    assert numeric['n_test_arrays']==110 and numeric['n_validation_only_calibrators']==95
    technical=dict(preprint_clean_build=True,mdpi_clean_build=True,numeric_cold_replay=True,shared_scientific_source=True,
        original_artifacts_unchanged=True,official_template_hash_matched=load(OUT/'template_verification.json')['matches_preserved_class'])
    # Existing author records still contain no official funding/signature/release proof.
    # Do not infer these from successful science, an old paper, or a user's wording request.
    old=load(PROJECT/'submission/author_records/316/author_facts.json')
    assert old['author_signoffs']==[] and old['official_funding_evidence'] is None
    state=evaluate(science,technical,{})
    state.update(work_order='TDS-WO-317-VALIDITY-A16-SUBMISSION',date='2026-10-30',
        identity=identities,n_original_artifacts_verified=len(frozen['files']),n_original_checkpoints_verified=cp['n'],
        n_original_vectors=65,n_new_fits=45,n_registered_vectors=110,n_validation_only_calibrators=95,
        numeric_candidate_sha256=release['sha256'],
        source_provenance_limit='month-level source coverage verified; exact cutoff policy unknown; claim restricted accordingly',
        pending=['both authors approve names/departments/correspondence/ORCID and exact PDF+supplement hashes',
                 'official IITP-ITRC award evidence and TDS funder role',
                 'TDS COI/originality/rights declarations and preprint permanence/license consent',
                 'numerical derivative redistribution approval plus immutable TDS release and anonymous replay, or publisher-accepted restriction',
                 'actual preprint posting receipt and DOI before journal submission',
                 'live correct SI portal selection and exact journal-file approval'])
    write(OUT/'submission_readiness_317.json',state)
    write(REPORTS/'frozen_evidence_verification.json',dict(status='PASS',original_files_verified=len(frozen['files']),
        original_checkpoints_verified=cp['n'],original_vectors_preserved=65,additional_316_source_zips_verified=True))
    authors=PROJECT/'submission/author_records/317';authors.mkdir(parents=True,exist_ok=True)
    write(authors/'approval_identity.json',dict(date='2026-10-30',files=identities,numeric_candidate=release['sha256'],
        a16_workflow_wording_authorized_for_TDS=True,author_signatures=[],public_release_approved=False))
    (authors/'author_approval_317.md').write_text('''# 317 저자 확인 및 승인 기록

상태: 미승인. A16 문구·작업 방식 적용에 관한 기존 사용자 지시는 유지하며,
이를 317 최종 원고의 공동저자 서명이나 공개 게시 동의로 대신하지 않습니다.
아래 해시는 `approval_identity.json`에 기계 판독 형식으로도 저장했습니다.

두 저자별로 이름(Seong-Su/SeongSu), 순서, 학과·소속, 이메일, 교신저자,
ORCID, 실제 CRediT 역할을 확인해야 합니다. 공식 IITP--ITRC 지원 증빙의 경로와
과제번호·사사 표현·TDS 지원관계·지원기관 역할도 확인해야 합니다.
TDS 전용 COI, 중복투고/독창성, IRB·동의 비해당, AI의 실제 코드/분석 기여,
원시·파생 자료 권리, 숫자 공개 범위와 라이선스는 각각 확인 대상입니다.

Preprint 공개 동의는 DOI의 영구성, 게시 후 완전 삭제가 보장되지 않는 점,
버전/철회 정책, 공개 라이선스와 아래 정확한 원고·부록 승인을 포함해야 합니다.
MDPI 파일도 별도 승인하고 실제 preprint DOI를 제출 시 고지해야 합니다.

승인 문구에 저자명·일자·대상 플랫폼·아래 article/supplement 해시와 함께
표본선택 해석 축소, 경쟁력 있는 MLP 결과, 라벨 발견시간 미상 및 비조기탐지
한계를 이해했다는 확인을 남깁니다. 현재 서명·영수증은 존재하지 않습니다.

```json
'''+json.dumps(identities,indent=2)+'''\n```\n''')
    report=f'''# TDS317 final local work report

Completed2026-10-30. **Gate S: PASS within explicitly restricted retrospective,
sampling-associated interpretation. Gate P/J: HOLD_AUTHOR_ACTION_AND_PUBLIC_RELEASE.**
Unknown exact collector policy is disclosed and is not marked verified. No claim
of identified behavioral drift, censoring causality, prospective early warning or
long-term transportability is made.

The work froze{len(frozen['files'])}prior artifacts,55original checkpoints and65score
arrays; independently recovered24,316contract lifetimes from268,282,924transfers;
diagnosed the reversed prevalence under first-time ordering; and ran45new bounded
fits including five-seed time ablation, tree/outlier controls and a separately built
24,179contract seven-day cohort. All original hashes remain unchanged.

Key findings: original last-order train/test39.85%/2.93% versus first-order
14.79%/56.99%; pooled positive/other lifetime medians0.671/608.447days. Controlled
MLP/GIN AP0.4751/0.4524; time-GAT/no-time-GAT AP0.3119/0.3897. Distinct seven-day
MLP/GIN AP0.9582/0.6358 at its own test prevalence56.16%, never a paired old/new gain.
Raw/TS GIN ECE0.1374/0.0322; MLP0.0424/0.0186. Polygon's one positive supports only
a point AP. Nineteen-hour-block sensitivity includes zero in the GIN--MLP interval.

Both venue articles are13pages and Supplements4pages. Four clean source builds
have zero LaTeX errors, undefined citations or missing figures; cold PDF text matches
local builds. Scientific body, bibliography and declarations match across venues.
The actual official MDPI ZIP/class was rechecked. Public A16 source was inspected,
and AI disclosure includes actual coding/analysis assistance without fabricated versions.

Local cold numeric replay checks110test arrays and95validation-only calibrators;
maximum metric error{numeric['maximum_point_metric_error']:.3g}. Candidate SHA-256:
`{release['sha256']}`. It excludes manuscript sources, identifiers, raw data and weights.
It is not an approved immutable public release or independent full retraining claim.

## Deliverables

- Private sources/PDFs: `projects/tds/paper/submission_317/` (shared, preprints, mdpi_applsci_anomaly).
- Private source ZIPs and platform files: `projects/tds/submission/{{preprints,applsci}}/317/`.
- Numeric outputs: `projects/tds/results/317/`; code: `projects/tds/scripts/revision317/`.
- Local numeric candidate: `projects/tds/publication/317/`; reports: this folder.
- Address-bearing lineage/cache: `projects/tds/private/317/`; signatures: private author_records/317.

## Required human/external closure

The exact current hashes are bound in the private approval record. Both-author
identity/correspondence/roles and file approval, official funding evidence, COI and
funder role, rights/license and posting consent remain unconfirmed. A real TDS
release with anonymous verification (or an accepted restriction), actual preprint
DOI, live SI selection and journal approval are still needed. The SI indexed
deadline is31December2026; the portal is not verified. No Git push/tag/release,
preprint submission/posting or journal submission was performed.

See the review response, collection audit, comparator audit, statistics, A16,
availability and policy reports for evidence and the limits of each conclusion.
'''
    (REPORTS/'work_report_317.md').write_text(report)
    (REPORTS/'README.md').write_text('# TDS317 evidence\n\n[Final work report](work_report_317.md) · [Review response](claude_review_response_matrix.md) · [Selection audit](collection_end_selection_audit.md) · [Gate state](../../results/317/submission_readiness_317.json)\n\nPaper sources and author approvals remain private. Numerical evidence and software are tracked separately.\n')
    print(state['GATE_S'],state['decision'])


if __name__=='__main__':main()
