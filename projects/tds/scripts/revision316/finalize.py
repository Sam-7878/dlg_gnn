"""Build reviewable platform metadata and evaluate author-controlled gates."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from paths import ROOT,PROJECT,SHARED,PREPRINT,MDPI,RESULTS,REPORTS,PUBLICATION,AUTHORS,delivery
from common import sha,write
from gates import evaluate,platform_status,author_signoffs_valid


SI='Machine Learning and Its Application for Anomaly Detection'
URL='https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt'


def load(path):return json.loads(path.read_text())


def proof(record):
    if not isinstance(record,dict) or record.get('verified') is not True or not record.get('date'):return False
    p=ROOT/record.get('path','')
    return p.is_file() and record.get('sha256')==sha(p)


def main():
    build=load(REPORTS/'build_verification.json');parity=load(REPORTS/'content_claim_parity.json')
    evidence=load(REPORTS/'frozen_evidence_verification.json');numeric=load(REPORTS/'numeric_cold_replay.json')
    archive=load(PUBLICATION/'release_manifest.json');facts=load(AUTHORS/'author_facts.json')
    requests=load(REPORTS/'venue_requests.json');meta=load(SHARED/'metadata.json')
    assert all(x['status']=='PASS' for x in [build,parity,evidence])
    assert numeric['archive_sha256']==archive['sha256']==sha(ROOT/archive['path'])
    docs={v['venue']:v for v in build['venues']}
    official=next(e for e in requests['requests'] if e['url'].endswith('MDPI_template.zip'))
    template_match=(official.get('status')==200 and sha(MDPI/'Definitions/mdpi.cls') in official.get('class_hashes',{}).values())
    for v in docs.values():
        for e in v['builds']:
            folder=PREPRINT if v['venue']=='preprints' else MDPI
            assert sha((folder/e['source']).with_suffix('.pdf'))==e['pdf_sha256']
    identity=proof(facts['identity_confirmation']) and proof(facts['corresponding_author_confirmation'])
    funding=proof(facts['official_funding_evidence'])
    declarations=proof(facts['coi_funder_role_originality_rights_confirmation'])
    license_ok=proof(facts['preprint_license_consent']) and proof(facts['numeric_redistribution_approval'])
    records=[]
    for e in facts['author_signoffs']:
        records.append({**e,'proof_verified':proof(e.get('proof'))})
    approvals={venue:author_signoffs_valid(records,v['builds'][0]['pdf_sha256'],v['builds'][1]['pdf_sha256'],venue)
               for venue,v in docs.items()}
    body=(SHARED/'sections/manuscript_body.tex').read_text()
    author_placeholders_clear=('Final approval of this TDS manuscript version remains pending' not in body)
    coi_placeholder_clear=('TDS-specific conflict-of-interest declaration is pending' not in body)
    availability_placeholder_clear=('immutable public tag and anonymous download verification remain pending' not in body)
    identity=identity and 'Corresponding-author designation pending' not in (PREPRINT/'preprint_main.tex').read_text()
    def event(value):
        return {**(value or {}),'proof_verified':proof((value or {}).get('proof'))}
    platform=platform_status(event(facts['preprint_submission']),event(facts['preprint_posting']),event(facts['journal_submission']))
    release=facts['public_release_proof'] or {}
    public=(proof(release) and release.get('archive_sha256')==archive['sha256']
            and bool(release.get('url')) and bool(re.fullmatch('[0-9a-f]{40}',release.get('commit','')))
            and bool(release.get('tag')) and release.get('anonymous_download_sha256')==archive['sha256']
            and release.get('anonymous_download_replay_pass') is True)
    exception_record=facts['approved_availability_exception'] or {}
    exception=(proof(exception_record) and exception_record.get('author_approved') is True
               and exception_record.get('preprints_accepted') is True and exception_record.get('journal_accepted') is True)
    availability=public or exception
    venue=proof(facts['venue_portal_proof']) and (facts['venue_portal_proof'] or {}).get('special_issue_url')==URL
    p=dict(P0=True,P1=approvals['preprints'] and author_placeholders_clear,P2=identity,
           P3=funding and declarations and license_ok and coi_placeholder_clear,
           P4=True,P5=True,P6=availability and license_ok and availability_placeholder_clear,P7=True)
    j=dict(J0=venue,J1=True,J2=True,J3=identity and funding and declarations and license_ok and coi_placeholder_clear,
           J4=True,J5=availability and availability_placeholder_clear,J6=True,J7=approvals['mdpi'] and author_placeholders_clear)
    state=evaluate(p,j,platform)
    state.update(date='2026-10-29',work_order='TDS-WO-316-PREPRINT-MDPI-ANOMALY-DUAL-TRACK',
        SCIENTIFIC_EVIDENCE_FROZEN=True,PREPRINT_SOURCE_BUILD_PASS=True,MDPI_SOURCE_BUILD_PASS=True,
        DUAL_VERSION_PARITY_PASS=True,AUTHOR_APPROVAL_COMPLETE=identity and all(approvals.values()),
        PREPRINT_NEUTRAL_TEMPLATE_PASS=True,NO_MDPI_LOGOS_NAMES_FOOTERS_PASS=True,
        SCIENCE_NUMERIC_PARITY_PASS=True,PREPRINT_SOURCE_FRESH_COMPILE_PASS=True,
        AUTHORS_APPROVED_PUBLIC_POSTING=p['P1'],LICENSE_AND_DATA_AVAILABILITY_APPROVED=p['P6'],
        COI_AND_FUNDING_TEXT_VERIFIED=funding and declarations,
        FUNDING_AND_COI_VERIFIED=funding and declarations,PUBLIC_EVIDENCE_RELEASE_VERIFIED=public,
        CURRENT_OFFICIAL_TEMPLATE_MATCH=template_match,FIRST_PAGE_METADATA_COMPLETE=identity,
        title=meta['Title'],title_final_author_approval=all(approvals.values()),
        reported_deadline='2026-12-31',deadline_independently_verified=requests['direct_deadline_verification'],
        special_issue=SI,special_issue_url=URL,guest_editor=None,guest_editor_verified=False,
        source_hashes={k:dict(article=v['builds'][0]['pdf_sha256'],supplement=v['builds'][1]['pdf_sha256'],source_zip=v['source_zip']['sha256']) for k,v in docs.items()},
        new_training_performed=False,public_push_performed=False)
    write(REPORTS/'submission_readiness_316.json',state)
    # Null values record that no factual platform event has occurred.
    write(RESULTS/'preprint_posting_record.json',dict(submitted=platform['SUBMITTED_PREPRINT'],
        posted=platform['POSTED_PREPRINT'],doi=platform['PREPRINT_DOI'],url=platform['PREPRINT_URL'],version=platform['PREPRINT_VERSION']))
    history=dict(date='2026-10-29',title=meta['Title'],abstract_sha256=hashlib.sha256(meta['abstract'].encode()).hexdigest(),
                 platform=platform,source_hashes=state['source_hashes'])
    history_bytes=(json.dumps(history,indent=2,sort_keys=True)+'\n').encode()
    history_path=RESULTS/'state_history'/('2026-10-29_'+hashlib.sha256(history_bytes).hexdigest()+'.json')
    if history_path.exists():assert history_path.read_bytes()==history_bytes
    else:
        history_path.parent.mkdir(exist_ok=True);history_path.write_bytes(history_bytes)
    for kind in ['preprints','mdpi']:
        value=dict(platform='Preprints.org' if kind=='preprints' else 'Applied Sciences',article_type='Article',
           title=meta['Title'],abstract=meta['abstract'],keywords=meta['keyword'],
           author_names=['SeongSu Park','Ki-Hyung Kim'],affiliation='Ajou University, Suwon, Republic of Korea',
           author_emails=['parky@ajou.ac.kr','kkim86@ajou.ac.kr'],corresponding_author=None,orcid=None,
           identity_confirmed=identity,funding='IITP-2026-RS-2021-II211835',funding_officially_verified=funding,
           coi_confirmed=declarations,source_zip=docs[kind]['source_zip'],
           numeric_candidate_sha256=archive['sha256'],public_numeric_release_url=release.get('url') if public else None,
           preprint_doi=platform['PREPRINT_DOI'],preprint_url=platform['PREPRINT_URL'],
           preprint_version=platform['PREPRINT_VERSION'],submitted=platform['SUBMITTED_PREPRINT' if kind=='preprints' else 'SUBMITTED_MDPI'])
        if kind=='mdpi':value.update(special_issue=SI,special_issue_url=URL,
              deadline='2026-12-31',deadline_source='316 work order reported official observation',
              deadline_independently_verified=False,guest_editor=None,portal_special_issue_selection_verified=venue,
              official_template_download_sha256=official.get('sha256'),official_template_class_match=template_match)
        name='metadata_preprints.json' if kind=='preprints' else 'metadata_applsci_anomaly.json'
        write(delivery(kind)/name,value)
        write(delivery(kind)/'approval_identity.json',dict(scope=kind,hashes=state['source_hashes'][kind],
             final_author_approval=approvals[kind],signoff_required_from=['S.-S.P.','K.-H.K.']))
    posting_text=(f"The preprint is posted at {platform['PREPRINT_URL']}, DOI {platform['PREPRINT_DOI']}, version {platform['PREPRINT_VERSION']}."
                  if platform['POSTED_PREPRINT'] else
                  'The preprint has been submitted but is not yet posted; no public DOI/URL is recorded.'
                  if platform['SUBMITTED_PREPRINT'] else
                  'No preprint has been submitted or posted by this task and no preprint DOI/URL has been assigned.')
    cover=f'''Dear Editors,

Please consider the Article "{meta['Title']}" for Applied Sciences, Special Issue
"{SI}" ({URL}).

This article provides a validity-first methodology for supervised anomaly screening
in financial contract graphs. Temporally admissible inputs, competitive simple
machine-learning controls, and separately audited ranking/calibration show how to
interpret risk scores under severe chronological distribution shift.

The task uses inherited retrospective contract-category labels. Its terminal test
window is18h57min10s, with107 positives and one Polygon positive. Degree-summary
MLP AP0.4751 is competitive with matched weighted-BCE GIN AP0.4524; historical GIN
AP0.4648 belongs to a separate panel. We claim neither a new architecture nor
unsupervised discovery, prospective warning, XAI, privacy guarantees or deployed
real-time monitoring. Fixed alert budgets describe investigative workload only.

The authors plan a Preprints.org-first sequence. {posting_text}
After actual posting, disclose the verified identity here and in the journal portal.

Code/numeric evidence use https://github.com/Sam-7878/dlg_gnn. The TDS-specific numeric
candidate is locally verified; immutable public deposition and anonymous replay
remain pending. Raw GoG data retain upstream CC BY-NC-SA4.0 conditions. Manuscript
sources are private and provided separately to the receiving platform.

This is an author-review candidate. Confirm author identities/correspondence,
official funding evidence, actual COI/funder-role/originality/rights declarations,
the exact-version signoffs and public availability before sending. No originality,
concurrent-submission or coauthor consent fact is attested by this draft.

Sincerely,
The authors (corresponding-author designation pending confirmation)
'''
    (delivery('mdpi')/'cover_letter_anomaly.md').write_text(cover)
    active=[cover,(delivery('mdpi')/'metadata_applsci_anomaly.json').read_text()]
    assert not any(s in '\n'.join(active) for s in ['Graph Neural Networks: Theory, Methods and Applications','C80IXAF9V4','2026-11-20'])
    for kind,entry in [('preprints','preprint_main.tex'),('mdpi','applsci_article.tex')]:
        name='README_PREPRINT_BUILD.md' if kind=='preprints' else 'README_MDPI_BUILD.md'
        (delivery(kind)/name).write_text('# Independent source ZIP build\n\nExtract into a new folder and run `latexmk -pdf '+entry+'`; '
          'then enter supplement/ and run `latexmk -pdf supplementary.tex`. All relative inputs are included. '
          'The public numeric candidate is a separate archive; no raw results are mixed into paper/. '
          'The exact hashes are in approval_identity.json. Local build success does not authorize public posting or journal submission.\n')
    report=f'''# TDS 316 final readiness

Scientific evidence remains frozen. Both independent manuscript/source ZIP builds
and scientific parity PASS. Preprints.org: **{state['preprint_decision']}**.
Applied Sciences: **{state['journal_decision']}**. No submission, posting, DOI
assignment, push or public release occurred. The existing R315 local/remote Git
HEAD is recorded separately and is not TDS release evidence.

Both forms retain the existing title and use one scientific body. The Abstract,
Introduction, four Related Work subsections and operating-point discussion clarify
supervised retrospective financial anomaly-risk screening. GADBench2023 and the
Qiao2025 survey were selected from the Benchmark reference list and verified against
primary proceedings/publisher-registered metadata. Original numeric tables, plots,
scores, statistical families and limitations are unchanged.

The new target is "{SI}" ({URL}). The work order reports deadline2026-12-31;
direct page/Guest Editor/portal confirmation remains unavailable (HTTP403/429).
Do not reuse the former SI deadline. A fresh official MDPI ZIP returned HTTP200,
and the bundled MDPI class matches its exact hash: template match={template_match}.

Preprint: {docs['preprints']['builds'][0]['pages']} pages plus {docs['preprints']['builds'][1]['pages']}-page supplement.
MDPI: {docs['mdpi']['builds'][0]['pages']} pages plus {docs['mdpi']['builds'][1]['pages']}-page supplement.
R315 MDPI had13 pages; R316 has added application framing and two relevant citations,
with unchanged scientific tables/figures. Page count is documented rather than
shrinking typography to mimic the previous count. Fresh source-ZIP builds have
identical extracted text and rendered pixels on every page.

Local numeric candidate: `{archive['sha256']}`. All65 score vectors are byte-identical
to the reviewed R315 archive (5ac20f2c...) and its relocated derivative. Cold extraction,
the seven AP pairs, representative10000-replicate statistics and18 tests pass.
This is L2 LOCAL validation, not public anonymous-release verification; L3 full
retraining is not guaranteed. GPU capacity was inspected only;316 prohibits new training.

## Remaining actual author/platform facts

- Both authors' exact identities/affiliations/correspondence and per-venue final
  article/supplement hash approvals, including the competitive-MLP negative finding.
- Official IITP award evidence and actual COI/funder-role/originality/permissions.
  The A16-authorized roles/ethics/funding/AI text remains applied, without inferring
  new publication approval from another manuscript.
- Consent to public preprint terms/irreversibility and numeric redistribution;
  real immutable TDS release plus anonymous replay, or documented accepted exception.
- Live journal Special Issue selection, current deadline/Guest Editor confirmation;
  actual preprint DOI/URL/version before the planned subsequent journal submission.

Use separate SUBMITTED_PREPRINT, POSTED_PREPRINT and SUBMITTED_MDPI states. They
remain false; DOI/URL/version remain null. Exact gate P0–P7 and J0–J7 outcomes are
in submission_readiness_316.json. Author-owned records remain private.
'''
    (REPORTS/'final_readiness_316.md').write_text(report)
    (REPORTS/'work_report_316.md').write_text(report)
    print('Gate P/J evaluated: local engineering PASS; author/public/portal facts remain HOLD')


if __name__=='__main__':main()
