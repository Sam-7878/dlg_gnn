"""Prepare reviewable R318 closure records. External publication is never inferred."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from gates import evaluate

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
OUT = PROJECT / 'results/318'
REPORTS = PROJECT / 'reports/318'
AUTH = PROJECT / 'submission/author_records/318'
RELEASE = PROJECT / 'publication/318/release'
PAPER = PROJECT / 'paper/submission_318'
DATE = '2026-10-30'
NUMERIC_SHA = 'a4f05674eef4d05913d897d3caa8e085026e7939e0d1d2c82c171d95a672e218'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def doc(name, text):
    (REPORTS / name).write_text(text.strip() + '\n')


def source_status():
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    files = []
    for revision in ['revision314','revision315','revision316','revision317']:
        for p in sorted((PROJECT / 'scripts' / revision).glob('*.py')):
            path = str(p.relative_to(ROOT))
            old = subprocess.run(['git','show', head+':'+path],cwd=ROOT,capture_output=True)
            files.append(dict(path=path, sha256=sha(p), at_local_head=old.returncode == 0,
                              matches_local_head=old.returncode == 0 and old.stdout == p.read_bytes()))
    public = load(OUT / 'public_source_check.json')
    return dict(local_head=head, files=files, all_current_r314_r317_code_at_local_head=all(f['matches_local_head'] for f in files),
                remote_commit_readable=public['commit'] == head and public['publicly_readable_at_commit'],
                independent_remote_file_check=public, source_317_false_commit_flag_is_historical=True,
                r318_new_code_publication_authorized=False, tds_release_verified=False)


def main():
    for p in [REPORTS, AUTH, RELEASE]:
        p.mkdir(parents=True, exist_ok=True)
    identity = load(OUT / 'candidate_identity.json')
    page_counts = {v['venue']:v['current_article_pages'] for v in identity['venues']}
    builds = load(OUT / 'build_verification.json')
    numeric = load(OUT / 'numeric_cold_replay.json')
    preservation = load(OUT / 'numeric_preservation.json')
    supplied = PROJECT / 'submission/author_records/317/author_confirmation_from_a16.json'
    author = load(supplied)
    assert author['confirmed_by'] == 'user instruction in this conversation'
    assert author['funding_statement_confirmed'] and author['ai_disclosure_confirmed']
    assert author['funding_identifier'] == 'IITP-2026-RS-2021-II211835'
    assert numeric['archive_sha256'] == NUMERIC_SHA
    checks = []
    final_hashes, source_hashes = {}, {}
    for v in identity['venues']:
        for name, e in v['current'].items():
            assert sha(ROOT / e['path']) == e['sha256']
            checks.append(e)
            if name.endswith('article.pdf'):
                final_hashes['mdpi' if v['venue']=='applsci' else v['venue']] = e['sha256']
            elif name.endswith('source.zip'):
                source_hashes['mdpi' if v['venue']=='applsci' else v['venue']] = e['sha256']
    for e in identity['shared_files']:
        assert sha(ROOT / e['path']) == e['sha256']
    expected = (PROJECT / 'publication/318/SHA256SUMS_FINAL.txt').read_text()
    for line in expected.splitlines():
        digest, path = line.split('  ',1)
        assert sha(ROOT / path) == digest
    for folder, entry in [('preprints','preprint_main.tex'),('mdpi_applsci_anomaly','applsci_article.tex')]:
        wrapper = (PAPER / folder / entry).read_text()
        for a in author['authors']:
            assert a['name'] in wrapper and a['orcid'] in wrapper and a['affiliation'] in wrapper and a['email'] in wrapper
        assert '0000-0002-2321-4475' not in wrapper
        assert author['funding_identifier'] in (PAPER / folder / 'declarations.tex').read_text()
    # Citation keys and rendered table/figure label order agree across wrappers.
    aux = [(PAPER / folder / entry).with_suffix('.aux').read_text() for folder,entry in
           [('preprints','preprint_main.tex'),('mdpi_applsci_anomaly','applsci_article.tex')]]
    citations = [re.findall(r'\\citation\{([^}]+)\}', s) for s in aux]
    # cleveref adds synthetic @cref companions in the MDPI template only.
    labels = [[k for k in re.findall(r'\\newlabel\{((?:tab|fig):[^}]+)\}', s) if not k.endswith('@cref')] for s in aux]
    assert citations[0] == citations[1] and labels[0] == labels[1]
    write(OUT / 'reference_and_caption_audit.json', dict(citation_order_identical=True, citation_groups=citations[0],
          table_figure_label_order_identical=True, table_figure_labels=labels[0],
          missing_or_undefined_references=0, scientific_bibliography_sha256=identity['original_bibliography_sha256']))
    source = source_status()
    write(OUT / 'source_commit_reconciliation.json', source)

    numeric_zip = PROJECT / 'publication/317/TDS_317_numeric_release_candidate.zip'
    copy = RELEASE / numeric_zip.name
    if copy.exists():
        assert sha(copy) == NUMERIC_SHA
    else:
        shutil.copy2(numeric_zip, copy)
    with zipfile.ZipFile(copy) as z:
        names = set(z.namelist())
    required = ['model_summary.csv','modelwise_calibration.csv','original_population.csv','horizon7_population.csv',
                'horizon7_dataset_manifest.json','sample_time_lineage.csv','first_last_ordering_diagnostics.csv',
                'lifespan_by_class.csv','experiment_preregistration.json','statistics_and_holm.csv',
                'time_block_sensitivity.csv','replay.py','manifest.json',
                'source/projects/tds/scripts/revision317/tables.py',
                'source/projects/tds/scripts/revision317/analyze.py',
                'source/projects/tds/tests/test_revision314.py']
    assert set(required) <= names
    license_bytes = (RELEASE / 'UPSTREAM_LICENSE_CC_BY_NC_SA_4_0.txt').read_bytes()
    upstream_blob = hashlib.sha1(b'blob '+str(len(license_bytes)).encode()+b'\0'+license_bytes).hexdigest()
    assert upstream_blob == 'c14a34c4954e217a88d27d206752db1c79dcaf47'
    (RELEASE / 'README_REPRODUCE.md').write_text('''# TDS317 evidence, staged by R318

Status: RELEASE_PENDING_AUTHOR_AUTHORIZATION. This directory has not been published.
The ZIP is the exact R317 candidate (SHA-256 below), with no numerical repackaging:

`'''+NUMERIC_SHA+'''`

## Numeric replay (CPU; no raw data, weights or paper sources)

Verify SHA256SUMS_RELEASE.txt from repository root, extract the ZIP into an empty
directory, then run `python replay.py` in that extracted directory. NumPy, pandas,
SciPy and scikit-learn are required; archived `environment.json` records the actual
runtime. R318 observed 110 test arrays, 95 validation-only calibrators and maximum
absolute point/calibrated metric error 1.1102230246251565e-16. This was a local
clean-directory replay, not an anonymous public download or full retraining.

## Source and figure/table reconstruction

`source/` contains repository-shaped R314/R317 code and tests, with R315/R316 gates.
`model_summary.csv`, `modelwise_calibration.csv`, `statistics_and_holm.csv`,
`operational_metrics.csv` and the sorting/lifetime CSVs feed the numerical tables.
`source/projects/tds/scripts/revision317/tables.py` creates those tables. Set its
module OUT variable to a writable copy of this extracted data directory before
calling main(); never regenerate into the sealed evidence directory. The selection
figure code is in `analyze.py:diagnostics`; its exact ECDF input is
`sample_time_lineage.csv`, and its prevalence input is
`first_last_ordering_diagnostics.csv`. The decision tree is a source-defined
procedural figure; no new fitted or empirical result is embedded in it. Archived
figure PDFs/PNGs and editable numeric tables are included. Full diagnostics also
refer to author-local graph inputs; numeric replay alone does not regenerate them.

## Full training reconstruction (separate, upstream-license-dependent)

Obtain upstream data from the GoG repository under its access/license terms, then
use the reconstruction and training source with graph-building inputs and the
documented PyTorch/PyG runtime. Raw 268-million-row transfer archives, contract
address mappings and weights are excluded. Old 18-regression cold tests require
PyTorch/PyG; new R317 integration tests additionally require author-local artifacts.
R318 ran no detector fit and did not repeat the raw-archive scan.

## Rights and identifiers

215 CSVs retain sample_id. Those IDs and timestamps can link to source contracts;
the package is pseudonymous, not anonymous, and has no formal privacy guarantee.
Read NOTICE.md and UPSTREAM_LICENSE_CC_BY_NC_SA_4_0.txt. Numeric derivatives do not
inherit an unrestricted repository software license. Manuscript licensing is a
separate author decision. Publication awaits rights and exact-file author approval.

There is no TDS release URL/tag/DOI yet. Local success must never be reported as
anonymous public verification. After approved publication, download the actual
immutable asset without login into another empty directory, verify these hashes,
run replay, and retain HTTP, asset, commit and replay receipts separately.
''')
    (RELEASE / 'NOTICE.md').write_text('''# License and attribution separation — staging notice

Upstream: Multi-Chain Graphs of Graphs: A New Approach to Analyzing Blockchain
Datasets (NeurIPS 2024), Xtra-Computing repository and the creators cited there:
https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs
Pinned reference: 7264f1bf510f7ba4f5041ac7a29b606abc12f262.
The upstream LICENSE is CC BY-NC-SA 4.0 (Git blob c14a34c4954e217a88d27d206752db1c79dcaf47),
retained verbatim in UPSTREAM_LICENSE_CC_BY_NC_SA_4_0.txt.

Changes: contract-category supervision, time-lineage/split diagnostics, derived
graph summary features, predictions, calibration and statistical aggregations.
Relevant upstream-derived material retains attribution/noncommercial/share-alike
terms. This staging notice does not newly grant rights on behalf of either author
or third parties. Confirm the lawful scope and authors' approval before release.
Software retains its applicable repository and third-party notices; do not assume
every upstream helper or numeric derivative is MIT merely because our repository
contains MIT software. No upstream endorsement is implied.

Raw transfers, direct contract mapping tables, checkpoints and manuscript sources
are excluded. sample_id and timing fields remain linkable to source records;
identifier reduction is not anonymization or automatic redistribution permission.
The intended manuscript CC BY publication and the upstream data's CC BY-NC-SA
terms are separate scopes. Do not silently relicense upstream-derived data/figures.
''')
    release_sums = ''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(RELEASE.iterdir()) if p.is_file())
    (PROJECT / 'publication/318/SHA256SUMS_RELEASE.txt').write_text(release_sums)
    release_manifest = dict(status='RELEASE_PENDING_AUTHOR_AUTHORIZATION', archive_path=str(copy.relative_to(ROOT)),
        sha256=NUMERIC_SHA, original_archive_sha256=NUMERIC_SHA, repackaged=False, public_release_verified=False,
        public_commit=source['independent_remote_file_check']['commit'], public_commit_scope='R317 code; not a TDS numerical release',
        tag=None, asset_url=None, release_doi=None, approved_for_push_tag_release=False,
        upstream_license_git_blob_sha=upstream_blob,
        required_payloads_present=required, no_raw_transfers=True, no_address_mapping=True,
        no_weights=True, no_manuscript_sources=True, pseudonymous_linkage_present=True,
        local_cold_replay='PASS', anonymous_public_download_replay='NOT_RUN')
    write(REPORTS / 'public_release_manifest.json', release_manifest)
    write(PROJECT / 'publication/318/release_manifest.json', release_manifest)

    facts = dict(exact_file_author_approvals=False, official_funding_evidence=author['official_award_document_verified'],
                 funder_role_confirmed=False, declarations_confirmed=True, rights_cleared=False,
                 public_release_verified=False, anonymous_replay_pass=False,
                 restricted_plan_accepted_by_platform=False, both_author_preprint_consent=False,
                 both_author_journal_consent=False, preprint_posted=False, preprint_doi=None,
                 preprint_disclosed_to_journal=False, special_issue_selection_verified=False)
    state = evaluate(numeric['status']=='PASS' and preservation['status']=='PASS',
                     builds['status']=='PASS' and identity['shared_content_identical'], facts)
    state.update(date=DATE, current_final_candidate=identity['current_final_candidate'],
        both_author_preprint_consent='PENDING', both_author_journal_consent='PENDING',
        coauthor_signatures_or_verifiable_approvals='PENDING', funding_official_evidence='PENDING',
        author_details_and_declarations_confirmed=True, author_confirmation_source=str(supplied.relative_to(ROOT)),
        author_confirmation_source_sha256=sha(supplied), funding_statement='CONFIRMED_BY_USER',
        coi_statement='CONFIRMED_BY_USER', funder_role='PENDING', funder_role_and_coi='PENDING',
        license_and_rights='HOLD', public_tds_release='PENDING', anonymous_download_replay='NOT_RUN',
        preprint_submission='NOT_SUBMITTED', preprint_posting='NOT_POSTED', preprint_doi=None,
        mdpi_portal_submission='NOT_SUBMITTED', special_issue_selection_verified=False,
        final_pdf_sha256=final_hashes, final_source_sha256=source_hashes,
        original_317_science_changed=False, original_numeric_zip_sha256=NUMERIC_SHA,
        SUBMITTED_PREPRINT=False, POSTED_PREPRINT=False, SUBMITTED_MDPI=False,
        PUBLIC_RELEASE_VERIFIED=False, detector_fits_performed=0,
        decision='TECHNICAL_WORK_COMPLETE_HOLD_AUTHOR_ACTION', facts=facts)
    regression = REPORTS / 'regression_verification.json'
    state['closure_regression_tests'] = load(regression)['status'] if regression.exists() else 'NOT_RUN'
    write(OUT / 'submission_readiness_318.json', state)
    write(AUTH / 'approval_identity.json', dict(candidate_id=identity['current_final_candidate'], artifacts=checks,
        supplied_author_record=str(supplied.relative_to(ROOT)), supplied_author_record_sha256=sha(supplied),
        author_details_and_declarations_confirmed=True, independent_coauthor_signatures_verified=False,
        permission_to_publish_or_submit=False, public_release_approved=False,
        funding_statement_confirmed=True, official_award_document_verified=False,
        authors=[dict(name=a['name'], orcid=a['orcid'], identity_and_declarations_confirmed=True,
                      final_manuscript_approval='PENDING', preprint_publication_consent='PENDING',
                      journal_submission_consent='PENDING', redistribution_rights_confirmation='PENDING',
                      verifiable_approval_reference=None) for a in author['authors']]))
    for a in author['authors']:
        slug = 'seong_su_park' if a['name']=='Seong-Su Park' else 'ki_hyung_kim'
        form = AUTH / ('author_signoff_'+slug+'.md')
        lines = ['# R318 exact-file signoff — '+a['name'], '',
                 'Status: PENDING_FINAL_FILE_SIGNOFF. This is an unsigned form, not consent.', '',
                 'Identity, affiliation, ORCID, CRediT, funding wording, IRB/consent, COI and AI',
                 'disclosure are already confirmed by the user-specified R317 JSON; do not reopen them.', '',
                 'Candidate: `'+identity['current_final_candidate']+'`.', '',
                 '|Artifact|SHA-256|','|---|---|']
        lines += ['|'+e['path']+'|`'+e['sha256']+'`|' for e in checks]
        lines += ['', 'Numeric package: `'+NUMERIC_SHA+'`.', '',
                  '|Approval scope|Status|','|---|---|',
                  '|Current manuscript/declarations, exact hashes above|PENDING|',
                  '|Preprints.org public posting and withdrawal-policy understanding|PENDING|',
                  '|MDPI submission and preprint disclosure|PENDING|',
                  '|Redistribution of linkable numerical evidence under applicable terms|PENDING|',
                  '|Funder role confirmation and official award evidence|PENDING|', '',
                  'Approver / date / verifiable approval reference: __________',
                  'No signature, portal receipt or approval is generated by this form.']
        expected_form = '\n'.join(lines)+'\n'
        if form.exists():
            assert form.read_text() == expected_form, 'Existing signoff differs: preserve it and audit a version transition before replacing it.'
        else:
            form.write_text(expected_form)

    table = '|Venue|Artifact|SHA-256|\n|---|---|---|\n' + '\n'.join(
        '|'+v['venue']+'|'+name+'|`'+e['sha256']+'`|' for v in identity['venues'] for name,e in v['current'].items())
    doc('final_candidate_ledger.md', '# Canonical final candidate ledger — '+DATE+'\n\n'
        'CURRENT_FINAL_CANDIDATE: **'+identity['current_final_candidate']+'** (exactly one).\n\n'
        'Retained predecessors: EARLIER_317_SCIENTIFIC_BUILD → FINAL_CANDIDATE_317_POST_DECLARATION → current R318.\n'
        'R317 files/reports remain unchanged; current hashes are never substituted into old reports.\n\n'+table+
        '\n\nCommon scientific body: `'+identity['original_scientific_body_sha256']+'`.\n'
        'Common bibliography: `'+identity['original_bibliography_sha256']+'`.\n'
        'The final SHA ledger includes every shared table, figure, declaration and bibliography.\n'
        'See `SHA256SUMS_FINAL.txt` (paths relative to repository root) and `results/318/candidate_identity.json`.\n'
        'Numerical ZIP remains `'+NUMERIC_SHA+'`. All earlier/post/current hashes are in the JSON ledger.\n')
    transition = ['# Version transition audit', '',
        'The requested uploaded 317-reports.zip / 317-results.zip / submission.zip are not present in the task318 directory.',
        'No checksum for unavailable upload containers is invented. Equivalent local deliveries match the exact',
        'post-declaration hashes quoted by the review; the original report and rollback ZIP provide earlier identities.', '',
        '|Venue|Earlier PDF / source prefix|Post-declaration PDF / source prefix|Current PDF / source prefix|',
        '|---|---|---|---|']
    for v in identity['venues']:
        pp = next(e['sha256'] for n,e in v['current'].items() if n.endswith('article.pdf'))
        zz = next(e['sha256'] for n,e in v['current'].items() if n.endswith('source.zip'))
        transition.append('|'+v['venue']+'|'+v['earlier']['local']['pdf_sha256'][:12]+' / '+v['earlier']['source_zip_sha256'][:12]+'|'+v['post_declaration']['local']['pdf_sha256'][:12]+' / '+v['post_declaration']['source_zip_sha256'][:12]+'|'+pp[:12]+' / '+zz[:12]+'|')
    transition += ['', 'Byte comparisons of earlier and post-declaration ZIP payloads isolate author/front matter,',
        'declarations, and the Supplement author line. Body, bibliography, numeric tables and figures match.',
        'R318 changes only wrapper array support, the checklist column layout (row text byte-identical),',
        'and Data Availability clarification of linkable IDs/reproduction scope. All supplied author facts remain intact.',
        'All displayed decimal quantities in each venue PDF preserve their exact multiplicities across R317→R318.',
        'Integer results are bound by byte-identical scientific source/row content; MDPI line/page counts may differ.',
        '271 archived numerical/table/figure files match local R317; all 309 manifest payloads verify in the cold archive.',
        'Original 496-file/55-checkpoint/65-vector checks are prior R317 evidence, not a repeated raw scan in R318.',
        'History copies are private/318/r317_history; original reports and their different hashes are retained.']
    doc('version_transition_audit.md', '\n'.join(transition))
    buildlines = ['# Fresh-build audit', '',
        'Each supplied post-declaration and new current venue ZIP was extracted into its own empty temporary directory.',
        'Article and Supplement were independently compiled: eight successful cold builds total.',
        'SOURCE_DATE_EPOCH pins 2026-10-30. Byte-hash differences are explicitly retained; extracted text must match.', '',
        '|Venue|Generation|Article / Supplement pages|Article bytes equal|Supplement bytes equal|Text equal|Overfull|',
        '|---|---|---|---|---|---|---|']
    for b in builds['fresh_builds']:
        buildlines.append(f"|{b['venue']}|{b['phase']}|{b['article']['pages']} / {b['supplement']['pages']}|{b['article_pdf_byte_identical']}|{b['supplement_pdf_byte_identical']}|{b['extracted_text_identical']}|{b['article']['overfull_boxes']+b['supplement']['overfull_boxes']}|")
    buildlines += ['', 'Undefined citations/references and missing-asset compilation failures: zero.',
                   'Citation key order and figure/table labels match across wrappers; shared Bib/figures/tables/declarations are byte-identical.',
                   'Build metadata and full per-artifact hashes: results/318/build_verification.json.']
    doc('fresh_build_audit.md', '\n'.join(buildlines))
    doc('final_scientific_claim_audit.md', '''# Final scientific claim audit — PASS_WITH_BOUNDED_CLAIMS

R317 body/Bib/scientific tables/figures/scores remain unchanged. No detector training occurred.
The same 24,316 contract observations reverse prevalence profiles under first/last sorting:
39.8508/8.4453/2.9331% versus 14.7935/71.3189/56.9901%. This is sampling-associated
composition sensitivity, not an identified right-censoring mechanism or behavioral drift.
Observed lifetimes (0.670625 vs 608.447361 days) are archive spans, not survival estimates.

The 24,179-observation seven-day cohort has test 3,627 / positive 2,037 (56.1621%):
MLP AP .958154 and LR .906827 cannot be paired as gains over the original cohort.
Original matched MLP .4751497 / GIN+BCE .4523830 remain central; archival GIN .4648
is separate. GAT time .3118749 / no-time .3897312 change both the normalized absolute
time branch and head dimension; no pure causal extrapolation effect is identified.
Isolation Forest .0169891 is one fixed unlabeled score orientation, with no probability metrics.
Deterministic LR/LightGBM repeated fits do not imply zero sampling uncertainty.

Label discovery times and exact collection stopping policy remain unknown. No prospective
early-warning, actor-independence, formal privacy or long-deployment claim is established.
The old 7-comparison and separate 6-comparison Holm families remain unchanged.
The 19-hour block GIN-minus-MLP interval [-.07361948,.04706323] includes zero;
Polygon has one positive, supporting point AP but no useful comparative inference.

R318 reran all 110 arrays/95 validation-only calibrators in a clean local directory;
maximum point/calibrated metric discrepancy 1.1102230246251565e-16. Independent
sklearn aggregation checks all 22 model groups, 464 Top-k rows and 33 chain rows.
This is numerical replay, not a new raw 268,282,924-row scan or GPU full retraining.
Raw reconstruction identity is retained in R317 lineage/source/freeze reports.
See results/318/numeric_preservation.json and independent_model_metrics.csv.
''')
    doc('figure_table_preflight.md', '''# Figure/table preflight — PASS

Rendered page 1 of both venues, the checklist pages and Figure 2 were visually inspected.
Private QA images are in private/318/qa, outside paper sources and public results.
The Preprint first page visibly contains title, both authors/affiliations, institutional
contacts, corresponding-author note, abstract and keywords, without publisher branding.
The official MDPI class is the preserved template whose class hash was checked in R317.

Checklist Table 6 now uses ragged-right paragraph cells; MDPI uses the existing full
page width, including its left margin. Row/caption wording is unchanged. Continued
headers repeat on the second page; cells and horizontal rules remain inside margins.
The cramped justified/hyphenated columns are resolved. Figure 2 and caption remain legible.
All fresh builds have zero overfull boxes and undefined references. Current article
and Supplement page counts are recorded per venue in candidate_identity.json and
fresh_build_audit.md; layout/disclosure reflow can change page counts without changing science.

MDPI's Version ... submitted to Appl. Sci. and empty doi.org template footer are
template output, not submission/DOI evidence. Administrative state remains NOT_SUBMITTED;
no DOI value, acceptance date or actual portal receipt is invented.
''')
    doc('rights_and_reuse_audit.md', '''# Rights and reuse audit — HOLD

The [upstream repository](https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs)
describes address-named per-contract CSVs and numeric-index/address mappings. Its
[pinned LICENSE](https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs/blob/7264f1bf510f7ba4f5041ac7a29b606abc12f262/LICENSE)
is CC BY-NC-SA 4.0, independently fetched through GitHub (blob c14a34c4954e217a88d27d206752db1c79dcaf47).
The license is not an unrestricted software grant: attribution, noncommercial and
share-alike conditions apply within its scope. The [official legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
also separates copyright/database rights from privacy and other rights. This audit
identifies obligations and unresolved scope; it does not supply author/third-party permission.

Actual ZIP scan: no direct 40-hex contract addresses in CSV/JSON, no raw transfers,
address mapping table, weights or manuscript source. However 215 CSVs contain sample_id;
all 24,316 lineage IDs join one-to-one to a private contract mapping. Chain/time/category
signatures can also support joins to upstream records. Identifier-reduced means
pseudonymous/linkable here; no anonymity or privacy guarantee is claimed. Only aggregate
linkage counts are recorded publicly; no private mapping is copied into public outputs.

The exact 13-MB R317 candidate remains sealed; a byte-identical release copy, attribution
NOTICE and the upstream license are staged in publication/318/release. No new derivative
license is imposed on behalf of unconsulted rights holders. Software and numeric
derivatives have distinct license scopes; manuscript CC BY intentions do not silently
relicense upstream-derived data or figures. The existing manuscript citation/upstream
link is retained. Authors must clear final distribution scope, including any derivative
figure permissions, before pushing/tagging/releasing. No history rewrite/removal occurred.

The R317 source manifest's new_code_committed=false was a historical working-tree
snapshot. Current R314–R317 files are now checked against local HEAD and a R317 code
file is publicly readable at that commit; this is not a numerical TDS release or
author release permission. See results/318/source_commit_reconciliation.json.
''')
    doc('anonymous_download_verification.md', '''# Anonymous public verification — NOT_RUN

An unauthenticated GitHub release-list GET returned HTTP 200 on 2026-10-30.
Only StreamMC/Benchmark releases are listed; none is reused as a TDS release.
The R317 replay source can be read at a public commit. These facts do not establish
that the exact TDS numeric asset is publicly available. No TDS tag/asset URL is
recorded. Anonymous download + checksum + replay for the candidate is NOT_RUN.

Local clean-directory replay PASS remains separate. After explicit rights/release
approval and actual publication, retain the actual immutable tag/commit/asset URL,
HTTP status, checksum and fresh-directory unauthenticated replay receipt, then
update the gate. API listing success alone cannot set PUBLIC_RELEASE_VERIFIED=true.
''')
    # These records are private because they contain exact-file author approval forms.
    (AUTH / 'ORCID_and_correspondence_audit.md').write_text('''# ORCID and correspondence audit

The user explicitly designated the R317 author_confirmation_from_a16.json as the
current author authority. Both PDFs/source ZIPs match every supplied identity field.
Kim: 0000-0001-6624-3003, Department of Computer Engineering, Graduate School,
Ajou University, Suwon, Republic of Korea. The old A16 ...4475 is historical only;
both Benchmark A16 venue sources were already corrected by explicit user request.

Official public API https://pub.orcid.org/v3.0/0000-0001-6624-3003/person returned
HTTP 200 and public name Ki-Hyung Kim on 2026-10-30. This corroborates the user's
correction beyond checksum; it is not authenticated account ownership or a signature.
Park's 0009-0008-4056-3875 follows the same supplied author record, checksum valid.
Contacts/order/affiliations follow that record; no email or phone verification
message was sent. Kim is corresponding author, kkim86@ajou.ac.kr, +82-31-219-2433.
No supplied author fact is reopened. Final exact-file approval/publication rights
remain separately unsigned, matching the record's explicit false permission flag.
''')
    (AUTH / 'official_funding_evidence_register.md').write_text('''# Official funding evidence register

Funding wording and identifier IITP-2026-RS-2021-II211835: CONFIRMED_BY_USER from
the designated R317 JSON. Applied without change to both current manuscripts.
Official award document independently verified: PENDING (the supplied record says false).
No official award/contract evidence file was found among supplied 318 task files or
the scoped author/submission records. Public searches returned related paper
acknowledgments; these are not the funding authority's award record. No alternative
award number or funder-role statement was invented.

Evidence location / issuing authority / document SHA / reviewer: not supplied.
Funder role in design, collection, analysis, writing or publication: PENDING specific
confirmation; the existing no-conflict declaration remains CONFIRMED_BY_USER.
''')
    (AUTH / 'declarations_venue_matrix.md').write_text('''# Declarations parity matrix

|Item|Preprints|MDPI|Basis|
|---|---|---|---|
|Authors/order/affiliations/ORCID/contact|Applied|Applied|Designated R317 JSON|
|CRediT / Funding / IRB / Consent / COI|Applied|Applied|Same confirmed allocation/wording|
|Generative AI use|Methods + Acknowledgments|Methods + Acknowledgments|Code/analysis/drafting disclosed; no invented versions|
|Data Availability|Same shared text|Same shared text|Real repo/upstream URLs; pseudonymous IDs; release pending|
|Supplementary Materials|Present|Present|Current venue Supplement|
|Official award evidence / funder role|Pending|Pending|Separate from confirmed wording|
|Exact-file final/publication signoff|Pending|Pending|Unsigned forms; hashes attached|

Shared declarations are byte-identical. AI use is not reduced to language editing.
No claim about a shared origin of papers is added to manuscripts or cover letters.
''')
    (PROJECT / 'submission/applsci/318/cover_letter.md').write_text('''Draft for author approval — NOT SUBMITTED

Dear Editors,

Please consider “Auditing Sampling-Associated Prevalence Shift in Supervised
Blockchain Graph Anomaly Screening” for Applied Sciences, Special Issue
“Machine Learning and Its Application for Anomaly Detection” (c67h3u9qgt).

The study audits how first-versus-last contract observation times reverse the
prevalence profile of the same blockchain graph population. Matched graph/tabular
controls, validation-only calibration and a separately registered seven-day cohort
bound the interpretation. We report sampling-associated composition sensitivity
without identifying a causal collection mechanism or claiming prospective early warning.

The article and Supplement are prepared as a distinct MDPI submission candidate.
The numerical package supports saved-score replay; full training requires upstream
licensed inputs. The corresponding author is Ki-Hyung Kim, kkim86@ajou.ac.kr.

Author-local completion before dispatch: secure both authors' exact-file submission
approval; verify official funding/funder role and redistribution scope; verify the
live Special Issue in the portal; insert a real preprint DOI/version only after
actual posting and disclose it to the journal. No exclusive-submission attestation,
portal acknowledgement or preprint DOI is fabricated in this staged letter.

Target: https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt
''')
    doc('venue_policy_and_authority_audit.md', '''# Venue policy and authority audit — 2026-10-30

The user-designated R317 author JSON is the operative source for author details and
declarations; its permission_to_publish_or_submit=false and lack of separate
signatures remain separate facts. No repeated generic author-information request is needed.

[Preprints.org instructions](https://www.preprints.org/instructions-for-authors)
were read: coauthor awareness/consent, rights and persistent posting matter; first-page
metadata and a complete LaTeX source archive are required; AI use belongs in Methods.
The current neutral wrapper and explicit Methods/acknowledgment disclosures meet
those formatting/disclosure checks. The authors' platform consent remains pending.

MDPI [publication ethics](https://www.mdpi.com/ethics) was available as an official
indexed excerpt on GenAI author responsibility; direct fetch returned HTTP 429.
The [target Special Issue](https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt)
also returned 429; no current deadline/editor/portal availability is newly asserted.
The requested topic/URL are used in metadata and cover letter. No previous GNN
Special Issue ID is introduced. Actual portal selection is NOT_VERIFIED.
The official MDPI template class matches the R317 verified class hash.

ORCID official API returned 200 for Kim's corrected number/name. Official award
evidence was not obtained; related paper acknowledgments do not replace it.
External response summaries are in results/318/external_checks.json. Platform
submission/posting/DOI and author signoff are never inferred from a successful build.
''')
    doc('submission_readiness_318.md', '# 318 최종 작업보고서 — '+DATE+'\n\n'
        '**기술 검증 완료: S=PASS, C=PASS. 출판·투고: P/J=HOLD_AUTHOR_ACTION.**\n\n'
        '317 초기본·저자정보 반영본·318 최종 후보를 해시 대장으로 연결했습니다. 과학 본문과 원래 수치·모델은 불변입니다. '
        '체크리스트 표를 개선하고 Data Availability에 가명 ID 연결 가능성과 재현 범위를 명시했습니다. '
        '저자 사항은 사용자가 지정한 author_confirmation_from_a16.json을 반영했으며 재확인을 요구하지 않습니다.\n\n'
        '110개 배열/95개 보정기 재생성 PASS, 최대 오차 1.11e-16. 22개 모델 집계·Top-k 464행·체인 AP 33행을 별도 검산했습니다. '
        '두 세대의 플랫폼 소스 ZIP에서 본문/부록 총 8건을 독립 빌드했습니다. '
        f'현재 Preprint {page_counts["preprints"]}쪽/MDPI {page_counts["applsci"]}쪽, 각 부록 4쪽, overfull/미해결 참조 0건입니다.\n\n'
        '정확한 최종 파일에 대한 양 저자의 승인, 공식 연구비 증빙·funder 역할, 재배포 권리, 실제 공개 릴리스·익명 검증, '
        '게시/DOI·실제 포털 접수는 미완료입니다. unsigned 승인서와 동일 바이트 릴리스 후보를 준비했습니다. '
        '작업지시서 G5 종료 규칙에 따라 추가 학습 없이 이 상태로 기술 작업을 종료합니다.\n\n'
        '```json\n'+json.dumps(state,indent=2,ensure_ascii=False)+'\n```\n\n'
        '7개 승인/정합성 회귀 검사의 실제 로그는 regression_tests_318.txt와 regression_verification.json에 보존했습니다.\n\n'
        '최종 파일: [해시 대장](final_candidate_ledger.md), [버전 전이](version_transition_audit.md), '
        '[독립 빌드](fresh_build_audit.md), [과학 주장](final_scientific_claim_audit.md), '
        '[권리 검토](rights_and_reuse_audit.md), [공개 검증 상태](anonymous_download_verification.md).\n\n'
        '미완료 근거: 사용자 지정 저자 기록은 permission_to_publish_or_submit=false, '
        'official_award_document_verified=false를 명시합니다. 318 지시서 D/G2, E/G3 및 G/G5는 '
        '사람의 정확한 파일 승인과 실제 공개·접수 증거를 요구합니다. 확인된 저자정보와 이 미완료 사항은 별개입니다.\n')
    doc('README.md', '''# R318 final publication-integrity closure

Start with [submission_readiness_318.md](submission_readiness_318.md).
One current candidate is bound by [final_candidate_ledger.md](final_candidate_ledger.md)
and SHA256SUMS_FINAL.txt. R317 histories are preserved, not rewritten.
Scientific/build gates PASS; publication/rights/actual submission remain separate.
Numeric inputs and machine records live in results/318; paper/venue/signoff files
remain private. Exact source/documentation is in scripts/revision318.
''')
    source_paths = [p for p in sorted(Path(__file__).parent.glob('*')) if p.is_file()]
    source_paths += [PROJECT / 'tests/test_closure318.py']
    sources = [dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in source_paths]
    write(OUT / 'source_manifest.json', dict(files=sources, scientific_source_commit=source['local_head'],
          r318_code_committed=False, note='Working-tree closure identity; not a release or fabricated publication commit'))
    print('R318 technical closure complete: S=PASS C=PASS P/J=HOLD_AUTHOR_ACTION')


if __name__ == '__main__':
    main()
