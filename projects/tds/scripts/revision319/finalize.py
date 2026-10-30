"""Seal R319 candidate identities and truthful publication gates after cold builds."""
import json
import re
import subprocess
from pathlib import Path
import pandas as pd
from common import ROOT,PROJECT,OUT,REPORTS,PRIVATE,PAPER,sha,write,require_builder

def pdftext(p):return subprocess.check_output(['pdftotext','-layout',str(p),'-'],text=True)
def md(path,s):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s)
def identify(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)

def main():
    require_builder()
    freeze=json.loads((OUT/'baseline_freeze_319.json').read_text());changed=[]
    for r in freeze['files']:
        p=ROOT/r['path']
        if not p.exists() or sha(p)!=r['sha256']:changed.append(r['path'])
    assert not changed,f'Frozen R314–318 files changed: {changed[:10]}'
    receipt=json.loads((OUT/'source_build_receipt_319.json').read_text());assert receipt['status']=='PASS'
    for name in ['seven_day_matched_models.csv','seven_day_training_histories.csv','seven_day_modelwise_calibration.csv',
                 'seven_day_operating_metrics.csv','seven_day_chain_support.csv','new_statistics_319.csv']:
        assert (OUT/name).exists(),name
    author=json.loads((PROJECT/'submission/author_records/319/author_confirmation_319.json').read_text())
    ai=author['ai_acknowledgments'];aff=author['authors'][0]['affiliation'];assert author['authors'][1]['affiliation']==aff
    extracts={};headers=[]
    for branch,entry in [('preprints','preprint_main.tex'),('mdpi_applsci_anomaly','applsci_article.tex')]:
        root=PAPER/branch;hdr=(root/entry).read_text();headers.append(hdr)
        decl=(root/'declarations.tex').read_text();assert ai in decl
        assert aff in hdr and hdr.count(aff)>=2
        for a in author['authors']:assert a['name'] in hdr
        text=pdftext(root/Path(entry).with_suffix('.pdf'));extracts[branch]=text
        for s in ['Seong-Su Park','Ki-Hyung Kim','Claude Opus 5.5','Google Gemini 4 Argon','OpenAI ChatGPT 6.1 Sol']:
            assert s in text,(branch,s)
        assert 'Department of Computer Engineering' in text and 'Suwon 16499' in text
        assert 'first_time_quarterly_composition.pdf' in (root/'data/r319_results.tex').read_text()
        assert not any(x in (root/'manuscript_body.tex').read_text() for x in ['Approve exact files and rights','Release before public submission'])
    assert 'Definitions/mdpi' not in headers[0]
    common=[]
    for name in ['manuscript_body.tex','declarations.tex','references.bib','supplement/supplement317.tex']:
        assert sha(PAPER/'preprints'/name)==sha(PAPER/'mdpi_applsci_anomaly'/name),name
        common.append(name)
    pdata=sorted((PAPER/'preprints/data').glob('*.tex'))
    assert pdata and all(sha(p)==sha(PAPER/'mdpi_applsci_anomaly/data'/p.name) for p in pdata)
    pfig=sorted((PAPER/'preprints/figures').glob('*.pdf'))
    assert all(sha(p)==sha(PAPER/'mdpi_applsci_anomaly/figures'/p.name) for p in pfig)
    for phrase in ['24,316','24,179','0.9445','0.9529','0.9582','0.9632','0.8943']:
        assert all(phrase in t for t in extracts.values()),phrase
    pdfqa=[]
    for branch,entry in [('preprints','preprint_main'),('mdpi_applsci_anomaly','applsci_article')]:
        root=PAPER/branch
        for item in [entry,'supplement/supplement317']:
            log=(root/(item+'.log')).read_text(errors='replace')
            bad=re.findall(r'Citation .* undefined|Reference .* undefined|LaTeX Error|Emergency stop',log)
            assert not bad,(branch,item,bad)
            pdfqa.append(dict(branch=branch,item=item,overfull_boxes=len(re.findall(r'Overfull \\hbox',log)),
                text_characters=len(pdftext(root/(item+'.pdf')))))
    by=pd.read_csv(OUT/'seven_day_matched_models.csv');means=by.groupby('arm')[['ap','roc_auc']].mean()
    stats=pd.read_csv(OUT/'new_statistics_319.csv')
    assert len(stats)==4
    preview='\n'.join(f'- {a}: AP {r.ap:.4f}, ROC-AUC {r.roc_auc:.4f}' for a,r in means.iterrows())
    md(REPORTS/'seven_day_failure_mode_report.md',f'''# Seven-day GNN/MLP gap: R319 evidence

The independent raw source scan reproduced all 24,316 original and 24,179 seven-day graph tensors exactly (zero mismatches), including edge multiplicity/order, labels, chain and cutoff. Input times never exceed the registered cutoff. This rules out a detected R317 source-to-tensor graph-builder discrepancy as the explanation for the frozen inputs; it does not prove all possible implementation bugs absent.

Five frozen seeds on the same seven-day split give:

{preview}

GIN-0 has mean AP {means.loc['gin0','ap']:.4f}; adding explicit cutoff-admissible graph size reaches {means.loc['gin_size','ap']:.4f}, while sum pooling reaches {means.loc['gin_sum','ap']:.4f}. The 24-feature MLP reaches {means.loc['mlp_full','ap']:.4f}; removing explicit counts/density reaches {means.loc['mlp_no_size','ap']:.4f}. The topology-summary MLP includes counts/density and reaches {means.loc['mlp_topology_only','ap']:.4f}. A single retained-node-count LR reaches AP 0.8943. The output strongly supports explicit size/readout information as a major contributor to the original GIN/MLP gap. It does not isolate the only causal factor: sum pooling changes capacity, tabular summaries expose degree statistics differently, and model optimization may still contribute. GIN and MLP never received identical raw information or parameter budgets. The 7-day test has 56.16% positive prevalence, and the labels are retrospective.

Each run's complete epoch train/validation loss, AP, ROC-AUC, selected epoch, parameter count, positive weight, checkpoint hash and validation/test prediction hash is retained. R319 test scores were read once after validation-selected checkpoints. The original R317 GIN and MLP scores remain unchanged. The separate four-comparison family is in `new_statistics_319.csv`; non-significance is not equivalence.
''')
    md(REPORTS/'results_value_diff_vs_318.md',f'''# R318 to R319 scientific value changes

R318's frozen seven-day summary reported weighted-BCE GIN AP 0.6358, summary MLP AP 0.9582, LR AP 0.9068, and time-input GATv2 AP 0.6126. These files and the R318 article PDF hashes remain unchanged. R319 repeats GIN-0 on the same fixed seven-day cohort and obtains mean AP {means.loc['gin0','ap']:.4f}; the small difference is a new run, not a correction to R317 scores. Explicit size raises GIN AP to {means.loc['gin_size','ap']:.4f}; sum pooling reaches {means.loc['gin_sum','ap']:.4f}; full MLP reaches {means.loc['mlp_full','ap']:.4f}; no-explicit-size MLP reaches {means.loc['mlp_no_size','ap']:.4f}. Validation-selected seven-day LightGBM reaches {means.loc['lightgbm','ap']:.4f}. The cutoff-admissible single retained-node count AP is 0.8943; the 7-day full-archive-span oracle AP 0.9464 is future dependent and not an operational control.

The new first-transfer quarter/chain composition and future-activity-conditioned subgroup sensitivity change interpretation. The abstract, Methods, Results, Discussion, Supplement and decision figure are refreshed in both candidates. No original versus seven-day AP difference is assigned a paired inferential test. Original R317 7+6 statistical families remain untouched; R319 adds a separate frozen four-comparison exploratory family.
''')
    md(REPORTS/'author_affiliation_ai_parity_audit.md',f'''# R319 author, affiliation and AI parity

Both source headers and extracted article PDFs contain `Seong-Su Park` and `Ki-Hyung Kim`. Both authors have the exact requested affiliation: `{aff}`. Kim retains corresponding-author status and corrected ORCID `0000-0001-6624-3003`; Park retains ORCID `0009-0008-4056-3875`. Both branches have byte-identical `declarations.tex` (SHA-256 `{receipt['sources_byte_identical']['declarations']}`) and a common Methods AI-use paragraph covering code and analysis. The user-specified Acknowledgments wording, including three model/product strings, occurs in both PDF text extractions. Historical model-version use was not independently verified; the declaration says earlier versions were not retained.

The A16-derived CRediT, Funding, IRB, Consent and COI statements are shared in both branch declarations. Official grant proof, author-specific declarations and exact-file signed approval remain pending. `projects/tds/CITATION.cff` reflects the R319 author names and ORCIDs. Earlier paper files are preserved.
''')
    md(REPORTS/'source_clean_build_audit.md',f'''# Independent source ZIP builds

Both venue source ZIPs were generated from private R319 trees and extracted into separate clean temporary directories. Article and Supplement compiled in each clean directory; extracted PDF text matched the local build. Shared body, declarations, bibliography, supplement, data tables and figures are byte-identical across venues; wrappers and MDPI class files are venue-specific. Preprints source has no MDPI class or logo references. Build receipt: `results/319/source_build_receipt_319.json`. Article/supplement hashes are in `candidate_identity_319.json`.

Scientific source parity is stronger than visual page equality: template page counts differ as expected. The LaTeX logs contain no unresolved citation/reference or fatal-error diagnostics. No public submission occurred.
''')
    md(REPORTS/'figure_table_pdf_preflight.md',f'''# R319 figure, table and PDF preflight

The first-transfer quarter figure is a vector PDF and visually reviewed. Its Wilson intervals display only chain-quarter support of at least 20, with every period retained in numeric CSV. The decision figure's last box now describes scientific evidence archiving rather than author approval or release. Both manuscripts include the same new size/lifetime table, model table, and separate-family statistics table, plus the same Supplement inputs.

Extracted PDF text contains both author names, requested affiliation, AI product strings and common key metric values. Four build logs report:

{chr(10).join(f'- {r['branch']} {r['item']}: {r['overfull_boxes']} overfull box notices, {r['text_characters']} extracted characters' for r in pdfqa)}

No unresolved citations/references or fatal LaTeX errors were observed. Critical path/hash extraction should be rechecked by authors on the final rendered pages before signoff; an overfull notice count alone is not a visual-quality verdict.
''')
    older={v:identify(PROJECT/'submission'/v/'318'/n) for v,n in [('preprints','preprint_article.pdf'),('applsci','applsci_article.pdf')]}
    artifacts={}
    for v,items in [('preprints',['preprint_article.pdf','supplement.pdf','preprint_source.zip']),('applsci',['applsci_article.pdf','supplement.pdf','mdpi_source.zip'])]:
        artifacts[v]={n:identify(PROJECT/'submission'/v/'319'/n) for n in items}
    candidate=dict(id='FINAL_CANDIDATE_319_SCIENTIFIC_RECONCILED',date='2026-10-30',
        previous_candidate='FINAL_CANDIDATE_318_RECONCILED',previous_articles=older,artifacts=artifacts,
        science_source_sha256=receipt['sources_byte_identical'],protocol_sha256=sha(OUT/'protocol_freeze_319.json'),
        builder_sha256=sha(OUT/'seven_day_builder_equivalence.json'),public_release_verified=False,preprint_doi=None,
        preprint_status='NOT_POSTED',journal_status='NOT_SUBMITTED',author_approval='PENDING_EXACT_FILE_SIGNOFF',
        rights='HOLD',official_funding_evidence='PENDING',submission_permission=False)
    write(OUT/'candidate_identity_319.json',candidate)
    for venue,title,zipname,pdf in [('preprints','Preprints.org','preprint_source.zip','preprint_article.pdf'),
                                    ('applsci','Applied Sciences Special Issue: Machine Learning and Its Application for Anomaly Detection','mdpi_source.zip','applsci_article.pdf')]:
        d=PROJECT/'submission'/venue/'319'
        write(d/'metadata.json',dict(candidate_id=candidate['id'],venue=title,
            article=artifacts[venue][pdf],supplement=artifacts[venue]['supplement.pdf'],source=artifacts[venue][zipname],
            authors=author['authors'],preprint_doi=None,posted=False,submitted=False,permission_to_submit=False,
            special_issue_url='https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt' if venue=='applsci' else None,
            portal_selection_verified=False))
        md(d/'README.md',f'# {title} R319 local candidate\n\nThe PDF, Supplement and clean-build source ZIP are author-local. They are not posted, submitted or approved for release. Exact hashes and prior-candidate linkage are in `results/319/candidate_identity_319.json`. The author signoff form is under `submission/author_records/319`; rights and official funding evidence remain pending.\n')
    md(PROJECT/'submission/applsci/319/cover_letter.md',f'''Draft for author approval — NOT SUBMITTED

Dear Editors,

Please consider “Auditing Sampling-Associated Prevalence Shift in Supervised Blockchain Graph Anomaly Screening” for Applied Sciences, Special Issue “Machine Learning and Its Application for Anomaly Detection” (c67h3u9qgt).

The article reconstructs 24,316 archived contract graphs and 24,179 seven-day graphs from source transfers, then measures sampling-associated class-composition, observed-activity and graph-size sensitivities. On the seven-day cohort, explicit size information changes GIN ranking substantially; a strong tabular control and retrospective lifetime oracle bound the interpretation. Unknown label discovery times prevent prospective early-warning claims. The corresponding author is Ki-Hyung Kim (kkim86@ajou.ac.kr), with Seong-Su Park as first author; both share `{aff}`.

This is a staged local candidate. Neither preprint posting/DOI nor journal portal selection/submission has been verified. The authors must approve the exact files and resolve official funding and reuse rights before dispatch. No exclusive-submission attestation is supplied by this draft.

Target: https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt
''')
    form=['# Exact-file R319 author approval — unsigned','',
          'Candidate ID: `FINAL_CANDIDATE_319_SCIENTIFIC_RECONCILED`. This form does not assert an approval.','',
          '| Venue | Artifact | SHA-256 |','|---|---|---|']
    for v,files in artifacts.items():
        for n,r in files.items():form.append(f'| {v} | `{n}` | `{r["sha256"]}` |')
    form+=['','Seong-Su Park: signature/date **PENDING**.','Ki-Hyung Kim: signature/date **PENDING**.',
           'Each author must also confirm CRediT, Funding, COI and submission permission for these exact hashes. Official award/funder-role evidence and public numerical-release scope remain **PENDING**.','']
    md(PROJECT/'submission/author_records/319/exact_file_approval_319.md','\n'.join(form))
    hashes=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS_319.txt':hashes.append((p,sha(p)))
    for v in ['preprints','applsci']:
        for p in sorted((PROJECT/'submission'/v/'319').glob('*')):
            if p.is_file():hashes.append((p,sha(p)))
    (OUT/'SHA256SUMS_319.txt').write_text(''.join(f'{h}  {p.relative_to(ROOT)}\n' for p,h in hashes))
    md(REPORTS/'final_readiness_319.md',f'''# TDS R319 final readiness

**SCIENTIFIC_PASS / SUBMISSION_HOLD.** Candidate identity: `FINAL_CANDIDATE_319_SCIENTIFIC_RECONCILED` (`results/319/candidate_identity_319.json`). R314–R318 selected baselines rehashed with **zero changes**. Source-to-tensor audit: **PASS** for 24,316 original and 24,179 seven-day graphs, 268,282,924 source transfer rows and zero mismatches. The frozen 5-seed/6-arm experiment, seven-day controls, subgroup/calendar diagnostics and separate 4-comparison exploratory family are complete. Both venue PDFs and Supplements compile from independently extracted source ZIPs; common science is byte-identical. Key old/new value differences and scientific limitations are in `results_value_diff_vs_318.md` and `seven_day_failure_mode_report.md`.

| Gate | Status |
|---|---|
| G0 baseline/protocol | PASS |
| G1 cutoff counts versus retrospective oracle | PASS |
| G2 builder identity and matched-input model diagnostics | PASS; representation contribution supported, residual mechanisms bounded |
| G3 first-time calendar/chain composition | PASS; mechanism unidentified |
| G4 seven-day controls, calibration, budgets and chain support | PASS for retrospective evaluation |
| G5 source/PDF author, affiliation, AI disclosure parity | PASS for draft; human exact-file approval pending |
| G6 dual cold source builds and scientific parity | PASS |
| G7 funding official proof, rights, author signature, public release and portal | HOLD |

No public tag, DOI, preprint posting or journal submission was created. Exact-file signoff form: `submission/author_records/319/exact_file_approval_319.md`. Rights status: `reports/319/rights_release_status.md`. All R319 hashes: `results/319/SHA256SUMS_319.txt`.
''')
    print('R319 final ledger and readiness prepared',flush=True)

if __name__=='__main__':main()
