"""Finalize byte-bound PRIVATE CANDIDATES, never publish or invent approvals."""
from __future__ import annotations
import csv
import difflib
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from r03_common import ROOT,PROJECT,R02,OLD,PAPER,OUT,RUN,artifact,sources,unchanged,now,read,write,sha256
from r03_paper import clean,validate_generated


def gates(author_approved=False,ai_final=False,roles_final=False,rights_final=False,portal_verified=False):
    """Truthful fail-closed status; no posting prerequisite falsely attributed to journal."""
    shared=[]
    for ok,name in ((author_approved,'two-author exact-version approval'),(ai_final,'actual AI-use declaration'),(roles_final,'confirmed CRediT'),(rights_final,'author rights/posting agreement')):
        if not ok:shared.append(name)
    return {'Gate_P':{'status':'HOLD' if shared or not portal_verified else 'PASS','blockers':shared+([] if portal_verified else ['target-journal preprint compatibility final recheck'])},
            'Gate_J':{'status':'HOLD' if shared or not portal_verified else 'PASS','blockers':shared+([] if portal_verified else ['final Applied Sciences portal/deadline/template/APC/AI/payload rules'])},
            'PUBLICATION_EXECUTED':False,'JOURNAL_SUBMISSION_EXECUTED':False,'user_workflow':'after author approval, Preprints.org first, then Applied Sciences',
            'preprint_posting_is_intrinsic_journal_gate_requirement':False,'DOI':None,'submission_receipt':None}


def check_zip(path,expected):
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(expected)
        for name,digest in expected.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name


def finalize():
    start=now();assert unchanged()==1103;validate_generated()
    audit=read(OUT/'paper_build_audit.json');decision=read(OUT/'visual_review_decision.json')
    assert decision['build_label']==audit['build_label'] and decision['technical_visual_status']=='VERIFIED'
    sheets=[]
    for r in audit['builds']:
        stem=Path(r['entry']).stem;assert r['PDF']['sha256'].startswith(decision['reviewed_pdf_sha256_prefixes'][stem])
        assert sha256(ROOT/r['PDF']['path'])==r['PDF']['sha256'] and r['pages']==decision['pages_viewed_on_contact_sheets'][stem]
        for a in r['rendered_pages']+r['contact_sheets']:assert sha256(ROOT/a['path'])==a['sha256']
        sheets.extend(r['contact_sheets'])
    assert len(sheets)==decision['contact_sheets_viewed']==7
    visual={'recorded_utc':now(),'decision':artifact(OUT/'visual_review_decision.json'),'build':artifact(OUT/'paper_build_audit.json'),
            'inspected_pdf_identities':{Path(r['entry']).stem:r['PDF'] for r in audit['builds']},'pages':decision['pages_viewed_on_contact_sheets'],
            'contact_sheets':sheets,'method':'agent actually viewed seven contact sheets/all41 pages plus journal9/10 and supplement3 individually',
            'status':'VERIFIED','human_author_approval':False}
    write(OUT/'final_build_visual_receipt.json',visual);write(OUT/'final_build_and_page_visual_audit.json',visual)
    source_records=[artifact(p) for p in sources(PAPER)];assert len(source_records)==33
    write(OUT/'source_set_manifest.json',{'run_id':RUN,'status':'candidate_not_approved','sources':source_records})
    source_set=artifact(OUT/'source_set_manifest.json')
    previous=read(PROJECT/'configs/author_declarations_r02.json');previous.update({'revision_id':RUN,'candidate_source_set':source_set,
        'credit_roles':'OPEN: existing draft only; user deferred final facts','ai_assistance_declaration':'OPEN: user deferred final facts; no tool/version inferred',
        'exact_final_source_pdf_evidence_version_approved_by_all_authors':False,'publication_execution_status':'paper not posted or submitted',
        'current_target':read(PROJECT/'configs/submission_target_r03.json'),'final_author_rights_confirmation':'OPEN'})
    write(OUT/'author_declarations_candidate.json',previous)
    for name in ('author_approval_exact_version.md','author_contributions_final.tex','ai_assistance_final.tex','preprint_final_package.zip','journal_final_package.zip'):
        assert not (OUT/name).exists(),'Do not relabel candidate as approved final: '+name
    with (R02/'verification/claim_to_raw_r02.csv').open(newline='') as h:
        reader=csv.DictReader(h);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        for pfield,hfield in (('raw_path','raw_sha256'),('output_path','output_sha256')):
            if row[pfield]:assert sha256(ROOT/row[pfield])==row[hfield],row[pfield]
        row.update({'R03_candidate_source_set_sha256':source_set['sha256'],'R03_action':'existing R02 evidence reused and local byte-rechecked; not rerun'})
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields+['R03_candidate_source_set_sha256','R03_action']);writer.writeheader();writer.writerows(rows)
    (OUT/'claim_to_raw_final.csv').write_text(buffer.getvalue())
    ledger=[]
    for p in sources(PAPER):
        old=OLD/p.relative_to(PAPER);different=sha256(old)!=sha256(p)
        ledger.append({'relative_path':str(p.relative_to(PAPER)),'R02_sha256':sha256(old),'R03_sha256':sha256(p),'changed':different,'reason':'R03 caption/protocol/date/candidate clarification' if different else 'immutable science input reused'})
    write(OUT/'R02_R03_source_change_ledger.json',{'source_set':source_set,'files':ledger})
    diff=[]
    for p in sources(PAPER):
        if p.suffix=='.png':continue
        old=OLD/p.relative_to(PAPER)
        diff.extend(difflib.unified_diff(old.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='R02/'+str(p.relative_to(PAPER)),tofile='R03/'+str(p.relative_to(PAPER))))
    (OUT/'editorial_source.diff').write_text(''.join(diff))
    build=PAPER/('build_r03_'+audit['build_label']);packages=[];package_records=[]
    for stem in ('preprint','journal'):
        path=OUT/(stem+'_candidate_package.zip');assert not path.exists(),'Existing package; never overwrite reviewed candidate'
        included=[p for p in sources(PAPER) if p.relative_to(PAPER).as_posix()!=(('journal' if stem=='preprint' else 'preprint')+'.tex')]
        document=OUT/('cover_letter_candidate.md' if stem=='journal' else 'preprint_disclosure_candidate.md')
        files=[(p,str(p.relative_to(PAPER))) for p in included]+[(build/(stem+'.pdf'),stem+'.pdf'),(build/'supplement.pdf','supplement.pdf'),(document,document.name)]
        expected={name:sha256(p) for p,name in files}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for p,name in files:z.write(p,name)
        check_zip(path,expected);packages.append(artifact(path))
        with tempfile.TemporaryDirectory(prefix='streammc_r03_pkg_') as temp:
            with zipfile.ZipFile(path) as z:z.extractall(temp)
            result=clean(Path(temp),stem+'.tex',OUT/'package_smoke_builds'/stem,'package_'+stem)
            assert (ROOT/result['extracted_text']['path']).read_text()==(build/(stem+'.txt')).read_text(),'Packaged sources compile to different text'
        package_records.append({'package':artifact(path),'member_sha256':expected,'source_count':len(included),'smoke_build':result,'identical_pdf_text':True})
    write(OUT/'private_package_identities_r03.json',{'packages':package_records,'approved_final':False,'paper_or_original939_uploaded':False})
    payloads={}
    for stem in ('preprint','journal'):
        fs=[OUT/(stem+'_candidate_package.zip'),build/(stem+'.pdf'),build/'supplement.pdf',OUT/('cover_letter_candidate.md' if stem=='journal' else 'preprint_disclosure_candidate.md')]
        payloads[stem]={'planned_files':[artifact(p) for p in fs],'total_bytes':sum(p.stat().st_size for p in fs),'large_public_evidence_included':False,
                        'size_count_scope':'conservative full upload set, even PDFs/letter also embedded in ZIP; external247MiB evidence excluded'}
        assert payloads[stem]['total_bytes']<120_000_000
    write(OUT/'submission_payload_bytes.json',{'payloads':payloads,'planning_budget_bytes':120_000_000,'live_portal_limit_verified':False,'portal_acceptance':'OPEN'})
    binding=read(OUT/'r02_final_identity_lock.json')
    identities={'run_id':RUN,'created_utc':now(),'status':'candidate_not_approved','source_set':source_set,'sources':source_records,
        'PDFs':[r['PDF'] for r in audit['builds']],'candidate_packages':packages,'baseline_git_commit':binding['baseline_git_commit'],
        'R03_tooling_git_status':'local new files/edits; exact source hashes provided, no fictional R03 release commit',
        'R03_scripts':[artifact(p) for p in sorted((PROJECT/'scripts').glob('r03_*.py'))],
        'R02_late_binding':artifact(OUT/'r02_final_source_and_receipt_binding.json'),'build_visual':artifact(OUT/'final_build_visual_receipt.json'),
        'generated_inputs':artifact(OUT/'generated_numeric_identity_r03.json'),'claim_to_raw':artifact(OUT/'claim_to_raw_final.csv'),
        'public_access':artifact(OUT/'public_access_final.json'),'diagnostic':artifact(OUT/'table_protocol_verification.json'),
        'all_author_approval':False,'PUBLICATION_EXECUTED':False,'JOURNAL_SUBMISSION_EXECUTED':False}
    write(OUT/'final_candidate_identities.json',identities)
    (OUT/'final_pdf_checksums.sha256').write_text(''.join(f"{a['sha256']}  {a['path']}\n" for a in identities['PDFs']))
    manifest_inputs=[ROOT/a['path'] for a in source_records]+[ROOT/r['PDF']['path'] for r in audit['builds']]+[ROOT/a['path'] for a in packages]
    manifest_inputs += [OUT/name for name in ('source_set_manifest.json','paper_build_audit.json','final_build_visual_receipt.json','generated_numeric_identity_r03.json','private_package_identities_r03.json','claim_to_raw_final.csv','public_access_final.json','table_protocol_verification.json','final_candidate_identities.json')]
    (OUT/'final_source_pdf_manifest.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(ROOT)}\n' for p in manifest_inputs))
    statuses={'R03-00':'VERIFIED','R03-01':'OPEN','R03-02':'VERIFIED','R03-03':'VERIFIED','R03-04':'OPEN','R03-05':'OPEN','R03-06':'VERIFIED'}
    reasons={'R03-00':'late R02 binding plus actual new candidate source/PDF/render identities',
        'R03-01':'user deferred final AI/CRediT; two-author exact-version approval not provided',
        'R03-02':'new20row CPU diagnostic; 90 matching rows checked; caption-only numeric preservation',
        'R03-03':'short UTC period/negative results/related-work boundaries preserved; no new scientific claim',
        'R03-04':'technical8asset full download/payload verification complete; author rights and final portal checks OPEN',
        'R03-05':'actual candidate builds/41page agent visual inspection/ZIP smoke builds complete; final approved artifacts unavailable',
        'R03-06':'honest per-gate HOLD and publication/submission false; closure of all gates is not claimed'}
    gate=gates();write(OUT/'GATE_P_J_R03.json',gate)
    closure={'run_id':RUN,'recorded_utc':now(),'status':'PARTIAL_CLOSEOUT_AUTHOR_AND_PORTAL_OPEN','tasks':{k:{'status':v,'reason':reasons[k]} for k,v in statuses.items()},
        'accepted_limitations':['retrospective unknown edge/label times','27–29February cohort and shared test seeds','single long run per policy and inherited-label diagnostics','simple-control F1 stronger','finite engine/local-file failure scope','Platt mirror access not guaranteed'],
        'FAILED':[],'gate':gate,'preserved_prior_files_rechecked':unchanged(),'new_fits':0,'new_GPU_replays':0,
        'previous6065_policy_and_MC40_verification_reused':True,'actual_new_stdlib_primary_rows':280,'claim_to_raw_rows':len(rows),
        'source_set':source_set,'candidate_identity':artifact(OUT/'final_candidate_identities.json')}
    write(OUT/'closure_r03.json',closure)
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=['item','status','scope']);w.writeheader()
    for k,v in statuses.items():w.writerow({'item':k,'status':v,'scope':reasons[k]})
    for name in closure['accepted_limitations']:w.writerow({'item':name,'status':'ACCEPTED_LIMITATION','scope':'not repaired by reusing original observations; claims restricted'})
    (OUT/'claim_ledger_r03.csv').write_text(buf.getvalue())
    write(OUT/'finalize_execution.json',{'run_id':RUN,'started_utc':start,'finished_utc':now(),'actual_exit_code':0,'script':artifact(__file__),'source_set':source_set})
    print('R03 CANDIDATE PACKAGES/SMOKE BUILDS/IDENTITIES PASS; GateP/J HOLD',flush=True)


if __name__=='__main__':finalize()
