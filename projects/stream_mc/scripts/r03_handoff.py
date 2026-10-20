"""Local handoff ZIP with detached identity; no GitHub or journal writes."""
from __future__ import annotations
import hashlib
import sys
import zipfile
from pathlib import Path
from r03_common import ROOT,PROJECT,R02,OLD,PAPER,OUT,RUN,sources,artifact,read,write,sha256,unchanged,command_record,now
from r03_finalize import check_zip


def main():
    start=now();assert unchanged()==1103
    test=command_record([sys.executable,'-m','unittest','discover','-s','projects/stream_mc/tests','-p','test_r03_closeout.py','-v'],ROOT,OUT/'r03_unit_tests.log')
    write(OUT/'r03_unit_tests.json',{'execution':test,'test_source':artifact(PROJECT/'tests/test_r03_closeout.py')});assert test['actual_exit_code']==0
    count=0
    for line in (OUT/'final_source_pdf_manifest.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha256(ROOT/name)==digest,name;count+=1
    identities=read(OUT/'final_candidate_identities.json')
    for a in identities['R03_scripts']:assert sha256(ROOT/a['path'])==a['sha256'],'Core R03 tool changed after candidate lock'
    for name in ('preprint','journal'):
        record=next(r for r in read(OUT/'private_package_identities_r03.json')['packages'] if Path(r['package']['path']).name==name+'_candidate_package.zip')
        check_zip(ROOT/record['package']['path'],record['member_sha256'])
    closure=read(OUT/'closure_r03.json');assert closure['gate']['Gate_P']['status']==closure['gate']['Gate_J']['status']=='HOLD'
    write(OUT/'final_integrity_recheck.json',{'run_id':RUN,'recorded_utc':now(),'manifest_targets':count,'prior_targets':unchanged(),
        'candidate_source_count':len(sources(PAPER)),'status':'VERIFIED','tests':artifact(OUT/'r03_unit_tests.json'),
        'scope':'byte-bound final local handoff validation; no new model fits, author approval or portal action'})
    archive=OUT/'r03_review_handoff_candidate.zip';detached=OUT/'review_handoff_archive_identity.json'
    manifest=OUT/'handoff_payload_manifest.json'
    assert not archive.exists() and not detached.exists(),'Preserve previous handoff ZIP; do not silently overwrite'
    included={p for p in OUT.rglob('*') if p.is_file() and p not in {archive,detached,manifest} and p.suffix!='.tmp'}
    # Both exact candidate and late R02 inputs/PDF/renders are author-local.
    # Do NOT include full originals, user-added r02.zip or old private bundles.
    included.update(sources(PAPER));included.update(sources(OLD))
    for base,build_name,visual_name in ((PAPER,'build_r03_candidate1','visual_r03_candidate1'),(OLD,'build_r02','visual_r02')):
        included.update(p for p in (base/build_name).rglob('*') if p.is_file())
        included.update(p for p in (base/visual_name).rglob('*') if p.is_file())
    required=('execution_paper_20261012T083831Z.json','paper_20261012T083831Z.log','paper_build_audit.json','visual_inspection_r02.json',
        'private_package_identities_r02.json','generated_numeric_identity.json','claim_to_raw_r02.csv','raw_lineage_binding_identity.json','model_registry_checked.jsonl','source_measurement_matrix.csv')
    included.update(R02/'verification'/name for name in required)
    included.update(PROJECT/'scripts'/name for name in ('r03_audit.py','r03_common.py','r03_paper.py','r03_finalize.py','r03_handoff.py'))
    included.update((PROJECT/'REPRODUCE_R03.md',PROJECT/'configs/submission_target_r03.json',PROJECT/'configs/author_declarations_r02.json',
                     PROJECT/'evidence/release_manifest_r02.json',PROJECT/'evidence/access_verification_r02.json',PROJECT/'docs/r02/reference_verification.md',PROJECT/'tests/test_r03_closeout.py'))
    assert all(p.is_file() for p in included)
    assert not any(p.name in {'r02.zip','original_r01_full_private.zip','author_approval_exact_version.md'} for p in included)
    records=[artifact(p) for p in sorted(included)]
    write(manifest,{'run_id':RUN,'recorded_utc':now(),'private_review_only':True,'members':records,
                    'scope':'R03 candidate + exact late R02 source/PDF/render/receipts, not all939 original private inputs; original939 locally hash-checked, not claimed delivered',
                    'self_inclusion':'this manifest and detached archive receipt are excluded from member list to avoid self-hash cycles'})
    included.add(manifest);expected={str(p.relative_to(ROOT)):sha256(p) for p in included}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(included):z.write(p,str(p.relative_to(ROOT)))
    check_zip(archive,expected)
    write(detached,{'run_id':RUN,'created_utc':now(),'archive':artifact(archive),'members':len(expected),'all_member_hashes_checked':True,
        'payload_manifest':artifact(manifest),'Gate_P':'HOLD','Gate_J':'HOLD','PUBLICATION_EXECUTED':False,'JOURNAL_SUBMISSION_EXECUTED':False,
        'contains_private_paper':True,'do_not_upload_to_public_Git_or_portal':True,'script':artifact(__file__),
        'started_utc':start,'finished_utc':now(),'actual_exit_code':0})
    assert unchanged()==1103
    print('R03 PRIVATE HANDOFF PASS',len(expected),archive.stat().st_size,'bytes; author/portal gates HOLD',flush=True)


if __name__=='__main__':main()
