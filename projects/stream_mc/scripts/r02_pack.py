"""Curated public R02 numeric sidecar, not original-ID or manuscript publishing."""
from __future__ import annotations
import json
import zipfile
from r02_common import ROOT,PROJECT,R01,R02,VERIFY,RUN,read,write,sha256,artifact,original_unchanged


def main():
    original_unchanged()
    numeric=read(VERIFY/'raw_numeric_validation.json');assert numeric['N_raw_metric_rows_recomputed']==6065
    public=read(VERIFY/'public_local_numeric/public_verification.json');assert public['full_policy_rows']==6065 and public['MC_rows']==40
    excluded={'handoff_baseline.json','paper_build_audit.json','private_package_identities_r02.json','github_access_preflight.json'}
    paths=[p for p in VERIFY.rglob('*') if p.is_file() and p.name not in excluded and 'negative_fixtures' not in p.parts]
    paths+=[VERIFY/'negative_fixtures/negative_fixtures.json']
    paths+=list((R02/'new_measurements/state_probe').glob('*.csv'))+list((R02/'new_measurements/state_probe').glob('*.json'))
    paths+=[p for p in (R02/'working/analysis').glob('*.csv') if not p.name.endswith('_paired_predictions.csv')]
    paths+=list((PROJECT/'docs/r02').glob('*.md'))
    payload={}
    for path in sorted(set(paths)):
        name=str(path.relative_to(R02)) if path.is_relative_to(R02) else 'docs/'+path.name
        data=path.read_bytes()
        assert path.suffix in ('.json','.csv','.jsonl','.xml','.log','.md'),path
        # No private manuscript contents or original account identifiers.
        import re
        assert not re.search(rb'0x[0-9a-fA-F]{40}\b',data),(path,'original account identifier')
        assert b'\\documentclass' not in data and b'\\begin{document}' not in data,(path,'manuscript source')
        payload[name]=data
    source_manifest={str(p.relative_to(ROOT)):sha256(p) for p in (PROJECT/'scripts').glob('r02_*.py')}
    manifest={'verification_id':RUN,'source_run_id':'selectivestream_r01_20261011','new_measurement_ids':['selectivestream_r02_20261012_state_probe'],
        'original_archive_sha256':read(PROJECT/'evidence/r01_release.json')['archive_sha256'],'r02_source_hashes':source_manifest,
        'scope':'public numeric verification/reaggregation sidecar; original pseudonymous raw bytes in unchanged R01 archive; private paper/addresses/checkpoints excluded',
        'files':{name:{'sha256':__import__('hashlib').sha256(data).hexdigest(),'bytes':len(data)} for name,data in payload.items()}}
    target=PROJECT/'evidence/r02_verification_evidence.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in payload.items():z.writestr(name,data)
        z.writestr('release_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    write(R02/'public_boundary_r02.json',{'public_archive':artifact(target),'payload_members':len(payload),'address_regex_hits':0,'manuscript_source_hits':0,'private_original_handoff_excluded':True,'original_R01_unmodified':True,
        'scope':'explicit curated selection/content checks; pseudonyms still linkable, not proof of anonymization'})
    files=[PROJECT/'handoff/r02/r02_scientific_source.zip',target,PROJECT/'evidence/r01_release.json']
    files+=sorted((PROJECT/'evidence').glob('r01_numeric_evidence.zip.part*'))
    lines=[sha256(p)+'  '+p.name for p in files]
    (PROJECT/'evidence/r02_release_checksums.sha256').write_text('\n'.join(lines)+'\n')
    files += [PROJECT/'evidence/r02_release_checksums.sha256',PROJECT/'evidence/RELEASE_R02_README.md']
    write(PROJECT/'evidence/release_manifest_r02.json',{'revision':'DLG-SelectiveStream-R02','tag':'stream-mc-r02-20261012','repository':'Sam-7878/dlg_gnn','release_status':'local exact assets; hosting not yet verified',
        'original_run_id':'selectivestream_r01_20261011','assets':[artifact(p) for p in files],'source_hashes':source_manifest,
        'public_boundary':read(R02/'public_boundary_r02.json'),'paper_publication_or_submission':False,'all_author_paper_approval':False})
    print('R02 PUBLIC PACK PASS',len(payload),target.stat().st_size,flush=True)


if __name__=='__main__':main()
