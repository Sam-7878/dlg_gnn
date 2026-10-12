"""Exact original handoff versus pseudonymous public derivative, explicitly separate."""
from __future__ import annotations
import json
import argparse
import subprocess
import zipfile
from pathlib import Path
from r02_common import ROOT,PROJECT,R01,R02,VERIFY,RUN,read,write,sha256,artifact,csv_write,environment
from r01_public import verified_archive


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    VERIFY.mkdir(parents=True,exist_ok=True)
    env=environment();write(VERIFY/'verification_environment.json',env)
    manifest=read(R01/'release_manifest.json');coverage=[]
    for name,expected in manifest['files'].items():
        path=R01/name;present=path.is_file();actual=sha256(path) if present else None
        coverage.append({'member':name,'present':present,'expected_sha256':expected,'actual_sha256':actual,'hash_matches':present and actual==expected,
            'external_access':'private exact bundle locally available; public pseudonymous derivative delivered separately','restriction':'author-private prior manuscript' if name=='input_snapshot.zip' else 'original identifiers retained privately; public aliases and disclosure boundary recorded separately'})
    csv_write(VERIFY/'handoff_missing_or_restricted.csv',coverage)
    missing=[r['member'] for r in coverage if not r['present']];bad=[r['member'] for r in coverage if r['present'] and not r['hash_matches']]
    if missing or bad:raise ValueError('original handoff missing/hash mismatch: '+repr((missing,bad)))
    assert len(coverage)==939,'the original manifest must not be reduced'
    with verified_archive()[0] as bundle:
        public_identity=artifact(PROJECT/'evidence/r01_numeric_evidence.zip')
        public_members=len(bundle.namelist())
    release=read(PROJECT/'evidence/r01_release.json')
    parts=[]
    for item in release['parts']:
        path=PROJECT/'evidence'/item['name'];assert sha256(path)==item['sha256'] and path.stat().st_size==item['bytes'];parts.append(artifact(path))
    assert sum(p['bytes'] for p in parts)==release['archive_bytes']
    handoff=PROJECT/'handoff/r02';handoff.mkdir(parents=True,exist_ok=True)
    # Contains original identifiers and an earlier author-private manuscript;
    # deliberately ignored by Git and never represented as a public archive.
    private=handoff/'original_r01_full_private.zip'
    if not args.source_only:
        with zipfile.ZipFile(private,'w',zipfile.ZIP_DEFLATED) as bundle:
            for name in manifest['files']:bundle.write(R01/name,name)
            for name in ('release_manifest.json','checksums.sha256'):bundle.write(R01/name,name)
    with zipfile.ZipFile(private) as bundle:
        for row in coverage:
            import hashlib
            assert hashlib.sha256(bundle.read(row['member'])).hexdigest()==row['expected_sha256']
    with verified_archive()[0] as public:source_manifest=json.loads(public.read('release_manifest.json'))
    paths=[ROOT/name for name in source_manifest['current_source_hashes']]
    paths += list((PROJECT/'scripts').glob('r02_*.py'))
    paths += [PROJECT/'reproduce.sh',ROOT/'scripts/reproduce_project.py']
    paths += list((PROJECT/'configs').glob('*'))+[PROJECT/'requirements-parquet.txt',PROJECT/'REPRODUCE.md',PROJECT/'README.md']
    paths += [PROJECT/'evidence'/name for name in ('closure_r01.json','claim_ledger_r01.csv','archive_boundary_r01.json','cleanroom_r01.json','r01_release.json')]
    paths += [R01/name for name in ('environment.json','environment.lock','training_source_manifest.json','execution_plan.json')]
    paths += [R01/name for name in ('training_sources.zip','runtime_sources_v1.tar','runtime_sources_v2.tar')]
    paths += list((PROJECT/'docs/r02').glob('*'))+[PROJECT/'REPRODUCE_R02.md']
    paths=sorted(set(p for p in paths if p.is_file()))
    source=handoff/'r02_scientific_source.zip'
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:bundle.write(path,str(path.relative_to(ROOT)))
    identity={'verification_id':RUN,'baseline_commit':env['git_commit'],'source_files':[artifact(p) for p in paths],
        'scientific_source_bundle':artifact(source),'frozen_executions':[artifact(R01/name) for name in ('training_sources.zip','runtime_sources_v1.tar','runtime_sources_v2.tar')],
        'source_status':'exact checkout bytes; R02 verifier is an added sidecar, not relabeled R01 executed training/timing source'}
    write(R02/'source_identity_r02.json',identity)
    report={'verification_id':RUN,'original_manifest':artifact(R01/'release_manifest.json'),'N_original_manifest_targets':len(coverage),
        'N_present_and_hash_matching':len(coverage),'N_missing':0,'N_hash_mismatch':0,'original_private_handoff':artifact(private),
        'original_private_member_count':941,'public_derivative':public_identity,'public_derivative_member_count':public_members,'public_parts':parts,
        'distinct_manifests':'original 939 targets and public derivative ZIP are different identities, not substituted',
        'review_input_zips_available_here':False,'reviewer_audit_zip_available_here':False,
        'incomplete_transfer_cause':'review documents establish that the canonical ZIP/parts were not attached and only35/939 original targets were supplied; the exact earlier compression/selection action is not locally evidenced and is not inferred',
        'external_delivery_or_hosting_performed':False}
    write(R02/'handoff_manifest_verification.json',report)
    if not args.source_only:
        write(VERIFY/'handoff_baseline.json',{'files':manifest['files'],'private_paper_files':{str(p.relative_to(ROOT)):sha256(p) for p in (PROJECT/'paper/current/r01').rglob('*') if p.is_file()}})
    print('R02 ORIGINAL HANDOFF PASS',len(coverage),public_members,flush=True)


if __name__=='__main__':main()
