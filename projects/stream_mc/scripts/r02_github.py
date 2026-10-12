"""Scoped GitHub access; reuse Git's credential helper without logging secrets."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import hashlib
import tempfile
import shutil
import sysconfig
import zipfile
from pathlib import Path
import urllib.error
import urllib.request
import urllib.parse
from r02_common import ROOT,PROJECT,R02,read,write,sha256

REPO='Sam-7878/dlg_gnn'
API='https://api.github.com/repos/'+REPO
TAG='stream-mc-r02-20261012'
ALLOWED={'r02_scientific_source.zip','r02_verification_evidence.zip','r01_release.json','r01_numeric_evidence.zip.part001','r01_numeric_evidence.zip.part002','r01_numeric_evidence.zip.part003','r02_release_checksums.sha256','RELEASE_R02_README.md'}


def credential():
    result=subprocess.run(['git','credential','fill'],cwd=ROOT,input='protocol=https\nhost=github.com\npath='+REPO+'.git\n\n',text=True,capture_output=True,
        env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'Never'},timeout=20)
    values=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    token=values.get('password')
    if result.returncode or not token:raise RuntimeError('No usable Git credential for the specified GitHub repository; no secret output retained')
    return token


def api(token,path='',method='GET',payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    request=urllib.request.Request(API+path,data=data,method=method,headers={
        'Accept':'application/vnd.github+json','Authorization':'Bearer '+token,'X-GitHub-Api-Version':'2026-03-10','User-Agent':'DLG-SelectiveStream-R02-audit','Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)


def publish(token):
    manifest=read(PROJECT/'evidence/release_manifest_r02.json')
    assert manifest['repository']==REPO and manifest['tag']==TAG
    assert set(Path(a['path']).name for a in manifest['assets'])==ALLOWED
    assert api(token).get('permissions',{}).get('push'),'No verified repository push permission'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    assert remote==commit,'Source must already be normally pushed to origin/main'
    for a in manifest['assets']:
        p=ROOT/a['path'];assert p.is_file() and sha256(p)==a['sha256'] and p.stat().st_size==a['bytes']
    try:release=api(token,'/releases/tags/'+TAG)
    except urllib.error.HTTPError as exc:
        if exc.code!=404:raise
        release=api(token,'/releases','POST',{'tag_name':TAG,'target_commitish':commit,'name':'DLG-SelectiveStream R02 — scientific code and evidence','draft':True,'prerelease':True,'make_latest':'false',
            'body':'Scientific source and pseudonymous numeric evidence for DLG-SelectiveStream R02. Original scientific run: selectivestream_r01_20261011; all 25 models/five seeds, frozen training/runtime-v1/offline-v2 identities. R02 reaggregation/verification and a distinct cheap state-boundary probe; no replacement of original training or 100k timings. See RELEASE_R02_README.md and exact checksums. Dataset-derived inputs retain provider CC BY-NC-SA 4.0 attribution. Private manuscripts/original mappings/full original-ID handoff are excluded. This is a scientific-code/evidence prerelease, not paper publication, peer review, author submission approval or production certification. Final AI wording and exact-version all-author paper approval remain OPEN.'})
    # Idempotent only for this exact release/source/assets; never overwrite.
    assert release['tag_name']==TAG and release['prerelease'] and release['target_commitish']==commit,'Existing release identity differs'
    existing={a['name']:a for a in release['assets']}
    assert set(existing)<=ALLOWED,'Unrecognized existing release assets; no overwrite'
    uploaded=[]
    for a in manifest['assets']:
        p=ROOT/a['path'];name=p.name
        if name in existing:
            item=existing[name];assert item['size']==a['bytes'] and item.get('digest')=='sha256:'+a['sha256'],'Existing asset identity differs; no overwrite'
        else:
            assert release['draft'],'Do not add missing assets to an already published release'
            url=release['upload_url'].split('{',1)[0]+'?name='+urllib.parse.quote(name)
            assert url.startswith('https://uploads.github.com/repos/'+REPO+'/releases/'),'Unexpected upload host/target'
            request=urllib.request.Request(url,data=p.read_bytes(),method='POST',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2026-03-10','Content-Type':'application/octet-stream','User-Agent':'DLG-SelectiveStream-R02-audit'})
            with urllib.request.urlopen(request,timeout=180) as response:item=json.load(response)
            assert item['size']==a['bytes'] and item.get('digest')=='sha256:'+a['sha256'],'Uploaded asset identity mismatch'
        uploaded.append({**a,'asset_id':item['id'],'download_url':item['browser_download_url'],'api_digest':item.get('digest')})
        print('GITHUB ASSET VERIFIED',name,a['bytes'],flush=True)
    if release['draft']:release=api(token,'/releases/'+str(release['id']),'PATCH',{'draft':False,'prerelease':True,'make_latest':'false'})
    manifest.update({'release_status':'published; independent unauthenticated downloads pending','release_url':release['html_url'],'release_id':release['id'],'target_source_commit':commit,'prerelease':True,'assets':uploaded})
    write(PROJECT/'evidence/release_manifest_r02.json',manifest)
    print('GITHUB RELEASE PUBLISHED',release['html_url'],flush=True)


def access():
    # Deliberately no credential() and no Authorization for public access.
    manifest=read(PROJECT/'evidence/release_manifest_r02.json');assert manifest.get('release_url')
    receipts=[];commands=[]
    with tempfile.TemporaryDirectory(prefix='stream_mc_r02_public_download_') as directory:
        base=Path(directory);download=base/'download';download.mkdir()
        for a in manifest['assets']:
            name=Path(a['path']).name;assert name in ALLOWED
            url=a['download_url'];assert url.startswith('https://github.com/'+REPO+'/releases/download/'+TAG+'/')
            request=urllib.request.Request(url,headers={'User-Agent':'DLG-SelectiveStream-public-access-check'})
            digest=hashlib.sha256();size=0
            with urllib.request.urlopen(request,timeout=30) as response,(download/name).open('wb') as handle:
                status=response.status
                for block in iter(lambda:response.read(2**20),b''):digest.update(block);handle.write(block);size+=len(block)
            assert size==a['bytes'] and digest.hexdigest()==a['sha256'],'Public download drift '+name
            receipts.append({'name':name,'url':url,'http_status':status,'bytes':size,'sha256':digest.hexdigest(),'authentication_sent':False})
            print('UNAUTHENTICATED DOWNLOAD VERIFIED',name,size,flush=True)
        stage=base/'source';stage.mkdir()
        with zipfile.ZipFile(download/'r02_scientific_source.zip') as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in z.namelist())
            assert all(not n.endswith(('.tex','.bib','.pdf')) and '/paper/' not in n for n in z.namelist())
            z.extractall(stage)
        evidence=stage/'projects/stream_mc/evidence';evidence.mkdir(parents=True,exist_ok=True)
        for name in ['r01_release.json']+[f'r01_numeric_evidence.zip.part{i:03d}' for i in range(1,4)]:shutil.copy2(download/name,evidence/name)
        with zipfile.ZipFile(download/'r02_verification_evidence.zip') as z:
            identity=json.loads(z.read('release_manifest.json'));assert set(identity['files'])==set(z.namelist())-{'release_manifest.json'}
            for name,i in identity['files'].items():data=z.read(name);assert len(data)==i['bytes'] and hashlib.sha256(data).hexdigest()==i['sha256']
            for name,value in identity['r02_source_hashes'].items():assert sha256(stage/name)==value
        def run(label,command,env):
            result=subprocess.run(command,cwd=stage,env=env,capture_output=True,text=True)
            path=R02/'verification'/('external_'+label+'.log');path.write_text(result.stdout+result.stderr)
            commands.append({'label':label,'command':[str(x) for x in command],'actual_exit_code':result.returncode,'log_sha256':sha256(path),'scope':'new temporary source/data path from real public downloads; same operator/host, not peer review'})
            assert result.returncode==0,'External-path command failed: '+label
        environment={**os.environ,'PYTHONPATH':str(stage/'src'),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        run('fresh_venv',[sys.executable,'-m','venv',str(base/'stdlib_venv')],environment)
        run('stdlib_public',[base/'stdlib_venv/bin/python','projects/stream_mc/scripts/r02_public.py'],environment)
        # Scientific dependencies reused explicitly; no network installs/venv mutations.
        # Source and all scientific input bytes are the fresh public downloads.
        run('full_policy',[sys.executable,'projects/stream_mc/scripts/r02_public.py','--full-policy'],environment)
        output=read(stage/'projects/stream_mc/results/public_recomputed_r02/public_verification.json')
        assert output['full_policy_rows']==6065 and output['MC_rows']==40
    report={'status':'VERIFIED','release_url':manifest['release_url'],'target_source_commit':manifest['target_source_commit'],'downloads':receipts,'commands':commands,'public_numeric':output,
        'scope':'actual unauthenticated hosted-byte downloads, fresh stdlib venv and isolated source/input full-policy execution using explicitly reused installed scientific dependencies; same implementer/host, not an external human scientific audit',
        'no_author_absolute_input_paths':True,'dependencies_reinstalled_in_fresh_scientific_venv':False,'paper_publication_or_submission':False}
    write(PROJECT/'evidence/access_verification_r02.json',report)
    manifest['release_status']='published; all assets publicly downloaded and verified';write(PROJECT/'evidence/release_manifest_r02.json',manifest)
    print('R02 PUBLIC ACCESS AND FULL RAW NUMERIC PASS',manifest['release_url'],flush=True)


def local():
    """Exercise the exact source bundle before any external publication."""
    with tempfile.TemporaryDirectory(prefix='stream_mc_r02_local_source_') as directory:
        stage=Path(directory)/'source';stage.mkdir()
        with zipfile.ZipFile(PROJECT/'handoff/r02/r02_scientific_source.zip') as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(stage)
        dest=stage/'projects/stream_mc/evidence';dest.mkdir(parents=True,exist_ok=True)
        for name in ['r01_release.json']+[f'r01_numeric_evidence.zip.part{i:03d}' for i in range(1,4)]:shutil.copy2(PROJECT/'evidence'/name,dest/name)
        env={**os.environ,'PYTHONPATH':str(stage/'src'),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        command=[sys.executable,'projects/stream_mc/scripts/r02_public.py','--full-policy'];result=subprocess.run(command,cwd=stage,env=env,capture_output=True,text=True)
        log=R02/'verification/local_isolated_source.log';log.write_text(result.stdout+result.stderr)
        write(R02/'verification/local_isolated_source.json',{'actual_exit_code':result.returncode,'source_bundle_sha256':sha256(PROJECT/'handoff/r02/r02_scientific_source.zip'),'log_sha256':sha256(log),
            'scope':'exact local source bundle and isolated input path, existing scientific dependencies, before upload; no external access asserted'})
        assert result.returncode==0,'Local isolated source failed; inspect recorded log before publishing'
    print('LOCAL ISOLATED SOURCE/PUBLIC RAW COVERAGE PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['inspect','local','publish','access'],default='inspect');args=parser.parse_args()
    try:
        if args.mode=='local':local();return
        if args.mode=='access':access();return
        if args.mode=='publish':publish(credential());return
        token=credential();repo=api(token);releases=api(token,'/releases?per_page=100')
        report={'repository':REPO,'repository_url':repo['html_url'],'credential_available':True,'push_permission':bool(repo.get('permissions',{}).get('push')),
            'existing_releases':[{'tag_name':r['tag_name'],'html_url':r['html_url'],'draft':r['draft'],'prerelease':r['prerelease']} for r in releases],
            'scope':'read-only credential/permission/release inspection; no token retained or external write'}
        write(R02/'verification/github_access_preflight.json',report);print(json.dumps(report,indent=2),flush=True)
    except (RuntimeError,subprocess.TimeoutExpired,urllib.error.HTTPError,urllib.error.URLError) as exc:
        report={'repository':REPO,'verified_write_access':False,'error_type':type(exc).__name__,'http_status':getattr(exc,'code',None),
            'scope':'preflight failed; no credential/exception-body output, no external write'}
        write(R02/'verification'/('github_'+args.mode+'_failure.json'),report);print(json.dumps(report,indent=2),flush=True);raise SystemExit(1)


if __name__=='__main__':main()
