"""Read-only platform, Git, GPU and primary-reference provenance."""
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
import zipfile
import io
from paths import ROOT,PROJECT,RESULTS,REPORTS,PRIVATE
from common import sha,write


def fetch(url):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'TDS-publication-preflight/316'})
        with urllib.request.urlopen(req,timeout=20) as f:
            data=f.read()
            return dict(url=url,status=f.status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),data
    except Exception as e:
        return dict(url=url,status=getattr(e,'code',None),error=str(e)),None


def main():
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,compute_cap',
                        '--format=csv,noheader'],capture_output=True,text=True)
    write(REPORTS/'gpu_capacity.json',dict(observed_on='2026-10-29',timezone='Asia/Seoul',
        devices=gpu.stdout.splitlines(),returncode=gpu.returncode,
        memory_is_per_device_not_a_single_pooled_32GB_heap=True,
        training_performed=False,scope='316 explicitly prohibits new training; RTX3090 capacity reserved for a later approved scientific amendment'))
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.run(['git','ls-remote','https://github.com/Sam-7878/dlg_gnn.git','HEAD','refs/heads/main','refs/tags/tds-*'],
                          cwd=ROOT,capture_output=True,text=True)
    changes=subprocess.check_output(['git','diff','--name-status','--','projects/tds','projects/results/tds','docs/architecture/tds'],cwd=ROOT,text=True)
    write(REPORTS/'git_remote_audit.json',dict(observed_on='2026-10-29',local_head=local,
        remote_query_returncode=remote.returncode,remote_refs=remote.stdout.splitlines(),
        tracked_worktree_changed_paths=changes.splitlines(),
        worktree_diff_name_status_sha256=hashlib.sha256(changes.encode()).hexdigest(),
        committed_head_is_not_a_tds_release=True,public_release_verified=False,pushed=False))
    requests=[]
    for url in ['https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt',
                'https://www.mdpi.com/journal/applsci/instructions',
                'https://mdpi-res.com/data/MDPI_template.zip']:
        record,data=fetch(url);requests.append(record)
        if data and url.endswith('.zip'):
            (PRIVATE/'official_MDPI_template.zip').write_bytes(data)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                record['class_hashes']={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('mdpi.cls')}
    write(REPORTS/'venue_requests.json',dict(observed_on='2026-10-29',requests=requests,
         reported_deadline='2026-12-31',deadline_source='316 work order/background reports official check on 2026-10-29',
         direct_deadline_verification=False,guest_editor=None,portal_special_issue_selection_verified=False,
         screenshot_available=False))
    url='https://api.crossref.org/works/10.1109%2FTKDE.2025.3581578'
    record,data=fetch(url)
    if data:
        value=json.loads(data)['message']
        record['publisher_registered_metadata']={k:value.get(k) for k in ['title','author','container-title','volume','issue','page','published','DOI']}
    write(REPORTS/'qiao_reference_verification.json',record)
    print('Read-only GPU/Git/policy/reference inspection recorded; no training/publication')


if __name__=='__main__':main()
