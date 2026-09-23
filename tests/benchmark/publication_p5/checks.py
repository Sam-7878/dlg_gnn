from __future__ import annotations
import hashlib, json, os, re, shlex, subprocess, sys, tempfile, urllib.request, zipfile
from pathlib import Path
import pytest, yaml
ROOT = Path(__file__).resolve().parents[3]
META = json.loads((ROOT/'publication/benchmark/publication_metadata.json').read_text())
COMMANDS = yaml.safe_load((ROOT/'docs/reproduction_commands.yaml').read_text())
ZIP = ROOT/'outputs/benchmark/manuscript_m5/release'/META['release_asset']
API = 'https://api.github.com/repos/Sam-7878/dlg_gnn'

def text(p): return (ROOT/p).read_text(encoding='utf-8')
def api(path):
    try:
        with urllib.request.urlopen(urllib.request.Request(API+path, headers={'User-Agent':'DLG-P5-audit'}), timeout=20) as r: return json.load(r)
    except Exception as e: pytest.skip(f'external GitHub check unavailable: {e}')
def check_lanl():
    m=json.loads(text('artifacts/manifests/lanl_canonical_manifest.json')); r=text('README.md')
    for value in (m['num_nodes'],m['num_edges'],m['num_features'],m['positive_count']): assert f'{value:,}' in r

def check_clone():
    if os.getenv('P5_EXTERNAL')!='1': pytest.skip('set P5_EXTERNAL=1 for public-clone gate')
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(['git','clone','--depth','1',META['repository_url'],d+'/dlg_gnn'],check=True)
        subprocess.run([sys.executable,'scripts/reproduce_frozen_artifacts.py'],cwd=d+'/dlg_gnn',check=True)
def check_command(key):
    c=COMMANDS[key]; assert c in text('README.md'); assert c in text('INSTALL.md')
def check_paths():
    for c in COMMANDS.values():
        for token in shlex.split(str(c)):
            if token.endswith(('.py','.yaml','.yml')): assert (ROOT/token).exists(), token
def check_tree():
    r=text('README.md'); assert 'Git repository structure' not in r or 'artifacts/' in r
    assert 'Raw third-party datasets' in r and 'local `outputs/` are not' in r
def manuscript_title():
    t=text('publication/benchmark/preprints/DLG-Benchmark-Preprint.tex')
    assert META['paper_title'] in t
    return META['paper_title']
def check_notes(): assert manuscript_title() in text('publication/benchmark/release_notes.md')
def env(): return json.loads(text('provenance/environment_manifest.json'))
def check_walkthrough():
    w=text('docs/work_reports/benchmark/221_p4/walkthrough.md'); e=env()
    for v in (e['python'],e['pytorch'],e['torch-geometric'],e['pygod']): assert v in w
def check_env_report():
    r=text('publication/benchmark/reports/04_p4_environment_versioning_audit.md'); e=env()
    for v in (e['python'],e['pytorch'],e['torch-geometric'],e['pygod']): assert v in r
def check_manuscript_env():
    for p in ('publication/benchmark/preprints/DLG-Benchmark-Preprint.tex','publication/benchmark/mdpi/DLG-Benchmark.tex'):
        s=text(p); e=env(); assert e['python'] in s and e['pytorch'].split('+')[0] in s and e['torch-geometric'] in s and e['pygod'] in s
def check_preprint_report():
    with zipfile.ZipFile(ROOT/'publication/benchmark/preprints/DLG_Benchmark_Preprints_Submission.zip') as z: names=set(z.namelist())
    r=text('publication/benchmark/reports/06_p4_preprints_final_freeze_audit.md')
    assert 'Not embedded in the source ZIP' in r and 'DLG-Benchmark-Preprint.pdf' not in names
def check_citation_date():
    c=yaml.safe_load(text('CITATION.cff')); rel=api('/releases/tags/v1.0.0-preprint')
    if rel: assert c.get('date-released') == rel['published_at'][:10]
    else: assert 'date-released' not in c
def check_status(): assert 'Prepared for Preprints.org deposit' in text('README.md')
def check_zip_readme():
    with zipfile.ZipFile(ZIP) as z: assert z.read('dlg_gnn/README.md').decode()==text('README.md')
def check_dryrun():
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(ZIP) as z:z.extractall(d)
        subprocess.run([sys.executable,'experiments/benchmark/run_sci_round5_final.py','--dry-run'],cwd=Path(d)/'dlg_gnn',check=True)
def check_remote_tag():
    tag=api('/git/ref/tags/v1.0.0-preprint'); head=api('/commits/main')
    obj=tag['object'];
    if obj['type']=='tag': obj=api('/git/tags/'+obj['sha'])['object']
    assert obj['sha']==head['sha']
def check_release_asset():
    rel=api('/releases/tags/v1.0.0-preprint'); assets={a['name']:a for a in rel['assets']}; assert META['release_asset'] in assets
    req=urllib.request.Request(assets[META['release_asset']]['browser_download_url'],method='HEAD',headers={'User-Agent':'DLG-P5-audit'}); assert urllib.request.urlopen(req,timeout=30).status in (200,302)
