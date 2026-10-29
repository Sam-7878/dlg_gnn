"""Read-only seal of original manuscript, result and source identities."""
import json
import subprocess
from common import ROOT,PROJECT,OUT,sha,write


def main():
    path=OUT/'evidence_freeze_manifest.json'
    if path.exists():
        for e in json.loads(path.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        print('317 input seal verified');return
    roots=['paper/submission_316','reports/316','publication/316','results/316_dual_submission',
           'results/main_final_v2','results/314_applsci','scripts/revision314']
    files=[]
    for rel in roots:
        base=PROJECT/rel
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or 'candidate' in p.relative_to(base).parts:continue
            if p.suffix in {'.aux','.bbl','.blg','.log','.out','.fls','.fdb_latexmk','.pyc'}:continue
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(path,dict(work_order='TDS-WO-317-VALIDITY-A16-SUBMISSION',files=files,
                   git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   prior_results_read_only=True))
    print('Sealed',len(files),'original artifacts')


if __name__=='__main__':main()
