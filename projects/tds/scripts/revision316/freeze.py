"""Seal R315 artifacts before publication-only revision; never train."""
import json
import subprocess
import zipfile
from paths import ROOT,PROJECT,RESULTS,REPORTS,PRIVATE
from common import sha,write


def main():
    target=RESULTS/'input_manifest.json'
    if target.exists():
        manifest=json.loads(target.read_text())
        for e in manifest['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        legacy=PROJECT/'private/315_before_venue_split/submission_315/evidence/TDS_315_numeric_candidate.zip'
        assert sha(legacy)==manifest['reviewed_candidate_sha256']
        print('Existing 316 input seal verified');return
    files=[]
    roots=[PROJECT/'results/main_final_v2',PROJECT/'results/314_applsci',
           PROJECT/'results/315_applsci',PROJECT/'paper/common/315',
           PROJECT/'paper/applsci/submission_315',PROJECT/'paper/preprints/submission_315',
           PROJECT/'reports/315',PROJECT/'publication/315']
    for base in roots:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or 'candidate' in p.relative_to(base).parts:continue
            if p.suffix in {'.aux','.bbl','.blg','.log','.out','.fls','.fdb_latexmk','.pyc'}:continue
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    original=PROJECT/'private/315_before_venue_split/submission_315'
    expected={'applsci_article.pdf':'ef507b23b2b8d10703a204e9622ec5edb3b245b2ea5df249ebc6526fe210a835',
              'supplement/supplementary.pdf':'f7b7275f7926856146e585e37cd0e018e3d5045f2542977d11f26ea6994d77de'}
    for rel,h in expected.items():
        assert sha(original/rel)==h
        files.append(dict(path=str((original/rel).relative_to(ROOT)),bytes=(original/rel).stat().st_size,sha256=h))
    latest=PROJECT/'publication/315/TDS_315_numeric_candidate.zip'
    legacy=PROJECT/'private/315_before_venue_split/submission_315/evidence/TDS_315_numeric_candidate.zip'
    assert sha(legacy)=='5ac20f2c0aafe3725faa428073b45065816c7549047f158699b20bef76b116db'
    for p in [latest,legacy]:
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
    write(target,dict(date='2026-10-29',files=files,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_reviewed_315_pdf_hashes=expected,reviewed_candidate_sha256=sha(legacy),
        current_relocated_candidate_sha256=sha(latest),
        candidate_difference='Relocation/portability documentation and sources changed after reviewed snapshot; score identity checked separately',
        new_training=False))
    print(f'Sealed {len(files)} current inputs and original reviewed-315 identities')


if __name__=='__main__':main()
