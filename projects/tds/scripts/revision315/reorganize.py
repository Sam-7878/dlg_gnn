"""One-time, hash-preserving separation of manuscript and numeric evidence."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'projects/tds'
LOG=P/'publication/layout/path_migrations.json'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    if LOG.exists():
        print('Migration already recorded; no moves repeated')
        return
    backup=P/'private/315_before_venue_split'
    paper=P/'paper/applsci/submission_315'
    shutil.copytree(paper,backup/'submission_315',dirs_exist_ok=True)
    shutil.copytree(P/'scripts/revision315',backup/'scripts_revision315',dirs_exist_ok=True)
    shutil.copytree(P/'reports/315',backup/'reports_315',dirs_exist_ok=True)
    moves=[]
    def move(src,dst):
        if not src.exists():return
        if src.is_dir():
            for f in sorted(src.rglob('*')):
                if f.is_file():move(f,dst/f.relative_to(src))
            for d in sorted(src.rglob('*'),reverse=True):
                if d.is_dir() and not any(d.iterdir()):d.rmdir()
            if not any(src.iterdir()):src.rmdir()
            return
        old=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            assert sha(dst)==old,(src,dst)
            src.unlink()
        else:shutil.move(src,dst)
        assert sha(dst)==old
        moves.append(dict(old=str(src.relative_to(ROOT)),new=str(dst.relative_to(ROOT)),sha256=old))
    r314=P/'results/314_applsci'
    old=P/'paper/applsci'
    for name in ['runs','reports']:
        move(old/name,r314/name)
    for f in sorted((old/'data').glob('*.csv')):move(f,r314/'data'/f.name)
    for f in sorted((old/'supplement').iterdir()):
        if f.is_file() and f.suffix in {'.csv','.json','.pt','.parquet','.npy','.npz'}:
            move(f,r314/'supplement'/f.name)
    move(old/'vendor/GoG_LICENSE.txt',r314/'licenses/GoG_LICENSE.txt')
    move(old/'packages',backup/'r314_packages')
    move(ROOT/'projects/results/tds/315_applsci',P/'results/315_applsci')
    for f in sorted((paper/'data').glob('*.csv')):
        move(f,P/'results/315_applsci/paper_inputs/data'/f.name)
    move(paper/'qa',backup/'latest_qa')
    move(paper/'evidence',backup/'legacy_evidence')
    authors={'author_approval_record.md','author_closure_status.json','author_section_provenance.json',
             'credit_confirmed.json','funding_evidence_register.md','declarations_confirmed.md'}
    for f in sorted(paper.iterdir()):
        if not f.is_file():continue
        if f.name in authors:move(f,P/'submission/author_records/315'/f.name)
        elif f.suffix=='.zip':move(f,backup/'legacy_packages'/f.name)
        elif f.suffix in {'.json','.csv'}:move(f,backup/'legacy_reports'/f.name)
        elif f.suffix=='.md' and f.name!='README_source.md':
            move(f,P/'submission/applsci/315'/f.name)
    for p in [ROOT/'projects/results/tds',ROOT/'projects/results']:
        if p.exists() and not any(p.iterdir()):p.rmdir()
    for f in sorted((P/'paper/current/data').glob('*.csv')):
        move(f,P/'results/historical_reviewed_draft/data'/f.name)
    LOG.parent.mkdir(parents=True,exist_ok=True)
    LOG.write_text(json.dumps(dict(date='2026-10-29',authority='Explicit user request: venue split and TDS-local results',
        file_moves=moves,all_moved_payload_bytes_preserved=True,private_backup=str(backup.relative_to(ROOT)),
        policy='Historical manifests retain original bytes and logical paths; resolve paths through this registry'),indent=2)+'\n')
    print(f'{len(moves)} files relocated; all SHA-256 checks pass; private rollback snapshot retained')


if __name__=='__main__':main()
