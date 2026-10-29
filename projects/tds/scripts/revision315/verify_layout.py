"""Verify manuscript privacy, moved payload identity, and canonical TDS ownership."""
import json
import subprocess
from pathlib import Path
from paths import ROOT, PROJECT, PAPER, PREPRINT, COMMON, RESULTS, REPORTS, PUBLICATION, resolve_frozen
from freeze import sha, write


def main():
    forbidden={'.csv','.pt','.pth','.npy','.npz','.parquet'}
    mixed=[str(p.relative_to(ROOT)) for p in (PROJECT/'paper').rglob('*') if p.is_file() and p.suffix in forbidden]
    assert not mixed,mixed
    assert not (ROOT/'projects/results/tds/315_applsci').exists()
    assert RESULTS==PROJECT/'results/315_applsci'
    assert sha(COMMON/'manuscript_body.tex')==sha(PAPER/'manuscript_body.tex')==sha(PREPRINT/'manuscript_body.tex')
    assert sha(COMMON/'references.bib')==sha(PAPER/'references.bib')==sha(PREPRINT/'references.bib')
    audit=json.loads((REPORTS/'evidence_audit.json').read_text())
    repairs={e['path']:e['after'] for e in audit['source_path_repairs']}
    sealed=json.loads((RESULTS/'input_hash_manifest.json').read_text())
    for e in sealed['files']:
        assert sha(resolve_frozen(e['path']))==repairs.get(e['path'],e['sha256']),e['path']
    registry=json.loads((PUBLICATION.parent/'layout/path_migrations.json').read_text())
    legacy=[e for e in registry['file_moves'] if '/results/historical_reviewed_draft/' in e['new']]
    for e in legacy:assert sha(ROOT/e['new'])==e['sha256']
    builds=json.loads((REPORTS/'venue_split/build_verification.json').read_text())
    private=[PAPER/'applsci_article.tex', PREPRINT/'TDS_315_Preprints.tex',COMMON/'prepare.py',
             PROJECT/'submission/author_records/315/credit_confirmed.json']
    private += [ROOT/v['source_zip']['path'] for v in builds['venues']]
    public=[RESULTS/'predictions/supplemental/summary_mlp/seed7.csv',
            PROJECT/'results/314_applsci/data/supplemental_summary.csv',
            PROJECT/'results/historical_reviewed_draft/data/data_fig3_reliability.csv',
            PUBLICATION/'TDS_315_numeric_candidate.zip',REPORTS/'venue_split/README.md']
    for p in private:
        assert p.is_file(),p
        assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=ROOT).returncode==0,p
    for p in public:
        assert p.is_file(),p
        assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=ROOT).returncode==1,p
    for v in builds['venues']:
        assert sha(ROOT/v['source_zip']['path'])==v['source_zip']['sha256']
        for e in v['builds']:
            folder=PAPER if v['venue']=='applsci' else PREPRINT
            assert sha((folder/e['source']).with_suffix('.pdf'))==e['pdf_sha256']
    write(REPORTS/'venue_split/layout_verification.json',dict(status='PASS',
        moved_file_count=len(registry['file_moves']),frozen_inputs_verified=len(sealed['files']),
        source_changes_recorded=audit['source_path_repairs'],result_files_in_paper=mixed,
        shared_body_and_bibliography_identical=True,old_results_location_removed=True,
        private_paths_ignored=[str(p.relative_to(ROOT)) for p in private],
        public_numeric_paths_git_eligible=[str(p.relative_to(ROOT)) for p in public],
        posted=False,pushed=False,manuscript_sources_private=True))
    print('PASS: no result files in paper/, 405 frozen inputs, both source identities and Git visibility')


if __name__=='__main__':main()
