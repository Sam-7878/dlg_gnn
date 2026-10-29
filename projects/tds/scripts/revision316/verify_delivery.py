"""Final ownership, source/archive identity and no-fiction checks."""
import json
import subprocess
import zipfile
from pathlib import Path
from paths import ROOT,PROJECT,BASE,SHARED,PREPRINT,MDPI,RESULTS,REPORTS,PUBLICATION,AUTHORS,delivery
from common import sha,write


def main():
    state=json.loads((REPORTS/'submission_readiness_316.json').read_text())
    build=json.loads((REPORTS/'build_verification.json').read_text())
    parity=json.loads((REPORTS/'content_claim_parity.json').read_text())
    release=json.loads((PUBLICATION/'release_manifest.json').read_text())
    cold=json.loads((REPORTS/'numeric_cold_replay.json').read_text())
    assert state['DUAL_VERSION_PARITY_PASS'] is True and parity['status']=='PASS'
    assert all(state[k] is False for k in ['READY_PREPRINT','READY_MDPI','SUBMITTED_PREPRINT','POSTED_PREPRINT','SUBMITTED_MDPI'])
    assert state['PREPRINT_DOI'] is None and state['PREPRINT_URL'] is None
    for v in build['venues']:
        folder=PREPRINT if v['venue']=='preprints' else MDPI
        for e in v['builds']:assert sha((folder/e['source']).with_suffix('.pdf'))==e['pdf_sha256']
        with zipfile.ZipFile(ROOT/v['source_zip']['path']) as z:assert z.testzip() is None
        assert sha(ROOT/v['source_zip']['path'])==v['source_zip']['sha256']
    assert sha(ROOT/release['path'])==release['sha256']==cold['archive_sha256']
    assert '18 passed' in cold['tests'] and release['score_vectors_exactly_preserved']==65
    forbidden=[str(p.relative_to(ROOT)) for p in BASE.rglob('*') if p.is_file() and p.suffix in {'.csv','.pt','.pth','.npy','.npz','.parquet'}]
    assert not forbidden,forbidden
    private=[SHARED/'prepare.py',SHARED/'sections/manuscript_body.tex',PREPRINT/'preprint_main.tex',MDPI/'applsci_article.tex',
             AUTHORS/'author_facts.json',delivery('preprints')/'preprint_source.zip',delivery('mdpi')/'mdpi_source.zip',
             delivery('preprints')/'metadata_preprints.json',delivery('mdpi')/'metadata_applsci_anomaly.json']
    public=[ROOT/release['path'],RESULTS/'input_manifest.json',REPORTS/'submission_readiness_316.json',
            PROJECT/'scripts/revision316/build.py',PROJECT/'tests/test_submission316.py']
    for p in private:assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=ROOT).returncode==0,p
    for p in public:assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=ROOT).returncode==1,p
    source_files=list((PROJECT/'scripts/revision316').glob('*.py'))
    for p in source_files:compile(p.read_text(),str(p),'exec')
    write(REPORTS/'delivery_verification.json',dict(status='PASS',manuscript_folder_result_files=forbidden,
        private_files_ignored=[str(p.relative_to(ROOT)) for p in private],public_files_git_eligible=[str(p.relative_to(ROOT)) for p in public],
        syntax_checked_files=len(source_files),frozen_scores_preserved=65,source_zip_crc_and_pdf_hashes=True,
        verified_public_release=False,no_platform_action_performed=True))
    print('PASS: final hashes, source ZIP CRC, paper/data separation, Git visibility and false/null publication states')


if __name__=='__main__':main()
