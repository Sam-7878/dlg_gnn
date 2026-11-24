"""A29 freeze: write the submission hash manifest and the evidence index after the final build.

  python a29_finalize.py        # requires a final build with no pending items and a passing a29_pdf_qa.py on the same PDFs
"""
from __future__ import annotations

import hashlib
import json

from a29_common import RESULTS, ROOT, dump, read_json

PUB = ROOT / 'projects/benchmark/publication/a29'
PAPER = ROOT / 'projects/benchmark/paper/a29'
FILES = ['applsci/DLG-Benchmark_A29_AppliedSciences.pdf', 'applsci/source_bundle.zip',
         'preprints/DLG-Benchmark_A29_Preprints.pdf', 'preprints/source_bundle.zip',
         'COVER_LETTER_A29.md', 'SI_EDITOR_INQUIRY_A29.md']


def main():
    b = read_json(RESULTS / 'manuscript_build.json')
    assert b.get('final') and not b.get('pending'), 'run a29_build_manuscript.py --final first'
    for v in ('applsci', 'preprints'):  # QA gate: a29_pdf_qa.py must have checked exactly these PDFs and found nothing
        q = read_json(PUB / f'{v}_qa.json')
        assert q['pdf_sha256'] == b[v]['pdf_sha256'], f'{v}: QA is for another build; rerun a29_pdf_qa.py'
        assert not q['leftovers'] and not q['log_warnings'] and q['a28_tables_all_identical'], f'{v}: QA failed'
        assert q['main_tables_in_order'] and q['appendix_tables_in_order'], f'{v}: table order'
    lines = [f"{hashlib.sha256((PAPER / f).read_bytes()).hexdigest()}  paper/a29/{f}" for f in FILES]
    (PUB / 'A29_SUBMISSION_HASHES.sha256').write_text('\n'.join(lines) + '\n')
    results = sorted(p for p in RESULTS.rglob('*') if p.is_file())
    dump(PUB / 'A29_RESULTS_INDEX.json', [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                           'bytes': p.stat().st_size} for p in results])
    manifest_sha = hashlib.sha256((PUB / 'A29_SUBMISSION_HASHES.sha256').read_bytes()).hexdigest()
    print('\n'.join(lines)); print('manifest sha256:', manifest_sha)
    dump(PUB / 'A29_FREEZE.json', {'manifest_sha256': manifest_sha, 'build': b})


if __name__ == '__main__':
    main()
