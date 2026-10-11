"""Inspect the actual candidate boundary; do not assert anonymization or hosting."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from r01_public import PROJECT, verified_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', action='store_true')
    args = parser.parse_args()
    candidate = PROJECT / 'evidence/r01_numeric_evidence.zip'
    with verified_archive()[0] as archive:
        names = archive.namelist()
        forbidden = [n for n in names if Path(n).suffix.lower() in ('.tex', '.bib', '.pdf')
                     or n.endswith('r01_build_paper.py') or n.startswith('paper/')]
        assert not forbidden, 'private manuscript payload found: ' + repr(forbidden)
        scanned = []
        exposed = []
        for name in names:
            if name.startswith(('code/', 'documentation/', 'protocols/')):
                continue
            if Path(name).suffix in ('.csv', '.json', '.jsonl', '.log', '.xml'):
                scanned.append(name)
                if re.search(rb'(?<![0-9a-fA-F])0x[0-9a-fA-F]{40}(?![0-9a-fA-F])', archive.read(name)):
                    exposed.append(name)
        assert not exposed, 'literal account/contract addresses in evidence text: ' + repr(exposed)
    report = {'archive_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
              'archive_bytes': candidate.stat().st_size, 'zip_members': len(names),
              'private_manuscript_members': forbidden,
              'evidence_text_members_scanned_for_literal_0x40_addresses': len(scanned),
              'literal_address_member_count': len(exposed),
              'scope': 'actual member boundaries and literal-address scan; hashes/aliases remain linkable, not anonymous; code and documentation are excluded from address scan',
              'hosted_public_release_verified': False}
    if args.paper:
        abstract = PROJECT / 'paper/current/r01/common/abstract.tex'
        record = []
        for entry in ('preprint', 'journal'):
            pdf = PROJECT / f'paper/current/r01/build/{entry}.pdf'
            text = subprocess.check_output(['pdftotext', '-f', '1', '-l', '1', str(pdf), '-'], text=True)
            body = text.split('Abstract', 1)[1].split('Keywords:', 1)[0]
            count = len(body.split())
            assert count <= 200, 'actual rendered abstract exceeds 200 words'
            record.append({'entry': entry, 'abstract_words_pdftotext': count,
                           'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()})
        report['private_rendered_abstracts'] = record
        report['source_abstract_words'] = len(abstract.read_text().split())
        report['word_count_scope'] = 'whitespace tokens in actual first-page PDF extraction; hyphenated line joins may affect counting'
    # The receipt binds the ZIP hash: keep it outside that ZIP and its local
    # per-run manifest to avoid a cyclic or immediately stale hash assertion.
    target = PROJECT / 'evidence/archive_boundary_r01.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('ARCHIVE BOUNDARY PASS', report['archive_bytes'], len(names), flush=True)


if __name__ == '__main__':
    main()
