"""Freeze of the manuscript release REV (a29_common): hash lists, sign-off form and its verification gate (A30-04).

Two stages, because the cover letter can only be completed after the preprint DOI exists:
  python a29_finalize.py               # after `a29_build_manuscript.py --final` and a passing `a29_pdf_qa.py`:
                                       # writes <REV>_MANUSCRIPT_HASHES.sha256 (two PDFs, two source bundles, editor inquiry),
                                       # <REV>_RESULTS_INDEX.json, <REV>_FREEZE.json and the sign-off form with the hashes filled in
  python a29_finalize.py cover-letter  # after the DOI is filled into the cover letter: <REV>_COVER_LETTER_HASH.sha256
  python a29_finalize.py verify        # gate: every signed row approves the actual SHA-256 of its hash list, is dated no
                                       # later than today, and every file in the lists still has its listed hash
Hashes in the sign-off form are never typed by hand. Rows marked "superseded" are kept as records and ignored by the gate.
"""
from __future__ import annotations

import datetime
import hashlib
import re
import sys

from a29_common import FREEZE_DATE, PAPER_OUT, PUB_OUT, REV, RESULTS, ROOT, dump, read_json
from a29_build_manuscript import STEM

R = REV.upper()
MANUSCRIPT_FILES = [f'applsci/{STEM.format("AppliedSciences")}.pdf', 'applsci/source_bundle.zip',
                    f'preprints/{STEM.format("Preprints")}.pdf', 'preprints/source_bundle.zip', f'SI_EDITOR_INQUIRY_{R}.md']
COVER = f'COVER_LETTER_{R}.md'
MAN_LIST = PUB_OUT / f'{R}_MANUSCRIPT_HASHES.sha256'
COVER_LIST = PUB_OUT / f'{R}_COVER_LETTER_HASH.sha256'
SIGNOFF = PUB_OUT / f'{R}_AUTHOR_SIGNOFF.md'
AUTHORS = ['Seong-Su Park', 'Ki-Hyung Kim, corresponding author']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return str(p.relative_to(ROOT))


def write_list(path, files):
    lines = [f'{sha(PAPER_OUT / f)}  {rel(PAPER_OUT / f)}' for f in files]
    path.write_text('\n'.join(lines) + '\n')
    return sha(path), lines


def signoff_form(man_sha, cover_sha=None):
    rows = '\n'.join(f'| {a} | | | `{man_sha}` |' for a in AUTHORS)
    cover = (f'| Ki-Hyung Kim, corresponding author | | | `{cover_sha}` |' if cover_sha else
             '| Ki-Hyung Kim, corresponding author | | | (run `python a29_finalize.py cover-letter` after the DOI is filled in) |')
    return f"""# {R} author approval

Release {R}, frozen {FREEZE_DATE}. The SHA-256 values below are written by `scripts/a29_finalize.py`; do not edit them by hand.
Fill in only the approval and date columns. `python a29_finalize.py verify` checks every signed row against the actual files
(hash of the approved list = actual hash, date no later than the day of the check). If any listed file changes, its hash list
changes and the approval must be renewed. Rows that turn out to be wrong are kept and marked "superseded", never deleted.

## Stage 1 — manuscript (both authors, before posting the preprint)

Approves [{MAN_LIST.name}]({MAN_LIST.name}): both PDFs, both source bundles and the inquiry to the editorial office.
The approval covers the manuscript and appendix of both versions, including the title; authorship, affiliations, author
contributions, funding, conflicts of interest and the generative-AI disclosure; the Data Availability Statement and the
internal-only scope of per-node scores, labels, split masks and node identifiers; posting the Preprints.org version under
CC BY 4.0 and submission to Applied Sciences, Special Issue C67H3U9QGT.

| Author | Written approval (yes/no) | Date (YYYY-MM-DD) | Approved list SHA-256 |
| --- | --- | --- | --- |
{rows}

## Stage 2 — cover letter (corresponding author, just before the SuSy submission)

Approves [{COVER_LIST.name}]({COVER_LIST.name}): the cover letter with the preprint URL, DOI and version filled in, and the
corresponding-author confirmations (exclusive consideration, any prior MDPI submission, APC responsibility).

| Author | Written approval (yes/no) | Date (YYYY-MM-DD) | Approved list SHA-256 |
| --- | --- | --- | --- |
{cover}
"""


def freeze():
    b = read_json(PUB_OUT / 'manuscript_build.json')
    assert b.get('final') and not b.get('pending'), 'run a29_build_manuscript.py --final first'
    for v in ('applsci', 'preprints'):  # QA gate: a29_pdf_qa.py must have checked exactly these PDFs and found nothing
        q = read_json(PUB_OUT / f'{v}_qa.json')
        assert q['pdf_sha256'] == b[v]['pdf_sha256'], f'{v}: QA is for another build; rerun a29_pdf_qa.py'
        assert not q['leftovers'] and not q['log_warnings'] and q['a28_tables_all_identical'], f'{v}: QA failed'
        assert q['main_tables_in_order'] and q['appendix_tables_in_order'], f'{v}: table order'
        assert q['pdf_date_matches_freeze'], f'{v}: PDF date is not the freeze date {FREEZE_DATE}'
    if SIGNOFF.exists() and re.search(r'\|\s*yes\s*\|', SIGNOFF.read_text(), re.I):
        raise SystemExit(f'{SIGNOFF.name} already holds approvals; mark them superseded before a new freeze')
    man_sha, lines = write_list(MAN_LIST, MANUSCRIPT_FILES)
    results = sorted(p for p in RESULTS.rglob('*') if p.is_file())
    dump(PUB_OUT / f'{R}_RESULTS_INDEX.json', [{'path': rel(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in results])
    dump(PUB_OUT / f'{R}_FREEZE.json', {'release': REV, 'freeze_date': FREEZE_DATE, 'manuscript_list_sha256': man_sha, 'build': b})
    SIGNOFF.write_text(signoff_form(man_sha))
    print('\n'.join(lines)); print(f'{MAN_LIST.name} sha256: {man_sha}')


def cover_letter():
    text = (PAPER_OUT / COVER).read_text()
    left = re.findall(r'\*\*\[[^\]]*\]\*\*|\[(?:Preprints\.org URL|preprint DOI|v1)\]', text)
    if left:
        raise SystemExit(f'cover letter still has placeholders: {left}')
    cover_sha, lines = write_list(COVER_LIST, [COVER])
    s = SIGNOFF.read_text()
    s = re.sub(r'\| Ki-Hyung Kim, corresponding author \| \| \| \(run `python a29_finalize.py cover-letter`[^\n]*\|',
               f'| Ki-Hyung Kim, corresponding author | | | `{cover_sha}` |', s)
    SIGNOFF.write_text(s)
    print('\n'.join(lines)); print(f'{COVER_LIST.name} sha256: {cover_sha}')


def verify(today=None):
    today = today or datetime.date.today()
    errors, ok = [], 0
    for lst in (MAN_LIST, COVER_LIST):
        if not lst.exists():
            continue
        for line in lst.read_text().splitlines():
            h, path = line.split('  ', 1)
            if sha(ROOT / path) != h:
                errors.append(f'{path}: file changed since {lst.name} was written')
    actual = {lst.name: sha(lst) for lst in (MAN_LIST, COVER_LIST) if lst.exists()}
    section = None
    for line in SIGNOFF.read_text().splitlines():
        if line.startswith('## Stage 1'):
            section = MAN_LIST.name
        elif line.startswith('## Stage 2'):
            section = COVER_LIST.name
        cells = [c.strip() for c in line.strip().strip('|').split('|')] if line.startswith('|') else []
        if len(cells) != 4 or cells[0] in ('Author', '---') or 'superseded' in cells[0].lower():
            continue
        author, approval, date, hsh = cells
        if approval.lower() != 'yes':
            continue
        h = re.search(r'[0-9a-f]{64}', hsh)
        if not h or h.group(0) != actual.get(section):
            errors.append(f'{author}: approved hash {hsh[:20]} is not the SHA-256 of {section} ({actual.get(section, "missing")[:16]}…)')
        try:
            if datetime.date.fromisoformat(date) > today:
                errors.append(f'{author}: date {date} is later than {today}')
        except ValueError:
            errors.append(f'{author}: date "{date}" is not YYYY-MM-DD')
        ok += 1
    if errors:
        raise SystemExit('sign-off verification FAILED:\n  ' + '\n  '.join(errors))
    print(f'sign-off verification passed: {ok} signed row(s), all files match their hash lists')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'freeze'
    {'freeze': freeze, 'cover-letter': cover_letter, 'verify': verify}[cmd]()
