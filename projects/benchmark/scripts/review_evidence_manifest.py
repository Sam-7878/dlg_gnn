#!/usr/bin/env python3
"""Hash or verify public revision evidence, independently of private manuscripts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'projects/benchmark/evidence/astra_revision'
MANIFEST = EVIDENCE / 'MANIFEST.json'


def inventory():
    return {p.relative_to(ROOT).as_posix(): {
        'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(EVIDENCE.rglob('*')) if p.is_file() and p != MANIFEST}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    actual = inventory()
    if not actual:
        raise RuntimeError('public evidence missing')
    if args.write:
        MANIFEST.write_text(json.dumps({'schema_version': 1,
            'scope': 'Public numeric evidence; unpublished manuscript excluded',
            'files': actual}, indent=2) + '\n')
    else:
        expected = json.loads(MANIFEST.read_text())['files']
        if expected != actual:
            changed = sorted(k for k in set(actual) | set(expected)
                             if actual.get(k) != expected.get(k))
            raise RuntimeError('public evidence drift: ' + ', '.join(changed))
    print(f'{"WROTE" if args.write else "PASS"} {len(actual)} public evidence hashes')


if __name__ == '__main__':
    main()
