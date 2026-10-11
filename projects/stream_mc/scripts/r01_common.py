"""Project-owned paths and exact-byte evidence helpers (no manuscript imports)."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'projects/stream_mc'
CONFIG = PROJECT / 'configs/r01.json'
# Keep optional Parquet support project-local; the shared CUDA venv is untouched.
sys.path.insert(0, str(PROJECT / '.deps'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    tmp.replace(path)


def config():
    cfg = read(CONFIG); cfg['config_sha256'] = sha256(CONFIG)
    return cfg


def output(): return ROOT / config()['output']


def git_sha():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def source_identity():
    paths = sorted(PROJECT.joinpath('scripts').glob('r01_*.py'))
    paths += [ROOT/'src/gog_fraud/data/level2/relation_builder.py', ROOT/'src/gog_fraud/streaming/selective.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths if p.exists()}
