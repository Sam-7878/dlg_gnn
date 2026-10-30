"""R319 paths and read-only access to preserved scientific implementations."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'projects/tds'
OUT = PROJECT / 'results/319'
REPORTS = PROJECT / 'reports/319'
PRIVATE = PROJECT / 'private/319'
PAPER = PROJECT / 'paper/submission_319'
RAW = ROOT.parent.parent / '_data/gog_round7_upstream/Token Data'
ORIGINAL = ROOT / 'archive/tds/data/gog_scimain_v1/graph.pt'
HORIZON = PROJECT / 'private/317/horizon7_graphs.pt'
SEEDS = (7,17,27,37,47)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+'\n')


def original_module(name):
    spec = importlib.util.spec_from_file_location('_tds319_'+name, PROJECT / f'scripts/revision314/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def population(cohort):
    import pandas as pd
    name = 'original_population.csv' if cohort=='original' else 'horizon7_population.csv'
    return pd.read_csv(PROJECT / 'results/317' / name)


def require_builder():
    state = json.loads((OUT / 'seven_day_builder_equivalence.json').read_text())
    assert state['status']=='PASS' and state['mismatches']==0, 'Stop fitting: source-to-tensor identity has not passed'
    assert sha(ORIGINAL)==state['original_graph_sha256'] and sha(HORIZON)==state['horizon_graph_sha256']
    return state
