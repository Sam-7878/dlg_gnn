"""Canonical TDS paths; shared library code must not depend on these."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'projects/tds'
INPUT = PROJECT / 'results/314_applsci'
ORIGINAL_PAPER = PROJECT / 'paper/applsci'
PAPER = ORIGINAL_PAPER / 'submission_315'
PREPRINT = PROJECT / 'paper/preprints/submission_315'
COMMON = PROJECT / 'paper/common/315'
PRIVATE = PROJECT / 'private/315_venue_split'
AUTHORS = PROJECT / 'submission/author_records/315'
RESULTS = PROJECT / 'results/315_applsci'
REPORTS = PROJECT / 'reports/315'
PUBLICATION = PROJECT / 'publication/315'
SUBMISSION = PROJECT / 'submission/applsci/315'
PRIMARY = PROJECT / 'results/main_final_v2'


def resolve_frozen(logical):
    """Resolve sealed logical paths without editing historical manifests."""
    registry = PROJECT / 'publication/layout/path_migrations.json'
    mapping = {e['old']: e['new'] for e in json.loads(registry.read_text())['file_moves']} if registry.exists() else {}
    return ROOT / mapping.get(str(logical), str(logical))


def manuscript_source():
    text = (PAPER/'applsci_article.tex').read_text()
    body = PAPER/'manuscript_body.tex'
    return text + ('\n' + body.read_text() if body.exists() else '')
