"""Canonical TDS paths; shared library code must not depend on these."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'projects/tds'
INPUT = PROJECT / 'paper/applsci'
PAPER = INPUT / 'submission_315'
RESULTS = ROOT / 'projects/results/tds/315_applsci'
REPORTS = PROJECT / 'reports/315'
PUBLICATION = PROJECT / 'publication/315'
SUBMISSION = PROJECT / 'submission/applsci/315'
PRIMARY = PROJECT / 'results/main_final_v2'
