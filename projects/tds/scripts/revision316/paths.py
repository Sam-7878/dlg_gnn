"""316 ownership: manuscript sources private, numeric results task-scoped."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROJECT=ROOT/'projects/tds'
BASE=PROJECT/'paper/submission_316'
SHARED=BASE/'shared'
PREPRINT=BASE/'preprints'
MDPI=BASE/'mdpi_applsci_anomaly'
RESULTS=PROJECT/'results/316_dual_submission'
REPORTS=PROJECT/'reports/316'
PUBLICATION=PROJECT/'publication/316'
PRIVATE=PROJECT/'private/316'
AUTHORS=PROJECT/'submission/author_records/316'
FROZEN=PROJECT/'results/main_final_v2'
SUPPLEMENTAL=PROJECT/'results/314_applsci'


def delivery(venue):
    return PROJECT/'submission'/('preprints' if venue=='preprints' else 'applsci')/'316'
