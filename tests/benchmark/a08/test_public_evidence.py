"""Public payload guards exercise actual boundary failures, without campaign writes."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('a08_public', ROOT / 'projects/benchmark/scripts/a08_public_evidence.py')
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


@pytest.mark.parametrize('name', [
    '/absolute.json', '../escape.json', 'nested/../../escape.json', 'windows\\escape.json',
    'paper.tex', 'nested/supplement.pdf', 'private/references.bib', 'private/Definitions/mdpi.cls',
    'local_storage/run/metrics.json', 'projects/benchmark/paper/current/numbers.csv',
    'projects/benchmark/scripts/a08_build_manuscript.py', 'tracked-working-tree.patch',
    'scores.npz', 'checkpoint.pt', 'wheel.whl',
])
def test_private_or_unsafe_payload_is_rejected(name):
    with pytest.raises(ValueError):
        public.safe_name(name)


def test_scientific_sources_numeric_records_and_audits_remain_public():
    for name in ['src/gog_fraud/data/crypto_raw.py', 'numeric/threshold_audit.csv',
                 'audit/private_environment_patch_disposition.json',
                 'projects/benchmark/scripts/a08_aggregate.py']:
        assert public.safe_name(name).as_posix() == name
