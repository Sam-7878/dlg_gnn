import hashlib
from pathlib import Path


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_round5_frozen_integrity(d3_gate):
    assert d3_gate["round5_frozen_unchanged"]
    assert digest("outputs/sci_round5_final/raw/benchmark_raw.csv") == "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
    assert digest("outputs/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv") == "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"
