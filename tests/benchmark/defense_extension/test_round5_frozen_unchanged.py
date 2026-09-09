"""Test frozen Round 5 benchmark raw and support matrix immutability."""
from pathlib import Path
import hashlib

FROZEN_RAW_SHA256 = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
FROZEN_SUPPORT_SHA256 = "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_round5_frozen_raw_csv_unmodified():
    raw_csv = Path("outputs/sci_round5_final/raw/benchmark_raw.csv")
    assert raw_csv.exists(), f"Missing frozen Round 5 raw CSV at {raw_csv}"
    actual_hash = sha256_file(raw_csv)
    assert actual_hash == FROZEN_RAW_SHA256, (
        f"Round 5 raw CSV was modified! Expected {FROZEN_RAW_SHA256}, got {actual_hash}"
    )


def test_round5_support_matrix_unmodified():
    support_file = Path("outputs/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv")
    assert support_file.exists(), f"Missing support matrix at {support_file}"
    actual_hash = sha256_file(support_file)
    assert actual_hash == FROZEN_SUPPORT_SHA256, (
        f"Round 5 support matrix was modified! Expected {FROZEN_SUPPORT_SHA256}, got {actual_hash}"
    )
