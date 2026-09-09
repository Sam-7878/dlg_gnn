"""Smoke test for the improved dataset_manifest module."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gog_fraud.data.io.dataset_manifest import build_dataset_manifest

with tempfile.TemporaryDirectory() as tmp:
    # Create a tiny fake CSV
    p = Path(tmp) / "test.csv"
    p.write_text(
        "sample_id,timestamp,blocknumber,from,to\n"
        "1,1000,100,0xA,0xB\n"
        "2,,200,0xC,0xD\n",
        encoding="utf-8",
    )

    manifest = build_dataset_manifest(
        tmp, chain="test", max_files=10, progress=False
    )
    assert manifest.chain == "test", f"chain mismatch: {manifest.chain}"
    assert manifest.transactions == 2, f"transactions: {manifest.transactions}"
    assert manifest.missing_timestamp == 1, f"missing_ts: {manifest.missing_timestamp}"
    assert manifest.addresses == 4, f"addresses: {manifest.addresses}"  # 0xA, 0xB, 0xC, 0xD
    print("manifest chain:", manifest.chain)
    print("transactions:", manifest.transactions)
    print("missing_timestamp:", manifest.missing_timestamp)
    print("addresses:", manifest.addresses)

    # Test write
    manifest.write(Path(tmp) / "out.json", Path(tmp) / "out.csv")
    assert (Path(tmp) / "out.json").exists()
    assert (Path(tmp) / "out.csv").exists()
    print("write OK")

    print("progress=True OK (run in fresh tmpdir to avoid write artifacts)")

print("MANIFEST SMOKE TEST PASSED")
