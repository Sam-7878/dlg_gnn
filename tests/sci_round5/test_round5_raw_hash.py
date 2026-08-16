from gog_fraud.experiments.round5_policy import sha256_file


def test_raw_hash_changes_with_bytes(tmp_path):
    path = tmp_path / "raw.csv"; path.write_bytes(b"a\n")
    first = sha256_file(path); path.write_bytes(b"b\n")
    assert first != sha256_file(path)
