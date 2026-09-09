from gog_fraud.experiments.round5_policy import sha256_file


def test_table_provenance_uses_frozen_raw_hash(tmp_path):
    raw=tmp_path/"benchmark_raw.csv";raw.write_text("dataset,model\nD,M\n",encoding="utf-8")
    comment=f"% source_raw_sha256={sha256_file(raw)}"
    assert sha256_file(raw) in comment
