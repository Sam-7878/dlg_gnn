import csv
import json

from gog_fraud.data.splits.artifact import build_split_artifacts


def test_split_artifact_is_immutable_and_truthful(tmp_path):
    chain_root = tmp_path / "transactions" / "ethereum"
    chain_root.mkdir(parents=True)
    labels = tmp_path / "labels.csv"
    labels.write_text("Chain,Contract,Category\n" + "".join(f"ethereum,c{i},{i % 2}\n" for i in range(12)), encoding="utf-8")
    for index in range(12):
        with (chain_root / f"c{index}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp"]); writer.writeheader(); writer.writerow({"timestamp": index + 1})
    manifest = tmp_path / "ethereum.json"; manifest.write_text('{"chain":"ethereum"}\n', encoding="utf-8")
    fixed, rolling, audit = build_split_artifacts(
        transaction_root=tmp_path / "transactions", labels_path=labels, chain="ethereum",
        source_manifest=manifest, output_dir=tmp_path / "splits",
    )
    fixed_data = json.loads(fixed.read_text())
    assert fixed_data["train"]["end_timestamp"] <= fixed_data["validation"]["start_timestamp"]
    assert len(json.loads(rolling.read_text())["folds"]) == 5
    audit_data = json.loads(audit.read_text())
    assert audit_data["raw_event_time_audit"] == "PASS"
    assert audit_data["status"] == "INCOMPLETE"

    before = {path: path.read_bytes() for path in (fixed, rolling, audit)}
    build_split_artifacts(
        transaction_root=tmp_path / "transactions", labels_path=labels, chain="ethereum",
        source_manifest=manifest, output_dir=tmp_path / "splits",
    )
    assert {path: path.read_bytes() for path in before} == before
