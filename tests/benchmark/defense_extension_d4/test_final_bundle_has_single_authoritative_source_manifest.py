import json


def test_final_bundle_has_single_authoritative_source_manifest(d4_root):
    manifests = list((d4_root / "manifests").glob("*source_of_truth*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["manifest_role"] == "single_authoritative_real_source_manifest"
    assert manifest["readiness"] == "PAPER_READY_10_PLUS_LANL_PLUS_THEIA_SCALABILITY"
