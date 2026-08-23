
def test_no_stale_gate_in_final_manifest_root(d4_root):
    names = {path.name for path in (d4_root / "manifests").iterdir()}
    stale = {"darpa_real_manifest.json", "lanl_real_manifest.json", "official_source_gate.json", "final_paper_gate.json"}
    assert not names.intersection(stale)
    archived = {path.name for path in (d4_root / "archive/pre_real_source_attempts").iterdir()}
    assert stale.issubset(archived)
