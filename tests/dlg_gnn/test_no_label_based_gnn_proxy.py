import _round3_bootstrap  # noqa: F401

from experiments.round3.artifact_paths import ROOT


BANNED = ("p_gnn = label", "label + noise", "0.7 * label", "simulated_gnn")


def test_no_label_proxy_in_paper_facing_pipeline():
    paths = [ROOT / "experiments" / "run_multiseed.py"]
    paths.extend((ROOT / "experiments" / "round3").glob("*.py"))
    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in BANNED:
            if pattern in text:
                violations.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert not violations, violations


def test_simulation_is_isolated_and_explicitly_named():
    path = ROOT / "experiments" / "simulation" / "run_multiseed_simulation.py"
    assert path.is_file()
