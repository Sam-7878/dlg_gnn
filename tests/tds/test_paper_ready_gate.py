import _round3_bootstrap  # noqa: F401

from experiments.round3.paper_ready_gate import evaluate_paper_ready


def test_gate_fails_closed_for_current_controlled_dataset():
    report = evaluate_paper_ready()
    assert report["paper_ready"] is False
    joined = "\n".join(report["failures"])
    assert "chronological_real" in joined
    assert "recorded transaction timestamps" in joined
    assert "label-conditioned context" in joined


def test_gate_reports_all_five_expected_seeds():
    assert evaluate_paper_ready()["expected_seeds"] == [7, 17, 27, 37, 47]
