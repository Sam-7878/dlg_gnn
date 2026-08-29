import _round3_bootstrap  # noqa: F401

from experiments.round3.artifact_paths import CHECKPOINT_DIR


def test_five_v3_real_checkpoints_exist():
    expected = {f"l1v3_seed{seed}_best.pt" for seed in (7, 17, 27, 37, 47)}
    actual = {path.name for path in CHECKPOINT_DIR.glob("l1v3_seed*_best.pt")}
    assert expected <= actual


def test_paper_facing_multiseed_requires_explicit_mode():
    source = (CHECKPOINT_DIR.parents[3] / "experiments" / "run_multiseed.py").read_text()
    assert "required=True" in source
    assert "--real-checkpoint" in source
    assert "--simulation-study" in source
