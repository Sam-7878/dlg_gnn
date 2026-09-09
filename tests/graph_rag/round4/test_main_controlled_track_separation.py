from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_main_controlled_track_separation():
    results = ROOT / "results/graphrag/round_4"
    main = pd.read_csv(results / "main_results.csv")
    controlled = pd.read_csv(results / "controlled_context_results.csv")
    assert set(main.track) == {"SCI Main Track"}
    assert set(main.method) == {"GNN Only", "MC-GNN (T=10)"}
    assert not main.context_used.astype(bool).any()
    assert set(controlled.track) == {"Controlled Context-Augmentation Study"}
    assert not controlled.paper_eligible.astype(bool).any()
    assert set(controlled.context_policy) == {"label-conditioned context"}

