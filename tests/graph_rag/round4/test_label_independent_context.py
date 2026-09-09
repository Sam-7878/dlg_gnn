from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_label_independent_context():
    context = pd.read_parquet(ROOT / "results/graphrag/round_4/context_provenance.parquet")
    assert len(context) == 24_316
    assert not context.label_accessed.astype(bool).any()
    assert set(context.source_fields.unique()) == {"chain_id,timestamp,num_nodes,num_edges"}
    forbidden = {"label", "is_fraud", "fraud_flag", "ground_truth", "test_y"}
    assert forbidden.isdisjoint(set(",".join(context.source_fields).split(",")))

