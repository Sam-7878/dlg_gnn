from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_gold_relevance_is_fixed_and_ranking_is_relevance_blind():
    queries = pd.read_parquet(RESULTS / "retrieval_queries_fixed.parquet")
    assert set(queries.hop) == {0, 1, 2}
    assert (queries.groupby("query_id").hop.nunique() == 3).all()
    assert (queries.groupby("query_id").relevance_set_sha256.nunique() == 1).all()
    assert not queries.ranking_uses_gold_membership.astype(bool).any()
    assert (queries.n_relevant > 0).all()
