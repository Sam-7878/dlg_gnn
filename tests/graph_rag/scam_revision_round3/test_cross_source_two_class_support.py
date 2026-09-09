from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_both_cross_source_directions_have_two_classes_and_finite_auc():
    metrics = pd.read_csv(RESULTS / "cross_source_bidirectional.csv")
    assert set(metrics.protocol) == {"CST_to_CSDB", "CSDB_to_CST"}
    assert (metrics.groupby("protocol").seed.nunique() == 5).all()
    assert (metrics.n_positive > 0).all() and (metrics.n_negative > 0).all()
    assert np.isfinite(metrics.roc_auc).all() and np.isfinite(metrics.auc_pr).all()
    manifest = pd.read_parquet(RESULTS / "evaluation_sample_manifests/cross_source.parquet")
    assert (manifest.groupby("protocol").label.nunique() == 2).all()
