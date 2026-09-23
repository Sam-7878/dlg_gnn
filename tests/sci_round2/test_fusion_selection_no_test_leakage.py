import inspect
import numpy as np
from gog_fraud.pipelines.run_sci_round1_ablation import select_fusion_weight


def test_fusion_selector_has_no_test_argument_and_uses_validation_only():
    assert all("test" not in name for name in inspect.signature(select_fusion_weight).parameters)
    weight, candidates = select_fusion_weight(
        np.array([0, 0, 1, 1]),
        np.array([0, .1, .9, 1]),
        np.array([1, .9, .1, 0]),
        [.2, .8],
    )
    assert weight == .8
    assert len(candidates) == 2
