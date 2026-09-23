import numpy as np
import torch

from gog_fraud.evaluation.reproducibility import seed_everything


def test_seed_repeats_numpy_and_torch_streams():
    seed_everything(44)
    first = (np.random.random(4), torch.rand(4))
    seed_everything(44)
    second = (np.random.random(4), torch.rand(4))
    assert np.array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])

