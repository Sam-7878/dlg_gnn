import torch
import pytest
from gog_fraud.models.extensions.mc.adaptive_mc import AdaptiveMCEngine, AdaptiveMCOutput


def test_adaptive_mc_engine_convergence():
    engine = AdaptiveMCEngine(t_min=3, t_max=10, mean_tol=0.05, var_tol=0.01)

    # Constant model: should converge immediately at t_min=3
    const_fn = lambda: torch.tensor([0.5, 0.8, 0.2])
    res = engine.forward_pass(const_fn)

    assert res.total_passes == 3
    assert res.t_effective == 3.0
    assert res.early_stopped_fraction == 1.0
    assert torch.allclose(res.mean, torch.tensor([0.5, 0.8, 0.2]))
    assert torch.allclose(res.variance, torch.zeros(3))


def test_adaptive_mc_engine_noisy():
    engine = AdaptiveMCEngine(t_min=3, t_max=6, mean_tol=1e-5, var_tol=1e-5)

    # High noise: will not converge, runs until t_max
    rng = torch.Generator().manual_seed(42)
    noisy_fn = lambda: torch.rand(4, generator=rng)
    res = engine.forward_pass(noisy_fn)

    assert res.total_passes == 6
    assert res.mean.shape == (4,)
    assert res.variance.shape == (4,)
    assert torch.all(res.variance > 0)
