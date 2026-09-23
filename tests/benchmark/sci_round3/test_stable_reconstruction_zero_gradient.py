import torch
from gog_fraud.models.pygod.stable_reconstruction import StableDOMINANTBase, stable_reconstruction_score


def test_exact_zero_residual_has_finite_zero_gradient():
    x = torch.zeros((2, 3), requires_grad=True)
    x_hat = torch.zeros((2, 3), requires_grad=True)
    s = torch.eye(2, requires_grad=True)
    s_hat = torch.eye(2, requires_grad=True)
    loss = stable_reconstruction_score(x, x_hat, s, s_hat).mean()
    loss.backward()
    for value in (x.grad, x_hat.grad, s.grad, s_hat.grad):
        assert torch.isfinite(value).all()
        assert torch.equal(value, torch.zeros_like(value))


def test_pygod_instance_assignment_is_replaced():
    model = StableDOMINANTBase(in_dim=3, hid_dim=4, num_layers=2)
    assert model.loss_func is stable_reconstruction_score
