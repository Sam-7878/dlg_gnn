"""Compare A08 rank-one KL with the actual upstream mathematical expression."""
import pytest
import torch
from pygod.nn.functional import KL_neighbor_loss
from gog_fraud.models.a08_gadnr_numerics import stable_kl_neighbor_loss


def dense_reference(predictions,targets,valid):
    x1=predictions.detach().cpu().reshape(-1,predictions.shape[-1])[:valid].double()
    x2=targets.detach().cpu().reshape(-1,targets.shape[-1])[:valid].double()
    a,b=x1.mean(0),x2.mean(0);width=x1.shape[1]
    c1=x1-a;c2=x2-b
    covariance1=torch.eye(width,dtype=torch.float64)+c1.T@c1/max(valid-1,1)
    covariance2=torch.eye(width,dtype=torch.float64)+c2.T@c2/max(valid-1,1)
    log1=torch.linalg.slogdet(covariance1);log2=torch.linalg.slogdet(covariance2)
    assert log1.sign==log2.sign==1
    d=b-a
    # Preserve PyGOD's implemented log(cov1/cov2) orientation exactly.
    return .5*(log1.logabsdet-log2.logabsdet-width+torch.trace(torch.linalg.solve(covariance2,covariance1))+d@torch.linalg.solve(covariance2,d))


@pytest.mark.parametrize('valid',[1,2])
@pytest.mark.parametrize('width',[8,64])
@pytest.mark.parametrize('scale',[.1,1.,100.])
def test_stable_kl_matches_dense_float64_objective(valid,width,scale):
    generator=torch.Generator().manual_seed(42)
    predictions=torch.randn(1,2,width,generator=generator,dtype=torch.float64)*scale
    targets=torch.randn(1,2,width,generator=generator,dtype=torch.float64)*scale
    actual=stable_kl_neighbor_loss(predictions,targets,valid,'cpu')
    expected=dense_reference(predictions,targets,valid)
    assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-8)
    assert not actual.requires_grad


@pytest.mark.parametrize('valid',[1,2])
def test_well_conditioned_actual_upstream_value_and_output_detachment(valid):
    torch.manual_seed(13)
    p=torch.randn(1,2,64,requires_grad=True)*.25;t=torch.randn(1,2,64)*.25
    old=KL_neighbor_loss(p,t,valid,'cpu');new=stable_kl_neighbor_loss(p,t,valid,'cpu')
    assert torch.allclose(old,new,atol=1e-5,rtol=1e-4)
    assert new.dtype==p.dtype and not new.requires_grad


def test_ill_conditioned_identity_regularized_covariance_remains_finite():
    # Rank-one SPD covariance has strictly positive exact determinant despite
    # a float32 dense determinant being unreliable at this scale.
    p=torch.zeros(1,2,64);p[0,0]=torch.linspace(1,2,64)*1e5;p[0,1]=-p[0,0]
    t=torch.zeros_like(p);t[0,0]=torch.linspace(.2,.3,64);t[0,1]=-t[0,0]
    actual=stable_kl_neighbor_loss(p,t,2,'cpu')
    assert torch.isfinite(actual) and actual.dtype==torch.float32
    assert float(actual)>1e8  # No clipping or fabricated nonnegative floor.


def test_nonfinite_sample_is_numerical_failure_not_fallback():
    p=torch.ones(1,2,8);p[0,0,0]=float('inf')
    with pytest.raises(FloatingPointError):stable_kl_neighbor_loss(p,torch.ones_like(p),2,'cpu')
