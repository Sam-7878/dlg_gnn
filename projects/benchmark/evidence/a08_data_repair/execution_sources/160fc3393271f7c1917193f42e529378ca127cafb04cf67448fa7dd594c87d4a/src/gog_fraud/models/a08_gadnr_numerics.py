"""A08-only stable evaluation of the existing detached GADNR KL term.

PyGOD 1.1.0 builds identity-regularized covariance matrices in float32,
then takes a determinant ratio. Its two-sample covariance is rank one;
roundoff can produce a negative determinant for a mathematically SPD matrix.
The determinant lemma and Sherman–Morrison identity evaluate the SAME
implemented term in float64, without clipping, extra jitter or changing its
log-ratio orientation. Detachment, output dtype and all sampling remain as
upstream. This is not a correction of the upstream Gaussian KL convention.
Historical shared GADNR is unchanged and does not import this module.
"""
import torch
from gog_fraud.models.pygod.gadnr import GADNR


def stable_kl_neighbor_loss(predictions, targets, mask_len, device):
    width = predictions.shape[-1]
    x1 = predictions.detach().cpu().reshape(-1, width)[:mask_len].double()
    x2 = targets.detach().cpu().reshape(-1, width)[:mask_len].double()
    if len(x1) != len(x2) or len(x1) not in (1, 2):
        raise ValueError('A08 stable KL requires the frozen one/two valid samples')
    if not torch.isfinite(x1).all() or not torch.isfinite(x2).all():
        raise FloatingPointError('nonfinite GADNR neighborhood samples')
    mean1, mean2 = x1.mean(0), x2.mean(0)
    diff = mean2 - mean1
    if len(x1) == 1:
        loss = 0.5 * diff.square().sum()  # Both covariance matrices are I.
    else:
        # Two-sample unbiased covariance = u u^T, u=(x[0]-x[1])/sqrt(2).
        u = (x1[0] - x1[1]) / (2.0 ** 0.5)
        v = (x2[0] - x2[1]) / (2.0 ** 0.5)
        uu, vv = u.dot(u), v.dot(v)
        denominator = 1.0 + vv
        trace_minus_width = uu - vv / denominator - u.dot(v).square() / denominator
        mean_quadratic = diff.dot(diff) - v.dot(diff).square() / denominator
        loss = 0.5 * (torch.log1p(uu) - torch.log1p(vv) + trace_minus_width + mean_quadratic)
    result = loss.to(device=device, dtype=predictions.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite GADNR stable KL value; no clipping fallback')
    return result


class A08NumericallyStableGADNR(GADNR):
    """Keep historical detector semantics isolated; override only detached KL."""
    def init_model(self, **kwargs):
        model = super().init_model(**kwargs)
        if model.full_batch or model.neigh_loss != 'KL' or model.sample_size != 2:
            raise ValueError('A08 numerical amendment covers sampled KL with sample_size=2 only')
        model.neighbor_loss = stable_kl_neighbor_loss
        return model
