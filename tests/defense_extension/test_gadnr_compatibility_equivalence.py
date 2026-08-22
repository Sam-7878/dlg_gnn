"""Round D2 equivalence checks for the GADNR compatibility correction."""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from pygod.detector import GADNR as UpstreamGADNR
from pygod.nn import GADNRBase as UpstreamGADNRBase
from gog_fraud.models.pygod.gadnr import GADNR as PatchedGADNR


class UpstreamMathematicalReference(UpstreamGADNR):
    """Upstream detector with only its obsolete PyG ``tot_nodes`` kwarg removed.

    PyGOD 1.1.0 forwards that kwarg into current PyG's GCN and fails before
    training. All preprocessing, forward, loss, scoring, and update logic remain
    the installed upstream implementation.
    """

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return UpstreamGADNRBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            encoder_layers=self.encoder_layers,
            deg_dec_layers=self.deg_dec_layers,
            fea_dec_layers=self.fea_dec_layers,
            sample_size=self.sample_size,
            sample_time=self.sample_time,
            neighbor_num_list=self.neighbor_num_list,
            neigh_loss=self.neigh_loss,
            lambda_loss1=self.lambda_loss1,
            lambda_loss2=self.lambda_loss2,
            lambda_loss3=self.lambda_loss3,
            full_batch=self.full_batch,
            backbone=self.backbone,
            device=self.device,
        ).to(self.device)


def _seed(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _connected_graph() -> Data:
    _seed()
    x = torch.rand(8, 5) + 0.1
    src = torch.arange(8)
    dst = torch.roll(src, -1)
    edge_index = torch.stack([src, dst])
    y = torch.tensor([0, 0, 0, 0, 0, 1, 0, 1])
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=8)


def _detector(detector_cls):
    return detector_cls(
        hid_dim=8,
        num_layers=1,
        deg_dec_layers=2,
        fea_dec_layers=2,
        sample_size=2,
        sample_time=1,
        epoch=1,
        batch_size=0,
        gpu=-1,
        verbose=0,
    )


def test_gadnr_original_vs_patched_connected_graph_equivalence():
    base = _connected_graph()

    upstream_pre, _, upstream_degree, _ = UpstreamGADNRBase.process_graph(base.clone())
    patched = _detector(PatchedGADNR)
    patched.batch_size = base.num_nodes
    patched.device = torch.device("cpu")
    patched_pre = patched.process_graph(base.clone())
    torch.testing.assert_close(upstream_pre.x, patched_pre.x, rtol=0, atol=0)
    assert torch.equal(upstream_pre.edge_index, patched_pre.edge_index)
    assert torch.equal(upstream_degree, patched.neighbor_num_list.cpu())

    _seed()
    upstream = _detector(UpstreamMathematicalReference).fit(base.clone())
    _seed()
    corrected = _detector(PatchedGADNR).fit(base.clone())

    upstream_score = upstream.decision_score_.detach().cpu()
    corrected_score = corrected.decision_score_.detach().cpu()
    torch.testing.assert_close(upstream_score, corrected_score, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        upstream.arg_min_loss_per_node,
        corrected.arg_min_loss_per_node,
        rtol=1e-6,
        atol=1e-6,
    )

    labels = base.y.numpy()
    assert abs(roc_auc_score(labels, upstream_score) - roc_auc_score(labels, corrected_score)) <= 1e-12
    assert abs(average_precision_score(labels, upstream_score) - average_precision_score(labels, corrected_score)) <= 1e-12


def test_gadnr_isolated_node_correctness():
    _seed()
    data = Data(
        x=torch.rand(5, 4) + 0.1,
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        y=torch.tensor([0, 0, 1, 0, 1]),
        num_nodes=5,
    )
    _, _, upstream_degree, _ = UpstreamGADNRBase.process_graph(data.clone())
    assert upstream_degree.numel() == 4, "Upstream path should omit the isolated highest-index node"

    corrected = _detector(PatchedGADNR)
    corrected.batch_size = data.num_nodes
    corrected.device = torch.device("cpu")
    corrected.process_graph(data.clone())
    assert corrected.neighbor_num_list.numel() == data.num_nodes
    assert torch.isfinite(corrected.neighbor_num_list).all()

    _seed()
    fitted = _detector(PatchedGADNR).fit(data.clone())
    assert fitted.decision_score_.numel() == data.num_nodes
    assert torch.isfinite(fitted.decision_score_).all()
