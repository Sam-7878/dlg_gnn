"""Numerically stable, mathematically equivalent SCI wrappers for PyGOD reconstruction models."""
from __future__ import annotations

import torch
from pygod.detector import CONAD as PyGODCONAD
from pygod.detector import DOMINANT as PyGODDOMINANT
from pygod.nn import DOMINANTBase


def stable_reconstruction_score(x, x_hat, s, s_hat, weight: float = 0.5):
    """Same Euclidean reconstruction score with a defined zero-residual gradient."""
    attr_error = torch.linalg.vector_norm(x - x_hat, ord=2, dim=1)
    struct_error = torch.linalg.vector_norm(s - s_hat, ord=2, dim=1)
    return weight * attr_error + (1.0 - weight) * struct_error


class StableDOMINANTBase(DOMINANTBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PyGOD assigns double_recon_loss as an instance attribute in __init__,
        # so a class-level override alone is ineffective.
        self.loss_func = stable_reconstruction_score


class DOMINANT(PyGODDOMINANT):
    """DOMINANT with stable Euclidean-norm backward at exact reconstruction."""

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return StableDOMINANTBase(in_dim=self.in_dim, hid_dim=self.hid_dim,
                                  num_layers=self.num_layers, dropout=self.dropout,
                                  act=self.act, sigmoid_s=self.sigmoid_s,
                                  backbone=self.backbone, **kwargs).to(self.device)


class CONAD(PyGODCONAD):
    """CONAD using the same stable reconstruction base as SCI DOMINANT."""

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return StableDOMINANTBase(in_dim=self.in_dim, hid_dim=self.hid_dim,
                                  num_layers=self.num_layers, dropout=self.dropout,
                                  act=self.act, sigmoid_s=self.sigmoid_s,
                                  backbone=self.backbone, **kwargs).to(self.device)

    def forward_model(self, data):
        edge_index = data.edge_index
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= data.x.size(0)):
            raise IndexError("CONAD sampled edge_index is outside the local node range")
        if hasattr(data, "n_id") and data.n_id.numel() != data.x.size(0):
            raise IndexError("CONAD n_id and sampled feature rows are misaligned")
        return super().forward_model(data)
