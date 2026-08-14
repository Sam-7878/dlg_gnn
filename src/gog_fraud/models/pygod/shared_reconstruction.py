"""Shared full-graph PyGOD-compatible detectors for SCI Round 4A.

These classes deliberately bypass ``NeighborLoader``: one model is trained on
one original sparse graph and every node score lives in the same embedding and
score space.  Only adjacency *reconstruction rows* may be chunked.
"""
from __future__ import annotations

import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from pygod.detector import AnomalyDAE as PyGODAnomalyDAE
from torch_geometric.nn import GCN
from torch.utils.checkpoint import checkpoint

from .dlg import DLG
from .dlg_base import DLGBase
from .dlg_full import DLGFull
from .dlg_full_base import DLGFullBase
from .exact_reconstruction import (
    BackendName,
    exact_double_reconstruction_score,
    iter_row_chunks,
    resolve_backend,
)
from .stable_reconstruction import CONAD, DOMINANT, StableDOMINANTBase


class CheckpointGCN(GCN):
    """GCN with activation recomputation and unchanged full-graph semantics."""

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(x, edge_index, edge_weight=edge_weight, **kwargs)
        if getattr(self, "jk_mode", None) is not None:
            raise NotImplementedError("CheckpointGCN currently requires jk=None")
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            final = index == self.num_layers - 1

            def layer(value, edges, conv=conv, norm=norm, final=final):
                if self.supports_edge_weight:
                    value = conv(value, edges, edge_weight=edge_weight)
                else:
                    value = conv(value, edges)
                if not final:
                    if self.act is not None and self.act_first:
                        value = self.act(value)
                    value = norm(value)
                    if self.act is not None and not self.act_first:
                        value = self.act(value)
                    value = self.dropout(value)
                return value

            x = checkpoint(layer, x, edge_index, use_reentrant=False)
        if hasattr(self, "lin"):
            x = self.lin(x)
        return x


class ExactDOMINANTBase(StableDOMINANTBase):
    """DOMINANT network returning its structure latent instead of ``N x N``."""

    def forward(self, x, edge_index):
        self.emb = self.shared_encoder(x, edge_index)
        x_hat = self.attr_decoder(self.emb, edge_index)
        z_structure = self.struct_decoder.nn(self.emb, edge_index)
        return x_hat, z_structure


class ExactDLGBase(DLGBase):
    def forward(self, x, edge_index):
        z_local = self.local_encoder(x, edge_index)
        z_global = self.global_encoder(z_local, edge_index)
        alpha = torch.sigmoid(self.alpha)
        z = alpha * z_local + (1.0 - alpha) * z_global
        self.emb = z.detach()
        return self.attr_decoder(z, edge_index), z


class ExactDLGFullBase(DLGFullBase):
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        self.emb = z.detach()
        return self.attr_decoder(z, edge_index), z


class ExactAnomalyDAEBase(torch.nn.Module):
    """Thin adapter around AnomalyDAEBase that omits the sigmoid Gram matrix."""

    def __init__(self, dense_model):
        super().__init__()
        self.dense_model = dense_model

    @property
    def emb(self):
        return self.dense_model.emb

    def forward(self, x, edge_index):
        model = self.dense_model
        h = model.dense_stru(x)
        if model.act is not None:
            h = model.act(h)
        h = F.dropout(h, model.dropout, training=model.training)
        model.emb = model.gat_layer(h, edge_index)

        attr = model.dense_attr_1(x[:model.num_nodes].T)
        if model.act is not None:
            attr = model.act(attr)
        attr = F.dropout(attr, model.dropout, training=model.training)
        attr = model.dense_attr_2(attr)
        attr = F.dropout(attr, model.dropout, training=model.training)
        return model.emb @ attr.T, model.emb


class SharedFullGraphMixin:
    """Exact full-batch training loop shared by reconstruction detectors."""

    reconstruction_backend: BackendName
    score_chunk_size: int
    positive_weight_attribute: float = 0.5
    positive_weight_structure: float = 0.5
    sigmoid_structure: bool = False

    def _init_exact(self, reconstruction_backend: BackendName, score_chunk_size: int):
        backend = resolve_backend(reconstruction_backend, score_chunk_size=score_chunk_size)
        if backend.name == "dense_reference":
            raise ValueError("shared full-graph classes require exact_sparse or chunked_exact")
        self.reconstruction_backend = backend.name
        self.score_chunk_size = backend.score_chunk_size
        self.reconstruction_metadata = backend.metadata

    def process_graph(self, data):
        # edge_index is already the sparse full-graph target and message graph.
        data._reconstruction_backend = self.reconstruction_backend

    def _targets(self, data):
        return data.x.to(self.device), data.x.to(self.device)

    def _forward_components(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        x_hat, z_structure = self.model(x, edge_index)
        x_target, _ = self._targets(data)
        return x_target, x_hat, z_structure, edge_index

    def _score_rows(self, components, rows):
        x, x_hat, z_structure, edge_index = components
        return exact_double_reconstruction_score(
            x, x_hat, z_structure, edge_index,
            weight=self.weight,
            positive_weight_attribute=self.positive_weight_attribute,
            positive_weight_structure=self.positive_weight_structure,
            sigmoid_structure=self.sigmoid_structure,
            rows=rows,
            backend=self.reconstruction_backend,
            chunk_size=self.score_chunk_size,
        )

    def _training_extra_loss(self, data, components):
        return None

    def _backward_exact(self, data, components, extra_loss=None):
        n = data.num_nodes
        chunks = list(iter_row_chunks(n, self.score_chunk_size, self.device))
        scores = torch.empty(n, device="cpu")
        for index, rows in enumerate(chunks):
            chunk_score = self._score_rows(components, rows)
            scores[rows.cpu()] = chunk_score.detach().cpu()
            loss = chunk_score.sum() / n
            if extra_loss is not None:
                # Add a graph-global auxiliary term exactly once.
                if index == len(chunks) - 1:
                    loss = loss + extra_loss
            loss.backward(retain_graph=index < len(chunks) - 1)
        return scores

    def fit(self, data, label=None):
        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        self.batch_size = self.num_nodes
        self.model = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.decision_score_ = torch.zeros(self.num_nodes)
        self.loss_history_ = []
        self.model.train()
        for _ in range(self.epoch):
            optimizer.zero_grad(set_to_none=True)
            components = self._forward_components(data)
            extra = self._training_extra_loss(data, components)
            scores = self._backward_exact(data, components, extra)
            optimizer.step()
            self.decision_score_ = scores
            self.loss_history_.append(float(scores.mean()))
        self._process_decision_score()
        return self

    @torch.no_grad()
    def decision_function(self, data, label=None):
        self.process_graph(data)
        self.model.eval()
        components = self._forward_components(data)
        scores = []
        for rows in iter_row_chunks(data.num_nodes, self.score_chunk_size, self.device):
            scores.append(self._score_rows(components, rows).cpu())
        result = torch.cat(scores) if scores else torch.empty(0)
        if self.save_emb:
            emb = getattr(self.model, "emb", None)
            self.emb = emb.detach().cpu() if emb is not None else None
        return result

    def backend_metadata(self) -> dict[str, object]:
        return dict(self.reconstruction_metadata)


class SharedDOMINANT(SharedFullGraphMixin, DOMINANT):
    def __init__(self, *args, reconstruction_backend="exact_sparse", score_chunk_size=8192,
                 gradient_checkpointing=True, **kwargs):
        if gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            kwargs.setdefault("cached", True)
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self.reconstruction_metadata["gradient_checkpointing"] = bool(gradient_checkpointing)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return ExactDOMINANTBase(
            in_dim=self.in_dim, hid_dim=self.hid_dim, num_layers=self.num_layers,
            dropout=self.dropout, act=self.act, sigmoid_s=False,
            backbone=self.backbone, **kwargs,
        ).to(self.device)


class SharedDLGBase(SharedFullGraphMixin, DLG):
    def __init__(self, *args, reconstruction_backend="exact_sparse", score_chunk_size=8192,
                 gradient_checkpointing=True, **kwargs):
        if gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            kwargs.setdefault("cached", True)
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self.reconstruction_metadata["gradient_checkpointing"] = bool(gradient_checkpointing)

    def init_model(self, **kwargs):
        kwargs.pop("subgraph_batch_size", None)
        return ExactDLGBase(
            in_dim=self.in_dim, hid_dim=self.hid_dim, num_layers=self.num_layers,
            dropout=self.dropout, act=self.act, alpha=self.alpha,
            sigmoid_s=False, backbone=self.backbone, **kwargs,
        ).to(self.device)


class SharedDLGFull(SharedFullGraphMixin, DLGFull):
    def __init__(self, *args, reconstruction_backend="exact_sparse", score_chunk_size=8192,
                 gradient_checkpointing=True, **kwargs):
        if gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            kwargs.setdefault("cached", True)
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self.reconstruction_metadata["gradient_checkpointing"] = bool(gradient_checkpointing)

    def process_graph(self, data):
        if hasattr(data, "_dlg_full_augmented") and data._dlg_full_augmented:
            data._reconstruction_backend = self.reconstruction_backend
            return
        self._orig_dim = data.x.size(1)
        data.dlg_original_x = data.x.clone()
        l1_embs = self._pretrain_level1(data)
        data.x = torch.cat([data.x, l1_embs.to(data.x.device)], dim=-1)
        data._dlg_full_augmented = True
        data._reconstruction_backend = self.reconstruction_backend

    def _targets(self, data):
        return data.dlg_original_x.to(self.device), data.x.to(self.device)

    def init_model(self, **kwargs):
        return ExactDLGFullBase(
            in_dim=self.in_dim, orig_dim=self._orig_dim, hid_dim=self.hid_dim,
            num_layers=self.num_layers, dropout=self.dropout, act=self.act,
            sigmoid_s=False, backbone=self.backbone, **kwargs,
        ).to(self.device)


class SharedAnomalyDAE(SharedFullGraphMixin, PyGODAnomalyDAE):
    def __init__(self, *args, reconstruction_backend="chunked_exact", score_chunk_size=512,
                 gradient_checkpointing=False, **kwargs):
        if gradient_checkpointing:
            raise ValueError("AnomalyDAE does not use the checkpointable GCN backbone")
        super().__init__(*args, **kwargs)
        self.weight = 1.0 - self.alpha
        self.positive_weight_attribute = self.eta / (1.0 + self.eta)
        self.positive_weight_structure = self.theta / (1.0 + self.theta)
        self.sigmoid_structure = True
        self._init_exact(reconstruction_backend, score_chunk_size)
        self.reconstruction_metadata["gradient_checkpointing"] = False

    def init_model(self, **kwargs):
        dense_model = super().init_model(**kwargs)
        return ExactAnomalyDAEBase(dense_model).to(self.device)


class SharedCONAD(SharedFullGraphMixin, CONAD):
    def __init__(self, *args, reconstruction_backend="exact_sparse", score_chunk_size=8192,
                 gradient_checkpointing=True, **kwargs):
        if gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            # CONAD alternates an augmented and the original graph in one
            # iteration; caching normalized adjacency would mix graph states.
            kwargs["cached"] = False
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self.reconstruction_metadata["gradient_checkpointing"] = bool(gradient_checkpointing)
        self._contrastive_extra = None

    def init_model(self, **kwargs):
        return ExactDOMINANTBase(
            in_dim=self.in_dim, hid_dim=self.hid_dim, num_layers=self.num_layers,
            dropout=self.dropout, act=self.act, sigmoid_s=False,
            backbone=self.backbone, **kwargs,
        ).to(self.device)

    def _sparse_data_augmentation(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        n = data.num_nodes
        prob = torch.rand(n, device=self.device)
        label = (prob < self.r).to(torch.int32)
        high = prob < self.r / 4
        outlying = (self.r / 4 <= prob) & (prob < self.r / 2)
        replace = high | outlying
        keep = ~replace[edge_index[0]]
        pieces = [edge_index[:, keep]]
        high_nodes = torch.nonzero(high, as_tuple=False).flatten()
        # Same independent Bernoulli(m/N) edge distribution as the dense code,
        # generated in bounded row blocks.
        probability = min(1.0, self.m / max(1, n))
        # Geometric skipping samples the exact iid Bernoulli row distribution
        # without drawing or storing ``high_degree_rows x N`` uniforms.
        for source in high_nodes.tolist():
            destinations = []
            position = -1
            while True:
                failures = int(torch.distributions.Geometric(
                    torch.tensor(probability, device=self.device)
                ).sample().item())
                position += failures + 1
                if position >= n:
                    break
                destinations.append(position)
            if destinations:
                dst = torch.tensor(destinations, device=self.device, dtype=torch.long)
                src = torch.full_like(dst, source)
                pieces.append(torch.stack((src, dst)))
        edge_aug = torch.cat(pieces, dim=1)
        x_aug = x.clone()
        deviated = (self.r / 2 <= prob) & (prob < self.r * 3 / 4)
        if deviated.any():
            candidates = x_aug[torch.randperm(n, device=self.device)[:min(self.k, n)]]
            distance = torch.cdist(x_aug[deviated], candidates)
            x_aug[deviated] = candidates[torch.argmax(distance, dim=1)]
        multiply = (self.r * 3 / 4 <= prob) & (prob < self.r * 7 / 8)
        divide = prob >= self.r * 7 / 8
        x_aug[multiply] *= self.f
        x_aug[divide] /= self.f
        return x_aug, edge_aug, label

    def _forward_components(self, data):
        if self.model.training:
            x_aug, edge_aug, labels = self._sparse_data_augmentation(data)
            self.model(x_aug, edge_aug)
            h_aug = self.model.emb
        components = super()._forward_components(data)
        if self.model.training:
            h = self.model.emb
            margin = self.margin_loss_func(h, h, h_aug) * labels
            self._contrastive_extra = (1.0 - self.eta) * margin.mean()
        return components

    def _score_rows(self, components, rows):
        # CONAD weights reconstruction by eta before adding contrastive loss.
        return self.eta * super()._score_rows(components, rows)

    def _training_extra_loss(self, data, components):
        return self._contrastive_extra
