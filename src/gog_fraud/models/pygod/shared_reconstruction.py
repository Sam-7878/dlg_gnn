"""Shared full-graph PyGOD-compatible detectors for SCI Round 4A.

These classes deliberately bypass ``NeighborLoader``: one model is trained on
one original sparse graph and every node score lives in the same embedding and
score space.  Only adjacency *reconstruction rows* may be chunked.
"""
from __future__ import annotations

import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
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
from .sparse_message import (
    MessageBackendName,
    MessageGraphCache,
    SparseFusedGCN,
    normalized_sparse_adjt,
    resolve_message_backend,
)


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

    def _init_message(self, message_backend: MessageBackendName):
        backend = resolve_message_backend(message_backend)
        self.message_backend = backend.name
        self.message_metadata = backend.metadata
        self._message_cache = MessageGraphCache()
        if hasattr(self, "reconstruction_metadata"):
            self.reconstruction_metadata.update(self.message_metadata)

    def _message_graph(self, data, dtype, *, edge_index=None, cache=True):
        edge_index = data.edge_index if edge_index is None else edge_index
        if self.message_backend == "pyg_coo_reference":
            return edge_index.to(self.device)
        edge_weight = getattr(data, "edge_weight", None) if edge_index is data.edge_index else None
        if cache:
            return self._message_cache.get(
                edge_index, data.num_nodes, edge_weight=edge_weight,
                dtype=dtype, device=self.device,
            )
        return normalized_sparse_adjt(
            edge_index, data.num_nodes, edge_weight=edge_weight,
            dtype=dtype, device=self.device,
        )

    def process_graph(self, data):
        # edge_index is already the sparse full-graph target and message graph.
        data._reconstruction_backend = self.reconstruction_backend
        data._message_backend = self.message_backend

    def _targets(self, data):
        return data.x.to(self.device), data.x.to(self.device)

    def _forward_components(self, data):
        x = data.x.to(self.device)
        message_graph = self._message_graph(data, x.dtype)
        x_hat, z_structure = self.model(x, message_graph)
        x_target, _ = self._targets(data)
        return x_target, x_hat, z_structure, data.edge_index

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
        if self.reconstruction_backend == "exact_sparse" and not self.sigmoid_structure:
            rows = torch.arange(n, device=self.device)
            score = self._score_rows(components, rows)
            loss = score.mean()
            if extra_loss is not None:
                loss = loss + extra_loss
            loss.backward()
            return score.detach().cpu()
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
        start_epoch = 0
        checkpoint_path_value = getattr(self, "training_checkpoint_path", None)
        checkpoint_path = Path(checkpoint_path_value) if checkpoint_path_value else None
        if checkpoint_path is not None and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = int(state["next_epoch"])
            self.decision_score_ = state["decision_score"].cpu()
            self.loss_history_ = list(state["loss_history"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
            np.random.set_state(state["numpy_rng_state"])
            random.setstate(state["python_rng_state"])
        self.resumed_from_epoch_ = start_epoch
        self.model.train()
        for epoch_index in range(start_epoch, self.epoch):
            optimizer.zero_grad(set_to_none=True)
            components = self._forward_components(data)
            extra = self._training_extra_loss(data, components)
            scores = self._backward_exact(data, components, extra)
            optimizer.step()
            self.decision_score_ = scores
            self.loss_history_.append(float(scores.mean()))
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                torch.save({
                    "next_epoch": epoch_index + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "decision_score": self.decision_score_,
                    "loss_history": self.loss_history_,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                }, temporary)
                os.replace(temporary, checkpoint_path)
                progress_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".progress.json")
                progress_temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
                progress_temporary.write_text(
                    json.dumps({"completed_epochs": epoch_index + 1}) + "\n",
                    encoding="utf-8",
                )
                os.replace(progress_temporary, progress_path)
        self.actual_epochs_ = len(self.loss_history_)
        self._process_decision_score()
        return self

    @torch.no_grad()
    def decision_function(self, data, label=None):
        self.process_graph(data)
        self.model.eval()
        components = self._forward_components(data)
        if self.reconstruction_backend == "exact_sparse" and not self.sigmoid_structure:
            rows = torch.arange(data.num_nodes, device=self.device)
            result = self._score_rows(components, rows).cpu()
            if self.save_emb:
                emb = getattr(self.model, "emb", None)
                self.emb = emb.detach().cpu() if emb is not None else None
            return result
        scores = []
        for rows in iter_row_chunks(data.num_nodes, self.score_chunk_size, self.device):
            scores.append(self._score_rows(components, rows).cpu())
        result = torch.cat(scores) if scores else torch.empty(0)
        if self.save_emb:
            emb = getattr(self.model, "emb", None)
            self.emb = emb.detach().cpu() if emb is not None else None
        return result

    def backend_metadata(self) -> dict[str, object]:
        return {**self.reconstruction_metadata, **self.message_metadata}


class SharedDOMINANT(SharedFullGraphMixin, DOMINANT):
    def __init__(self, *args, reconstruction_backend="exact_sparse", score_chunk_size=8192,
                 gradient_checkpointing=True, message_backend="sparse_fused", **kwargs):
        if message_backend == "sparse_fused":
            kwargs["backbone"] = SparseFusedGCN
            gradient_checkpointing = False
        elif gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            kwargs.setdefault("cached", True)
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self._init_message(message_backend)
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
                 gradient_checkpointing=True, message_backend="sparse_fused", **kwargs):
        if message_backend == "sparse_fused":
            kwargs["backbone"] = SparseFusedGCN
            gradient_checkpointing = False
        elif gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            kwargs.setdefault("cached", True)
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self._init_message(message_backend)
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
                 gradient_checkpointing=True, message_backend="sparse_fused", **kwargs):
        if message_backend == "sparse_fused":
            kwargs["backbone"] = SparseFusedGCN
            gradient_checkpointing = False
        elif gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            kwargs.setdefault("cached", True)
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self._init_message(message_backend)
        self.reconstruction_metadata["gradient_checkpointing"] = bool(gradient_checkpointing)

    def _pretrain_level1(self, data):
        """Full-graph local pretraining using the selected message backend."""
        in_dim = data.x.size(1)
        backbone = SparseFusedGCN if self.message_backend == "sparse_fused" else GCN
        l1_encoder = backbone(
            in_channels=in_dim, hidden_channels=self.l1_hid_dim,
            num_layers=self.l1_hops, out_channels=self.l1_hid_dim,
        ).to(self.device)
        l1_decoder = torch.nn.Linear(self.l1_hid_dim, in_dim).to(self.device)
        x = data.x.to(self.device)
        message_graph = self._message_graph(data, x.dtype)
        optimizer = torch.optim.Adam(
            list(l1_encoder.parameters()) + list(l1_decoder.parameters()), lr=.01
        )
        l1_encoder.train()
        for _ in range(self.l1_epochs):
            optimizer.zero_grad(set_to_none=True)
            z = l1_encoder(x, message_graph)
            x_hat = l1_decoder(z)
            loss = F.mse_loss(x_hat, x)
            loss.backward(); optimizer.step()
        l1_encoder.eval()
        with torch.no_grad():
            z = l1_encoder(x, message_graph)
            x_hat = l1_decoder(z)
            data.dlg_l1_score = (x_hat - x).square().mean(dim=1).cpu()
            embeddings = z.cpu()
        del l1_encoder, l1_decoder, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeddings

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
        data._message_backend = self.message_backend

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
        self._init_message("pyg_coo_reference")
        self.reconstruction_metadata["gradient_checkpointing"] = False

    def init_model(self, **kwargs):
        dense_model = super().init_model(**kwargs)
        return ExactAnomalyDAEBase(dense_model).to(self.device)


class SharedCONAD(SharedFullGraphMixin, CONAD):
    def __init__(self, *args, reconstruction_backend="exact_sparse", score_chunk_size=8192,
                 gradient_checkpointing=True, message_backend="sparse_fused", **kwargs):
        if message_backend == "sparse_fused":
            kwargs["backbone"] = SparseFusedGCN
            gradient_checkpointing = False
        elif gradient_checkpointing:
            kwargs["backbone"] = CheckpointGCN
            # CONAD alternates an augmented and the original graph in one
            # iteration; caching normalized adjacency would mix graph states.
            kwargs["cached"] = False
        super().__init__(*args, **kwargs)
        self.sigmoid_structure = bool(self.sigmoid_s)
        self._init_exact(reconstruction_backend, score_chunk_size)
        self._init_message(message_backend)
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
        # Geometric skipping samples the exact iid Bernoulli row process.  It
        # is vectorized in blocks, so expected work is O(high_rows * m), not
        # O(high_rows * N), while incomplete rare rows are extended exactly.
        block_rows, width = 4096, max(64, int(self.m * 2 + 32))
        for sources in high_nodes.split(block_rows):
            active_sources = sources
            offsets = torch.full((sources.numel(),), -1, device=self.device, dtype=torch.long)
            while active_sources.numel():
                trials = torch.empty(
                    (active_sources.numel(), width), device=self.device
                ).geometric_(probability).long()
                positions = trials.cumsum(dim=1) + offsets[:, None]
                valid = positions < n
                local, column = torch.nonzero(valid, as_tuple=True)
                if local.numel():
                    pieces.append(torch.stack((active_sources[local], positions[local, column])))
                new_offsets = positions[:, -1]
                incomplete = new_offsets < n
                active_sources = active_sources[incomplete]
                offsets = new_offsets[incomplete]
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
            augmented_graph = self._message_graph(
                data, x_aug.dtype, edge_index=edge_aug, cache=False
            )
            self.model(x_aug, augmented_graph)
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
