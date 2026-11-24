"""Compact re-implementation of the original CoLA design (Liu et al., IEEE TNNLS 2021),
used only as an A29-04 sanity reference for the PyGOD 1.1.0 `CoLA` class.

Follows the authors' public code (GRAND-Lab/CoLA):
  * row-normalised features; symmetric-normalised adjacency with self-loops;
  * per target: random walk with restart (restart prob 1.0, 12 steps; retry with
    0.9 and 20 steps if fewer than 3 distinct context nodes; pad by repetition);
  * subgraph = 3 context nodes + anonymised target (zero features), then an extra
    row that aggregates all five positions with the target's own features;
  * 1-layer GCN (64 units, PReLU), average readout over the first four positions,
    bilinear discriminator, one negative pair per node (readout shifted in the batch);
  * BCE loss, Adam lr 1e-3, batch 300, 100 epochs; anomaly score = mean over 256
    rounds of (negative logit - positive logit).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class _Model(nn.Module):
    def __init__(self, n_in, n_h=64):
        super().__init__()
        self.fc = nn.Linear(n_in, n_h, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_h))
        self.act = nn.PReLU()
        self.disc = nn.Bilinear(n_h, n_h, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.disc.weight)
        nn.init.zeros_(self.disc.bias)

    def forward(self, feats, adj):  # feats [B,5,F], adj [B,5,5]
        h = self.act(torch.bmm(adj, self.fc(feats)) + self.bias)
        c = h[:, :-1, :].mean(1)
        hm = h[:, -1, :]
        c_neg = torch.cat((c[-2:-1], c[:-1]), 0)
        return self.disc(c, hm).squeeze(-1), self.disc(c_neg, hm).squeeze(-1)


class CoLAReference:
    def __init__(self, hid=64, epoch=100, lr=1e-3, batch=300, rounds=256, size=4, device='cuda', seed=42):
        self.hid, self.epoch, self.lr, self.batch, self.rounds, self.size = hid, epoch, lr, batch, rounds, size
        self.device, self.seed = device, seed
        self.loss_history_ = []

    def _prepare(self, x, edge_index, n):
        x = np.asarray(x, np.float64)
        rs = x.sum(1, keepdims=True); rs[rs == 0] = 1.
        self.x = torch.tensor(x / rs, dtype=torch.float32, device=self.device)
        a = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n)).tocsr()
        a = ((a + a.T) > 0).astype(np.float64)
        a.setdiag(0); a.eliminate_zeros()
        self.indptr, self.indices = a.indptr.astype(np.int64), a.indices.astype(np.int64)
        self.deg = np.diff(self.indptr)
        ah = a + sp.eye(n)
        d = np.asarray(ah.sum(1)).ravel() ** -0.5
        self.norm = (sp.diags(d) @ ah @ sp.diags(d)).tocsr()
        self.n = n

    def _walk(self, targets, restart, steps):
        cur = targets.copy()
        out = np.empty((len(targets), steps), np.int64)
        for t in range(steps):
            back = self.rng.random(len(targets)) < restart
            src = np.where(back, targets, cur)
            dg = self.deg[src]
            pick = self.indptr[src] + np.floor(self.rng.random(len(targets)) * np.maximum(dg, 1)).astype(np.int64)
            nxt = np.where(dg > 0, self.indices[np.minimum(pick, len(self.indices) - 1)], targets)
            out[:, t] = nxt
            cur = nxt
        return out

    def _context(self, targets):
        k = self.size - 1
        ctx = np.full((len(targets), k), -1, np.int64)
        todo = np.arange(len(targets))
        for restart, steps in ((1.0, 3 * self.size), (0.9, 5 * self.size), (0.9, 5 * self.size)):
            if len(todo) == 0:
                break
            v = self._walk(targets[todo], restart, steps)
            first = (v != targets[todo, None])
            eq = v[:, :, None] == v[:, None, :]
            earlier = np.tril(np.ones((steps, steps), bool), -1)
            first &= ~(eq & earlier[None]).any(2)
            rank = np.cumsum(first, 1)
            got = np.minimum(rank[:, -1], k)
            for j in range(k):
                sel = first & (rank == j + 1)
                has = sel.any(1)
                ctx[todo[has], j] = v[has][sel[has]].reshape(-1) if has.any() else ctx[todo[has], j]
            todo = todo[got < k]
        # pad short contexts by repeating what was found (target itself if nothing)
        for i in np.flatnonzero((ctx < 0).any(1)):
            found = ctx[i][ctx[i] >= 0]
            fill = found if len(found) else np.array([targets[i]])
            ctx[i] = np.resize(fill, k)
        return ctx

    def _batch(self, targets):
        ctx = self._context(targets)
        nodes = np.concatenate([ctx, targets[:, None]], 1)  # [B,4], target last
        b = len(targets)
        r = np.repeat(nodes, self.size, 1).ravel(); c = np.tile(nodes, (1, self.size)).ravel()
        sub = np.asarray(self.norm[r, c]).reshape(b, self.size, self.size)
        adj = np.zeros((b, self.size + 1, self.size + 1), np.float32)
        adj[:, :self.size, :self.size] = sub
        adj[:, -1, :] = 1.
        nt = torch.as_tensor(nodes, device=self.device)
        f = self.x[nt]                                  # [B,4,F]
        feats = torch.cat([f[:, :-1], torch.zeros_like(f[:, -1:]), f[:, -1:]], 1)  # anonymised target + raw target row
        return feats, torch.as_tensor(adj, device=self.device)

    def fit(self, x, edge_index, n):
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self._prepare(x, edge_index, n)
        self.model = _Model(self.x.shape[1], self.hid).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.epoch):
            perm = self.rng.permutation(n)
            tot = 0.
            self.model.train()
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                feats, adj = self._batch(idx)
                pos, neg = self.model(feats, adj)
                logits = torch.cat([pos, neg]); lab = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
                loss = bce(logits, lab)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx)
            self.loss_history_.append(tot / n)
        return self

    @torch.no_grad()
    def decision_function(self):
        self.model.eval()
        score = np.zeros(self.n)
        for _ in range(self.rounds):
            perm = self.rng.permutation(self.n)
            for s in range(0, self.n, self.batch):
                idx = perm[s:s + self.batch]
                feats, adj = self._batch(idx)
                pos, neg = self.model(feats, adj)
                score[idx] += (neg - pos).cpu().numpy()
        return score / self.rounds
