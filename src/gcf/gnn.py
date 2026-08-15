"""A small GCN plus a GNNExplainer-style edge mask.

Written against dense adjacency rather than torch-geometric. The graphs here are
~800 nodes, so dense is fast, and it removes a heavy dependency whose install
story is the most common reason a GNN repo does not run on someone else's
machine. The propagation rule is the standard one; nothing is approximated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, n_feat: int, n_hidden: int = 32, n_class: int = 3):
        super().__init__()
        self.l1 = nn.Linear(n_feat, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_class)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = F.relu(a @ self.l1(x))
        h = F.relu(a @ self.l2(h))
        return self.out(a @ h)


def train(x, a, y, train_mask, epochs: int = 300, seed: int = 0, lr: float = 0.01):
    torch.manual_seed(seed)
    m = GCN(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = F.cross_entropy(m(x, a)[train_mask], y[train_mask])
        loss.backward()
        opt.step()
    m.train(False)
    return m


@torch.no_grad()
def accuracy(m, x, a, y, mask) -> float:
    return float((m(x, a)[mask].argmax(1) == y[mask]).float().mean())


def explain_edges(m: GCN, x: torch.Tensor, a: torch.Tensor, node: int,
                  epochs: int = 200, seed: int = 0) -> torch.Tensor:
    """GNNExplainer: learn a soft mask over adjacency that preserves the
    prediction for `node` while being sparse.

    Same objective as the paper -- maximise the predicted-class log-probability
    under the masked graph, penalise mask mass and entropy. Optimising the dense
    adjacency directly is equivalent to an edge mask here because the graph is
    dense-represented.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        target = int(m(x, a)[node].argmax())
    mask = torch.nn.Parameter(torch.randn_like(a) * 0.1)
    opt = torch.optim.Adam([mask], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        s = torch.sigmoid(mask)
        s = (s + s.T) / 2                       # keep it undirected
        logits = m(x, a * s)
        loss = -F.log_softmax(logits[node], dim=-1)[target]
        loss = loss + 0.005 * s.sum() + 0.1 * (-s * torch.log(s + 1e-9)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = torch.sigmoid(mask)
        return ((s + s.T) / 2)


def top_edges(mask: torch.Tensor, candidates: list[int], k: int = 8
              ) -> list[tuple[int, int]]:
    """The k highest-scoring edges among a candidate node set."""
    scores = []
    for i, u in enumerate(candidates):
        for v in candidates[i + 1:]:
            scores.append((float(mask[u, v]), u, v))
    scores.sort(reverse=True)
    return [(u, v) for _, u, v in scores[:k]]
