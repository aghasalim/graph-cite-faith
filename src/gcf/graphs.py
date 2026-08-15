"""BA-Shapes-style graphs with planted motifs and exact ground truth.

A Barabasi-Albert base graph with `house` and `cycle` motifs attached at random
points. A node's class is the motif it belongs to (or none), so the causally
relevant subgraph for any classified node is known exactly -- which is the whole
reason for using synthetic graphs. On a real citation network there is no
ground-truth "reason", so a faithfulness claim would have nothing to check
against.

Two motif types rather than the usual one, because the experiment needs a
*decoy*: to test whether a narrative tracks the subgraph it was given, you have
to be able to hand it a different, equally plausible structure and see whether
the story changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

HOUSE = "house"
CYCLE = "cycle"
NONE = "none"
MOTIFS = [NONE, HOUSE, CYCLE]


@dataclass
class Dataset:
    graph: nx.Graph
    x: np.ndarray            # (N, F) node features
    y: np.ndarray            # (N,) class index into MOTIFS
    motif_nodes: dict[int, list[int]]   # node -> the motif instance it belongs to

    @property
    def n(self) -> int:
        return self.graph.number_of_nodes()


def _house(g: nx.Graph, base: int) -> list[int]:
    """5-node house: a square with a roof. Classic BA-Shapes motif."""
    ns = [g.number_of_nodes() + i for i in range(5)]
    g.add_nodes_from(ns)
    a, b, c, d, e = ns
    g.add_edges_from([(a, b), (b, c), (c, d), (d, a),  # square
                      (a, e), (b, e)])                 # roof
    g.add_edge(base, a)
    return ns


def _cycle(g: nx.Graph, base: int) -> list[int]:
    """6-node ring. Same order of size as the house, different topology --
    so a model cannot separate them by node count alone."""
    ns = [g.number_of_nodes() + i for i in range(6)]
    g.add_nodes_from(ns)
    for i in range(6):
        g.add_edge(ns[i], ns[(i + 1) % 6])
    g.add_edge(base, ns[0])
    return ns


def make(n_base: int = 300, n_each: int = 40, n_feat: int = 8,
         seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    g = nx.barabasi_albert_graph(n_base, 3, seed=seed)
    y = {i: 0 for i in g.nodes}
    motif_nodes: dict[int, list[int]] = {}

    for kind, fn, cls in ((HOUSE, _house, 1), (CYCLE, _cycle, 2)):
        for _ in range(n_each):
            base = int(rng.integers(0, n_base))
            ns = fn(g, base)
            for v in ns:
                y[v] = cls
                motif_nodes[v] = ns

    # Uninformative features on purpose. If features carried the signal the GNN
    # could ignore structure entirely, and this project is about whether a
    # narrative tracks *structural* evidence.
    x = rng.normal(0, 1, (g.number_of_nodes(), n_feat)).astype(np.float32)
    yy = np.array([y[i] for i in range(g.number_of_nodes())], dtype=np.int64)
    return Dataset(g, x, yy, motif_nodes)


def adjacency(g: nx.Graph) -> np.ndarray:
    """Symmetric-normalised adjacency with self-loops (the GCN propagation rule)."""
    a = nx.to_numpy_array(g, nodelist=range(g.number_of_nodes())).astype(np.float32)
    a += np.eye(len(a), dtype=np.float32)
    d = a.sum(1)
    dinv = np.diag((d ** -0.5).astype(np.float32))
    return dinv @ a @ dinv


def khop(g: nx.Graph, node: int, k: int = 3) -> list[int]:
    return sorted(nx.single_source_shortest_path_length(g, node, cutoff=k))


def describe_subgraph(g: nx.Graph, nodes: list[int]) -> str:
    """Edge list rendering handed to the LLM. Deliberately raw: no motif name,
    no hint. If the narrative names a motif it has to have read the structure."""
    sub = g.subgraph(nodes)
    edges = sorted(tuple(sorted(e)) for e in sub.edges())
    return "; ".join(f"{u}-{v}" for u, v in edges)
