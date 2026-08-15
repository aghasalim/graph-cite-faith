"""Tests for the instrument. Two of these encode bugs that actually shipped and
would have produced a confident, entirely fake result."""
import networkx as nx
import numpy as np
import torch

from src.gcf import gnn, graphs, narrate


def test_planted_motifs_have_exact_ground_truth():
    ds = graphs.make(n_base=60, n_each=5, seed=1)
    for node, members in ds.motif_nodes.items():
        assert ds.y[node] in (1, 2)
        assert node in members


def test_house_and_cycle_are_topologically_distinct():
    """If the two motifs were isomorphic the whole decoy design collapses."""
    ds = graphs.make(n_base=40, n_each=3, seed=2)
    houses = [m for n, m in ds.motif_nodes.items() if ds.y[n] == 1]
    cycles = [m for n, m in ds.motif_nodes.items() if ds.y[n] == 2]
    h = ds.graph.subgraph(houses[0])
    c = ds.graph.subgraph(cycles[0])
    assert sum(nx.triangles(h).values()) > 0      # house has a roof
    assert sum(nx.triangles(c).values()) == 0     # ring has no chords


def test_features_carry_no_class_signal():
    """Features are noise on purpose: if they leaked the label the GNN could
    ignore structure and every structural claim downstream would be void."""
    ds = graphs.make(n_base=120, n_each=15, seed=3)
    for cls in (0, 1, 2):
        m = ds.x[ds.y == cls].mean(0)
        assert np.abs(m).max() < 0.6


def test_gnn_learns_structure():
    ds = graphs.make(seed=0)
    a = torch.from_numpy(graphs.adjacency(ds.graph))
    x, y = torch.from_numpy(ds.x), torch.from_numpy(ds.y)
    # Random, not contiguous. Node ids are assigned in construction order --
    # base graph, then every house, then every cycle -- so a contiguous 60%
    # split puts zero cycle nodes in training and accuracy collapses to 9%.
    # This is not hypothetical; it is what the first version of this test did.
    idx = np.random.default_rng(0).permutation(ds.n)
    tr = torch.zeros(ds.n, dtype=torch.bool)
    tr[idx[: int(0.6 * ds.n)]] = True
    m = gnn.train(x, a, y, tr, epochs=150, seed=0)
    assert gnn.accuracy(m, x, a, y, ~tr) > 0.7


def test_contiguous_split_is_degenerate():
    """Guards the trap above: if node ordering ever stops correlating with
    class, the warning in the sibling test becomes misleading and should go."""
    ds = graphs.make(seed=0)
    cut = int(0.6 * ds.n)
    assert len(set(ds.y[:cut].tolist())) < 3, "ordering no longer class-degenerate"


def test_parse_extracts_both_commitments():
    motif, nodes = narrate.parse("blah\nMOTIF: house\nNODES: 3, 4, 5")
    assert motif == "house" and nodes == [3, 4, 5]


def test_parse_survives_a_missing_block():
    """An empty reply must not silently score as a valid answer."""
    motif, nodes = narrate.parse("no structured tail here")
    assert motif == "" and nodes == []


def test_prompt_does_not_leak_the_answer():
    """The class names shown to the model are deliberately uninformative. If the
    prompt said 'house-class', naming the motif would prove nothing."""
    assert "motif-A" not in narrate.PROMPT and "house-class" not in narrate.PROMPT
