"""Tests for the instrument. Two of these encode bugs that actually shipped and
would have produced a confident, entirely fake result."""
import math

import networkx as nx
import numpy as np
import pytest
import torch

from experiments import counterfactual
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


def test_parse_tolerates_markdown_around_the_commitment():
    """Shipped bug: a strict `MOTIF:\\s*(\\w+)` blanked 35-50% of three models'
    replies in a live run because they bolded the tag. A blank agrees with
    neither structure nor label, so the models read as evasive when the parser
    was simply not looking at what they wrote."""
    for text in ("**MOTIF:** cycle\n**NODES:** 1, 2, 3",
                 "MOTIF: **cycle**\nNODES: **1, 2, 3**",
                 "MOTIF:cycle\nNODES:1,2,3",
                 "motif: Cycle\nnodes: 1 2 3"):
        motif, nodes = narrate.parse(text)
        assert motif == "cycle", text
        assert nodes == [1, 2, 3], text


def test_parse_rejects_answers_outside_the_closed_set():
    """The closed set is the fix for an earlier bug that scored 40 replies
    calling a six-node ring a "hexagon" as wrong. Anything outside the set must
    be reported unparsed, not silently marked incorrect."""
    motif, _ = narrate.parse("MOTIF: hexagon\nNODES: 1,2")
    assert motif == "" and motif not in narrate.ANSWERS


def test_prompt_does_not_leak_the_answer():
    """The class names shown to the model are deliberately uninformative. If the
    prompt said 'house-class', naming the motif would prove nothing."""
    assert "motif-A" not in narrate.PROMPT and "house-class" not in narrate.PROMPT


def test_control_prompt_mentions_no_prediction():
    """The control exists to measure edge reading with nothing to
    post-rationalise from. A predicted class in it would measure the same thing
    the main condition does."""
    p = narrate.CONTROL_PROMPT
    assert "{label}" not in p
    for leak in ("predict", "class", "motif-A", "neural"):
        assert leak not in p.lower()


def test_explanations_contain_only_real_edges():
    """Shipped bug: top_edges ranked every candidate *pair*. A GNNExplainer mask
    entry for a non-edge gets no gradient and keeps its ~0.5 init, and saliency
    gives non-edges a real gradient, so phantom edges reached the prompt -- 71%
    of the saliency baseline's output. The LLM cannot be asked to read a
    structure that includes edges the graph does not have."""
    ds = graphs.make(n_base=60, n_each=5, seed=4)
    a = torch.from_numpy(graphs.adjacency(ds.graph))
    x, y = torch.from_numpy(ds.x), torch.from_numpy(ds.y)
    tr = torch.ones(ds.n, dtype=torch.bool)
    m = gnn.train(x, a, y, tr, epochs=30, seed=0)
    node = next(i for i in range(ds.n) if ds.y[i] == 1)
    cand = graphs.khop(ds.graph, node, 3)
    for mask in (gnn.explain_edges(m, x, a, node, epochs=20, seed=0),
                 gnn.saliency_edges(m, x, a, node)):
        edges = gnn.top_edges(mask, ds.graph, cand, k=8)
        assert edges
        for u, v in edges:
            assert ds.graph.has_edge(u, v), f"{u}-{v} is not an edge"


def test_saliency_and_gnnexplainer_disagree():
    """If the two explainers returned the same subgraph the contrast is a
    no-op and the extra 200 calls per model buy nothing."""
    ds = graphs.make(n_base=60, n_each=5, seed=5)
    a = torch.from_numpy(graphs.adjacency(ds.graph))
    x, y = torch.from_numpy(ds.x), torch.from_numpy(ds.y)
    tr = torch.ones(ds.n, dtype=torch.bool)
    m = gnn.train(x, a, y, tr, epochs=60, seed=0)
    diff = 0
    for node in [i for i in range(ds.n) if ds.y[i] in (1, 2)][:5]:
        cand = graphs.khop(ds.graph, node, 3)
        g = set(gnn.top_edges(gnn.explain_edges(m, x, a, node, epochs=40, seed=0),
                              ds.graph, cand, k=8))
        s = set(gnn.top_edges(gnn.saliency_edges(m, x, a, node), ds.graph, cand, k=8))
        diff += g != s
    assert diff > 0


def test_wilson_interval_brackets_and_handles_the_edges():
    """Normal approximation runs outside [0,1] at p=1, which is exactly where
    citation validity sits. Wilson must not."""
    lo, hi = counterfactual.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.192, abs=0.01)
    lo, hi = counterfactual.wilson(100, 100)
    assert 0.0 <= lo <= 1.0 and hi == pytest.approx(1.0) and lo > 0.95
    lo, hi = counterfactual.wilson(0, 24)
    assert lo == 0.0 and hi < 0.15
    assert all(math.isnan(v) for v in counterfactual.wilson(0, 0))
    # More data must not widen the interval.
    w = [counterfactual.wilson(n // 2, n) for n in (24, 100, 400)]
    assert (w[0][1] - w[0][0]) > (w[1][1] - w[1][0]) > (w[2][1] - w[2][0])


def test_decoys_are_not_all_the_same_subgraph():
    """Shipped bug: the decoy was `the first node of the other class`, the same
    one for every node, so 96 decoy narrations rested on 2 distinct stimuli."""
    ds = graphs.make(seed=0)
    rng = np.random.default_rng(0)
    # Shuffled, not the first 60: node ids run base graph, then every house,
    # then every cycle, so a head slice is single-class.
    pool = [i for i in range(ds.n) if ds.y[i] in (1, 2)]
    rng.shuffle(pool)
    chosen = pool[:60]
    by_class = {c: [n for n in chosen if int(ds.y[n]) == c] for c in (1, 2)}
    decoy = {nd: int(rng.choice(by_class[2 if int(ds.y[nd]) == 1 else 1]))
             for nd in chosen}
    assert len(set(decoy.values())) > 10


def _rows(model, node, kinds):
    """kinds: list of (kind, subgraph, label, motif_claimed)."""
    return [{"model": model, "explainer": "gnnexplainer", "node": node,
             "kind": k, "subgraph": s, "label": l, "motif_claimed": m}
            for k, s, l, m in kinds]


FULL = [("narrate", "true", "true", "house"),
        ("narrate", "true", "flipped", "house"),
        ("narrate", "decoy", "true", "cycle"),
        ("narrate", "decoy", "flipped", "cycle"),
        ("control", "true", "none", "house")]


def test_partial_nodes_are_excluded_from_every_cell():
    """The daily token budget stops a model mid-node. Keeping the conditions it
    did finish would leave the 2x2 unbalanced, and an unbalanced 2x2 cannot be
    compared across cells -- which is the entire design."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_rows("m", 1, FULL)          # complete
                      + _rows("m", 2, FULL[:3])    # cut off mid-node
                      + _rows("m", 3, FULL[:4]))   # narrate done, no control
    assert counterfactual.complete_nodes(df) == {("m", "gnnexplainer", 1)}


def test_label_sensitivity_separates_a_follower_from_a_guesser():
    """The point of the measure. Both models below score 0.5 on label agreement
    in the decisive cell; only one of them ever consulted the label."""
    pd = pytest.importorskip("pandas")
    follower = _rows("follower", 1, [
        ("narrate", "true", "true", "house"),
        ("narrate", "true", "flipped", "cycle")])      # moved with the label
    ignorer = _rows("ignorer", 1, [
        ("narrate", "true", "true", "house"),
        ("narrate", "true", "flipped", "house")])      # same answer either way
    s = counterfactual.label_sensitivity(pd.DataFrame(follower + ignorer))
    assert s.loc[("follower", "gnnexplainer"), "sum"] == 1
    assert s.loc[("ignorer", "gnnexplainer"), "sum"] == 0
    assert s.loc[("follower", "gnnexplainer"), "size"] == 1


def test_rate_limit_reset_headers_parse():
    """Shipped bug: '577ms' parsed as 577 minutes and put four of five worker
    threads to sleep for nine hours mid-run, silently. Nothing raised; the run
    just stopped producing rows."""
    assert narrate._secs("3.885s") == pytest.approx(3.885)
    assert narrate._secs("1m26.4s") == pytest.approx(86.4)
    assert narrate._secs("2m") == pytest.approx(120.0)
    assert narrate._secs("577ms") == pytest.approx(0.577)
    assert narrate._secs("205ms") == pytest.approx(0.205)
    # Nothing the per-minute budget reports should ever exceed a minute.
    for h in ("577ms", "3.885s", "59s", "junk"):
        assert narrate._secs(h) <= 60.0
