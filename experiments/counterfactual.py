"""Does the narrative track the subgraph it was given, or the label it was told?

A 2x2, per node:

                      | true label            | flipped label
  ----------------------------------------------------------------
  true subgraph       | the normal case       | label contradicts structure
  decoy subgraph      | structure swapped     | both swapped

The decoy is a real explanation subgraph taken from a node of the *other* motif
class, so it is a genuine, equally plausible structure -- not noise. A narrative
faithful to its inputs must describe whatever it was actually handed.

Two scores, neither needing an LLM judge:

  **structure agreement** -- does the named motif match the shape actually
  present in the provided edges?
  **label agreement** -- does the named motif match the class it was told?

In the swapped cells these two point at different answers, which is what makes
the design able to separate reading from post-rationalising. High label
agreement with a decoy subgraph is post-rationalisation: the model is narrating
the answer rather than the evidence.

**citation validity** -- the share of cited node ids that appear in the provided
edges -- ports the RAG-attribution measure (Wallat et al., arXiv 2412.18004) to
graphs, where it can be checked exactly rather than by entailment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gcf import gnn, graphs, narrate  # noqa: E402

N_NODES = 24          # nodes analysed; each costs 4 LLM calls per model
# Two models, because a single one cannot separate "this pipeline
# post-rationalises" from "this model cannot read an edge list". On a trivial
# square-plus-triangle, qwen answers house and llama-3.3-70b answers cycle --
# so structural reading ability is a variable here, not a constant.
MODELS = ["qwen/qwen3.6-27b", "llama-3.3-70b-versatile"]
K_EDGES = 8
OUT = Path(__file__).resolve().parents[1] / "reports"
CLASS_NAME = {1: "motif-A", 2: "motif-B"}   # deliberately uninformative names
TRUE_MOTIF = {1: "house", 2: "cycle"}


def motif_recovery(edges: list[tuple[int, int]], motif: list[int]) -> float:
    """Share of the planted motif's own edges present in the extracted subgraph.

    Reported because it bounds everything downstream: if the explainer did not
    surface the motif, no narrative could describe it, and a low
    structure-agreement score would be the explainer's failure rather than the
    LLM's. Separating the two is the only way the LLM number means anything.
    """
    ms = set(motif)
    got = {tuple(sorted(e)) for e in edges if e[0] in ms and e[1] in ms}
    # Planted motifs are 5-6 nodes; count their internal edges as the target.
    return len(got)


# An earlier version inferred the shown shape from the extracted edges with a
# "contains a triangle => house" heuristic. The k-hop neighbourhood of a
# Barabasi-Albert graph is full of incidental triangles, so it labelled 88 of 96
# subgraphs "house" and the resulting agreement score was measuring nothing.
# Ground truth now comes from which planted motif the source node belongs to,
# which is known by construction.


def main() -> None:
    ds = graphs.make(seed=0)
    a = torch.from_numpy(graphs.adjacency(ds.graph))
    x, y = torch.from_numpy(ds.x), torch.from_numpy(ds.y)
    rng = np.random.default_rng(0)
    idx = rng.permutation(ds.n)
    tr = torch.zeros(ds.n, dtype=torch.bool)
    tr[idx[: int(0.6 * ds.n)]] = True
    model = gnn.train(x, a, y, tr, seed=0)
    print(f"GNN test accuracy {gnn.accuracy(model, x, a, y, ~tr):.3f}")

    # Pick correctly-classified motif nodes; an explanation of a wrong
    # prediction is a different question and would muddy the measurement.
    with torch.no_grad():
        pred = model(x, a).argmax(1)
    pool = [i for i in range(ds.n)
            if ds.y[i] in (1, 2) and int(pred[i]) == ds.y[i] and not tr[i]]
    rng.shuffle(pool)
    chosen = pool[:N_NODES]

    # Pre-compute one explanation subgraph per node.
    expl: dict[int, list[tuple[int, int]]] = {}
    for nd in chosen:
        mask = gnn.explain_edges(model, x, a, nd, seed=0)
        expl[nd] = gnn.top_edges(mask, graphs.khop(ds.graph, nd, 3), k=K_EDGES)

    rows, failures = [], []
    for nd in chosen:
        cls = int(ds.y[nd])
        other = 2 if cls == 1 else 1
        decoy_src = next((m for m in chosen if int(ds.y[m]) == other), None)
        if decoy_src is None:
            continue
        for sub_kind in ("true", "decoy"):
            src = nd if sub_kind == "true" else decoy_src
            edges = expl[src]
            shown = TRUE_MOTIF[int(ds.y[src])]
            recovered = motif_recovery(edges, ds.motif_nodes.get(src, []))
            edge_str = "; ".join(f"{u}-{v}" for u, v in edges)
            node_ids = {u for e in edges for u in e}
            for lab_kind in ("true", "flipped"):
                lab_cls = cls if lab_kind == "true" else other
                for mdl in MODELS:
                  try:
                    text = narrate.narrate(nd, CLASS_NAME[lab_cls], edge_str, model=mdl)
                  except Exception as e:
                    failures.append(f"{nd}/{sub_kind}/{lab_kind}/{mdl}: {type(e).__name__}")
                    continue
                  motif, cited = narrate.parse(text)
                  rows.append({
                    "model": mdl,
                    "node": nd, "subgraph": sub_kind, "label": lab_kind,
                    "shape_shown": shown,
                    "motif_edges_recovered": recovered,
                    "motif_claimed": motif,
                    "agrees_with_structure": motif == shown,
                    "agrees_with_label": motif == TRUE_MOTIF[lab_cls],
                    "citation_validity": (
                        float(np.mean([c in node_ids for c in cited])) if cited else float("nan")
                    ),
                    "n_cited": len(cited),
                  })
        print(f"  node {nd}: {len([r for r in rows if r['node'] == nd])}/4 conditions")

    expected = len(chosen) * 4 * len(MODELS)
    if failures:
        print(f"\n{len(failures)} of {expected} calls failed: {failures[:3]}")
    # Refuse to report a mean over a decimated sample. The first run produced a
    # tidy-looking table from 5 of 96 calls; that is how a broken harness
    # launders itself into a result.
    if len(rows) < 0.8 * expected:
        raise SystemExit(
            f"only {len(rows)}/{expected} conditions completed -- too few to "
            f"report. Re-run when the rate limit clears.")

    OUT.mkdir(exist_ok=True)
    (OUT / "counterfactual.json").write_text(json.dumps(rows, indent=2))
    import pandas as pd

    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "subgraph", "label"]).agg(
        n=("node", "size"),
        structure_agreement=("agrees_with_structure", "mean"),
        label_agreement=("agrees_with_label", "mean"),
        citation_validity=("citation_validity", "mean"),
    ).round(3)
    agg.to_csv(OUT / "counterfactual_summary.csv")
    print()
    print(agg.to_string())


if __name__ == "__main__":
    main()
