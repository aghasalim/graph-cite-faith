"""Does the narrative track the subgraph it was given, or the label it was told?

A 2x2, per node:

                      | true label            | flipped label
  ----------------------------------------------------------------
  true subgraph       | the normal case       | label contradicts structure
  decoy subgraph      | structure swapped     | both swapped

The decoy is a real explanation subgraph taken from a *randomly drawn* node of
the other motif class, so it is a genuine, equally plausible structure -- not
noise. A narrative faithful to its inputs must describe whatever it was handed.

Two scores, neither needing an LLM judge:

  **structure agreement** -- does the named motif match the shape actually
  present in the provided edges?
  **label agreement** -- does the named motif match the class it was told?

In the swapped cells these two point at different answers, which is what makes
the design able to separate reading from post-rationalising. High label
agreement with a decoy subgraph looks like post-rationalisation: the model
narrating the answer rather than the evidence.

It only looks like it. A model that never answers "neither" has label agreement
identically equal to 1 - structure agreement in those cells, so the two are one
number wearing two hats, and a model guessing at chance scores 0.5 on
"post-rationalisation" without ever having consulted the label. Hence:

  **label sensitivity** -- for one node and one subgraph, does the named motif
  change when the *only* thing that changes in the prompt is the label? At
  temperature 0 an indifferent model scores 0 and a pure label-follower scores
  1. This is the measure that can actually see the label being used.

**citation validity** -- the share of cited node ids that appear in the provided
edges -- ports the RAG-attribution measure (Wallat et al., arXiv 2412.18004) to
graphs, where it can be checked exactly rather than by entailment.

**the control** -- the same subgraphs with no predicted class in the prompt at
all. "The model falls back on the label when it cannot read the evidence" is
only a measurement if *cannot read* has a number attached, and that number has
to come from a prompt with no label in it.

Two explainers, because the subgraph is an input to the narration and swapping
the method that produced it changes what the LLM is asked to describe.
"""
from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gcf import gnn, graphs, narrate  # noqa: E402

N_NODES = 100         # nodes in the primary 2x2; each costs 4 calls per model
N_SALIENCY = 50       # paired subset re-run through the second explainer
K_EDGES = 8
MIN_NODES = 30        # below this an arm is dropped rather than reported

# Five models spanning roughly an order of magnitude of capability. Two models
# cannot separate "this pipeline post-rationalises" from "this model cannot read
# an edge list": that needs a spread wide enough for edge-reading ability to
# vary, plus the control task below to measure it directly.
MODELS = ["llama-3.1-8b-instant",
          "openai/gpt-oss-20b",
          "qwen/qwen3.6-27b",
          "openai/gpt-oss-120b",
          "llama-3.3-70b-versatile"]
EXPLAINERS = ("gnnexplainer", "saliency")

OUT = Path(__file__).resolve().parents[1] / "reports"
LOG = OUT / "runs.jsonl"
CLASS_NAME = {1: "motif-A", 2: "motif-B"}   # deliberately uninformative names
TRUE_MOTIF = {1: "house", 2: "cycle"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than normal-approximation because several cells here sit at
    or near 0 and 1, where the textbook interval runs outside [0, 1] and its
    coverage collapses. No scipy: the closed form is four lines.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p, z2 = k / n, z * z
    d = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def motif_recovery(g, edges: list[tuple[int, int]], motif: list[int]) -> float:
    """Share of the planted motif's own edges present in the extracted subgraph.

    Reported because it bounds everything downstream: if the explainer did not
    surface the motif, no narrative could describe it, and a low
    structure-agreement score would be the explainer's failure rather than the
    LLM's. Separating the two is the only way the LLM number means anything --
    and with two explainers of visibly different recovery, it is the only way
    the explainer contrast means anything either.
    """
    ms = set(motif)
    planted = g.subgraph(motif).number_of_edges()
    got = {tuple(sorted(e)) for e in edges if e[0] in ms and e[1] in ms}
    return len(got) / planted if planted else float("nan")


# An earlier version inferred the shown shape from the extracted edges with a
# "contains a triangle => house" heuristic. The k-hop neighbourhood of a
# Barabasi-Albert graph is full of incidental triangles, so it labelled 88 of 96
# subgraphs "house" and the resulting agreement score was measuring nothing.
# Ground truth now comes from which planted motif the source node belongs to,
# which is known by construction.


def build_stimuli():
    """Train the GCN, pick nodes, and pre-compute every subgraph shown.

    Cached: 100 GNNExplainer optimisations take ~7 minutes, and the free-tier
    daily token budget makes several restarts a certainty rather than a
    contingency. Everything here is a deterministic function of seed 0.
    """
    cache = OUT / "stimuli.json"
    ds = graphs.make(seed=0)
    a = torch.from_numpy(graphs.adjacency(ds.graph))
    x, y = torch.from_numpy(ds.x), torch.from_numpy(ds.y)
    rng = np.random.default_rng(0)
    idx = rng.permutation(ds.n)
    tr = torch.zeros(ds.n, dtype=torch.bool)
    tr[idx[: int(0.6 * ds.n)]] = True
    model = gnn.train(x, a, y, tr, seed=0)
    acc = gnn.accuracy(model, x, a, y, ~tr)
    print(f"GNN test accuracy {acc:.3f}")

    # Pick correctly-classified motif nodes; an explanation of a wrong
    # prediction is a different question and would muddy the measurement.
    with torch.no_grad():
        pred = model(x, a).argmax(1)
    pool = [i for i in range(ds.n)
            if ds.y[i] in (1, 2) and int(pred[i]) == ds.y[i] and not tr[i]]
    rng.shuffle(pool)
    chosen = pool[:N_NODES]
    if len(chosen) < N_NODES:
        raise SystemExit(f"only {len(chosen)} eligible nodes; need {N_NODES}")

    # One decoy drawn per node, not one per class. The published run took the
    # first eligible node of the other class for *every* node, so 96 decoy
    # narrations rested on 2 distinct subgraphs -- pseudo-replication that an
    # n=24 confidence interval would have badly overstated.
    by_class = {c: [n for n in chosen if int(ds.y[n]) == c] for c in (1, 2)}
    decoy = {nd: int(rng.choice(by_class[2 if int(ds.y[nd]) == 1 else 1]))
             for nd in chosen}

    if cache.exists():
        raw = json.loads(cache.read_text())
        if raw["n"] == [N_NODES, N_SALIENCY, K_EDGES]:
            stim = {(e, int(n)): {**v, "ids": set(v["ids"])}
                    for e, d in raw["stim"].items() for n, v in d.items()}
            print(f"  reusing {len(stim)} cached subgraphs")
            return ds, chosen, decoy, stim, float(acc)

    stim: dict[tuple[str, int], dict] = {}
    for expl in EXPLAINERS:
        nodes = chosen if expl == "gnnexplainer" else chosen[:N_SALIENCY]
        # a decoy may come from outside the saliency subset, so stimuli are
        # needed for it too.
        need = sorted(set(nodes) | {decoy[n] for n in nodes})
        for nd in need:
            mask = (gnn.explain_edges(model, x, a, nd, seed=0)
                    if expl == "gnnexplainer"
                    else gnn.saliency_edges(model, x, a, nd))
            edges = gnn.top_edges(mask, ds.graph, graphs.khop(ds.graph, nd, 3),
                                  k=K_EDGES)
            stim[(expl, nd)] = {
                "edges": "; ".join(f"{u}-{v}" for u, v in edges),
                "ids": {u for e in edges for u in e},
                "shape": TRUE_MOTIF[int(ds.y[nd])],
                "recovery": motif_recovery(ds.graph, edges,
                                           ds.motif_nodes.get(nd, [])),
            }
        print(f"  {expl}: {len(need)} subgraphs, mean motif recovery "
              f"{np.mean([stim[(expl, n)]['recovery'] for n in need]):.3f}")
    OUT.mkdir(exist_ok=True)
    out: dict[str, dict] = {e: {} for e in EXPLAINERS}
    for (e, n), v in stim.items():
        out[e][str(n)] = {**v, "ids": sorted(v["ids"])}
    cache.write_text(json.dumps({"n": [N_NODES, N_SALIENCY, K_EDGES],
                                 "stim": out}))
    return ds, chosen, decoy, stim, float(acc)


def jobs_for(ds, chosen, decoy) -> list[dict]:
    """Every condition, once per model. 750 calls per model at the defaults --
    Groq's free tier meters 1000 requests per day per model, so the run has to
    fit inside one day's allowance with room for retries."""
    js = []
    for expl in EXPLAINERS:
        nodes = chosen if expl == "gnnexplainer" else chosen[:N_SALIENCY]
        for nd in nodes:
            for sub in ("true", "decoy"):
                for lab in ("true", "flipped"):
                    js.append({"kind": "narrate", "explainer": expl, "node": nd,
                               "subgraph": sub, "label": lab})
            js.append({"kind": "control", "explainer": expl, "node": nd,
                       "subgraph": "true", "label": "none"})
    return js


def key(mdl: str, j: dict) -> str:
    return f"{mdl}|{j['kind']}|{j['explainer']}|{j['node']}|{j['subgraph']}|{j['label']}"


def score(ds, mdl: str, j: dict, decoy: dict, stim: dict, text: str) -> dict:
    src = j["node"] if j["subgraph"] == "true" else decoy[j["node"]]
    s = stim[(j["explainer"], src)]
    cls = int(ds.y[j["node"]])
    other = 2 if cls == 1 else 1
    lab_cls = cls if j["label"] == "true" else other
    motif, cited = narrate.parse(text)
    return {
        "model": mdl, "kind": j["kind"], "explainer": j["explainer"],
        "node": j["node"], "subgraph": j["subgraph"], "label": j["label"],
        "shape_shown": s["shape"],
        "motif_edges_recovered": round(s["recovery"], 3),
        "motif_claimed": motif,
        "parsed": motif in narrate.ANSWERS,
        "agrees_with_structure": motif == s["shape"],
        "agrees_with_label": (motif == TRUE_MOTIF[lab_cls]
                              if j["kind"] == "narrate" else None),
        "cited_valid": sum(c in s["ids"] for c in cited),
        "n_cited": len(cited),
        # Kept so an unparsed reply can be read rather than guessed at. A run
        # that reports a parse-failure rate without the failures is asking to be
        # taken on trust.
        "reply": text[:500],
    }


def run(ds, chosen, decoy, stim, js: list[dict]) -> list[str]:
    """One thread per model. Groq meters each model separately, so five models
    in parallel cost the wall-clock of the slowest, not the sum."""
    # Only a parsed reply counts as done. An unparsed one is retried on the next
    # run: it is a missing measurement, not a measurement of zero, and leaving
    # it in place would let a parser bug harden into a result.
    done = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            if r["parsed"]:
                done.add(key(r["model"], r))
    OUT.mkdir(exist_ok=True)
    fh = LOG.open("a")
    lock, failures = threading.Lock(), []

    def worker(mdl: str) -> None:
        todo = [j for j in js if key(mdl, j) not in done]
        for i, j in enumerate(todo):
            src = j["node"] if j["subgraph"] == "true" else decoy[j["node"]]
            s = stim[(j["explainer"], src)]
            try:
                if j["kind"] == "control":
                    text = narrate.read_shape(s["edges"], model=mdl)
                else:
                    cls = int(ds.y[j["node"]])
                    lab_cls = cls if j["label"] == "true" else (2 if cls == 1 else 1)
                    text = narrate.narrate(j["node"], CLASS_NAME[lab_cls],
                                           s["edges"], model=mdl)
            except Exception as e:                       # noqa: BLE001
                with lock:
                    failures.append(f"{key(mdl, j)}: {type(e).__name__}")
                continue
            row = score(ds, mdl, j, decoy, stim, text)
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  {mdl}: {i + 1}/{len(todo)}")

    ts = [threading.Thread(target=worker, args=(m,), daemon=True) for m in MODELS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    fh.close()
    return failures


def complete_nodes(df):
    """The (model, explainer, node) triples where every condition landed.

    A model that ran out of daily tokens half way through a node would otherwise
    contribute to some cells of its 2x2 and not others, and an unbalanced 2x2 is
    not a 2x2 -- comparing cells is the whole design. Dropping partial nodes
    keeps every cell of an arm at the same n, whatever n it reached.
    """
    per = df.groupby(["model", "explainer", "node"])["kind"].agg(
        narrate=lambda s: (s == "narrate").sum(),
        control=lambda s: (s == "control").sum())
    return set(per[(per.narrate == 4) & (per.control == 1)].index)


def label_sensitivity(nar):
    """Per arm, how often the named motif moves when *only* the label moves.

    Same node, same edges, temperature 0; the two prompts differ in one word.
    An indifferent model scores 0, a pure label-follower 1.

    This exists because the obvious measure -- label agreement in the cell where
    structure and label disagree -- is not independent evidence. A model that
    never answers "neither" has label agreement identically equal to
    1 - structure agreement there, so reading it as "how much the model leans on
    the label" is reading structure agreement backwards and calling it a second
    finding. It scores a coin-flipping model at 0.5 for post-rationalisation.
    """
    pv = nar.pivot_table(index=["model", "explainer", "node", "subgraph"],
                         columns="label", values="motif_claimed",
                         aggfunc="first").dropna()
    pv["moved"] = pv["true"] != pv["flipped"]
    return pv.groupby(["model", "explainer"])["moved"].agg(["size", "sum"])


def ci(col) -> str:
    lo, hi = wilson(int(col.sum()), len(col))
    return f"{col.mean():.3f} [{lo:.3f},{hi:.3f}]"


def pooled_ci(valid, total) -> str:
    k, n = int(valid.sum()), int(total.sum())
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f},{hi:.3f}]"


def report(js: list[dict]) -> None:
    import pandas as pd

    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["model", "kind", "explainer", "node", "subgraph", "label"],
        keep="last")   # a retry supersedes the unparsed attempt it replaces
    expected = len(js) * len(MODELS)
    print(f"\n{len(df)}/{expected} planned conditions collected")

    complete = complete_nodes(df)
    before = len(df)
    df = df[[t in complete for t in
             zip(df["model"], df["explainer"], df["node"])]]
    print(f"{before - len(df)} rows dropped from partially-collected nodes")

    # Refuse to report an arm built on a handful of nodes. The first run of this
    # project produced a tidy-looking table from 5 of 96 calls; that is how a
    # broken harness launders itself into a result. The bar is now per arm and
    # in nodes, because the budget stops arms at different points.
    sizes = df[df.kind == "narrate"].groupby(
        ["model", "explainer"])["node"].nunique()
    for (m, e), n in sizes.items():
        if n < MIN_NODES:
            print(f"DROPPED {m}/{e}: only {n} complete nodes (< {MIN_NODES})")
    keep = {k for k, n in sizes.items() if n >= MIN_NODES}
    df = df[[(m, e) in keep for m, e in zip(df["model"], df["explainer"])]]
    if not keep:
        raise SystemExit("no arm reached the minimum node count -- nothing to "
                         "report. Re-run when the daily token budget resets.")
    print("nodes per cell: " + ", ".join(
        f"{m.split('/')[-1]}/{e}={n}" for (m, e), n in sizes.items()
        if (m, e) in keep))
    unparsed = (~df["parsed"]).sum()
    print(f"{unparsed} replies gave no answer from the closed set "
          f"({unparsed / len(df):.1%})")

    nar = df[df["kind"] == "narrate"]
    main = nar.groupby(["model", "explainer", "subgraph", "label"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "structure_agreement": ci(g["agrees_with_structure"]),
            "label_agreement": ci(g["agrees_with_label"].astype(bool)),
            "citation_validity": pooled_ci(g["cited_valid"], g["n_cited"]),
        }), include_groups=False)

    ctl = df[df["kind"] == "control"].groupby(["model", "explainer"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "edge_reading_accuracy": ci(g["agrees_with_structure"]),
        }), include_groups=False)

    sens = label_sensitivity(nar)

    # The joint view: edge-reading ability against label-following in the one
    # cell where structure and label disagree and the structure is the decoy.
    dec = nar[(nar["subgraph"] == "decoy") & (nar["label"] == "true")]
    joint = pd.DataFrame([{
        "model": m, "explainer": e,
        "n": len(g),
        "edge_reading": g0["agrees_with_structure"].mean(),
        "follows_structure": g["agrees_with_structure"].mean(),
        "follows_label": g["agrees_with_label"].astype(bool).mean(),
        "label_sensitivity": sens.loc[(m, e), "sum"] / sens.loc[(m, e), "size"],
        "sens_n": int(sens.loc[(m, e), "size"]),
        # An unparsed reply agrees with neither, so a model that cannot hold the
        # output format would read as "follows neither" rather than as broken.
        "unparsed": 1 - nar[(nar["model"] == m)
                            & (nar["explainer"] == e)]["parsed"].mean(),
    } for (m, e), g in dec.groupby(["model", "explainer"])
        if len(g0 := df[(df["kind"] == "control") & (df["model"] == m)
                        & (df["explainer"] == e)])]).round(3)

    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "counterfactual.json", orient="records", indent=2)
    main.to_csv(OUT / "counterfactual_summary.csv")
    ctl.to_csv(OUT / "edge_reading_control.csv")
    joint.to_csv(OUT / "competence_vs_label.csv", index=False)
    for name, t in (("2x2 (proportion [95% Wilson CI])", main),
                    ("edge-reading control, no label in prompt", ctl),
                    ("decoy subgraph + true label vs control", joint)):
        print(f"\n=== {name} ===")
        print(t.to_string())


def main() -> None:
    ds, chosen, decoy, stim, _ = build_stimuli()
    js = jobs_for(ds, chosen, decoy)
    print(f"{len(js)} conditions x {len(MODELS)} models = {len(js) * len(MODELS)} calls")
    failures = run(ds, chosen, decoy, stim, js)
    if failures:
        print(f"\n{len(failures)} calls failed: {failures[:3]}")
    report(js)


if __name__ == "__main__":
    main()
