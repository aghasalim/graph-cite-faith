# GraphCiteFaith — perfect citations, wrong explanation

[![ci](https://github.com/aghasalim/graph-cite-faith/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/graph-cite-faith/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A GNN classifies a node, an explainer extracts the subgraph it used, and an LLM
turns that into a sentence a human reads. This tests whether the sentence
describes the subgraph it was handed — or the answer it was told. Built by a
third-year Applied Computer Science (AI) student.

**Headline: citation validity is 1.000 across all 192 conditions and both
models, while one of those models names the correct structure only 33% of the
time.** Every cited node id is real. The story built from them is wrong.

---

## The design

Synthetic graphs with planted `house` and `cycle` motifs, so the causally
relevant subgraph for every node is known exactly. A GCN reaches 94.3% test
accuracy on structure alone — node features are pure noise, so it cannot be
reading anything else.

For each node, a 2×2:

| | true label | flipped label |
|---|---|---|
| **true subgraph** | the normal case | label contradicts structure |
| **decoy subgraph** | structure swapped | both swapped |

The decoy is a *real* explanation subgraph from a node of the other motif class
— a genuine alternative structure, not noise. The model is asked to commit to a
motif name and a list of supporting node ids, both checkable against the edges
it was given, so nothing is scored by a second LLM. A judge would reproduce the
exact failure under study: one fluent model agreeing with another.

The class names shown to the model are `motif-A` / `motif-B`. Nothing in the
prompt reveals which shape belongs to which class.

---

## Results

24 nodes × 4 conditions × 2 models = 192 narrations (`make counterfactual`).

| model | subgraph | label | structure agr. | label agr. | citation validity |
|---|---|---|---|---|---|
| **llama-3.3-70b** | true | true | 0.792 | 0.792 | **1.000** |
| | decoy | flipped | 0.917 | 0.917 | **1.000** |
| | **decoy** | **true** | **0.833** | **0.125** | **1.000** |
| | **true** | **flipped** | **0.500** | **0.500** | **1.000** |
| **qwen3.6-27b** | true | true | 0.458 | 0.458 | **1.000** |
| | decoy | flipped | 0.292 | 0.292 | **1.000** |
| | **decoy** | **true** | **0.333** | **0.583** | **1.000** |
| | **true** | **flipped** | **0.542** | **0.208** | **1.000** |

The bold rows are the ones that carry information. In the other two, structure
and label point at the same answer, so agreement with one is agreement with the
other and the cell cannot distinguish anything.

**1. Citation validity is perfect everywhere — and means almost nothing.**
Across all 192 narrations, every node id either model cited appeared in the
edges it was shown. Zero fabrication. On the measure ported from
[Wallat et al.'s RAG attribution work](https://arxiv.org/abs/2412.18004), where
up to 57% of citations were post-rationalised, this pipeline scores flawlessly.

It is still frequently wrong. qwen names the correct shape 29–46% of the time —
at or below chance for a binary choice — while citing exclusively real nodes.
**Citing the right evidence and describing it correctly are separate
properties, and the standard attribution metric only tests the first.** That is
the finding: a pipeline can pass a citation-faithfulness audit and still hand an
investigator a false account of the structure.

**2. Post-rationalisation is model-dependent, and it tracks competence.** In the
decisive cell — decoy subgraph, true label, where structure and label disagree —
llama names the shown structure 83.3% against the told label 12.5%. It reads the
evidence and ignores the misleading answer. qwen inverts it: 33.3% structure
against 58.3% label. It narrates the answer instead.

The two are not equally able to read an edge list. qwen's structure agreement is
near chance *even when structure and label agree*, so it often cannot recover
the shape at all. The pattern is that **the label is what a model falls back on
when it cannot read the evidence** — post-rationalisation looks less like
deception and more like a competence floor.

**3. llama's `true`/`flipped` cell is an exact coin flip (0.500 / 0.500).** I
have no account of why it reads the decoy cleanly but splits here, and with
n=24 I am not going to invent one.

---

## Four instrument bugs, found before any result was reported

Each would have produced a confident, entirely fake number.

**The first run silently discarded 95% of its samples.** 96 requests fired back
to back hit Groq's per-minute cap; 91 failed and the harness printed a tidy
summary table over **n=2**. It now backs off, and *refuses to report* below 80%
completion. A run that drops most of its data and still prints a mean is worse
than one that crashes, because the output looks like a result.

**Scoring exact-matched the string `cycle`.** 40 replies calling a six-node ring
a "hexagon" were counted wrong. The answer set is now closed and defined in the
prompt.

**The ground-truth labeller was broken.** It inferred the shown shape from the
extracted edges via "contains a triangle → house". Barabási–Albert
neighbourhoods are full of incidental triangles, so it labelled **88 of 96**
subgraphs "house" and the agreement score measured nothing at all. Ground truth
now comes from planted-motif membership, which is known by construction.

**A contiguous train/test split is degenerate here.** Node ids follow
construction order — base graph, then every house, then every cycle — so the
first 60% contains zero cycle nodes and accuracy collapses to 9%. There is a
test guarding this.

A fifth issue was performance, not correctness: at `max_tokens=2000` the
per-minute token budget allowed about four calls, and a 96-call run took over
two hours. Naming a shape in an edge list needs no hidden reasoning, so the cap
is now 400 with reasoning disabled where the model permits. The same run
finishes in minutes.

---

## Running it

```bash
make setup && make test
```

8 tests, all on the generator, the split and the parser — the instrument, not
the model.

```bash
export GROQ_API_KEY=...   # or leave it in ~/eu-ai-act-rag/.env
make counterfactual
```

GCN and GNNExplainer are written against dense adjacency rather than
torch-geometric: 740-node graphs make dense fast, and it removes the dependency
most likely to stop this running on someone else's machine.

## Limitations

- **n=24 nodes per cell.** Enough to separate 0.833 from 0.125; not enough to
  explain a 0.500/0.500 cell.
- **Synthetic graphs only.** Exact ground truth is the point, and a real
  citation network has no ground-truth "reason" to check against.
- **Two motif classes**, so chance is 0.5 and the metric is coarse.
- **Two models, one provider.** The competence-floor reading is a hypothesis
  consistent with two data points, not an established result.
- **GNNExplainer only.** PGExplainer and SubgraphX may surface different
  subgraphs, which would change what the LLM is asked to describe.

## License

MIT — see [LICENSE](LICENSE).
