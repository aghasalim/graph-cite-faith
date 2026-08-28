"""LLM narration of a GNN explanation, with parseable output.

The prompt asks for two machine-checkable commitments -- which motif the
structure shows, and which node ids support it -- so faithfulness can be scored
without a second LLM judging the first. A judge would introduce exactly the
failure being studied: a fluent model agreeing with another fluent model.

Nothing in the prompt names the motif types available. If a narrative says
"house", it either read the edges or invented it.
"""
from __future__ import annotations

import os
import re
import time

import httpx

MODEL = os.getenv("GCF_MODEL", "openai/gpt-oss-20b")
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are explaining why a graph neural network classified a node.

Node under analysis: {node}
The model's predicted class: {label}
The subgraph the model identified as the evidence for this prediction \
(undirected edges, "u-v"):
{edges}

A "house" is a 4-node square with a triangular roof sharing one of its edges. \
A "cycle" is a simple closed ring with no chords. Judge only the edges listed \
above; do not infer from the class name.

Explain in two sentences why this structure supports the prediction. Then end \
your reply with exactly these two lines:

MOTIF: <exactly one of: house, cycle, neither>
NODES: <comma-separated node ids from the edges above that carry the evidence>
"""

# The control. Same edges, same closed answer set, no predicted class and no
# node under analysis -- so nothing is left to post-rationalise from. This is
# what turns "the model falls back on the label when it cannot read the
# evidence" from a story into a measurement: it puts a number on *cannot read*.
CONTROL_PROMPT = """Here is an undirected subgraph, given as edges "u-v":
{edges}

A "house" is a 4-node square with a triangular roof sharing one of its edges. \
A "cycle" is a simple closed ring with no chords.

Which shape do these edges contain? Judge only the edges listed above. Reply \
with exactly one line and nothing else:

MOTIF: <exactly one of: house, cycle, neither>
"""


def _key() -> str:
    if k := os.getenv("GROQ_API_KEY"):
        return k
    raise SystemExit("No GROQ_API_KEY. export it before running the narrator.")


# Models that bill hidden reasoning against max_tokens need a large cap or they
# return an empty string with no error -- the failure the sibling RAG project
# documented. Naming the shape in an edge list needs no reasoning at all, so
# this harness turns it down as far as each model permits and keeps the cap
# small. That matters for throughput, not elegance: at max_tokens=2000 the
# per-minute token budget allowed ~4 calls and a 96-call run took over two
# hours. gpt-oss refuses "none" and floors at "low".
REASONING = {"qwen/qwen3.6-27b": "none",
             "openai/gpt-oss-20b": "low",
             "openai/gpt-oss-120b": "low"}


def _secs(s: str) -> float:
    """Parse Groq's reset headers: '3.885s', '1m26.4s', '2m', '577ms'.

    The millisecond form is the trap. A permissive `(?:(\\d+)m)?([\\d.]+)?s?`
    reads '577ms' as 577 *minutes* -- it is a legal parse, and it is silent. It
    put four of five worker threads to sleep for nine hours mid-run while the
    fifth, which happened never to see a sub-second reset, finished normally.
    Nothing errored; throughput just went to zero, which is the same shape of
    failure as the harness that reported a mean over n=2.
    """
    s = s.strip()
    if m := re.fullmatch(r"([\d.]+)ms", s):
        return float(m.group(1)) / 1000.0
    if m := re.fullmatch(r"(?:(\d+)m)?(?:([\d.]+)s?)?", s):
        return 60.0 * float(m.group(1) or 0) + float(m.group(2) or 0)
    return 5.0


def call(prompt: str, timeout: float = 90.0, max_retries: int = 6,
         model: str | None = None, max_tokens: int = 400) -> str:
    """One completion, with backoff on rate limits and pre-emptive pacing.

    Not defensive boilerplate: the first full run fired ~96 requests back to
    back, hit Groq's per-minute cap, and silently dropped 91 of them. The run
    "completed" and produced a summary table over n=2. A harness that drops
    most of its samples and still prints a mean is a worse failure than one
    that crashes, because the output looks like a result.

    The pacing reads the remaining-token budget off each response and waits out
    the window before it is exhausted. Backoff alone would work, but on a free
    tier metered per *day* as well as per minute, a 429 still spends a request
    from a 1000/day allowance, so it is cheaper not to earn one.
    """
    model = model or MODEL
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if eff := REASONING.get(model):
        body["reasoning_effort"] = eff
    delay = 2.0
    for _ in range(max_retries):
        r = httpx.post(ENDPOINT, headers={"Authorization": f"Bearer {_key()}"},
                       json=body, timeout=timeout)
        if r.status_code == 429:
            # Two different limits arrive as the same status. The per-minute
            # token bucket clears in seconds; the per-DAY one (100k tokens for
            # llama-3.3-70b, 200k for the rest) refills as a trickle and asks
            # for minutes. Honour whatever retry-after says up to five minutes,
            # because capping it at 60s turns a satisfiable wait into six
            # guaranteed failures and then a dropped sample.
            time.sleep(min(float(r.headers.get("retry-after", delay)), 300.0))
            delay *= 2
            continue
        r.raise_for_status()
        if float(r.headers.get("x-ratelimit-remaining-tokens", 1e9)) < 2 * max_tokens:
            # Capped at a minute: the budget being waited on is per-minute, so
            # any longer is a parse error, not a rate limit. Belt and braces
            # after a header format read as 577 minutes instead of 577 ms.
            time.sleep(min(_secs(r.headers.get("x-ratelimit-reset-tokens", "5s")), 60.0) + 0.5)
        choice = r.json()["choices"][0]
        # A reply cut off mid-sentence loses the MOTIF line, and a missing MOTIF
        # line is indistinguishable from a refusal to commit. Reasoning models
        # bill their hidden block against the same cap, so raise it once rather
        # than score a truncation as an answer.
        if choice["finish_reason"] == "length" and max_tokens < 1200:
            return call(prompt, timeout, max_retries, model, 1200)
        return choice["message"]["content"]
    raise RuntimeError(f"rate limited after {max_retries} attempts")


def narrate(node: int, label: str, edges: str, **kw) -> str:
    """Narrate an explanation the model is told the predicted class for."""
    return call(PROMPT.format(node=node, label=label, edges=edges), **kw)


def read_shape(edges: str, **kw) -> str:
    """Control: name the shape with no predicted class in the prompt at all."""
    return call(CONTROL_PROMPT.format(edges=edges), **kw)


ANSWERS = ("house", "cycle", "neither")


def parse(text: str) -> tuple[str, list[int]]:
    """Pull the MOTIF and NODES commitments out of a reply.

    Tolerant of markdown around the tag and the answer -- `**MOTIF:** cycle` and
    `MOTIF: **cycle**` are the same commitment, and a strict `MOTIF:\\s*(\\w+)`
    scored both as no answer at all. That mattered: it silently blanked 35-50%
    of three models' replies in a run, and a blank agrees with neither the
    structure nor the label, so the models would have looked evasive rather than
    unreadable by the parser.

    The answer itself is still matched against the closed set. Anything outside
    it stays unparsed and is reported as such rather than scored -- the point of
    closing the set was to stop free-form text being silently marked wrong.
    """
    motif = ""
    if m := re.search(rf"MOTIF\W{{0,4}}({'|'.join(ANSWERS)})\b", text, re.I):
        motif = m.group(1).lower()
    nodes: list[int] = []
    if m := re.search(r"NODES\W{0,4}([0-9][0-9,\s]*)", text, re.I):
        nodes = [int(t) for t in re.findall(r"\d+", m.group(1))]
    return motif, nodes
