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
from pathlib import Path

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


def _key() -> str:
    if k := os.getenv("GROQ_API_KEY"):
        return k
    # Fall back to the sibling project's .env so the harness runs without
    # re-entering a key. Read, never echoed.
    env = Path.home() / "eu-ai-act-rag" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No GROQ_API_KEY. export it, or put it in ~/eu-ai-act-rag/.env")


def narrate(node: int, label: str, edges: str, timeout: float = 90.0,
            max_retries: int = 6) -> str:
    """One narration, with backoff on rate limits.

    Not defensive boilerplate: the first full run fired ~96 requests back to
    back, hit Groq's per-minute cap, and silently dropped 91 of them. The run
    "completed" and produced a summary table over n=2. A harness that drops
    most of its samples and still prints a mean is a worse failure than one
    that crashes, because the output looks like a result.
    """
    body = {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 2000,
        "messages": [{"role": "user",
                      "content": PROMPT.format(node=node, label=label, edges=edges)}],
    }
    delay = 2.0
    for attempt in range(max_retries):
        r = httpx.post(ENDPOINT, headers={"Authorization": f"Bearer {_key()}"},
                       json=body, timeout=timeout)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", delay))
            time.sleep(min(wait, 60.0))
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"rate limited after {max_retries} attempts")


def parse(text: str) -> tuple[str, list[int]]:
    """Pull the MOTIF and NODES commitments out of a reply."""
    motif = ""
    if m := re.search(r"MOTIF:\s*([A-Za-z\-]+)", text):
        motif = m.group(1).strip().lower()
    nodes: list[int] = []
    if m := re.search(r"NODES:\s*([0-9,\s]+)", text):
        nodes = [int(t) for t in re.findall(r"\d+", m.group(1))]
    return motif, nodes
