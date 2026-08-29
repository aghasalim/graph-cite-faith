"""Guard the markdown against the formatting damage a bulk text pass causes.

Every failure here has actually happened in this repo. A punctuation pass once
closed the space between a table pipe and the code span next to it, ran a bold
run into the code that followed, and collapsed whole tables onto one line, which
GitHub renders as a wall of pipe characters. None of it is visible in a diff
unless you know to look, so it is checked instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", "node_modules", "site-packages"}
SEPARATOR = re.compile(r"^[\s:|-]+$")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def markdown_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.md")
        if not SKIP & set(p.parts) and "adapter" not in p.parts
    )


def prose_lines(path: Path):
    """Yield (line number, text) outside fenced and indented code blocks."""
    fenced = False
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.startswith("    "):
            continue
        yield n, line


def outside_code(line: str):
    """Yield (index, char) for characters not inside a backtick span."""
    tick, i = 0, 0
    while i < len(line):
        if line[i] == "`":
            j = i
            while j < len(line) and line[j] == "`":
                j += 1
            run = j - i
            tick = 0 if tick == run else (tick or run)
            i = j
            continue
        yield i, line[i]
        i += 1


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p))
def test_no_collapsed_tables(path: Path):
    """A table squeezed onto one line renders as literal pipes on GitHub."""
    bad = [n for n, l in prose_lines(path)
           if len(l) > 200 and "|" in l and "---" in l and not SEPARATOR.match(l.strip())]
    assert not bad, f"{path.name}: table collapsed onto one line at {bad}"


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p))
def test_code_spans_are_not_fused(path: Path):
    """A code span run into the pipe, bullet or bold beside it reads as one word."""
    bad = []
    for n, line in prose_lines(path):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            for i, ch in outside_code(line):
                if ch == "|" and i + 1 < len(line) and line[i + 1] == "`":
                    bad.append((n, "pipe"))
        if re.match(r"^\s*-`", line):
            bad.append((n, "bullet"))
        if "**`" in line:
            bad.append((n, "bold"))
    assert not bad, f"{path.name}: code span fused to what precedes it at {bad}"


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p))
def test_plain_ascii_punctuation(path: Path):
    """No em dashes, en dashes or emoji. Maths symbols are fine."""
    bad = [(n, c) for n, l in prose_lines(path)
           for c in l if c in "—–" or EMOJI.match(c)]
    assert not bad, f"{path.name}: non-plain punctuation at {bad[:5]}"
