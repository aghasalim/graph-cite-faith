"""Draw the README figures from reports/*.csv and reports/runs.jsonl.

Reads the saved runs only, no API calls. The CI columns in these CSVs are
strings of the form ``0.550 [0.452,0.644]``; they are parsed here rather than
recomputed, so the intervals shown are the ones already quoted in the tables.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CI = re.compile(r"([\d.]+)\s*\[([\d.]+)[,\s]+([\d.]+)\]")

# Two colours carry meaning in the README prose and are used the same way in
# every figure here: structure-following against label-following, and an
# interval that still contains chance against one that does not.
STRUCTURE, LABEL, CLEAR = PALETTE[0], PALETTE[1], PALETTE[2]
CHANCE = "#8c8c8c"

# One colour per model, fixed here so a model is the same colour in every figure
# and in the GIF, whatever order its rows happen to arrive in.
ORDER = ["llama-3.1-8b", "llama-3.3-70b", "qwen3.6-27b", "gpt-oss-120b", "gpt-oss-20b"]
COLOUR = dict(zip(ORDER, PALETTE))
MARKER = dict(zip(ORDER, ["o", "s", "v", "^", "D"]))
MIN_NODES = 30   # the reporting floor the harness enforces


def parse_ci(value: str) -> tuple[float, float, float]:
    """Turn ``0.550 [0.452,0.644]`` into (point, low, high)."""
    match = CI.search(str(value))
    if not match:
        return float(value), float("nan"), float("nan")
    return tuple(float(g) for g in match.groups())  # type: ignore[return-value]


def short(model: str) -> str:
    """Name the model the way the README tables name it."""
    name = model.split("/")[-1]
    for suffix in ("-instant", "-versatile"):
        name = name.removesuffix(suffix)
    return name


def points(column) -> list[float]:
    return [parse_ci(v)[0] for v in column]


def edge_reading(out: Path) -> Path:
    """Can the narrator read the subgraph it is describing?

    Chance is 0.5. The 8B model sits on it; the 20B model reads the structure 90%
    of the time. Whether narration is faithful is a capability question before it
    is an explainability question.
    """
    table = pd.read_csv(REPORTS / "edge_reading_control.csv")
    parsed = table["edge_reading_accuracy"].map(parse_ci)
    table["point"] = [p[0] for p in parsed]
    table["low"] = [p[1] for p in parsed]
    table["high"] = [p[2] for p in parsed]
    table["row"] = [f"{short(m)}\n{e}, n={n}"
                    for m, e, n in zip(table["model"], table["explainer"], table["n"])]
    table = table.sort_values("point")

    figure, ax = plt.subplots(figsize=(8.8, 4.6))
    # One call per row: matplotlib's ecolor takes a single colour, and the point
    # of this figure is which intervals still contain chance.
    for index, row in enumerate(table.itertuples()):
        includes_chance = row.low <= 0.5 <= row.high
        ax.errorbar(
            row.point, index,
            xerr=[[row.point - row.low], [row.high - row.point]],
            fmt="o", markersize=7, color="#222222",
            ecolor=LABEL if includes_chance else CLEAR,
            elinewidth=2.6, capsize=4, zorder=3,
        )
        ax.text(row.high + 0.018, index, f"{row.point:.3f}", va="center",
                fontsize=9, color="#5a5a5a")
    ax.axvline(0.5, color=CHANCE, ls="--", lw=1.2, zorder=1)
    ax.text(0.5, -0.85, "chance", ha="center", va="center", fontsize=9.5, color=CHANCE)
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["row"], fontsize=8.5)
    ax.set_ylim(-1.25, len(table) - 0.35)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("edge-reading accuracy (fraction of subgraphs named correctly)")
    ax.grid(axis="y", visible=False)
    titled(ax, "Two of the five narrators cannot say which shape they were shown",
           "control probe, no predicted class in the prompt, six arms over five "
           "models, red still contains chance")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def structure_or_label(out: Path) -> Path:
    """Does the narration follow the subgraph, or the predicted label?

    These are the same narrations scored two ways, in the one cell where the two
    answers differ: the decoy subgraph shown with the model's real label.
    """
    table = pd.read_csv(REPORTS / "competence_vs_label.csv")
    table["row"] = [f"{short(m)}\n{e}, n={n}"
                    for m, e, n in zip(table["model"], table["explainer"], table["n"])]
    table = table.sort_values("follows_structure")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9.4, 4.9))
    ax.barh(positions + 0.19, table["follows_structure"], 0.38,
            label="names the structure it was shown", color=STRUCTURE)
    ax.barh(positions - 0.19, table["follows_label"], 0.38,
            label="names the label it was told", color=LABEL)
    ax.axvline(0.5, color=CHANCE, ls="--", lw=1.2)
    ax.text(0.5, -0.9, "chance", ha="center", va="center", fontsize=9.5, color=CHANCE)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["row"], fontsize=8.5)
    ax.set_ylim(-1.3, len(table) - 0.35)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of narrations in that cell")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    titled(ax, "The narrators that cannot read the graph name the label instead",
           "decoy subgraph shown with the model's real label, so the two answers "
           "point at different shapes")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def label_sensitivity(out: Path) -> Path:
    """How much the narration changes when only the label is flipped.

    Zero means the text names the same shape whichever label the model was
    handed, which is what the 8B narrator does. It is not reading the label
    either, it is producing boilerplate that agrees with it half the time.
    """
    table = pd.read_csv(REPORTS / "competence_vs_label.csv")
    table["row"] = [f"{short(m)}\n{e}"
                    for m, e in zip(table["model"], table["explainer"])]
    table = table.sort_values("label_sensitivity")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9.0, 4.4))
    colours = [LABEL if v == 0 else STRUCTURE for v in table["label_sensitivity"]]
    ax.barh(positions, table["label_sensitivity"], color=colours)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["row"], fontsize=8.5)
    ax.set_xlabel("fraction of narration pairs whose named motif changes")
    ax.grid(axis="y", visible=False)
    for index, row in enumerate(table.itertuples()):
        ax.text(row.label_sensitivity + 0.006, index,
                f"{row.label_sensitivity:.3f}  (n={row.sens_n} pairs)",
                va="center", fontsize=8.5, color="#5a5a5a")
    ax.set_xlim(0, max(table["label_sensitivity"]) * 1.30)
    titled(ax, "One narrator never changes its answer when the label is flipped",
           "same node, same edges, temperature 0, one word of the prompt different; "
           "both zero rows are llama-3.1-8b")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def counterfactual(out: Path) -> Path:
    """Structure agreement across the 2x2 of decoy subgraph and flipped label.

    Only two of the four cells can tell structure-following from label-following.
    In the other two the label points at the shape that is actually there, so
    agreeing with one is agreeing with the other; those are drawn pale.
    """
    table = pd.read_csv(REPORTS / "counterfactual_summary.csv")
    table = table[table["explainer"] == "gnnexplainer"]
    table["point"] = points(table["structure_agreement"])

    control = pd.read_csv(REPORTS / "edge_reading_control.csv")
    control = control[control["explainer"] == "gnnexplainer"]
    control["point"] = points(control["edge_reading_accuracy"])
    models = list(control.sort_values("point")["model"])

    cells = [
        (("true", "true"), "#cfcfcf", "true subgraph, true label (aligned)"),
        (("decoy", "flipped"), "#9e9e9e", "decoy subgraph, flipped label (aligned)"),
        (("decoy", "true"), STRUCTURE, "decoy subgraph, true label (conflicting)"),
        (("true", "flipped"), LABEL, "true subgraph, flipped label (conflicting)"),
    ]

    figure, ax = plt.subplots(figsize=(10.2, 5.0))
    width = 0.2
    base = np.arange(len(models))
    for offset, ((subgraph, label), colour, name) in enumerate(cells):
        values = [table[(table.model == m) & (table.subgraph == subgraph)
                        & (table.label == label)]["point"].iloc[0] for m in models]
        ax.bar(base + (offset - 1.5) * width, values, width, label=name, color=colour)
    ax.axhline(0.5, color=CHANCE, ls="--", lw=1.2)
    ax.text(len(models) - 0.45, 0.512, "chance", fontsize=9.5, color=CHANCE,
            ha="right", va="bottom")
    ax.set_xticks(base)
    ax.set_xticklabels([f"{short(m)}\nn={table[table.model == m]['n'].iloc[0]} per cell"
                        for m in models], fontsize=9)
    ax.set_ylabel("structure agreement (fraction of narrations)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    titled(ax, "Once the label contradicts the subgraph, both Llama narrators fall to chance",
           "GNNExplainer subgraphs, models ordered by edge-reading accuracy; the pale "
           "cells cannot separate structure from label")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def citation_validity(out: Path) -> Path:
    """Every narrator cites validly, including the ones describing the wrong graph.

    Citation validity checks that a cited node exists in the subgraph handed over.
    It never falls below 0.987, while structure agreement over the same narrations
    ranges from 0.450 to 0.891. Both axes are drawn on the same scale, so the flat
    band at the top is not an artefact of zooming in on it.
    """
    table = pd.read_csv(REPORTS / "counterfactual_summary.csv")
    table["citation"] = points(table["citation_validity"])
    table["structure"] = points(table["structure_agreement"])
    models = list(dict.fromkeys(table["model"]))

    figure, ax = plt.subplots(figsize=(9.6, 4.4))
    for model in models:
        rows = table[table.model == model]
        name = short(model)
        ax.scatter(rows["structure"], rows["citation"], s=66, marker=MARKER[name],
                   color=COLOUR[name], alpha=0.85, edgecolor="white", lw=0.7,
                   zorder=3, label=name)
    ax.axvline(0.5, color=CHANCE, lw=1.2, ls="--", zorder=1)
    ax.text(0.5, 0.425, " chance", fontsize=9.5, color=CHANCE, ha="left")
    ax.set_xlabel("structure agreement (fraction naming the right shape)")
    ax.set_ylabel("citation validity\n(fraction of cited ids that exist)")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.03)
    ax.text(1.01, 0.94, "lowest cell is 0.987: qwen3.6-27b invented 8 ids out of 919",
            fontsize=8.5, color="#5a5a5a", ha="right", va="top")
    ax.legend(loc="lower right", ncol=3)
    titled(ax, "Citation validity sits at the ceiling whether the description is right or not",
           "one point per cell, 24 cells over 6 narrator arms, both axes on the same scale")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def competence_vs_label(out: Path) -> Path:
    """The two ways of asking whether a model leans on the label.

    Label agreement in the decisive cell is not independent of edge reading: a
    model that never answers "neither" has label agreement identically equal to
    1 - structure agreement there. The within-node contrast is, and it finds
    nothing.
    """
    table = pd.read_csv(REPORTS / "competence_vs_label.csv")
    table = table[table["explainer"] == "gnnexplainer"].reset_index(drop=True)

    panels = (
        ("follows_label", "label agreement (fraction of narrations)",
         "The naive measure just tracks reading ability",
         "label agreement in the decoy-subgraph cell"),
        ("label_sensitivity", "label sensitivity (fraction of pairs that move)",
         "The within-node measure finds nothing",
         "same node and edges, one word of the prompt different"),
    )
    # A few labels would otherwise land on a neighbour's marker.
    nudge = {("follows_label", "llama-3.3-70b"): (8, -14),
             ("label_sensitivity", "gpt-oss-120b"): (0, 13),
             ("label_sensitivity", "gpt-oss-20b"): (7, -13)}

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, (column, ylabel, title, subtitle) in zip(axes, panels):
        x, y = table["edge_reading"].to_numpy(), table[column].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        grid = np.linspace(0.45, 0.95, 2)
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(grid, slope * grid + intercept, color="#aaaaaa", ls="--", lw=1.2, zorder=1)
        for row in table.itertuples():
            name = short(row.model)
            ax.scatter(row.edge_reading, getattr(row, column), s=70,
                       color=COLOUR[name], zorder=3)
            offset = nudge.get((column, name), (7, 7))
            ax.annotate(name, (row.edge_reading, getattr(row, column)),
                        textcoords="offset points", xytext=offset, fontsize=8.5,
                        color="#5a5a5a",
                        ha="center" if offset[0] == 0 else
                        ("right" if offset[0] < 0 else "left"))
        ax.set_xlim(0.42, 1.06)
        ax.set_ylim(-0.08, 0.72)
        ax.set_xlabel("edge-reading accuracy (no label in the prompt)")
        ax.set_ylabel(ylabel)
        titled(ax, title, f"{subtitle}, 5 models, r = {r:+.2f}")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def control_runs() -> dict[str, np.ndarray]:
    """Per model, the control probes in the order they came back.

    Same filtering as the report: a retry supersedes the unparsed attempt it
    replaces, a node counts only if all five of its conditions landed, and an arm
    counts only if it reached the 30-node floor.
    """
    rows = [json.loads(line) for line in (REPORTS / "runs.jsonl").read_text().splitlines()]
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["model", "kind", "explainer", "node", "subgraph", "label"], keep="last")
    per = frame.groupby(["model", "explainer", "node"])["kind"].agg(
        narrate=lambda s: (s == "narrate").sum(),
        control=lambda s: (s == "control").sum())
    complete = set(per[(per.narrate == 4) & (per.control == 1)].index)
    frame = frame[[t in complete for t in
                   zip(frame["model"], frame["explainer"], frame["node"])]]
    sizes = frame[frame.kind == "narrate"].groupby(["model", "explainer"])["node"].nunique()
    keep = {k for k, n in sizes.items() if n >= MIN_NODES}
    frame = frame[[(m, e) in keep for m, e in zip(frame["model"], frame["explainer"])]]
    control = frame[(frame.kind == "control") & (frame.explainer == "gnnexplainer")]
    return {model: group["agrees_with_structure"].to_numpy(dtype=float)
            for model, group in control.groupby("model", sort=False)}


def anim_edge_reading(out: Path, fps: int = 16, hold: int = 16) -> Path:
    """The control arm accumulating one probe at a time.

    Nothing is sampled or fitted here, so the GIF is the same on every run. The
    curve is a running mean of the committed per-probe outcomes and the assert
    below is what says so: each curve has to land on the number in
    edge_reading_control.csv.
    """
    series = control_runs()
    final = pd.read_csv(REPORTS / "edge_reading_control.csv")
    final = final[final["explainer"] == "gnnexplainer"].set_index("model")

    curves = {}
    for model, values in series.items():
        run = np.cumsum(values) / np.arange(1, len(values) + 1)
        point, low, high = parse_ci(final.loc[model, "edge_reading_accuracy"])
        assert abs(run[-1] - point) < 5e-4, f"{model}: {run[-1]:.3f} != {point:.3f}"
        curves[model] = (run, point, low, high)

    longest = max(len(run) for run, *_ in curves.values())
    figure, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, longest * 1.10)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("control probes scored")
    ax.set_ylabel("cumulative edge-reading accuracy")
    titled(ax, "Both Llama narrators answer cycle to every probe",
           "control arm in the order the replies landed, the two Llama curves "
           "coincide, bars are the committed 95% intervals")

    ax.axvspan(0, MIN_NODES, color="#f2f2f2", zorder=0)
    ax.text(MIN_NODES / 2, 0.045, "below the 30-node\nreporting floor", fontsize=8.5,
            color="#8a8a8a", ha="center", va="center")
    ax.axhline(0.5, color=CHANCE, ls="--", lw=1.2, zorder=1)
    ax.text(longest * 0.85, 0.487, "chance", fontsize=9.5, color=CHANCE,
            ha="right", va="top")

    art = {}
    for model, (run, point, low, high) in curves.items():
        colour = COLOUR[short(model)]
        # The two Llama arms answer the same shape to every probe, so their
        # curves coincide exactly. Dash the upper one or one of them vanishes.
        dashed = model.startswith("llama-3.3")
        art[model] = {
            "line": ax.plot([], [], color=colour, lw=1.8 if dashed else 3.0,
                            zorder=4 if dashed else 3,
                            ls=(0, (3, 2.5)) if dashed else "-", label=short(model))[0],
            "head": ax.plot([], [], "o", color=colour, markersize=6, zorder=5)[0],
            "ci": ax.plot([], [], color=colour, lw=2.6, zorder=5, solid_capstyle="butt")[0],
            "text": ax.text(len(run) + longest * 0.012, point, "", fontsize=8.5,
                            color=colour, va="center"),
        }
    ax.legend(loc="lower right", ncol=2)
    readout = ax.text(0.985, 0.965, "", transform=ax.transAxes, fontsize=9.5,
                      color="#5a5a5a", ha="right", va="top")

    def draw(frame: int):
        i = min(frame + 1, longest)
        readout.set_text(f"probes scored: {i}")
        for model, (run, point, low, high) in curves.items():
            k = min(i, len(run))
            a = art[model]
            a["line"].set_data(np.arange(1, k + 1), run[:k])
            a["head"].set_data([k], [run[k - 1]])
            if i >= len(run):
                a["ci"].set_data([len(run), len(run)], [low, high])
                a["text"].set_text(f"{point:.3f}")
        return []

    anim = FuncAnimation(figure, draw, frames=longest + hold,
                         interval=1000 // fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return shrink(out)


def shrink(path: Path) -> Path:
    """Rewrite every frame onto one shared palette. Roughly halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], loop=0,
                      duration=durations, optimize=True)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        edge_reading(FIGURES / "edge-reading.png"),
        structure_or_label(FIGURES / "structure-or-label.png"),
        label_sensitivity(FIGURES / "label-sensitivity.png"),
        counterfactual(FIGURES / "counterfactual.png"),
        citation_validity(FIGURES / "citation-validity.png"),
        competence_vs_label(FIGURES / "competence-vs-label.png"),
        anim_edge_reading(FIGURES / "edge-reading-accumulates.gif"),
    ):
        print(f"-> {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
