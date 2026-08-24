"""Draw the README figures from reports/*.csv.

Reads the saved runs only -- no API calls. The CI columns in these CSVs are
strings of the form ``0.550 [0.452,0.644]``; they are parsed here rather than
recomputed, so the intervals shown are the ones already quoted in the tables.

    python scripts/make_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CI = re.compile(r"([\d.]+)\s*\[([\d.]+)[,\s]+([\d.]+)\]")


def parse_ci(value: str) -> tuple[float, float, float]:
    """Turn ``0.550 [0.452,0.644]`` into (point, low, high)."""
    match = CI.search(str(value))
    if not match:
        return float(value), float("nan"), float("nan")
    return tuple(float(g) for g in match.groups())  # type: ignore[return-value]


def short(model: str) -> str:
    return model.split("/")[-1]


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
    table["label"] = table["model"].map(short) + "\n" + table["explainer"]
    table = table.sort_values("point")

    figure, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(table))
    # One call per row: matplotlib's ecolor takes a single colour, and the point
    # of this figure is which intervals still contain chance.
    for index, row in enumerate(table.itertuples()):
        includes_chance = row.low <= 0.5 <= row.high
        ax.errorbar(
            row.point, index,
            xerr=[[row.point - row.low], [row.high - row.point]],
            fmt="o", markersize=7, color="0.15",
            ecolor="#b2182b" if includes_chance else "#1a9850",
            elinewidth=2.4, capsize=4,
        )
    ax.axvline(0.5, color="0.35", ls="--", lw=1.3)
    ax.text(0.5, len(table) - 0.4, "  chance", fontsize=9, color="0.4")
    ax.set_yticks(positions)
    ax.set_yticklabels(table["label"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("edge-reading accuracy")
    ax.set_title(
        "Red intervals include chance. Two of six narrators cannot read the "
        "subgraph they are asked to describe.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def structure_or_label(out: Path) -> Path:
    """Does the narration follow the subgraph, or the predicted label?

    These are the same 100 narrations scored two ways. A model that describes the
    structure it was given tracks the left bar; one that paraphrases the label
    tracks the right.
    """
    table = pd.read_csv(REPORTS / "competence_vs_label.csv")
    table["label"] = table["model"].map(short) + "\n" + table["explainer"]
    table = table.sort_values("follows_structure")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.barh(positions - 0.2, table["follows_structure"], 0.4,
            label="follows the structure", color="#2166ac", edgecolor="0.3", lw=0.5)
    ax.barh(positions + 0.2, table["follows_label"], 0.4,
            label="follows the label", color="#b2182b", edgecolor="0.3", lw=0.5)
    ax.axvline(0.5, color="0.4", ls="--", lw=1.1)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["label"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("fraction of narrations")
    ax.set_title(
        "The weakest narrators track the label they were told and ignore the "
        "graph;\nthe strongest do the opposite.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(table.itertuples()):
        ax.text(1.03, index, f"n={row.n}", va="center", fontsize=7, color="0.45")
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def label_sensitivity(out: Path) -> Path:
    """How much the narration changes when only the label is flipped.

    Zero means the text is identical whichever label the model was handed, which
    is what the 8B narrator does -- it is not reading the label either, it is
    producing boilerplate that happens to agree with it half the time.
    """
    table = pd.read_csv(REPORTS / "competence_vs_label.csv")
    table["label"] = table["model"].map(short) + "\n" + table["explainer"]
    table = table.sort_values("label_sensitivity")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9.5, 4.6))
    colours = ["#b2182b" if v == 0 else "#2166ac" for v in table["label_sensitivity"]]
    ax.barh(positions, table["label_sensitivity"], color=colours,
            edgecolor="0.3", lw=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["label"], fontsize=8)
    ax.set_xlabel("fraction of narrations that change when the label is flipped")
    ax.set_title(
        "A narration that never changes is not an explanation of anything.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(table.itertuples()):
        ax.text(row.label_sensitivity + 0.008, index,
                f"{row.label_sensitivity:.3f}  (n={row.sens_n})",
                va="center", fontsize=8, color="0.35")
    ax.set_xlim(0, max(table["label_sensitivity"]) * 1.35 + 0.02)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def counterfactual(out: Path) -> Path:
    """Structure agreement across the 2x2 of decoy subgraph and flipped label."""
    table = pd.read_csv(REPORTS / "counterfactual_summary.csv")
    table["point"] = [parse_ci(v)[0] for v in table["structure_agreement"]]
    models = list(dict.fromkeys(table["model"]))
    cells = [("true", "true"), ("true", "flipped"), ("decoy", "true"), ("decoy", "flipped")]

    figure, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.2
    base = np.arange(len(models))
    for offset, (subgraph, label) in enumerate(cells):
        values = []
        for model in models:
            rows = table[(table.model == model) & (table.subgraph == subgraph)
                         & (table.label == label)]
            values.append(rows["point"].mean() if len(rows) else np.nan)
        ax.bar(base + (offset - 1.5) * width, values, width,
               label=f"subgraph={subgraph}, label={label}", edgecolor="0.3", lw=0.4)
    ax.axhline(0.5, color="0.4", ls="--", lw=1.1)
    ax.set_xticks(base)
    ax.set_xticklabels([short(m) for m in models], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("structure agreement")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Swapping in a decoy subgraph should move this. For the 8B narrator it "
        "does not.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def citation_validity(out: Path) -> Path:
    """Every narrator cites validly, including the ones describing the wrong graph.

    Citation validity checks that a cited node exists in the subgraph handed over.
    It never falls below 0.987, and sits at exactly 1.000 in 20 of 24 cells, while
    structure agreement over the same narrations ranges from 0.450 to 0.891. A
    metric that is pinned near its ceiling regardless cannot be evidence that the
    narration is faithful.
    """
    table = pd.read_csv(REPORTS / "counterfactual_summary.csv")
    table["citation"] = [parse_ci(v)[0] for v in table["citation_validity"]]
    table["structure"] = [parse_ci(v)[0] for v in table["structure_agreement"]]
    models = list(dict.fromkeys(table["model"]))

    figure, ax = plt.subplots(figsize=(9.5, 5.0))
    for model in models:
        rows = table[table.model == model]
        ax.scatter(rows["structure"], rows["citation"], s=70, alpha=0.8,
                   label=short(model), edgecolor="0.3", lw=0.5)
    ax.axhline(1.0, color="#1a9850", lw=1.4, ls="--")
    ax.axvline(0.5, color="0.5", lw=1.1, ls=":")
    ax.set_xlabel("structure agreement (is the description right?)")
    ax.set_ylabel("citation validity (does the cited node exist?)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.9, 1.02)
    ax.set_title(
        "Citation validity never drops below 0.987, including where the "
        "description\nis at chance. A valid citation is not a faithful "
        "explanation.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        edge_reading(FIGURES / "edge-reading.png"),
        structure_or_label(FIGURES / "structure-or-label.png"),
        label_sensitivity(FIGURES / "label-sensitivity.png"),
        counterfactual(FIGURES / "counterfactual.png"),
        citation_validity(FIGURES / "citation-validity.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
