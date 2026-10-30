"""Two-panel figure of the kappa between the models and the groups (baseline prompt), one per dataset (section 7.2).

Panel (a) is a heatmap of the kappa of every group and model, with its 95% range; panel (b) shows, for every pair of groups of a characteristic, the difference in
kappa of each model with its 95% range. Only numbers of ``kappa_by_group.csv`` and ``kappa_gaps.csv`` are drawn; nothing is recomputed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .kappa_tables import MODEL_NAME

WIDTH = 6.3                                   # inches: one text column of the ACM manuscript format
MODEL_COLOR = {"chatgpt": "#0072B2", "deepseek": "#D55E00", "gemini": "#009E73", "mistral": "#CC79A7"}          # Okabe-Ito
NO_EFFECT = "#BDBDBD"                         # dots and bars whose range includes 0
GROUP_LABEL = {"gen x+": "Gen X+", "gen z": "Gen Z", "millenial": "millennial", "not_high": "not high"}          # labels in the figure; the csv keeps the data's names
MINUS = "−"
BLOCK_GAP = 0.35                              # rows left empty between two characteristics
LEFT = 2.45                                   # inches left of both panels: characteristic label and group labels
CELL_UNIT, GAP_UNIT = 0.29, 0.17              # inches per row of panel (a) and of panel (b); shrunk (down to 90%) when the figure would be too tall
FONT = 8


def _number(value: float) -> str:
    shown = round(float(value), 2)
    return "0.00" if shown == 0 else (MINUS + f"{abs(shown):.2f}" if shown < 0 else f"{shown:.2f}")


def _group(name: str) -> str:
    return GROUP_LABEL.get(name, name)


def _characteristic(name: str, kind: str, one_line: bool = False) -> str:
    """``Gender (D)`` on two lines, so that the label fits the column left of the group names; on one line for a block of a single row (two lines would touch the next label)."""
    return f"{name.replace('_', ' ').capitalize()}{' ' if one_line else chr(10)}({'D' if kind == 'demographic' else 'N'})"


def _luminance(rgb) -> float:
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _text_color(rgb) -> str:
    """White or near-black text, whichever has the higher contrast ratio on the cell color."""
    light, dark = _luminance(rgb), _luminance((0.1, 0.1, 0.1))
    white_ratio, dark_ratio = 1.05 / (light + 0.05), (light + 0.05) / (dark + 0.05)
    return "white" if white_ratio >= dark_ratio else "#1a1a1a"


def _positions(counts: Sequence[int]) -> tuple:
    """The top row of every item (in row units, items of consecutive blocks separated by ``BLOCK_GAP``) and the total height."""
    tops, cursor = [], 0.0
    for i, count in enumerate(counts):
        tops.append(cursor)
        cursor += count + (BLOCK_GAP if i < len(counts) - 1 else 0.0)
    return tops, cursor


def kappa_figure(by_group: pd.DataFrame, gaps: pd.DataFrame, dataset: str, max_height: float = 8.0, prompt: str = "baseline", note: Optional[str] = None):
    """The figure of one dataset and a dict with what was drawn. ``by_group`` and ``gaps`` are the tables of ``compute_kappa`` (or their csv files)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    kappa = by_group[(by_group["dataset"] == dataset) & (by_group["prompt"] == prompt)]
    pair_table = gaps[(gaps["dataset"] == dataset) & (gaps["prompt"] == prompt)].copy()
    models = [m for m in MODEL_NAME if m in set(kappa["model"])]
    kind = kappa.drop_duplicates("characteristic").set_index("characteristic")["type"].to_dict()
    cells = {(r.characteristic, r.group, r.model): r for r in kappa.itertuples()}
    pair_rows = {(r.characteristic, r.group_a, r.group_b, r.model): r for r in pair_table.itertuples()}
    heat_blocks = [(c, list(part["group"].drop_duplicates())) for c, part in kappa.groupby("characteristic", sort=False)]
    pair_blocks = [(c, list(part[["group_a", "group_b"]].drop_duplicates().itertuples(index=False, name=None))) for c, part in pair_table.groupby("characteristic", sort=False)]

    all_pairs = sum(len(p) for _, p in pair_blocks)
    heat_units = _positions([len(g) for _, g in heat_blocks])[1]
    extras = 0.1 + 0.2 + 0.25 + (0.22 if note else 0.0) + 0.3 + 0.25 + 0.45 + 0.3 + 0.05                  # title, model names, note, gap, title, axis, legend, margins

    def height(pair_count_blocks, scale=1.0):
        return extras + scale * (CELL_UNIT * heat_units + GAP_UNIT * _positions(pair_count_blocks)[1])

    counts = [len(p) for _, p in pair_blocks]
    scale, filtered = 1.0, None
    if height(counts) > max_height:
        scale = (max_height - extras) / (height(counts) - extras)
        if scale < 0.9:                                              # shrinking is not enough: keep only the pairs with a range that excludes 0 for some model
            keep = {(r.characteristic, r.group_a, r.group_b) for r in pair_table.itertuples() if r.interval_excludes_zero}
            filtered = (all_pairs, len(keep))
            pair_blocks = [(c, [p for p in pairs if (c,) + tuple(p) in keep]) for c, pairs in pair_blocks]
            pair_blocks = [(c, pairs) for c, pairs in pair_blocks if pairs]
            counts = [len(p) for _, p in pair_blocks]
            scale = min(1.0, max(0.9, (max_height - extras) / (height(counts) - extras)))
    cell, unit = CELL_UNIT * scale, GAP_UNIT * scale
    heat_tops, heat_rows = _positions([len(g) for _, g in heat_blocks])
    pair_tops, pair_rows_total = _positions(counts)
    total = extras + cell * heat_rows + unit * pair_rows_total

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": FONT, "axes.unicode_minus": True}):
        fig = plt.figure(figsize=(WIDTH, total))
        W, H = WIDTH, total

        def rect(x, top, w, h):
            return [x / W, (H - top - h) / H, w / W, h / H]

        cursor = 0.1
        fig.text(0.05 / W, 1 - (cursor + 0.08) / H, "(a) Cohen's kappa between the model and each group", fontweight="bold", ha="left", va="center")
        cursor += 0.2 + 0.25
        width_a = W - LEFT - 0.5
        ax = fig.add_axes(rect(LEFT, cursor, width_a, cell * heat_rows))
        cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#2166AC", "#F7F7F7", "#E08214"])
        norm = Normalize(-1, 1)
        outlined = 0
        for (characteristic, groups), top in zip(heat_blocks, heat_tops):
            for i, group in enumerate(groups):
                for j, model in enumerate(models):
                    row = cells[(characteristic, group, model)]
                    color = cmap(norm(row.kappa))
                    ax.add_patch(Rectangle((j, top + i), 1, 1, facecolor=color, edgecolor="white", linewidth=0.6))
                    dagger = "†" if row.constant_model else ""
                    ax.text(j + 0.5, top + i + 0.5, f"{_number(row.kappa)}{dagger}\n[{_number(row.ci_low)}, {_number(row.ci_high)}]", ha="center", va="center",
                            color=_text_color(color), fontsize=FONT, linespacing=0.95)
                    if row.ci_low > 0 or row.ci_high < 0:
                        outlined += 1
                        ax.add_patch(Rectangle((j + 0.03, top + i + 0.03), 0.94, 0.94, fill=False, edgecolor="black", linewidth=1.8, zorder=5))
                ax.text(-0.02, top + i + 0.5, _group(group), ha="right", va="center", transform=ax.get_yaxis_transform())
            ax.text(-(LEFT - 0.05) / width_a, top + len(groups) / 2, _characteristic(characteristic, kind[characteristic]), ha="left", va="center",
                    fontweight="bold", transform=ax.get_yaxis_transform())
        ax.set_xlim(0, len(models))
        ax.set_ylim(heat_rows, 0)
        ax.set_xticks([j + 0.5 for j in range(len(models))])
        ax.set_xticklabels([MODEL_NAME[m] for m in models])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=2)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes(rect(LEFT + width_a + 0.12, cursor, 0.12, cell * heat_rows))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-1, 0, 1])
        bar.ax.set_yticklabels([MINUS + "1", "0", "1"])
        bar.set_label("Cohen's kappa", fontsize=FONT)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2)
        cursor += cell * heat_rows
        if note:
            fig.text(LEFT / W, 1 - (cursor + 0.13) / H, note, ha="left", va="center", fontsize=FONT, style="italic")
            cursor += 0.22

        cursor += 0.3
        fig.text(0.05 / W, 1 - (cursor + 0.08) / H, "(b) Difference in kappa between two groups of a characteristic", fontweight="bold", ha="left", va="center")
        cursor += 0.25
        width_b = W - LEFT - 0.1
        bx = fig.add_axes(rect(LEFT, cursor, width_b, unit * pair_rows_total))
        offsets = np.linspace(-0.27, 0.27, len(models))
        colored, low, high = 0, 0.0, 0.0
        for (characteristic, pairs), top in zip(pair_blocks, pair_tops):
            for i, (a, b) in enumerate(pairs):
                for offset, model in zip(offsets, models):
                    row = pair_rows[(characteristic, a, b, model)]
                    excluded = bool(row.interval_excludes_zero)
                    colored += excluded
                    low, high = min(low, row.ci_low), max(high, row.ci_high)
                    color = MODEL_COLOR[model] if excluded else NO_EFFECT
                    y = top + i + 0.5 + offset
                    bx.plot([row.ci_low, row.ci_high], [y, y], color=color, linewidth=1.3, solid_capstyle="butt", zorder=3 if excluded else 2)
                    bx.plot([row.gap], [y], "o", color=color, markersize=2.8, zorder=4 if excluded else 3)
                bx.text(-0.02, top + i + 0.5, f"{_group(a)} – {_group(b)}", ha="right", va="center", transform=bx.get_yaxis_transform())
            bx.text(-(LEFT - 0.05) / width_b, top + len(pairs) / 2, _characteristic(characteristic, kind[characteristic], one_line=len(pairs) == 1), ha="left", va="center",
                    fontweight="bold", transform=bx.get_yaxis_transform())
        pad = 0.05 * (high - low)
        bx.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)
        bx.set_xlim(low - pad, high + pad)
        bx.set_ylim(pair_rows_total, 0)
        bx.set_yticks([])
        bx.set_xlabel("Difference in kappa (group A – group B)", fontsize=FONT)
        bx.tick_params(axis="x", length=3, labelsize=FONT)
        for side in ("top", "right", "left"):
            bx.spines[side].set_visible(False)
        handles = [Line2D([0], [0], marker="o", color=MODEL_COLOR[m], markersize=4, linewidth=1.3, label=MODEL_NAME[m]) for m in models]
        handles.append(Line2D([0], [0], marker="o", color=NO_EFFECT, markersize=4, linewidth=1.3, label="range includes 0"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT, bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.0)
    info = {"dataset": dataset, "height_inches": round(total, 2), "outlined_cells": outlined, "colored_bars": int(colored),
            "colored_bars_in_csv": int(pair_table["interval_excludes_zero"].sum()), "pairs_in_csv": all_pairs, "pairs_drawn": sum(counts), "filtered": filtered}
    return fig, info


def toxic_note(by_group: pd.DataFrame, dataset: str, prompt: str = "baseline") -> str:
    """``Models label 92-98% of texts toxic``: the range over the models of the share of toxic labels (mean over the groups)."""
    part = by_group[(by_group["dataset"] == dataset) & (by_group["prompt"] == prompt)]
    share = part.groupby("model")["pct_toxic_model"].mean()
    return f"Models label {share.min():.0f}–{share.max():.0f}% of texts toxic"


def save_kappa_figures(by_group: pd.DataFrame, gaps: pd.DataFrame, folder: Path, datasets: Optional[Sequence[str]] = None, formats=("pdf", "png"),
                       max_height: Optional[Dict[str, float]] = None, prompt: str = "baseline") -> pd.DataFrame:
    """Write ``kappa_<dataset>.pdf`` and ``.png`` (300 dpi) of every dataset into ``folder`` (created if needed). Returns what was drawn, one row per dataset.
    MHS gets a line under panel (a) with the range of its toxic labels; POPQUORN may be 10 inches tall (the others 8) before its pairs are filtered."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    limits = {"popquorn": 10.0, **(max_height or {})}
    rows = []
    for dataset in datasets or list(dict.fromkeys(by_group["dataset"])):
        note = toxic_note(by_group, dataset, prompt) if dataset == "mhs" else None
        fig, info = kappa_figure(by_group, gaps, dataset, max_height=limits.get(dataset, 8.0), prompt=prompt, note=note)
        for extension in formats:
            fig.savefig(folder / f"kappa_{dataset}.{extension}", dpi=300)
        plt.close(fig)
        rows.append(info)
    return pd.DataFrame(rows)
