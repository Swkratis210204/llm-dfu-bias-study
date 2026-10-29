"""Exploring the AU texts across all the datasets. Nothing here asks a model for anything, and nothing changes the AU analysis: it only reads what
the analysis already computed (and the table of the rows given to the models).

``save_total_au(results, models, folder)`` takes ``{dataset name: DatasetResult}`` and ``models`` (the table ``aposteriori_for_models.csv``) and writes
into ``folder``:

    au_summary.csv              per dataset: AU texts and rows, how many dimensions a text is AU in, ratings and DFU of the AU texts, rows given to the models
    au_funnel.csv               what happens to the texts of every dimension (not polarized, too few groups, a group not unimodal, AU), merged
    au_groups.csv               per dataset, dimension and group: the AU texts it takes part in (at the start, at the end), the status under the rule of
                                at least 10 AU texts, and the ratings and share of toxic ratings in the AU texts
    au_label_patterns.csv       per dataset and dimension: the AU texts whose groups give different labels, and the group labels decided by the mean
    au_dimension_overlap.csv    per dataset: how many AU texts are AU in two dimensions at once
    au_rating_distribution.csv  the ratings of the AU texts, per dataset
    au_models_input.csv         per dataset, dimension and group: its share of all the AU rows and of the rows given to the models
    *.png / *.pdf               the funnel, the AU texts per group, the label patterns, the ratings and toxic share of the AU texts, the overlap of
                                the dimensions, and the sample given to the models for Kumar
    README.md                   what each file is, the AU rule and what to be careful about
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .dfu_total import _ordered_results
from .eda import PLOT_LABEL, REASONS, eda_tables
from .eda_total import (BLUE, DATASET_LABEL, GROUP_PLOT_LABEL, RED, _by_dimension_figure, _ordered, _rating_figure, _save_figure,
                        toxicity_by_group)
from .pipeline import DatasetResult

__all__ = ["au_summary", "au_groups", "au_label_patterns", "au_dimension_overlap", "au_models_input", "save_total_au", "save_kumar_sample_figure"]

ORANGE = "#DD8452"
GREY = "#BBBBBB"
FUNNEL_COLORS = ["#DDDDDD", "#BBBBBB", "#888888", ORANGE]


def _tables(results: Dict[str, DatasetResult]) -> Dict[str, dict]:
    return {name: eda_tables(result) for name, result in results.items()}


def _models_keys(models: pd.DataFrame, dataset: str) -> pd.DataFrame:
    part = models[models["dataset"] == dataset]
    return part.assign(item_id=part["item_id"].astype(str))


def au_summary(results: Dict[str, DatasetResult], models: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the AU texts (distinct texts, and rows = text and dimension), the dimensions a text is AU in, the ratings and the DFU of the AU texts, the
    share of toxic ratings in them, and the rows given to the models."""
    rows = {}
    for name, result in _ordered_results(results).items():
        au = result.au
        per_text = au.groupby("item_id").agg(ratings=("n_ratings", "first"), dfu=("dfu", "first"), dimensions=("dimension", "size"))
        clean = result.clean[result.clean["item_id"].isin(per_text.index)]
        dims = per_text["dimensions"]
        rows[name] = {
            "texts": result.clean["item_id"].nunique(), "AU texts": len(per_text), "AU texts %": round(100 * len(per_text) / result.clean["item_id"].nunique(), 1),
            "AU rows (text x dimension)": len(au), "AU in 1 dimension": int((dims == 1).sum()), "AU in 2 dimensions": int((dims == 2).sum()),
            "AU in 3 or more dimensions": int((dims >= 3).sum()), "rows given to the models": len(_models_keys(models, name)),
            "ratings in the AU texts": len(clean), "ratings per AU text (median)": float(per_text["ratings"].median()),
            "ratings called toxic % (AU texts)": round(100 * float((clean["rating"] >= result.spec.toxic_threshold).mean()), 1),
            "mean DFU (AU texts)": round(float(per_text["dfu"].mean()), 3)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("dataset")           # from_dict keeps whole numbers whole


def au_groups(results: Dict[str, DatasetResult], tables: Dict[str, dict] = None) -> pd.DataFrame:
    """Per dataset, dimension and group: the AU texts the group takes part in at the start and at the end of the removal of thin groups, the status under the
    rule (a group in fewer than ``min_au_texts`` AU texts is removed, unless every group is thin or only two groups are left), and the ratings and the share of
    toxic ratings in all texts and in the AU texts. The AU columns are left empty for a removed group: it took no part in the comparison."""
    tables = tables or _tables(results)
    rows = []
    for name, result in _ordered_results(results).items():
        groups = tables[name]["groups"]
        minimum = result.spec.min_au_texts
        final = {dimension: int(pr.history.iloc[-1]["AU texts"]) for dimension, pr in result.pruned.items()}
        for (dimension, group), r in result.starting_list.iterrows():
            removed = bool(r["removed"])
            end = None if removed else int(r["AU texts at the end"])
            status = (f"removed (fewer than {minimum} AU texts)" if removed
                      else f"kept, below {minimum} (the rule could not remove it)" if end < minimum else "kept")
            g = groups.loc[(dimension, group)]
            rows.append({"dataset": name, "dimension": dimension, "group": group, "AU texts at the start": int(r["AU texts at start"]),
                         "AU texts at the end": end, "AU texts of the dimension (final)": final[dimension], "status": status,
                         "ratings (all texts)": g["ratings"], "toxic % (all texts)": g["toxic %"],
                         "ratings (AU texts)": None if removed else g["AU ratings"], "toxic % (AU texts)": None if removed else g["AU toxic %"]})
    table = pd.DataFrame(rows)
    for column in ("AU texts at the end", "ratings (AU texts)"):                      # whole numbers with gaps for the removed groups, not floats
        table[column] = table[column].astype("Int64")
    return table


def au_label_patterns(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """Per dataset and dimension: the AU texts, how many have groups with different labels (the texts that separate perspectives), how many have all groups
    toxic or all civil, and how many group labels were decided by the mean (a tie between toxic and civil ratings)."""
    rows = []
    for name, result in _ordered_results(results).items():
        for dimension, pr in result.pruned.items():
            au = pr.result[pr.result["au"]]
            sets = [set(labels.values()) for labels in au["group_label"]]
            labels_total, by_mean = sum(len(l) for l in au["group_label"]), sum(len(t) for t in au["tie_broken"])
            disagree = sum(len(s) > 1 for s in sets)
            rows.append({"dataset": name, "dimension": dimension, "AU texts": len(au), "groups disagree": disagree, "all toxic": sum(s == {"toxic"} for s in sets),
                         "all civil": sum(s == {"civil"} for s in sets), "groups disagree %": round(100 * disagree / len(au), 1) if len(au) else np.nan,
                         "group labels": labels_total, "decided by the mean": by_mean,
                         "decided by the mean %": round(100 * by_mean / labels_total, 1) if labels_total else np.nan})
    return pd.DataFrame(rows)


def au_dimension_overlap(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """Per dataset and pair of dimensions: the texts that are AU in the first, in the second, and in both."""
    rows = []
    for name, result in _ordered_results(results).items():
        sets = {dimension: set(part["item_id"]) for dimension, part in result.au.groupby("dimension")}
        for a, b in combinations(list(result.spec.dimensions), 2):
            sa, sb = sets.get(a, set()), sets.get(b, set())
            rows.append({"dataset": name, "dimension A": a, "dimension B": b, "AU in A": len(sa), "AU in B": len(sb), "AU in both": len(sa & sb)})
    return pd.DataFrame(rows)


def au_models_input(results: Dict[str, DatasetResult], models: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, dimension and group: how many of all the AU rows, and of the rows given to the models, contain the group, and the share of all the group
    appearances in the dimension. They differ only for Kumar, where the models got a sample, not every AU row."""
    rows = []
    for name, result in _ordered_results(results).items():
        given = _models_keys(models, name)
        all_counts, given_counts = {}, {}
        for dimension, group_label in zip(result.au["dimension"], result.au["group_label"]):
            for group in group_label:
                all_counts[(dimension, group)] = all_counts.get((dimension, group), 0) + 1
        for dimension, group_label in zip(given["dimension"], given["group_label"]):
            for group in json.loads(group_label):
                given_counts[(dimension, group)] = given_counts.get((dimension, group), 0) + 1
        for dimension in result.spec.dimensions:
            keys = sorted({g for d, g in all_counts if d == dimension} | {g for d, g in given_counts if d == dimension}, key=str)
            total_all = sum(all_counts.get((dimension, g), 0) for g in keys)
            total_given = sum(given_counts.get((dimension, g), 0) for g in keys)
            for group in keys:
                a, b = all_counts.get((dimension, group), 0), given_counts.get((dimension, group), 0)
                rows.append({"dataset": name, "dimension": dimension, "group": group, "AU rows with the group (all)": a,
                             "share of the group appearances % (all)": round(100 * a / total_all, 1) if total_all else np.nan,
                             "AU rows with the group (given to the models)": b,
                             "share of the group appearances % (given to the models)": round(100 * b / total_given, 1) if total_given else np.nan})
    return pd.DataFrame(rows)


def _rating_shares_au(tables: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, tab in tables.items():
        dist = tab["rating_distribution"]
        for rating, r in dist.iterrows():
            rows.append({"dataset": name, "rating": int(rating), "ratings": int(r["AU texts: ratings"]), "%": r["AU texts: %"]})
    return pd.DataFrame(rows)


def _merged_funnel(results: Dict[str, DatasetResult], tables: Dict[str, dict]) -> pd.DataFrame:
    frames = [tables[name]["funnel"].reset_index().rename(columns={"index": "dimension"}).assign(dataset=name) for name in _ordered_results(results)]
    table = pd.concat(frames, ignore_index=True)
    return table[["dataset"] + [c for c in table.columns if c != "dataset"]]


# ---------------------------------------------------------------------------------------------------------------------------------------------
def _grid(plt, n: int, height: float, width: float = 13):
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(width, height * rows), squeeze=False)
    return fig, axes.flatten()


def _funnel_figure(plt, funnel: pd.DataFrame):
    """One panel per dataset: the share of the texts of every dimension that is not polarized, has too few groups, has a group that is not unimodal, or is AU."""
    names = _ordered(funnel["dataset"])
    fig, axes = _grid(plt, len(names), 3.2)
    for ax, name in zip(axes, names):
        part = funnel[funnel["dataset"] == name].reset_index(drop=True)
        left = np.zeros(len(part))
        for reason, color in zip(REASONS, FUNNEL_COLORS):
            share = 100 * part[reason].to_numpy() / part["texts"].to_numpy()
            ax.barh(part["dimension"], share, left=left, color=color)
            left += share
        ax.invert_yaxis()
        ax.set(title=DATASET_LABEL.get(name, name), xlabel="% of the texts")
        ax.set_xlim(0, 100)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in FUNNEL_COLORS], [PLOT_LABEL.get(r, r) for r in REASONS], loc="lower center", ncol=4)
    return fig


def _groups_figure(plt, table: pd.DataFrame, minimum: int):
    """One panel per dataset: the AU texts every group takes part in (at the end of the removal of thin groups; at the start for a removed group), the
    groups removed under the rule in grey and the groups kept below the minimum in orange; the dashed line is the minimum."""
    names = _ordered(table["dataset"])
    fig, axes = _grid(plt, len(names), 5.2, width=14)
    colors = {"kept": BLUE, "removed": GREY, "below": ORANGE}
    for ax, name in zip(axes, names):
        part = table[table["dataset"] == name].reset_index(drop=True)
        kind = ["removed" if s.startswith("removed") else "below" if s.startswith("kept, below") else "kept" for s in part["status"]]
        counts = part["AU texts at the end"].fillna(part["AU texts at the start"]).astype(float)
        labels = part["dimension"] + ": " + part["group"].astype(str).map(lambda g: GROUP_PLOT_LABEL.get(g, g))
        ax.barh(labels, counts, color=[colors[k] for k in kind])
        ax.axvline(minimum, color="black", linestyle="--", linewidth=0.8)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
        ax.set(title=DATASET_LABEL.get(name, name), xlabel="AU texts the group takes part in")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("kept", "removed", "below")],
               ["kept", f"removed (fewer than {minimum} AU texts)", f"kept, below {minimum} (the rule could not remove it)"], loc="lower center", ncol=3)
    return fig


def _patterns_figure(plt, patterns: pd.DataFrame):
    """One panel per dataset: for every dimension, the share of the AU texts whose groups give different labels, or are all toxic, or all civil."""
    names = _ordered(patterns["dataset"])
    fig, axes = _grid(plt, len(names), 3.2)
    columns, colors = ["groups disagree", "all toxic", "all civil"], [ORANGE, RED, BLUE]
    for ax, name in zip(axes, names):
        part = patterns[patterns["dataset"] == name].reset_index(drop=True)
        left = np.zeros(len(part))
        for column, color in zip(columns, colors):
            share = 100 * part[column].to_numpy() / part["AU texts"].to_numpy()
            ax.barh(part["dimension"], share, left=left, color=color)
            left += share
        ax.invert_yaxis()
        ax.set(title=DATASET_LABEL.get(name, name), xlabel="% of the AU texts")
        ax.set_xlim(0, 100)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], ["groups disagree", "all groups toxic", "all groups civil"], loc="lower center", ncol=3)
    return fig


def _overlap_figure(plt, summary: pd.DataFrame):
    """The share of the AU texts of every dataset that are AU in one, two, or three or more dimensions."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    columns = ["AU in 1 dimension", "AU in 2 dimensions", "AU in 3 or more dimensions"]
    colors, left = ["#DDDDDD", "#999999", "#555555"], np.zeros(len(summary))
    for column, color in zip(columns, colors):
        share = 100 * summary[column].to_numpy() / summary["AU texts"].to_numpy()
        ax.barh([DATASET_LABEL.get(n, n) for n in summary.index], share, left=left, color=color)
        left += share
    ax.invert_yaxis()
    ax.set(xlabel="% of the AU texts")
    ax.set_xlim(0, 100)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], ["AU in 1 dimension", "in 2", "in 3 or more"], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3)
    return fig


def _sample_figure(plt, models_input: pd.DataFrame):
    """Kumar: the share of the group appearances of every dimension in all the AU rows against the sample given to the models."""
    part = models_input[models_input["dataset"] == "kumar"]
    dimensions = list(dict.fromkeys(part["dimension"]))
    rows = math.ceil(len(dimensions) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.3 * rows), squeeze=False, sharey=True)
    for ax, dimension in zip(axes.flatten(), dimensions):
        sub = part[part["dimension"] == dimension].reset_index(drop=True)
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["share of the group appearances % (all)"], width=0.4, color=BLUE, label="all AU rows")
        ax.bar(x + 0.2, sub["share of the group appearances % (given to the models)"], width=0.4, color=ORANGE, label="sample given to the models")
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_PLOT_LABEL.get(g, g) for g in sub["group"]], fontsize=9)
        ax.set_title(dimension)
    for ax in axes.flatten()[len(dimensions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of the group appearances")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)                              # below the panels: inside one it covered the bars
    return fig


README = """# AU texts across the datasets (exploratory)

A text is **polarized** when its ratings are not unimodal (DFU above 0). It is **AU** (a-posteriori unimodal) for a characteristic when it is polarized, at
least two annotator groups of that characteristic have two or more ratings on it, and every one of those groups is internally unimodal. The AU texts of a
characteristic are then the texts on which the disagreement is explained by who the annotators are. A text can be AU for several characteristics; a **row**
is one text under one characteristic. A group in fewer than 10 AU texts (counted per characteristic) is removed and the analysis is repeated, unless every
group is below 10 or only two groups are left (then nothing is removed). All numbers use the cleaned ratings.

## Files

| file | content |
|---|---|
| `au_summary.csv` | per dataset: AU texts and rows, the dimensions a text is AU in, ratings, DFU and toxic share of the AU texts, rows given to the models |
| `au_funnel.csv`, `au_funnel.png` | what happens to the texts of every dimension: not polarized, too few groups, a group not unimodal, AU |
| `au_groups.csv`, `au_texts_per_group.png` | per group: the AU texts it takes part in, its status under the rule of 10, its ratings and toxic share in the AU texts |
| `au_label_patterns.csv`, `au_label_patterns.png` | AU texts whose groups give different labels, all toxic, all civil; the group labels decided by the mean |
| `au_dimension_overlap.csv`, `au_dimension_overlap.png` | how many AU texts are AU in more than one dimension |
| `au_rating_distribution.csv`, `au_rating_distribution.png/.pdf` | the ratings of the AU texts (toxic ratings in red) |
| `au_toxicity_by_dimension.png/.pdf` | the share of toxic ratings per group in the AU texts, the datasets side by side (groups removed under the rule are left out) |
| `au_models_input.csv`, `au_kumar_sample.png/.pdf` | each group's share of all AU rows against the rows given to the models (they differ only for Kumar) |

## Be careful

- **The AU set is small**, and for some groups very small: Attitudes has 11 to 41 AU ratings per group. Read small gaps in the AU toxic shares with care.
- **"Kept, below 10"** (orange in `au_texts_per_group.png`) are groups that the rule could not remove: all groups of the characteristic were below 10 (Attitudes
  age) or only two groups were left (Attitudes traditionalism). They are in the analysis with fewer than 10 AU texts.
- **Kumar:** the AU set has 6,090 rows; the models were given a 65-row sample (every group at least 10 times, not proportional). `au_models_input.csv`
  and `au_kumar_sample.png` show how the sample differs.
- **Group labels** are the majority of the group's binarized ratings; an exact tie is decided by the group's mean rating (`decided by the mean`).
- **AU depends on the number of ratings:** a text needs at least four ratings, and with few ratings DFU takes few values (see `new_results/total_dfu/`).
"""


def save_total_au(results: Dict[str, DatasetResult], models: pd.DataFrame, folder: Path) -> Dict[str, Path]:
    """Write the tables, the figures and the README described in the module docstring into ``folder``. Returns the files written, by file name.
    ``models`` is the table of the rows given to the models (``aposteriori_for_models.csv``, with ``item_id`` as text)."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results = _ordered_results(results)
    tables = _tables(results)
    summary, groups, patterns = au_summary(results, models), au_groups(results, tables), au_label_patterns(results)
    funnel, shares, models_input = _merged_funnel(results, tables), _rating_shares_au(tables), au_models_input(results, models)
    written = {}
    for file, frame, index in (("au_summary.csv", summary, True), ("au_funnel.csv", funnel, False), ("au_groups.csv", groups, False),
                               ("au_label_patterns.csv", patterns, False), ("au_dimension_overlap.csv", au_dimension_overlap(results), False),
                               ("au_rating_distribution.csv", shares, False), ("au_models_input.csv", models_input, False)):
        frame.to_csv(folder / file, index=index, encoding="utf-8")
        written[file] = folder / file
    minimum = next(iter(results.values())).spec.min_au_texts
    thresholds = {name: result.spec.toxic_threshold for name, result in results.items()}
    _save_figure(plt, _funnel_figure(plt, funnel), folder, "au_funnel", written, rect=(0, 0.05, 1, 1))
    _save_figure(plt, _groups_figure(plt, groups, minimum), folder, "au_texts_per_group", written, rect=(0, 0.04, 1, 1))
    _save_figure(plt, _patterns_figure(plt, patterns), folder, "au_label_patterns", written, rect=(0, 0.05, 1, 1))
    _save_figure(plt, _overlap_figure(plt, summary), folder, "au_dimension_overlap", written)
    _save_figure(plt, _rating_figure(plt, shares, thresholds), folder, "au_rating_distribution", written, formats=("pdf", "png"))
    by_group = toxicity_by_group({name: tab["groups"] for name, tab in tables.items()})
    _save_figure(plt, _by_dimension_figure(plt, by_group, "AU toxic %", True), folder, "au_toxicity_by_dimension", written, formats=("pdf", "png"),
                 rect=(0, 0.04, 1, 1))
    if "kumar" in results:
        _save_figure(plt, _sample_figure(plt, models_input), folder, "au_kumar_sample", written, formats=("pdf", "png"), rect=(0, 0.05, 1, 1))
    (folder / "README.md").write_text(README, encoding="utf-8")
    written["README.md"] = folder / "README.md"
    return written


def save_kumar_sample_figure(folder: Path) -> Dict[str, Path]:
    """Redraw only the figure of the Kumar sample (``au_kumar_sample.pdf`` and ``.png``) from ``au_models_input.csv`` in ``folder``; nothing is recomputed."""
    import matplotlib.pyplot as plt

    folder, written = Path(folder), {}
    _save_figure(plt, _sample_figure(plt, pd.read_csv(folder / "au_models_input.csv")), folder, "au_kumar_sample", written, formats=("pdf", "png"),
                 rect=(0, 0.05, 1, 1))
    return written
