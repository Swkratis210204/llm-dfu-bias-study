"""A first description of one dataset (a ``DatasetResult``), for the paper: all its texts and its AU texts, together and apart. Nothing here asks
a model for anything.

``eda_tables(result)`` returns the tables as a dict, ``eda_figures(result, folder)`` draws the figures (it needs matplotlib) and
``save_eda(result, folder)`` writes both: every table as a csv and every figure as a png. "AU texts" are the texts that are AU in at least one
dimension (in ``groups``, per dimension: the texts that are AU in that dimension). The tables:

    preprocessing        what the cleaning removed: ratings and texts before and after, ratings with an excluded group value per dimension
    overview             all texts and AU texts: texts, ratings, ratings per text (and the texts rated far more than usual), mean rating,
                         share of toxic ratings, polarisation, words
    rating_distribution  how many ratings (and which share) each score got, for all texts and AU texts
    funnel               per dimension: how many texts are not polarised, have too few groups, have a group that is not unimodal, or are AU
    groups               per dimension and group: ratings, mean rating and share of toxic ratings, for all texts and for the AU texts of that dimension
    au_labels            per dimension: the AU texts, and how many have groups with different labels (the texts that tell perspectives apart)
    text_length          words per text, for all texts and AU texts

The figures: ``ratings.png`` (the ratings), ``dfu.png`` (the polarisation) and ``groups.png`` (toxic ratings per group) put all texts and AU texts
side by side, and the ``_all`` and ``_au`` versions (e.g. ``dfu_au.png``) show each of them alone; ``funnel.png`` and ``au_labels.png`` describe the step from one to the other.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .eda_dataset import save_dataset_eda
from .pipeline import DatasetResult

__all__ = ["eda_tables", "eda_figures", "save_eda"]

REASONS = ["not polarised", "too few groups", "a group is not unimodal", "AU"]       # the order of the funnel (and the columns of funnel.csv)
PLOT_LABEL = {"not polarised": "not polarized"}                                        # American spelling in the plots; the csv columns stay as they are
ALL, AU = "all texts", "AU texts"
HEAVY = 3                                  # a text with more than HEAVY times the median number of ratings is flagged in the overview
TOXIC_COLUMN = {ALL: "toxic %", AU: "AU toxic %"}          # the columns of the groups table that the figures draw
COLORS = {ALL: "#4C72B0", AU: "#DD8452"}


def _reason(text: str) -> str:
    if text in ("AU", "not polarised"):
        return text
    return "too few groups" if text.startswith("fewer") else "a group is not unimodal"


def _texts(result: DatasetResult) -> pd.DataFrame:
    """One row per text: its ratings, mean, share of toxic ratings, DFU, whether it is polarised, its words, and whether it is AU (any dimension)."""
    flagged = result.clean.assign(toxic=result.clean["rating"] >= result.spec.toxic_threshold)        # vectorised: Kumar has 107k texts
    texts = flagged.groupby("item_id", sort=False).agg(
        text=("text", "first"), ratings=("rating", "size"), mean=("rating", "mean"), toxic=("toxic", "mean"))
    overall = next(iter(result.pruned.values())).result.set_index("item_id")[["dfu", "polarised"]]      # the same in every dimension
    texts = texts.join(overall)
    texts["words"] = texts["text"].astype(str).str.split().str.len()
    texts["au"] = texts.index.isin(set(result.au["item_id"]))
    return texts


def _toxic_pct(ratings: pd.Series, threshold: float) -> float:
    return round(100 * float((ratings >= threshold).mean()), 1) if len(ratings) else np.nan


def _mean(ratings: pd.Series) -> float:
    return round(float(ratings.mean()), 2) if len(ratings) else np.nan


def _preprocessing(result: DatasetResult) -> pd.DataFrame:
    """What the cleaning removed: ratings and texts before and after, and the ratings with an excluded group value, per dimension."""
    r = result.report
    rows = [("ratings (raw)", r["ratings raw"]), ("removed by the prefilter", r["ratings raw"] - r["ratings after prefilter"]),
            ("removed for an excluded group value (any dimension)", r["ratings after prefilter"] - r["ratings clean"]),
            ("ratings (clean)", r["ratings clean"]),
            ("texts (raw)", r["items raw"]), ("texts (clean)", r["items clean"]), ("texts lost", r["items raw"] - r["items clean"])]
    rows += [(f"ratings with an excluded {dimension} value (after the prefilter)", n) for dimension, n in r["ratings with a removed value"].items()]
    table = pd.DataFrame(rows, columns=["step", "count"]).set_index("step")
    is_ratings = table.index.str.startswith(("ratings", "removed"))
    table["% of raw ratings"] = np.where(is_ratings, (100 * table["count"] / r["ratings raw"]).round(1), np.nan)
    return table


def eda_tables(result: DatasetResult) -> Dict[str, pd.DataFrame]:
    """The tables of the description of one dataset (see the module docstring)."""
    clean, threshold = result.clean, result.spec.toxic_threshold
    texts = _texts(result)
    in_au = clean["item_id"].isin(texts.index[texts["au"]])
    subsets = {ALL: (texts, clean), AU: (texts[texts["au"]], clean[in_au])}
    heavy_cut = HEAVY * texts["ratings"].median()          # a text rated far more than usual (MHS has 70 comments with about 800 ratings each)

    def describe(part, ratings):
        per_text = part["ratings"]
        heavy = per_text[per_text > heavy_cut]
        return {"texts": len(part), "ratings": len(ratings),
                "ratings per text (min / median / max)": f"{per_text.min()} / {per_text.median():g} / {per_text.max()}" if len(part) else "-",
                f"texts with > {HEAVY}x the median ratings (% of the ratings)": f"{len(heavy)} ({100 * heavy.sum() / max(len(ratings), 1):.1f}%)",
                "mean rating": _mean(ratings["rating"]), "ratings called toxic %": _toxic_pct(ratings["rating"], threshold),
                "polarised texts %": round(100 * float(part["polarised"].mean()), 1) if len(part) else np.nan,
                "mean DFU": round(float(part["dfu"].mean()), 3) if len(part) else np.nan,
                "words per text (median)": float(part["words"].median()) if len(part) else np.nan}

    overview = pd.DataFrame({name: describe(*subset) for name, subset in subsets.items()})

    scale = range(1, result.spec.scale_max + 1)
    distribution = {}
    for name, (_, ratings) in subsets.items():
        counts = ratings["rating"].value_counts().reindex(scale, fill_value=0)
        distribution[f"{name}: ratings"] = counts
        distribution[f"{name}: %"] = (100 * counts / counts.sum()).round(1)
    rating_distribution = pd.DataFrame(distribution).rename_axis("rating")

    funnel = pd.DataFrame({dimension: pr.result["reason"].map(_reason).value_counts().reindex(REASONS, fill_value=0)
                           for dimension, pr in result.pruned.items()}).T
    funnel.insert(0, "texts", funnel.sum(axis=1))
    funnel["AU %"] = (100 * funnel["AU"] / funnel["texts"]).round(1)

    rows = []
    for dimension, pr in result.pruned.items():
        is_au = clean["item_id"].isin(set(pr.result.loc[pr.result["au"], "item_id"]))           # AU in THIS dimension
        for group, ratings in clean.groupby(dimension)["rating"]:
            row = {"dimension": dimension, "group": group, "ratings": len(ratings), "mean rating": _mean(ratings),
                   "toxic %": _toxic_pct(ratings, threshold)}
            part = clean.loc[is_au & (clean[dimension] == group), "rating"]
            row.update({"AU ratings": len(part), "AU mean rating": _mean(part), "AU toxic %": _toxic_pct(part, threshold)})
            row["kept in AU analysis"] = group in pr.counts
            rows.append(row)
    groups = pd.DataFrame(rows).set_index(["dimension", "group"])

    rows = {}
    for dimension, pr in result.pruned.items():
        sets = [set(labels.values()) for labels in pr.result.loc[pr.result["au"], "group_label"]]
        rows[dimension] = {"AU texts": len(sets), "groups disagree": sum(len(s) > 1 for s in sets),
                           "all toxic": sum(s == {"toxic"} for s in sets), "all civil": sum(s == {"civil"} for s in sets)}
    au_labels = pd.DataFrame(rows).T
    au_labels["groups disagree %"] = (100 * au_labels["groups disagree"] / au_labels["AU texts"].where(au_labels["AU texts"] > 0)).round(1)

    text_length = pd.DataFrame({name: part["words"].describe() for name, (part, _) in subsets.items()}).round(1)

    return {"preprocessing": _preprocessing(result), "overview": overview, "rating_distribution": rating_distribution, "funnel": funnel,
            "groups": groups, "au_labels": au_labels, "text_length": text_length}


def _ratings_figure(plt, tables: Dict[str, pd.DataFrame], title: str, populations: List[str]):
    """How often each rating was given, for all texts, the AU texts, or both side by side."""
    fig, ax = plt.subplots(figsize=(7, 4))
    dist = tables["rating_distribution"]
    x, width = np.arange(len(dist)), 0.8 / len(populations)
    for i, name in enumerate(populations):
        ax.bar(x + (i - (len(populations) - 1) / 2) * width, dist[f"{name}: %"], width=width, color=COLORS[name],
               label=f"{name} ({int(dist[f'{name}: ratings'].sum()):,} ratings)")
    ax.set_xticks(x)
    ax.set_xticklabels(dist.index.astype(int))
    ax.set(xlabel="rating", ylabel="% of ratings", title=title)
    ax.legend()
    return fig


def _dfu_figure(plt, texts: pd.DataFrame, title: str, populations: List[str]):
    """The polarisation (DFU) of the texts, for all texts, the AU texts, or both side by side."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    for name in populations:
        part = texts[texts["au"]] if name == AU else texts
        ax.hist(part["dfu"], bins=bins, weights=np.full(len(part), 100 / max(len(part), 1)), alpha=0.6 if len(populations) > 1 else 0.9,
                color=COLORS[name], label=f"{name} ({len(part):,} texts)")
    ax.set(xlabel="DFU of the text (0 = unimodal)", ylabel="% of texts", title=title)
    ax.legend()
    return fig


def _groups_figure(plt, tables: Dict[str, pd.DataFrame], title: str, populations: List[str]):
    """The share of toxic ratings of every group, one panel per dimension, for all texts, the AU texts, or both side by side."""
    groups = tables["groups"].reset_index()
    dimensions = list(dict.fromkeys(groups["dimension"]))
    rows = math.ceil(len(dimensions) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.4 * rows + 0.8), squeeze=False)
    height = 0.8 / len(populations)
    for ax, dimension in zip(axes.flatten(), dimensions):
        part = groups[groups["dimension"] == dimension].reset_index(drop=True)
        y = np.arange(len(part))
        for i, name in enumerate(populations):
            ax.barh(y + (i - (len(populations) - 1) / 2) * height, part[TOXIC_COLUMN[name]], height=height, color=COLORS[name], label=name)
        ax.set_yticks(y)
        ax.set_yticklabels(part["group"].astype(str) + part["kept in AU analysis"].map({True: "", False: " (removed)"}))
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(dimension)
    for ax in axes.flatten()[len(dimensions):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("% of ratings called toxic")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=len(populations))
    fig.suptitle(title, fontweight="bold")
    return fig


def eda_figures(result: DatasetResult, folder: Path) -> List[Path]:
    """Draw the figures of the description into ``folder`` and return their paths: ``ratings.png`` (how often each rating was given), ``dfu.png``
    (the polarisation of the texts) and ``groups.png`` (the share of toxic ratings of every group), each with all texts and AU texts side by side,
    and for each alone (``_all`` and ``_au``, e.g. ``ratings_au.png``); ``funnel.png`` (what happens to the texts of every dimension) and
    ``au_labels.png`` (the AU texts: do the groups agree?). The ratings and the DFU are separate figures: they measure different things."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables, texts, name = eda_tables(result), _texts(result), result.spec.name
    paths = []

    def finish(fig, file):
        fig.tight_layout()
        fig.savefig(folder / file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(folder / file)

    for suffix, populations in (("", [ALL, AU]), ("_all", [ALL]), ("_au", [AU])):
        label = " against ".join(populations)
        finish(_ratings_figure(plt, tables, f"{name}: the ratings, {label}", populations), f"ratings{suffix}.png")
        finish(_dfu_figure(plt, texts, f"{name}: polarization of the texts, {label}", populations), f"dfu{suffix}.png")
        finish(_groups_figure(plt, tables, f"{name}: toxic ratings per group, {label}", populations), f"groups{suffix}.png")

    funnel = tables["funnel"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(funnel) + 2))
    left = pd.Series(0.0, index=funnel.index)
    for reason, color in zip(REASONS, ["#DDDDDD", "#BBBBBB", "#888888", COLORS[AU]]):
        share = 100 * funnel[reason] / funnel["texts"]
        ax.barh(funnel.index, share, left=left, label=PLOT_LABEL.get(reason, reason), color=color)
        left += share
    ax.invert_yaxis()
    ax.set(xlabel="% of texts", title=f"{name}: what happens to the texts of every dimension")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    finish(fig, "funnel.png")

    au_labels = tables["au_labels"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(au_labels) + 2))
    left = pd.Series(0.0, index=au_labels.index)
    for column, color in zip(["groups disagree", "all toxic", "all civil"], [COLORS[AU], "#C44E52", COLORS[ALL]]):
        ax.barh(au_labels.index, au_labels[column], left=left, label=column, color=color)
        left += au_labels[column]
    ax.invert_yaxis()
    ax.set(xlabel="AU texts", title=f"{name}: the AU texts, do the groups give different labels?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    finish(fig, "au_labels.png")
    return paths


def save_eda(result: DatasetResult, folder: Path) -> Dict[str, Path]:
    """Write every table of ``eda_tables`` as ``<name>.csv`` and the figures of ``eda_figures`` into ``folder``.
    Returns the files written, by file name (``groups.csv`` and ``groups.png`` are different files)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in eda_tables(result).items():
        table.to_csv(folder / f"{name}.csv", encoding="utf-8")
        paths.append(folder / f"{name}.csv")
    paths += eda_figures(result, folder)
    written = {path.name: path for path in paths}
    if result.raw is not None:                       # the description of the dataset itself (annotators, group mapping, dataset info)
        written.update(save_dataset_eda(result, folder))
    return written
