"""A first description of one dataset (a ``DatasetResult``), for the paper: all its texts next to its AU texts. Nothing here asks a model for anything.

``eda_tables(result)`` returns the tables as a dict, ``eda_figures(result, folder)`` draws the figures (it needs matplotlib) and
``save_eda(result, folder)`` writes both: every table as a csv and every figure as a png. "AU texts" are the texts that are AU in at
least one dimension (in ``groups`` and ``au_labels`` the texts that are AU in that dimension). The tables:

    preprocessing        what the cleaning removed: ratings and texts before and after, ratings with an excluded group value per dimension
    overview             all texts against AU texts: texts, ratings, ratings per text (and the texts rated far more than usual), mean rating,
                         share of toxic ratings, polarisation, words
    rating_distribution  how many ratings (and which share) each score got, in all texts and in the AU texts
    funnel               per dimension: how many texts are not polarised, have too few groups, have a group that is not unimodal, or are AU
    groups               per dimension and group: ratings, mean rating and share of toxic ratings in all texts and in the AU texts of that dimension
    au_labels            per dimension: the AU texts, and how many have groups with different labels (the texts that tell perspectives apart)
    text_length          words per text, all texts against the AU texts
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .pipeline import DatasetResult

__all__ = ["eda_tables", "eda_figures", "save_eda"]

REASONS = ["not polarised", "too few groups", "a group is not unimodal", "AU"]       # the order of the funnel
ALL, AU = "all texts", "AU texts"
HEAVY = 3                                  # a text with more than HEAVY times the median number of ratings is flagged in the overview


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
        au_here = clean[clean["item_id"].isin(set(pr.result.loc[pr.result["au"], "item_id"]))]
        for group, ratings in clean.groupby(dimension)["rating"]:
            au_ratings = au_here.loc[au_here[dimension] == group, "rating"]
            rows.append({"dimension": dimension, "group": group, "ratings": len(ratings), "mean rating": _mean(ratings),
                         "toxic %": _toxic_pct(ratings, threshold), "AU ratings": len(au_ratings), "AU mean rating": _mean(au_ratings),
                         "AU toxic %": _toxic_pct(au_ratings, threshold), "kept in AU analysis": group in pr.counts})
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


def eda_figures(result: DatasetResult, folder: Path) -> List[Path]:
    """Draw the figures of the description into ``folder`` and return their paths:
    ``ratings.png`` (the ratings and the polarisation of all texts and of the AU texts), ``groups.png`` (the share of toxic ratings of every
    group, in all texts and in the AU texts), ``funnel.png`` (what happens to the texts of every dimension) and ``au_labels.png`` (the AU
    texts: do the groups agree?)."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables, texts, name = eda_tables(result), _texts(result), result.spec.name
    colors = {ALL: "#4C72B0", AU: "#DD8452"}
    paths = []

    def finish(fig, file):
        fig.tight_layout()
        fig.savefig(folder / file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(folder / file)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dist, x = tables["rating_distribution"], np.arange(len(tables["rating_distribution"]))
    for shift, group in zip((-0.2, 0.2), (ALL, AU)):
        axes[0].bar(x + shift, dist[f"{group}: %"], width=0.4, color=colors[group], label=f"{group} ({int(dist[f'{group}: ratings'].sum()):,} ratings)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dist.index.astype(int))
    axes[0].set(xlabel="rating", ylabel="% of ratings", title="Ratings")
    axes[0].legend()
    bins = np.linspace(0, 1, 21)
    for group, part in ((ALL, texts), (AU, texts[texts["au"]])):
        axes[1].hist(part["dfu"], bins=bins, weights=np.full(len(part), 100 / max(len(part), 1)), alpha=0.6, color=colors[group],
                     label=f"{group} ({len(part):,})")
    axes[1].set(xlabel="DFU of the text (0 = unimodal)", ylabel="% of texts", title="Polarisation of the texts")
    axes[1].legend()
    fig.suptitle(f"{name}: all texts against the AU texts", fontweight="bold")
    finish(fig, "ratings.png")

    groups = tables["groups"].reset_index()
    labels = groups["dimension"] + ": " + groups["group"].astype(str) + groups["kept in AU analysis"].map({True: "", False: " (removed)"})
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(groups) + 1.5))
    ax.barh(y - 0.2, groups["toxic %"], height=0.4, color=colors[ALL], label=ALL)
    ax.barh(y + 0.2, groups["AU toxic %"], height=0.4, color=colors[AU], label=AU)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set(xlabel="% of ratings called toxic", title=f"{name}: toxic ratings per group")
    ax.legend()
    finish(fig, "groups.png")

    funnel = tables["funnel"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(funnel) + 2))
    left = pd.Series(0.0, index=funnel.index)
    for reason, color in zip(REASONS, ["#DDDDDD", "#BBBBBB", "#888888", colors[ALL]]):
        share = 100 * funnel[reason] / funnel["texts"]
        ax.barh(funnel.index, share, left=left, label=reason, color=color)
        left += share
    ax.invert_yaxis()
    ax.set(xlabel="% of texts", title=f"{name}: what happens to the texts of every dimension")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    finish(fig, "funnel.png")

    au_labels = tables["au_labels"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(au_labels) + 2))
    left = pd.Series(0.0, index=au_labels.index)
    for column, color in zip(["groups disagree", "all toxic", "all civil"], [colors[AU], "#C44E52", colors[ALL]]):
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
    return {path.name: path for path in paths}
