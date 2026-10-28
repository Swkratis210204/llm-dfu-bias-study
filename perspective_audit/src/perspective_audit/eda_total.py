"""General description of all the datasets together, after the preprocessing (the cleaned ratings), for the paper. Nothing here asks a model for
anything. ``save_total_eda(results, folder)`` takes ``{dataset name: DatasetResult}`` and writes into ``folder``:

    rating_distribution.png / .csv   one panel per dataset: the share of each rating
    toxicity.png / .csv              one panel per dataset: the share of ratings called civil and toxic (toxic = the dataset's own threshold)
    ratings_per_text.csv / .txt      per dataset: the minimum, median and maximum number of ratings of a text, and the texts below four ratings
                                     (a text needs four ratings to be able to be AU: two groups of at least two); the .txt has one short line each

``save_toxicity_by_dimension(tables, folder)`` adds the share of toxic ratings of every group, the datasets side by side, from the ``groups.csv`` that
the per-dataset EDA saved (``read_group_tables``): ``toxicity_by_group.csv`` and one figure each for all texts and for the AU texts.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .pipeline import DatasetResult

__all__ = ["rating_shares", "toxicity_shares", "ratings_per_text", "ratings_per_text_lines", "save_total_eda",
           "read_group_tables", "toxicity_by_group", "save_toxicity_by_dimension", "save_appendix_figures"]

MIN_RATINGS = 4                                    # a text with fewer ratings cannot be AU (2 + 2)
BLUE, RED = "#4C72B0", "#C44E52"
DATASET_LABEL = {"popquorn": "POPQUORN", "attitudes": "Attitudes", "mhs": "MHS", "kumar": "Kumar"}      # the names in the figures
DATASET_ORDER = ["attitudes", "popquorn", "mhs", "kumar"]                                               # the order of the panels and the legends
DATASET_COLOR = {"attitudes": "#4C72B0", "kumar": "#55A868", "mhs": "#C44E52", "popquorn": "#8172B3"}


def _ordered(names) -> list:
    """The dataset names in the order of the figures (a name that is not listed goes last, in the order it comes)."""
    names = list(dict.fromkeys(names))
    return sorted(names, key=lambda n: DATASET_ORDER.index(n) if n in DATASET_ORDER else len(DATASET_ORDER))


def _save_figure(plt, fig, folder: Path, stem: str, written: dict, formats=("png",), rect=None) -> None:
    """Save a figure as every format in ``formats`` (e.g. ``("pdf", "png")``) with a tight box, close it, and record the files.
    ``rect`` reserves a part of the figure (left, bottom, right, top; 0-1) outside the layout, e.g. for a legend."""
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    for extension in formats:
        fig.savefig(Path(folder) / f"{stem}.{extension}", dpi=150, bbox_inches="tight")
        written[f"{stem}.{extension}"] = Path(folder) / f"{stem}.{extension}"
    plt.close(fig)


def rating_shares(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """For every dataset and every rating of its scale: how many (cleaned) ratings, and which share of the dataset's ratings."""
    rows = []
    for name, result in results.items():
        counts = result.clean["rating"].value_counts().reindex(range(1, result.spec.scale_max + 1), fill_value=0)
        for rating, n in counts.items():
            rows.append({"dataset": name, "rating": int(rating), "ratings": int(n), "%": round(100 * n / counts.sum(), 1)})
    return pd.DataFrame(rows)


def toxicity_shares(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """For every dataset: the share of (cleaned) ratings called civil and toxic (toxic = a rating at or above the dataset's threshold)."""
    rows = []
    for name, result in results.items():
        toxic = result.clean["rating"] >= result.spec.toxic_threshold
        for label, mask in (("civil", ~toxic), ("toxic", toxic)):
            rows.append({"dataset": name, "label": label, "ratings": int(mask.sum()), "%": round(100 * float(mask.mean()), 1),
                         "toxic is a rating >=": result.spec.toxic_threshold})
    return pd.DataFrame(rows)


def ratings_per_text(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """For every dataset: the minimum, median and maximum number of (cleaned) ratings of a text, and how many texts have fewer than four."""
    rows = {}
    for name, result in results.items():
        per_text = result.clean.groupby("item_id").size()
        below = int((per_text < MIN_RATINGS).sum())
        rows[name] = {"texts": len(per_text), "min": int(per_text.min()), "median": float(per_text.median()), "max": int(per_text.max()),
                      f"texts with fewer than {MIN_RATINGS} ratings": below, "% of texts": round(100 * below / len(per_text), 1)}
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("dataset")          # from_dict keeps whole numbers whole (a transpose made them floats)
    table["median"] = [int(m) if float(m).is_integer() else m for m in table["median"]]       # 8, not 8.0; 4.5 stays 4.5
    return table


def ratings_per_text_lines(table: pd.DataFrame) -> str:
    """One short line per dataset, from the table of ``ratings_per_text``."""
    column = f"texts with fewer than {MIN_RATINGS} ratings"
    return "\n".join(f"{name}: ratings per text min {int(row['min'])}, median {row['median']:g}, max {int(row['max'])}; "
                     f"{int(row[column]):,} of {int(row['texts']):,} texts ({row['% of texts']}%) have fewer than {MIN_RATINGS} ratings"
                     for name, row in table.iterrows()) + "\n"


def _panels(plt, n: int):
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.6 * rows + 0.6), sharey=True, squeeze=False)
    return fig, axes.flatten()


def _rating_figure(plt, shares: pd.DataFrame, thresholds: Dict[str, float]):
    """One panel per dataset (in the order of ``DATASET_ORDER``): the share of each rating, the toxic ratings (at or above the dataset's
    threshold) in red and the civil ones in blue. No overall title: the caption of the paper replaces it."""
    from matplotlib.patches import Patch

    names = _ordered(shares["dataset"])
    fig, axes = _panels(plt, len(names))
    for ax, name in zip(axes, names):
        part = shares[shares["dataset"] == name]
        bars = ax.bar(part["rating"], part["%"], color=[RED if rating >= thresholds[name] else BLUE for rating in part["rating"]])
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in part["%"]], fontsize=8, padding=2)
        ax.set_xticks(part["rating"])
        ax.set(title=f"{DATASET_LABEL.get(name, name)} ({int(part['ratings'].sum()):,} ratings)", xlabel="rating", ylabel="% of ratings")
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].set_ylim(0, shares["%"].max() * 1.18)
    axes[min(1, len(names) - 1)].legend([Patch(color=BLUE), Patch(color=RED)], ["civil", "toxic"], loc="upper right", fontsize=9)
    return fig


def _toxicity_figure(plt, shares: pd.DataFrame):
    names = list(dict.fromkeys(shares["dataset"]))
    fig, axes = _panels(plt, len(names))
    for ax, name in zip(axes, names):
        part = shares[shares["dataset"] == name]
        bars = ax.bar(part["label"], part["%"], color=[BLUE if label == "civil" else RED for label in part["label"]])
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in part["%"]], fontsize=8, padding=2)
        ax.set(title=f"{name} (toxic = rating >= {part['toxic is a rating >='].iloc[0]:g})", ylabel="% of ratings")
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].set_ylim(0, 100)
    fig.suptitle("Share of ratings called civil and toxic, after the preprocessing", fontweight="bold")
    return fig


def save_total_eda(results: Dict[str, DatasetResult], folder: Path) -> Dict[str, Path]:
    """Write the figures, tables and lines described in the module docstring into ``folder``. Returns the files written, by file name."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}

    ratings, toxicity, per_text = rating_shares(results), toxicity_shares(results), ratings_per_text(results)
    thresholds = {name: result.spec.toxic_threshold for name, result in results.items()}
    _save_figure(plt, _rating_figure(plt, ratings, thresholds), folder, "rating_distribution", written, formats=("pdf", "png"))
    fig = _toxicity_figure(plt, toxicity)
    fig.tight_layout()
    fig.savefig(folder / "toxicity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    written["toxicity.png"] = folder / "toxicity.png"
    for table, file, index in ((ratings, "rating_distribution.csv", False), (toxicity, "toxicity.csv", False), (per_text, "ratings_per_text.csv", True)):
        table.to_csv(folder / file, index=index, encoding="utf-8")
        written[file] = folder / file
    (folder / "ratings_per_text.txt").write_text(ratings_per_text_lines(per_text), encoding="utf-8")
    written["ratings_per_text.txt"] = folder / "ratings_per_text.txt"
    return written


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The share of toxic ratings of every group, the datasets side by side (read from the groups tables that the per-dataset EDA already saved)
DIMENSION_NAMES = {"ideology": "politics"}                              # MHS calls its political dimension "ideology"
GROUP_NAMES = {                                                         # the same group written differently in different datasets (compared in lower case)
    "gender": {"man": "man", "male": "man", "woman": "woman", "female": "woman"},
    "race": {"white": "White", "black": "Black", "black or african american": "Black", "asian": "Asian"},
    "politics": {"left": "left / liberal", "liberal": "left / liberal", "center": "center / neutral / independent",
                 "neutral": "center / neutral / independent", "independent": "center / neutral / independent",
                 "right": "right / conservative", "conservative": "right / conservative"},
}
GROUP_ORDER = {"gender": ["man", "woman"], "race": ["White", "Black", "Asian"], "age": ["gen z", "millenial", "gen x+"],
               "politics": ["left / liberal", "center / neutral / independent", "right / conservative"]}
DIMENSION_ORDER = ["gender", "race", "age", "politics", "religion", "education", "occupation", "traditionalism"]      # demographic first
DEMOGRAPHIC = ("gender", "race", "age")
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
GROUP_PLOT_LABEL = {"millenial": "millennial",                                  # labels in the figures only; the csv keeps the names of the data
                    "left / liberal": "left", "center / neutral / independent": "center", "right / conservative": "right"}


def read_group_tables(results_dir: Path, datasets) -> Dict[str, pd.DataFrame]:
    """The ``groups.csv`` that the EDA of every dataset saved in ``<results_dir>/<dataset>/eda/``."""
    return {name: pd.read_csv(Path(results_dir) / name / "eda" / "groups.csv", index_col=[0, 1]) for name in datasets}


def toxicity_by_group(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per dataset, dimension and group: ratings and the share of toxic ratings in all texts and in the AU texts of the dimension.
    ``tables`` is ``{dataset: groups table}`` (``eda_tables(result)["groups"]`` or the saved ``groups.csv``). The same group can be written
    differently in different datasets ("Man", "male", ...), so the table has the names as in the dataset and the harmonized ones (``GROUP_NAMES``);
    MHS's ``ideology`` is shown as ``politics``. Groups removed from the AU analysis keep their all-texts numbers (``kept in AU analysis`` is False)."""
    rows = []
    for dataset, table in tables.items():
        for r in table.reset_index().to_dict("records"):
            dimension = DIMENSION_NAMES.get(r["dimension"], r["dimension"])
            group = GROUP_NAMES.get(dimension, {}).get(str(r["group"]).lower(), r["group"])
            rows.append({"dataset": dataset, "dimension": dimension, "group": group, "group (as in the dataset)": r["group"],
                         "dimension (as in the dataset)": r["dimension"], "ratings": r["ratings"], "toxic %": r["toxic %"],
                         "AU ratings": r["AU ratings"], "AU toxic %": r["AU toxic %"], "kept in AU analysis": r["kept in AU analysis"]})
    table = pd.DataFrame(rows)
    table["_d"] = table["dimension"].map(lambda d: DIMENSION_ORDER.index(d) if d in DIMENSION_ORDER else len(DIMENSION_ORDER))
    table["_g"] = [GROUP_ORDER.get(d, []).index(g) if g in GROUP_ORDER.get(d, []) else len(GROUP_ORDER.get(d, []))
                   for d, g in zip(table["dimension"], table["group"])]
    return table.sort_values(["_d", "_g", "dataset"], kind="stable").drop(columns=["_d", "_g"]).reset_index(drop=True)


def _by_dimension_figure(plt, table: pd.DataFrame, column: str, hide_removed: bool):
    """One panel per dimension: the share of toxic ratings of each group, one bar per dataset (in the order of ``DATASET_ORDER``, with the names of
    ``DATASET_LABEL``; the groups with the labels of ``GROUP_PLOT_LABEL``). No overall title: the caption of the paper replaces it."""
    part = table[table["kept in AU analysis"]] if hide_removed else table
    dimensions = [d for d in DIMENSION_ORDER if d in set(part["dimension"])]
    datasets = _ordered(table["dataset"])
    colors = {name: DATASET_COLOR.get(name, PALETTE[i % len(PALETTE)]) for i, name in enumerate(datasets)}
    rows = math.ceil(len(dimensions) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.5 * rows + 0.9), sharey=True, squeeze=False)
    for ax, dimension in zip(axes.flatten(), dimensions):
        sub = part[part["dimension"] == dimension]
        groups = list(dict.fromkeys(sub["group"]))
        present = [d for d in datasets if d in set(sub["dataset"])]
        x, width = np.arange(len(groups)), 0.8 / len(present)
        for i, name in enumerate(present):
            values = sub[sub["dataset"] == name].set_index("group")[column]
            ax.bar(x + (i - (len(present) - 1) / 2) * width, [values.get(g, np.nan) for g in groups], width=width, color=colors[name])
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_PLOT_LABEL.get(g, g) for g in groups], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{dimension} ({'demographic' if dimension in DEMOGRAPHIC else 'non-demographic'})")
    for ax in axes.flatten()[len(dimensions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of ratings called toxic")
    axes[0][0].set_ylim(0, 100)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=colors[d]) for d in datasets], [DATASET_LABEL.get(d, d) for d in datasets],
               loc="lower center", ncol=len(datasets))
    return fig


def save_toxicity_by_dimension(tables: Dict[str, pd.DataFrame], folder: Path) -> Dict[str, Path]:
    """Write ``toxicity_by_group.csv`` and the two figures ``toxicity_by_dimension_all_texts`` and ``toxicity_by_dimension_au_texts``, each as pdf
    and png (one panel per dimension, the datasets side by side; in the AU figure the groups removed from the AU analysis are left out)."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    table = toxicity_by_group(tables)
    table.to_csv(folder / "toxicity_by_group.csv", index=False, encoding="utf-8")
    written = {"toxicity_by_group.csv": folder / "toxicity_by_group.csv"}
    for column, stem, hide in (("toxic %", "toxicity_by_dimension_all_texts", False), ("AU toxic %", "toxicity_by_dimension_au_texts", True)):
        _save_figure(plt, _by_dimension_figure(plt, table, column, hide), folder, stem, written, formats=("pdf", "png"), rect=(0, 0.04, 1, 1))
    return written


def save_appendix_figures(folder: Path) -> Dict[str, Path]:
    """Redraw the two appendix figures, ``rating_distribution`` and ``toxicity_by_dimension_all_texts`` (each as pdf and png), from the csv files
    already in ``folder`` (``rating_distribution.csv``, ``toxicity.csv`` and ``toxicity_by_group.csv``). No number is recomputed: only the labels,
    the order and the styling come from the code."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    written = {}
    toxicity = pd.read_csv(folder / "toxicity.csv")
    thresholds = toxicity.groupby("dataset")["toxic is a rating >="].first().to_dict()
    _save_figure(plt, _rating_figure(plt, pd.read_csv(folder / "rating_distribution.csv"), thresholds), folder, "rating_distribution", written,
                 formats=("pdf", "png"))
    _save_figure(plt, _by_dimension_figure(plt, pd.read_csv(folder / "toxicity_by_group.csv"), "toxic %", False), folder,
                 "toxicity_by_dimension_all_texts", written, formats=("pdf", "png"), rect=(0, 0.04, 1, 1))
    return written
