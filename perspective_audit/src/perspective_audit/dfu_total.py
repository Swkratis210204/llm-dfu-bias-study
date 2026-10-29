"""Exploring the DFU (distance from unimodality) of the texts across all the datasets. Nothing here asks a model for anything, and nothing changes
the AU analysis: it only reads the DFU values that the analysis already computed.

``save_total_dfu(results, folder)`` takes ``{dataset name: DatasetResult}`` and writes into ``folder``:

    dfu_text_level.csv           one row per dataset and text: ratings, DFU, polarized (DFU above the tolerance), AU (in any dimension), AU dimensions
    dfu_summary.csv              per dataset: how many texts are polarized, and statistics of the DFU of all texts, the polarized texts and the AU texts
    dfu_histogram.csv            the DFU in bins of 0.05, for all texts, the polarized texts and the AU texts (the numbers behind the histograms)
    dfu_by_ratings_per_text.csv  the share of polarized texts and the mean DFU by the number of ratings of a text
    dfu_group_level.csv          per dataset, dimension and group: how often the group's own ratings on a text are unimodal (the condition of AU)
    dfu_pairwise_ks.csv          the Kolmogorov-Smirnov distance between the DFU distributions of every two datasets (descriptive only)
    dfu_funnel.csv               what happens to the texts of every dimension (not polarized, too few groups, a group not unimodal, AU), merged
    *.png                        histograms, cumulative curves, DFU by number of ratings, boxplots (all texts and AU texts, one box per dataset)
    README.md                    what each file is, and what to be careful about when comparing the datasets
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .eda import eda_tables, _texts
from .eda_total import BLUE, DATASET_COLOR, DATASET_LABEL, RED, _ordered, _panels
from .pipeline import DatasetResult

__all__ = ["dfu_text_table", "dfu_summary", "dfu_histogram", "dfu_by_ratings_per_text", "dfu_group_level", "dfu_pairwise_ks", "save_total_dfu"]

BINS = np.round(np.arange(0, 1.0001, 0.05), 2)
RATING_CLASSES = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 6, "6"), (7, 10, "7 to 10"), (11, 10 ** 9, "11 or more")]
ORANGE = "#DD8452"
MIN_TEXTS_PLOTTED = 20                  # in the figure of the DFU by ratings per text, a point is drawn only when it rests on at least this many texts


def _ordered_results(results: Dict[str, DatasetResult]) -> Dict[str, DatasetResult]:
    return {name: results[name] for name in _ordered(results)}


def dfu_text_table(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """One row per dataset and text: the number of (cleaned) ratings, the DFU of all its ratings, whether it is polarized (DFU above the tolerance of
    the dataset), whether it is AU in at least one dimension, and in how many dimensions."""
    frames = []
    for name, result in _ordered_results(results).items():
        texts = _texts(result)
        au_dimensions = result.au.groupby("item_id").size()
        frames.append(pd.DataFrame({
            "dataset": name, "item_id": texts.index.astype(str), "ratings": texts["ratings"].to_numpy(), "dfu": texts["dfu"].to_numpy(),
            "polarized": texts["polarised"].to_numpy().astype(bool), "AU": texts["au"].to_numpy().astype(bool),
            "AU dimensions": pd.Series(texts.index).map(au_dimensions).fillna(0).astype(int).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def _quantile(values: pd.Series, q: float) -> float:
    return round(float(values.quantile(q)), 3) if len(values) else np.nan


def dfu_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the texts, how many are polarized, and statistics of the DFU of all texts, of the polarized texts and of the AU texts.
    ``distinct DFU values`` shows how coarse the DFU is (few ratings per text give few possible values)."""
    rows = {}
    for name in _ordered(table["dataset"]):
        part = table[table["dataset"] == name]
        polarized, au = part[part["polarized"]]["dfu"], part[part["AU"]]["dfu"]
        rows[name] = {
            "texts": len(part), "ratings per text (median)": float(part["ratings"].median()),
            "polarized texts": len(polarized), "polarized %": round(100 * len(polarized) / len(part), 1),
            "DFU = 0 %": round(100 * float((part["dfu"] == 0).mean()), 1), "distinct DFU values": int(part["dfu"].round(10).nunique()),
            "mean DFU (all texts)": round(float(part["dfu"].mean()), 3), "median (all texts)": _quantile(part["dfu"], 0.5),
            "90th percentile (all texts)": _quantile(part["dfu"], 0.9), "99th percentile (all texts)": _quantile(part["dfu"], 0.99),
            "max DFU": round(float(part["dfu"].max()), 3),
            "mean DFU (polarized texts)": round(float(polarized.mean()), 3) if len(polarized) else np.nan,
            "25th percentile (polarized)": _quantile(polarized, 0.25), "median (polarized)": _quantile(polarized, 0.5),
            "75th percentile (polarized)": _quantile(polarized, 0.75),
            "AU texts": len(au), "mean DFU (AU texts)": round(float(au.mean()), 3) if len(au) else np.nan, "median (AU texts)": _quantile(au, 0.5)}
    return pd.DataFrame(rows).T.rename_axis("dataset")


def dfu_histogram(table: pd.DataFrame) -> pd.DataFrame:
    """The DFU in bins of 0.05 (the last bin includes 1.0) per dataset: the texts and their share, for all texts, the polarized texts and the AU texts."""
    rows = []
    for name in _ordered(table["dataset"]):
        part = table[table["dataset"] == name]
        subsets = {"all texts": part["dfu"], "polarized texts": part[part["polarized"]]["dfu"], "AU texts": part[part["AU"]]["dfu"]}
        counts = {label: np.histogram(values, bins=BINS)[0] for label, values in subsets.items()}
        for i in range(len(BINS) - 1):
            row = {"dataset": name, "bin from": BINS[i], "bin to": BINS[i + 1]}
            for label, c in counts.items():
                row[f"{label}"] = int(c[i])
                row[f"{label} %"] = round(100 * c[i] / max(len(subsets[label]), 1), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _rating_class(n: int) -> str:
    return next(label for low, high, label in RATING_CLASSES if low <= n <= high)


def dfu_by_ratings_per_text(table: pd.DataFrame) -> pd.DataFrame:
    """By the number of ratings of a text: the texts, the polarized ones and their share, and the mean DFU of all texts and of the polarized texts."""
    labels = [label for _, _, label in RATING_CLASSES]
    rows = []
    for name in _ordered(table["dataset"]):
        part = table[table["dataset"] == name].assign(**{"ratings per text": lambda d: d["ratings"].map(_rating_class)})
        for label in labels:
            sub = part[part["ratings per text"] == label]
            if len(sub) == 0:
                continue
            polarized = sub[sub["polarized"]]
            rows.append({"dataset": name, "ratings per text": label, "texts": len(sub), "polarized texts": len(polarized),
                         "polarized %": round(100 * len(polarized) / len(sub), 1), "mean DFU (all texts)": round(float(sub["dfu"].mean()), 3),
                         "mean DFU (polarized texts)": round(float(polarized["dfu"].mean()), 3) if len(polarized) else np.nan})
    return pd.DataFrame(rows)


def dfu_group_level(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """Per dataset, dimension and group: on how many texts the group has enough ratings to be compared, the mean DFU of the group's own ratings,
    and how often they are unimodal (DFU 0), on all those texts and on the polarized ones. A group removed from the AU analysis is not here."""
    rows = []
    for name, result in _ordered_results(results).items():
        for dimension, pr in result.pruned.items():
            stats = defaultdict(lambda: [0, 0.0, 0, 0, 0.0, 0])           # texts, sum, unimodal, polarized texts, polarized sum, polarized unimodal
            for group_dfu, polarized in zip(pr.result["group_dfu"], pr.result["polarised"]):
                for group, value in group_dfu.items():
                    s = stats[group]
                    s[0] += 1
                    s[1] += value
                    s[2] += value == 0
                    if polarized:
                        s[3] += 1
                        s[4] += value
                        s[5] += value == 0
            for group in sorted(stats, key=str):
                texts, total, unimodal, p_texts, p_total, p_unimodal = stats[group]
                rows.append({"dataset": name, "dimension": dimension, "group": group, "texts with enough ratings": texts,
                             "mean group DFU": round(total / texts, 3), "unimodal %": round(100 * unimodal / texts, 1),
                             "polarized texts": p_texts, "mean group DFU (polarized texts)": round(p_total / p_texts, 3) if p_texts else np.nan,
                             "unimodal % (polarized texts)": round(100 * p_unimodal / p_texts, 1) if p_texts else np.nan})
    return pd.DataFrame(rows)


def dfu_pairwise_ks(table: pd.DataFrame) -> pd.DataFrame:
    """The Kolmogorov-Smirnov distance (0 = same distribution, 1 = disjoint) between the DFU distributions of every two datasets, for all texts and for
    the polarized texts. With these sample sizes a p-value says nothing, so only the distance is given: it is descriptive."""
    from scipy.stats import ks_2samp

    names = _ordered(table["dataset"])
    rows = []
    for a, b in combinations(names, 2):
        da, db = table[table["dataset"] == a], table[table["dataset"] == b]
        rows.append({"dataset A": a, "dataset B": b,
                     "KS distance (all texts)": round(float(ks_2samp(da["dfu"], db["dfu"]).statistic), 3),
                     "KS distance (polarized texts)": round(float(ks_2samp(da[da["polarized"]]["dfu"], db[db["polarized"]]["dfu"]).statistic), 3),
                     "texts A": len(da), "texts B": len(db)})
    return pd.DataFrame(rows)


def _merged_funnel(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    frames = [eda_tables(result)["funnel"].reset_index().rename(columns={"index": "dimension"}).assign(dataset=name)
              for name, result in _ordered_results(results).items()]
    table = pd.concat(frames, ignore_index=True)
    return table[["dataset"] + [c for c in table.columns if c != "dataset"]]


# ---------------------------------------------------------------------------------------------------------------------------------------------
def _percent_weights(n_values: int, n_base: int) -> np.ndarray:
    return np.full(n_values, 100 / max(n_base, 1))


def _histogram_figure(plt, table: pd.DataFrame, polarized_only: bool):
    """One panel per dataset: the DFU of all texts, or of the polarized texts with the AU texts among them (as a share of the polarized texts)."""
    names = _ordered(table["dataset"])
    fig, axes = _panels(plt, len(names))
    for ax, name in zip(axes, names):
        part = table[table["dataset"] == name]
        base = part[part["polarized"]] if polarized_only else part
        ax.hist(base["dfu"], bins=BINS, weights=_percent_weights(len(base), len(base)), color=BLUE, alpha=0.85,
                label=f"{'polarized texts' if polarized_only else 'all texts'} ({len(base):,})")
        if polarized_only:
            au = part[part["AU"]]
            ax.hist(au["dfu"], bins=BINS, weights=_percent_weights(len(au), len(base)), color=ORANGE, alpha=0.9, label=f"of which AU ({len(au):,})")
        ax.set(title=f"{DATASET_LABEL.get(name, name)} ({len(part):,} texts, {100 * part['polarized'].mean():.1f}% polarized)",
               xlabel="DFU of the text (0 = unimodal)", ylabel="% of the polarized texts" if polarized_only else "% of the texts")
        if polarized_only:                                  # the legend tells the two colors apart; for all texts the panel title already gives the count
            ax.legend(loc="upper right", fontsize=8)
    for ax in axes[len(names):]:
        ax.axis("off")
    return fig


def _ecdf_figure(plt, table: pd.DataFrame):
    """The cumulative share of texts by DFU, one line per dataset: all texts on the left, the polarized texts on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, only_polarized) in zip(axes, (("all texts", False), ("polarized texts", True))):
        for name in _ordered(table["dataset"]):
            part = table[table["dataset"] == name]
            values = np.sort((part[part["polarized"]] if only_polarized else part)["dfu"].to_numpy())
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=DATASET_COLOR.get(name), label=DATASET_LABEL.get(name, name))
        ax.set(title=title, xlabel="DFU of the text (0 = unimodal)", ylabel="cumulative share of the texts")
        ax.legend(loc="lower right")
    return fig


def _by_ratings_figure(plt, by_ratings: pd.DataFrame):
    """By the number of ratings of a text: the share of polarized texts (left) and the mean DFU of the polarized texts (right), one line per dataset."""
    labels = [label for _, _, label in RATING_CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, ylabel in ((axes[0], "polarized %", "% of the texts that are polarized"), (axes[1], "mean DFU (polarized texts)", "mean DFU of the polarized texts")):
        for name in _ordered(by_ratings["dataset"]):
            part = by_ratings[by_ratings["dataset"] == name].set_index("ratings per text").reindex(labels)
            values = part[column].where(part["texts"] >= MIN_TEXTS_PLOTTED)                 # a point that rests on a handful of texts is not drawn
            ax.plot(range(len(labels)), values, marker="o", color=DATASET_COLOR.get(name), label=DATASET_LABEL.get(name, name))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set(xlabel="ratings of the text", ylabel=ylabel)
        ax.legend()
    return fig


def _boxplot_figure(plt, table: pd.DataFrame, au_only: bool):
    """The DFU of all the texts, or of the AU texts only: one box per dataset (the line is the median, the diamond the mean; whiskers reach 1.5
    times the interquartile range and points beyond them are not drawn)."""
    names = _ordered(table["dataset"])
    data, labels = [], []
    for name in names:
        part = table[table["dataset"] == name]
        values = part[part["AU"]]["dfu"] if au_only else part["dfu"]
        data.append(values.to_numpy())
        labels.append(f"{DATASET_LABEL.get(name, name)}\n({len(values):,} texts)")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    boxes = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})
    for patch in boxes["boxes"]:
        patch.set_facecolor(ORANGE if au_only else BLUE)
    for line in boxes["medians"]:
        line.set_color("black")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(labels)
    ax.set(ylabel="DFU of the text (0 = unimodal)")
    ax.set_ylim(-0.03, 1.05)
    return fig


README = """# DFU across the datasets (exploratory)

DFU is the normalized distance from unimodality of the histogram of a text's ratings (the `ndfu` package): 0 means the ratings are unimodal, larger
values mean the ratings split into separate peaks (1 is the largest). A text is **polarized** when its DFU is above the tolerance (0 for all four
datasets, so any DFU above 0), and **AU** when, in addition, at least two annotator groups have two or more ratings on it and every one of those
groups is unimodal. All numbers use the cleaned ratings (after the preprocessing); the DFU values are the ones the AU analysis computed.

## Files

| file | content |
|---|---|
| `dfu_text_level.csv` | one row per dataset and text: ratings, DFU, polarized, AU (in any dimension), number of AU dimensions |
| `dfu_summary.csv` | per dataset: texts, polarized share, statistics of the DFU of all texts, polarized texts and AU texts |
| `dfu_histogram.csv` | the DFU in bins of 0.05 (all, polarized, AU): the numbers behind the histograms |
| `dfu_by_ratings_per_text.csv` | share of polarized texts and mean DFU by the number of ratings of a text |
| `dfu_group_level.csv` | per dimension and group: how often the group's own ratings on a text are unimodal (the second condition of AU) |
| `dfu_pairwise_ks.csv` | Kolmogorov-Smirnov distance between the DFU distributions of every two datasets |
| `dfu_funnel.csv` | the funnel of every dimension (not polarized, too few groups, a group not unimodal, AU), merged |
| `dfu_all_texts.png` | the DFU of all texts, one panel per dataset (the spike at 0 is the texts that are not polarized) |
| `dfu_polarized_texts.png` | the DFU of the polarized texts, with the AU texts among them |
| `dfu_ecdf.png` | cumulative share of texts by DFU, all datasets in one panel (all texts, and polarized texts) |
| `dfu_by_ratings_per_text.png` | share of polarized texts and mean DFU by the number of ratings of a text |
| `dfu_boxplot_all_texts.png` | the DFU of all texts, one box per dataset (the median is 0 in every dataset: more than half of the texts are not polarized) |
| `dfu_boxplot_au_texts.png` | the DFU of the AU texts, one box per dataset |

## Be careful when comparing datasets

- **The DFU depends on how many ratings a text has.** With few ratings only a few DFU values are possible (see `distinct DFU values` in
  `dfu_summary.csv`), and a text with few ratings is more often unimodal by chance. Kumar texts have at most 5 ratings; popquorn up to 10; MHS up to 777
  (70 reference comments, see the dataset notes); `dfu_by_ratings_per_text` shows the effect. In `dfu_by_ratings_per_text.png` a point is drawn only
  when it rests on at least 20 texts; the csv has all of them.
- **The scales differ.** MHS is 1-3, the others 1-5, so the same DFU does not mean the same disagreement.
- **A group is not in `dfu_group_level.csv`** when it was removed from the AU analysis (fewer than 10 AU texts).
- **The Kolmogorov-Smirnov distance is descriptive:** with thousands of texts a p-value is meaningless, and the datasets differ in more than the DFU.
"""


def save_total_dfu(results: Dict[str, DatasetResult], folder: Path) -> Dict[str, Path]:
    """Write the tables, the figures and the README described in the module docstring into ``folder``. Returns the files written, by file name."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results = _ordered_results(results)
    table = dfu_text_table(results)
    by_ratings = dfu_by_ratings_per_text(table)
    written = {}
    for file, frame, index in (("dfu_text_level.csv", table, False), ("dfu_summary.csv", dfu_summary(table), True), ("dfu_histogram.csv", dfu_histogram(table), False),
                               ("dfu_by_ratings_per_text.csv", by_ratings, False), ("dfu_group_level.csv", dfu_group_level(results), False),
                               ("dfu_pairwise_ks.csv", dfu_pairwise_ks(table), False), ("dfu_funnel.csv", _merged_funnel(results), False)):
        frame.to_csv(folder / file, index=index, encoding="utf-8")
        written[file] = folder / file
    for file, fig in (("dfu_all_texts.png", _histogram_figure(plt, table, False)), ("dfu_polarized_texts.png", _histogram_figure(plt, table, True)),
                      ("dfu_ecdf.png", _ecdf_figure(plt, table)), ("dfu_by_ratings_per_text.png", _by_ratings_figure(plt, by_ratings)),
                      ("dfu_boxplot_all_texts.png", _boxplot_figure(plt, table, False)), ("dfu_boxplot_au_texts.png", _boxplot_figure(plt, table, True))):
        fig.tight_layout()
        fig.savefig(folder / file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written[file] = folder / file
    (folder / "README.md").write_text(README, encoding="utf-8")
    written["README.md"] = folder / "README.md"
    return written
