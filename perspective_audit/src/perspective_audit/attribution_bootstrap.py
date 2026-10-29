"""Section 7.1: do non-demographic characteristics attribute a larger share of the testable texts than demographic ones?

Nothing here asks a model for anything, and nothing changes the AU analysis: it reads the final results after the removal of thin groups.

For one dataset the unit is a **polarized text with at least four ratings** (a text needs two groups of two ratings to be compared). For every
characteristic of the dataset, such a text is

* **testable** when at least two groups have two or more ratings on it (after the removal of thin groups), and
* **attributed** when it is AU for that characteristic.

The characteristics gender, race and age are *demographic* (D); every other one (politics or ideology, traditionalism, religion, education, occupation) is
*non-demographic* (N). The pooled attribution rate of a type is attributed / testable summed over its characteristics, and the gap is ``N - D`` in
percentage points. The texts are resampled with replacement (``reps`` times, fixed ``seed``); in every resample both rates and the gap are computed again from
the same drawn texts, so the two types move together. The 95% interval is the 2.5th to 97.5th percentile of the gaps.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .eda import _texts
from .eda_total import _ordered
from .pipeline import DatasetResult

__all__ = ["attribution_table", "bootstrap_gap", "attribution_gap"]

DEMOGRAPHIC = ("gender", "race", "age")
MIN_RATINGS = 4
COLUMNS = ["testable D", "attributed D", "testable N", "attributed N"]


def attribution_table(result: DatasetResult) -> pd.DataFrame:
    """One row per polarized text with at least four ratings: in how many demographic (D) and non-demographic (N) characteristics it is testable and in how many
    it is attributed (AU), from the final results after the removal of thin groups."""
    texts = _texts(result)
    keep = texts.index[texts["polarised"] & (texts["ratings"] >= MIN_RATINGS)]
    table = pd.DataFrame(0, index=keep, columns=COLUMNS, dtype=int)
    for dimension, pr in result.pruned.items():
        kind = "D" if dimension in DEMOGRAPHIC else "N"
        rows = pr.result.set_index("item_id")
        testable = (rows["polarised"] & (rows["kept_groups"].map(len) >= 2)).reindex(keep, fill_value=False)
        attributed = rows["au"].reindex(keep, fill_value=False)
        table[f"testable {kind}"] += testable.to_numpy().astype(int)
        table[f"attributed {kind}"] += attributed.to_numpy().astype(int)
    return table


def _rates(sums: np.ndarray):
    """The pooled rates (D, N) of a vector of the four sums; NaN when a type has nothing testable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return sums[1] / sums[0], sums[3] / sums[2]


def bootstrap_gap(table: pd.DataFrame, reps: int = 10_000, seed: int = 42) -> dict:
    """The observed gap ``N - D`` of the pooled attribution rates (percentage points), its 95% percentile interval over ``reps`` resamples of the texts, and
    the share of the resamples in which the gap is zero or negative."""
    values = table[COLUMNS].to_numpy(dtype=float)
    n = len(values)
    rate_d, rate_n = _rates(values.sum(axis=0))
    rng = np.random.default_rng(seed)
    gaps = np.empty(reps)
    for b in range(reps):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)                  # how often each text is drawn
        d, m = _rates(counts @ values)
        gaps[b] = 100 * (m - d)
    finite = gaps[np.isfinite(gaps)]
    return {"texts": n, "testable D": int(values[:, 0].sum()), "attributed D": int(values[:, 1].sum()), "rate D %": 100 * rate_d,
            "testable N": int(values[:, 2].sum()), "attributed N": int(values[:, 3].sum()), "rate N %": 100 * rate_n,
            "N - D (pp)": 100 * (rate_n - rate_d), "95% interval low": float(np.percentile(finite, 2.5)), "95% interval high": float(np.percentile(finite, 97.5)),
            "share of resamples with N - D <= 0 (%)": 100 * float(np.mean(finite <= 0)), "resamples": reps, "resamples without a rate": int(reps - len(finite))}


def attribution_gap(results: Dict[str, DatasetResult], reps: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """``bootstrap_gap`` for every dataset (in the order Attitudes, POPQUORN, MHS, Kumar). Every dataset starts from the same ``seed``, so its result does
    not depend on the other datasets."""
    rows = {name: bootstrap_gap(attribution_table(results[name]), reps, seed) for name in _ordered(results)}
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("dataset")
    for column in ("rate D %", "rate N %", "N - D (pp)", "95% interval low", "95% interval high"):
        table[column] = table[column].round(2)
    table["share of resamples with N - D <= 0 (%)"] = table["share of resamples with N - D <= 0 (%)"].round(2)
    return table
