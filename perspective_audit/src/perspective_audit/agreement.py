"""How much two sets of ratings of the same texts agree: two prompts, two models, a model and a group of people.

``cohen_kappa`` is for labels (civil / toxic), ``weighted_kappa`` and ``spearman`` for scores. For all of them 1 means perfect
agreement, 0 means no more agreement than chance would give, and a negative value means less. A pair where one side is missing
(NaN) is left out. A value that cannot be computed (everything rated the same by both sides, no pairs left) is NaN.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["cohen_kappa", "weighted_kappa", "spearman"]


def _pairs(a: Iterable, b: Iterable):
    a, b = np.asarray(list(a), dtype=object), np.asarray(list(b), dtype=object)
    if len(a) != len(b):
        raise ValueError(f"the two sets of ratings must have the same length ({len(a)} and {len(b)})")
    keep = ~(pd.isna(a) | pd.isna(b))
    return a[keep], b[keep]


def cohen_kappa(a: Iterable, b: Iterable) -> float:
    """Cohen's kappa of two sets of labels (any values that can be compared, e.g. ``"toxic"``/``"civil"`` or True/False)."""
    a, b = _pairs(a, b)
    if len(a) == 0:
        return float("nan")
    chance = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in set(a) | set(b))
    return float("nan") if chance == 1 else (float(np.mean(a == b)) - chance) / (1 - chance)


def weighted_kappa(a: Iterable, b: Iterable, scale: Sequence[int] = range(1, 6)) -> float:
    """Quadratic weighted kappa of two sets of scores on ``scale`` (default 1-5): a disagreement of 2 points counts four times as much as
    one of 1 point. A score outside ``scale`` raises ``ValueError``."""
    a, b = _pairs(a, b)
    scale = list(scale)
    if len(a) == 0 or len(scale) < 2:
        return float("nan")
    position = {int(v): i for i, v in enumerate(scale)}
    try:
        seen = np.zeros((len(scale), len(scale)))
        for x, y in zip(a, b):
            seen[position[int(x)], position[int(y)]] += 1
    except KeyError as missing:
        raise ValueError(f"score {missing} is not on the scale {scale[0]}-{scale[-1]}") from None
    expected = np.outer(seen.sum(axis=1), seen.sum(axis=0)) / seen.sum()
    steps = np.arange(len(scale))
    weight = (steps[:, None] - steps[None, :]) ** 2 / (len(scale) - 1) ** 2
    spread = float((weight * expected).sum())
    return float("nan") if spread == 0 else 1 - float((weight * seen).sum()) / spread


def spearman(a: Iterable, b: Iterable) -> float:
    """Spearman rank correlation of two sets of scores: do they put the texts in the same order?"""
    a, b = _pairs(a, b)
    if len(a) < 2:
        return float("nan")
    ra, rb = pd.Series(a.astype(float)).rank(), pd.Series(b.astype(float)).rank()
    return float("nan") if ra.nunique() < 2 or rb.nunique() < 2 else float(ra.corr(rb))      # one side never changes: undefined
