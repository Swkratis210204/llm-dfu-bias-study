"""Samples of AU texts.

``minimal_cover`` picks, for one dimension, the **smallest number of AU texts in which every group appears at least k times**.
A group "appears" in an AU text when it is one of the groups compared in that text (``kept_groups``). The number is found
exactly (an integer program over the combinations of groups that occur), and the rows of each combination are drawn at
random with a fixed seed, so the same input always gives the same sample.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

__all__ = ["minimal_cover", "cover_per_dimension", "appearances"]


def _groups_of(value) -> Tuple[str, ...]:
    return tuple(sorted(str(g) for g in value))


def appearances(table: pd.DataFrame, groups_column: str = "kept_groups") -> Dict[str, int]:
    """In how many rows of ``table`` each group appears."""
    counts: Dict[str, int] = {}
    for value in table[groups_column]:
        for g in _groups_of(value):
            counts[g] = counts.get(g, 0) + 1
    return dict(sorted(counts.items()))


def minimal_cover(table: pd.DataFrame, k: int = 10, seed: int = 42, groups_column: str = "kept_groups") -> pd.DataFrame:
    """The fewest rows of ``table`` (the AU texts of ONE dimension) in which every group appears at least ``k`` times.

    ``groups_column`` holds, for every row, the list of the groups compared in that text. A group that appears in fewer than
    ``k`` rows of the whole table is taken in all the rows it has. Rows of the same combination of groups are interchangeable,
    so they are drawn at random (``seed``). The result keeps the columns and the index of ``table``.
    """
    combos = table[groups_column].map(_groups_of)
    types = combos.value_counts()
    type_list = list(types.index)
    groups = sorted({g for t in type_list for g in t})
    coverage = np.array([[1.0 if g in t else 0.0 for t in type_list] for g in groups])
    available = types.to_numpy(dtype=float)
    need = np.minimum(k, coverage @ available)                       # a group with fewer than k rows cannot reach k
    result = milp(c=np.ones(len(type_list)), constraints=LinearConstraint(coverage, lb=need, ub=np.inf),
                  integrality=np.ones(len(type_list)), bounds=Bounds(0, available))
    if not result.success:                                           # pragma: no cover - cannot happen: taking every row is feasible
        raise RuntimeError(f"no cover found: {result.message}")
    take = np.rint(result.x).astype(int)

    rng = np.random.default_rng(seed)
    chosen = []
    for combo, n in sorted(zip(type_list, take), key=lambda pair: pair[0]):
        if n:
            rows = table.index[combos == combo]
            chosen.extend(rng.choice(np.asarray(rows), size=int(n), replace=False).tolist())
    return table.loc[chosen]


def cover_per_dimension(table: pd.DataFrame, dimensions: Optional[Sequence[str]] = None, k: int = 10, seed: int = 42,
                        groups_column: str = "kept_groups") -> pd.DataFrame:
    """``minimal_cover`` for every dimension of an AU table (rows have a ``dimension`` column), stacked in the order given."""
    dimensions = list(dimensions) if dimensions is not None else list(dict.fromkeys(table["dimension"]))
    return pd.concat([minimal_cover(table[table["dimension"] == d], k, seed, groups_column) for d in dimensions], ignore_index=True)
