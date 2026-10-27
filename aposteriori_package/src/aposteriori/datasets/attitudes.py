"""Attitudes: 1-5 ratings of 626 tweets (`toyou`); the annotator attributes are lists parallel to the ratings."""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..spec import DatasetSpec
from .groups import age_generations

PARALLEL = {"rating": "toyou", "gender": "annotatorGender", "race": "annotatorRace", "age": "annotatorAge",
            "politics": "annotatorPolitics", "traditionalism": "traditionalism"}


def _as_list(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return []
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else []


def load(root: Path, path: Optional[Path] = None) -> pd.DataFrame:
    """One row per rating. ``path`` (or ``APOSTERIORI_ATTITUDES_FILE``) overrides the default location.
    NOTE: despite its .csv extension, the file is a pickled DataFrame."""
    path = Path(path or os.environ.get("APOSTERIORI_ATTITUDES_FILE") or Path(root) / "Thesis" / "Code" / "attitudes_embedded.csv")
    raw = pd.read_pickle(path)
    lists = {name: raw[col].map(_as_list) for name, col in PARALLEL.items()}
    lengths = pd.DataFrame({name: s.map(len) for name, s in lists.items()})
    if not ((lengths.nunique(axis=1) == 1).all() and (lengths["rating"] > 0).all()):
        raise ValueError("ratings and annotator attributes do not line up in the attitudes file")
    rows = [(idx, raw.at[idx, "tweet"]) + tuple(lists[name].at[idx][k] for name in PARALLEL)
            for idx in raw.index for k in range(lengths.at[idx, "rating"])]
    return pd.DataFrame(rows, columns=["item_id", "text"] + list(PARALLEL))


def _more_than_three_ratings(table: pd.DataFrame) -> pd.DataFrame:
    """Texts with only 3 annotations are removed."""
    return table[table.groupby("item_id")["rating"].transform("size") > 3]


def _politics_group(score: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(score, errors="coerce"), [-10, -0.25, 0.25, 10], labels=["left", "center", "right"]).astype(object)


def _traditionalism_group(score: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(score, errors="coerce"), [0, 3, 5], labels=["not_high", "high"]).astype(object)


SPEC = DatasetSpec(
    name="attitudes",
    load=load,
    dimensions={
        "gender": ("gender", {"man": "man", "woman": "woman"}),
        "race": ("race", {"white": "white", "black": "black"}),
        "age": ("age", age_generations),
        "politics": ("politics", _politics_group),
        "traditionalism": ("traditionalism", _traditionalism_group),
    },
    prefilter=_more_than_three_ratings,
    item_word="tweet",
    description="Attitudes, 1-5 `toyou` ratings",
)
