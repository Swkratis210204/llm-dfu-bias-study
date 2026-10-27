"""Group definitions shared by several datasets."""
from __future__ import annotations

import pandas as pd

__all__ = ["age_generations"]


def age_generations(ages: pd.Series) -> pd.Series:
    """Age in years -> ``gen z`` (under 25), ``millenial`` (25-44), ``gen x+`` (45 and over). Missing stays missing.
    (The spelling ``millenial`` is the one used in all the results.)"""
    return pd.cut(pd.to_numeric(ages, errors="coerce"), [0, 25, 45, 200], right=False,
                  labels=["gen z", "millenial", "gen x+"]).astype(object)
