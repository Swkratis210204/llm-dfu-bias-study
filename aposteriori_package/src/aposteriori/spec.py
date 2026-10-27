"""The description of one dataset: how to load it and how its annotator groups are defined."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd

__all__ = ["DatasetSpec"]

# a dimension is defined by a column of the raw table and a mapping raw value -> group.
# The mapping is a dict (a value that is not listed is removed) or a function Series -> Series (missing = removed).
Mapping = Union[Dict, Callable[[pd.Series], pd.Series]]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    load: Callable[..., pd.DataFrame]               # load(root) -> long table with item_id, text, rating and the raw attribute columns
    dimensions: Dict[str, Tuple[str, Mapping]]      # dimension -> (raw column, mapping)
    scale_max: int = 5                              # ratings run from 1 to scale_max
    toxic_threshold: float = 3
    tolerance: float = 0.0
    min_au_texts: int = 10                          # a group in fewer AU texts than this is removed
    prefilter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None     # applied to the raw table before the cleaning
    item_word: str = "tweet"                        # how a text is called in prompts ("tweet", "comment", ...)
    description: str = ""
