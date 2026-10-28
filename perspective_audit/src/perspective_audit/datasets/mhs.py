"""Measuring Hate Speech (Hugging Face `ucberkeley-dlab/measuring-hate-speech`): the `hatespeech` item, No / Unclear / Yes -> 1-3."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..spec import DatasetSpec
from .groups import age_generations

URL = "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
COLUMNS = ["comment_id", "annotator_id", "text", "hatespeech", "annotator_gender", "annotator_age", "annotator_ideology"]


def load(root: Path, path: Optional[Path] = None) -> pd.DataFrame:
    """Reads ``datasets/mhs/train.parquet`` when it exists (or ``path``), otherwise downloads the file from Hugging Face."""
    local = Path(path) if path else Path(root) / "datasets" / "mhs" / "train.parquet"
    raw = pd.read_parquet(local if local.exists() else URL, columns=COLUMNS)
    out = raw.rename(columns={"comment_id": "item_id"})
    out["rating"] = out["hatespeech"].astype(int) + 1                       # 1 = No, 2 = Unclear, 3 = Yes
    return out


def _at_least_four_ratings(table: pd.DataFrame) -> pd.DataFrame:
    """A comment needs 4 ratings to have 2 + 2; the others cannot be AU."""
    return table[table.groupby("item_id")["rating"].transform("size") >= 4]


IDEOLOGY = {"extremely_liberal": "liberal", "liberal": "liberal", "slightly_liberal": "liberal", "neutral": "neutral",
            "slightly_conservative": "conservative", "conservative": "conservative", "extremely_conservative": "conservative"}

SPEC = DatasetSpec(
    name="mhs",
    load=load,
    dimensions={
        "gender": ("annotator_gender", {"male": "male", "female": "female"}),
        "age": ("annotator_age", age_generations),
        "ideology": ("annotator_ideology", IDEOLOGY),
    },
    scale_max=3,
    prefilter=_at_least_four_ratings,
    item_word="comment",
    description="Measuring Hate Speech, `hatespeech` item (1 = No, 2 = Unclear, 3 = Yes)",
    annotator_column="annotator_id",
    rating_column="hatespeech",
)
