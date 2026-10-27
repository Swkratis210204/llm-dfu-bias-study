"""POPQUORN: 1-5 offensiveness ratings of 1,500 texts, with the rater's gender, race, age band, education and occupation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..spec import DatasetSpec

# the ten age bands of the file, grouped like the generations of DICES
AGE_BANDS = {"18-24": "gen z",
             "25-29": "millenial", "30-34": "millenial", "35-39": "millenial", "40-44": "millenial",
             "45-49": "gen x+", "50-54": "gen x+", "54-59": "gen x+", "60-64": "gen x+", ">65": "gen x+"}


def load(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(Path(root) / "datasets" / "popquorn" / "popquorn.csv")
    return raw.rename(columns={"instance_id": "item_id", "offensiveness": "rating"})


def _keep(*values):
    return {v: v for v in values}               # keep these values as they are; every other value is removed


SPEC = DatasetSpec(
    name="popquorn",
    load=load,
    dimensions={
        "gender": ("gender", _keep("Man", "Woman")),
        "race": ("race", _keep("White", "Black or African American", "Asian")),
        "age": ("age", AGE_BANDS),
        "education": ("education", _keep("College degree", "High school diploma or equivalent", "Graduate degree")),
        "occupation": ("occupation", _keep("Employed", "Self-employed", "Unemployed", "Retired", "Student", "Homemaker")),
    },
    item_word="text",
    description="POPQUORN, 1-5 offensiveness",
)
