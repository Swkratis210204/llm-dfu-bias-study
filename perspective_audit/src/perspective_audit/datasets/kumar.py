"""Kumar et al.: 0-4 toxicity scores (shifted to 1-5) of 107,620 comments, 5 raters each, with the rater's attributes."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..spec import DatasetSpec

COLUMNS = ["toxic_score", "gender", "race", "age_range", "political_affilation", "religion_important"]


def load(root: Path) -> pd.DataFrame:
    """The file is JSON lines: one comment per line with its list of ratings (about 450 MB, read line by line)."""
    rows = []
    with open(Path(root) / "datasets" / "kumar" / "toxicity_ratings.json", encoding="utf-8") as f:
        for line in f:
            comment = json.loads(line)
            for rating in comment["ratings"]:
                rows.append((comment["comment_id"], comment["comment"]) + tuple(rating.get(c) for c in COLUMNS))
    out = pd.DataFrame(rows, columns=["item_id", "text"] + COLUMNS)
    out["rating"] = out["toxic_score"] + 1                                  # 0-4 -> 1-5
    return out


SPEC = DatasetSpec(
    name="kumar",
    load=load,
    dimensions={
        "gender": ("gender", {"Male": "Male", "Female": "Female"}),
        "race": ("race", {"White": "White", "Black or African American": "Black", "Asian": "Asian"}),
        "age": ("age_range", {"Under 18": "gen z", "18 - 24": "gen z", "25 - 34": "millenial", "35 - 44": "millenial",
                              "45 - 54": "gen x+", "55 - 64": "gen x+", "65 or older": "gen x+"}),
        "politics": ("political_affilation", {"Liberal": "liberal", "Independent": "independent", "Conservative": "conservative"}),
        "religion": ("religion_important", {"Not important": "not_high", "Not too important": "not_high",
                                            "Somewhat important": "not_high", "Very important": "high"}),
    },
    item_word="comment",
    description="Kumar, toxic_score + 1 (1-5)",
)
