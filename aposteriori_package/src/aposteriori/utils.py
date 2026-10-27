"""Small helpers shared by the modules and the notebooks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

__all__ = ["find_root", "ensure_dir", "as_json", "from_json", "write_csv", "read_au_csv", "JSON_COLUMNS"]

JSON_COLUMNS = ("kept_groups", "group_sizes", "group_label")      # columns of an AU table that hold lists / dicts


def find_root(marker: str = "datasets", start: Optional[Path] = None) -> Path:
    """The repository root: the first folder, from ``start`` (default: working directory) upwards, that contains ``marker``.
    ``THESIS_ROOT`` in the environment overrides the starting point."""
    here = Path(os.environ.get("THESIS_ROOT", start or Path.cwd())).resolve()
    for folder in (here, *here.parents):
        if (folder / marker).is_dir():
            return folder
    raise FileNotFoundError(f"Could not find a folder containing '{marker}' above {here}; set THESIS_ROOT.")


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def as_json(value) -> str:
    """A list or dict (numpy numbers allowed) as a JSON string."""
    return json.dumps(value, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, np.integer) else str(o))


def from_json(text: str):
    return json.loads(text)


def write_csv(table: pd.DataFrame, path: Path) -> Path:
    """Write a table as UTF-8 CSV, creating the folder."""
    path = Path(path)
    ensure_dir(path.parent)
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def read_au_csv(path: Path, json_columns: Iterable[str] = JSON_COLUMNS) -> pd.DataFrame:
    """Read an AU table written by ``export``: ``item_id`` stays text and the JSON columns become lists / dicts again."""
    table = pd.read_csv(path, dtype={"item_id": str})
    for column in json_columns:
        if column in table:
            table[column] = table[column].map(from_json)
    return table
