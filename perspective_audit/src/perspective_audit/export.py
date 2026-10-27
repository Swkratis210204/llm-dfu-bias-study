"""Save the AU tables: one CSV per dataset, one with everything, one for the models, and a random sample of a large dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from .sampling import cover_per_dimension
from .utils import JSON_COLUMNS, as_json, ensure_dir, read_au_csv, write_csv

__all__ = ["csv_ready", "sample_per_dimension", "save_au_tables"]

CSV_COLUMNS = ["row_id", "dataset", "dimension", "item_id", "text", "n_ratings", "dfu", "toxic_pole", *JSON_COLUMNS]


def csv_ready(table: pd.DataFrame) -> pd.DataFrame:
    """An AU table as it is written: ``item_id`` as text, a unique ``row_id``, lists and dicts as JSON."""
    table = table.copy()
    table["item_id"] = table["item_id"].astype(str)
    table["row_id"] = table["dataset"] + "_" + table["dimension"] + "_" + table["item_id"]
    for column in JSON_COLUMNS:
        table[column] = table[column].map(as_json)
    return table[CSV_COLUMNS]


def sample_per_dimension(table: pd.DataFrame, dimensions: Sequence[str], n: int = 20, seed: int = 42) -> pd.DataFrame:
    """``n`` random rows of every dimension (fixed seed, dimensions in the order given)."""
    return pd.concat([table[table["dimension"] == d].sample(n, random_state=seed) for d in dimensions], ignore_index=True)


def save_au_tables(tables: Dict[str, pd.DataFrame], out_dir: Path, exclude_for_models: Iterable[str] = ("kumar",),
                   sample: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """Write the AU tables of several datasets (``{name: table}``) under ``out_dir``:

    ``<name>/<name>_au.csv`` for each, ``aposteriori_all.csv`` with all of them, and, for each dataset in ``sample``
    (``{name: {"n": 20, "seed": 42, "dimensions": [...]}}``), ``<name>/<name>_au_sample.csv``.

    ``aposteriori_for_models.csv`` has the whole table of every dataset not in ``exclude_for_models``; a dataset that is
    excluded but has a sample contributes **its sample** (a large dataset is kept as a reference and only a sample of it is
    given to the models). Every file is read back to check it. Returns a table of the files written.
    """
    out_dir = ensure_dir(out_dir)
    exclude = set(exclude_for_models)
    ready = {name: csv_ready(t) for name, t in tables.items()}
    samples = {}
    allowed = {"cover": {"method", "k", "seed", "dimensions"}, "random": {"method", "n", "seed", "dimensions"}}
    for name, o in (sample or {}).items():
        method = o.get("method", "random")
        if method not in allowed:
            raise ValueError(f"sample[{name!r}]: unknown method {method!r}; use 'cover' or 'random'")
        if set(o) - allowed[method]:
            raise ValueError(f"sample[{name!r}]: the method {method!r} does not take {sorted(set(o) - allowed[method])}")
        if method == "cover":                                               # fewest texts so that every group appears >= k times
            samples[name] = csv_ready(cover_per_dimension(tables[name], o.get("dimensions"), o.get("k", 10), o.get("seed", 42)))
        else:                                                               # n random texts per dimension
            samples[name] = sample_per_dimension(ready[name], o["dimensions"], o.get("n", 20), o.get("seed", 42))
    files = {f"{name}/{name}_au.csv": t for name, t in ready.items()}
    files["aposteriori_all.csv"] = pd.concat(ready.values(), ignore_index=True)
    for_models = [t for name, t in ready.items() if name not in exclude]
    for_models += [s for name, s in samples.items() if name in exclude]
    files["aposteriori_for_models.csv"] = pd.concat(for_models, ignore_index=True)
    for name, s in samples.items():
        files[f"{name}/{name}_au_sample.csv"] = s

    rows = []
    for relative, table in files.items():
        path = write_csv(table, out_dir / relative)
        back = read_au_csv(path)
        assert len(back) == len(table) and back["row_id"].is_unique
        rows.append({"file": relative, "rows": len(table), "different texts": table.groupby("dataset")["item_id"].nunique().sum(),
                     "MB": round(path.stat().st_size / 1e6, 2)})
    return pd.DataFrame(rows).set_index("file")
