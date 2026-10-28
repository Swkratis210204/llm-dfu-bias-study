"""From a dataset description to its AU texts: clean, find the AU texts of every dimension, remove thin groups, summarise."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .spec import DatasetSpec
from .unimodality import PrunedAU, Unimodality
from .utils import find_root

__all__ = ["DatasetResult", "make_tool", "clean", "run_dataset"]

AU_COLUMNS = ["dataset", "dimension", "item_id", "text", "n_ratings", "dfu", "toxic_pole", "kept_groups", "group_sizes", "group_label"]


def make_tool(spec: DatasetSpec) -> Unimodality:
    """The AU tool with the rating scale and rules of the dataset (2 ratings per group, 2 groups)."""
    return Unimodality(scale=range(1, spec.scale_max + 1), min_ratings=2, min_groups=2,
                       toxic_threshold=spec.toxic_threshold, tolerance=spec.tolerance)


def clean(spec: DatasetSpec, raw: pd.DataFrame):
    """Apply the prefilter and map every dimension to its groups; a rating with a removed value in any dimension is dropped.

    Returns ``(clean table, report)``. The clean table has ``item_id``, ``text``, ``rating`` and one column per dimension.
    """
    table = spec.prefilter(raw) if spec.prefilter else raw
    out = table[["item_id", "text", "rating"]].copy()
    for dimension, (column, mapping) in spec.dimensions.items():
        out[dimension] = mapping(table[column]) if callable(mapping) else table[column].map(mapping)
    dims = list(spec.dimensions)
    removed = {dimension: int(out[dimension].isna().sum()) for dimension in dims}
    out = out.dropna(subset=dims).reset_index(drop=True)
    report = {
        "ratings raw": len(raw), "ratings after prefilter": len(table), "ratings clean": len(out),
        "items raw": raw["item_id"].nunique(), "items clean": out["item_id"].nunique(),
        "ratings with a removed value": removed,
        "ratings left per item": out.groupby("item_id").size().value_counts().sort_index().to_dict(),
    }
    return out, report


@dataclass
class DatasetResult:
    spec: DatasetSpec
    clean: pd.DataFrame
    report: dict
    pruned: Dict[str, PrunedAU]
    tool: Unimodality
    raw: Optional[pd.DataFrame] = None               # the table as loaded, before any filter (for the description of the data)

    def describe(self) -> str:
        """The cleaning in a few lines: ratings and items before and after, what the cleaning removed."""
        r = self.report
        lines = [f"ratings {r['ratings raw']:,} -> {r['ratings clean']:,} | items {r['items raw']:,} -> {r['items clean']:,}"]
        if r["ratings after prefilter"] != r["ratings raw"]:
            lines.append(f"prefilter removed {r['ratings raw'] - r['ratings after prefilter']:,} ratings (before the group cleaning)")
        lines.append(f"ratings with a removed value, per dimension: {r['ratings with a removed value']}")
        lines.append(f"ratings left per item: {r['ratings left per item']}")
        return "\n".join(lines)

    @property
    def groups(self) -> pd.DataFrame:
        """Ratings per group after the cleaning."""
        table = {dim: self.clean[dim].value_counts() for dim in self.spec.dimensions}
        return pd.DataFrame(table).stack().dropna().astype(int).rename("ratings").to_frame()

    @property
    def au(self) -> pd.DataFrame:
        """All AU texts: one row per text and dimension."""
        parts = []
        for dimension, pr in self.pruned.items():
            part = pr.result[pr.result["au"]].copy()
            part["dimension"] = dimension
            parts.append(part)
        table = pd.concat(parts, ignore_index=True)
        table["dataset"] = self.spec.name
        return table[AU_COLUMNS]

    @property
    def starting_list(self) -> pd.DataFrame:
        """Every group with the AU texts it takes part in before and after the thin groups were removed."""
        rows = []
        for dimension, pr in self.pruned.items():
            for group, n in pr.history.iloc[0]["AU texts per group"].items():
                rows.append({"dimension": dimension, "group": group, "AU texts at start": n,
                             "AU texts at the end": pr.counts.get(group), "removed": group in pr.removed})
        return pd.DataFrame(rows).set_index(["dimension", "group"])

    @property
    def per_dimension(self) -> pd.DataFrame:
        n_items = self.clean["item_id"].nunique()
        table = pd.DataFrame({
            "texts": {dim: n_items for dim in self.pruned},
            "survived (AU)": {dim: int(pr.history.iloc[-1]["AU texts"]) for dim, pr in self.pruned.items()},
            "groups kept": {dim: len(pr.counts) for dim, pr in self.pruned.items()},
            "groups removed": {dim: ", ".join(map(str, pr.removed)) or "-" for dim, pr in self.pruned.items()},
        })
        table.insert(2, "not survived", table["texts"] - table["survived (AU)"])
        return table

    @property
    def per_group(self) -> pd.DataFrame:
        rows = []
        for dimension, pr in self.pruned.items():
            au = pr.result[pr.result["au"]]
            for group in sorted(pr.counts, key=str):
                labels = [label[group] for label in au["group_label"] if group in label]
                rows.append({"dimension": dimension, "group": group, "AU texts": len(labels),
                             "toxic": labels.count("toxic"), "civil": labels.count("civil")})
        return pd.DataFrame(rows).set_index(["dimension", "group"])

    @property
    def removal_cost(self) -> pd.DataFrame:
        """What removing a group cost: AU texts at the start (no group removed) against the end."""
        rows = []
        for dimension, pr in self.pruned.items():
            if not pr.removed:
                continue
            start = self.tool.analyse(self.clean, "item_id", "rating", dimension)
            start_au, final_au = set(start.loc[start["au"], "item_id"]), set(pr.result.loc[pr.result["au"], "item_id"])
            for group in pr.removed:
                rows.append({"dimension": dimension, "group removed": group, "ratings": int((self.clean[dimension] == group).sum()),
                             "AU texts it took part in": int(sum(group in lab for lab in start.loc[start["au"], "group_label"])),
                             "AU texts at start": len(start_au), "AU texts at end": len(final_au),
                             "lost": len(start_au - final_au), "gained": len(final_au - start_au)})
        return pd.DataFrame(rows).set_index(["dimension", "group removed"]) if rows else pd.DataFrame()


def run_dataset(spec: DatasetSpec, root: Optional[Path] = None, raw: Optional[pd.DataFrame] = None,
                min_au_texts: Optional[int] = None, cache: Optional[Path] = None) -> DatasetResult:
    """Load (unless ``raw`` is given), clean, and find the AU texts of every dimension of a dataset.

    ``cache`` is a file for a large dataset (Kumar takes minutes): when it exists and was made for the same dataset and ``min_au_texts``, the
    result is read from it instead of being computed again; otherwise the result is computed and saved there. It is only a convenience: delete
    the file to run the analysis again (after the data, the rules or the package changed), and use the same Python that made it."""
    threshold = spec.min_au_texts if min_au_texts is None else min_au_texts
    key = (spec.name, threshold)
    if cache is not None and Path(cache).exists():
        with open(cache, "rb") as f:
            saved_key, saved = pickle.load(f)
        if saved_key == key and hasattr(saved, "raw"):
            return saved
    raw = spec.load(Path(root) if root else find_root()) if raw is None else raw
    cleaned, report = clean(spec, raw)
    tool = make_tool(spec)
    pruned = {dim: tool.au_pruned(cleaned, "item_id", "rating", dim, min_au_texts=threshold, keep_cols=["text"])
              for dim in spec.dimensions}
    result = DatasetResult(spec, cleaned, report, pruned, tool, raw)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump((key, result), f, protocol=4)
    return result
