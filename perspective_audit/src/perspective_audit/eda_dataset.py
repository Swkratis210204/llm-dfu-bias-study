"""Describing the datasets themselves, for the paper: who the annotators are, how the raw attribute values become groups, and what the data is.
These are additions to ``eda_tables``; nothing that exists is changed, and nothing here asks a model for anything.

``dataset_tables(result)`` returns four tables (``save_dataset_eda`` writes them as csv, and ``save_eda`` calls it):

    annotators         annotators and ratings in the raw data, after the prefilter and after the cleaning; ratings per annotator; annotators whose
                       attribute changes between ratings; repeated (annotator, text) pairs. "not available" when the data has no annotator id
    groups_annotators  per dimension and group: ratings and distinct annotators after the cleaning, and whether the group stayed in the AU analysis
                       (same rows as ``groups.csv``, so the two can be joined on dimension and group)
    group_mapping      one row per raw value of every dimension: the group it becomes, kept or excluded, why, and its ratings and annotators in
                       the raw data (a numeric attribute with many values, such as an age in years, is shown as a range per group)
    dataset_info       a short fixed table: what can be read from the data is filled in, the rest says TODO and is for you to write

``eda_overview_all(results)`` puts the datasets in one table (texts, ratings, annotators, ratings per text, scale, dimensions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .pipeline import DatasetResult

__all__ = ["dataset_tables", "save_dataset_eda", "eda_overview_all"]

NA = "not available"
TODO = "TODO"
DEMOGRAPHIC = ("gender", "race", "age")        # every other dimension (politics, religion, education, occupation, ...) is non-demographic
MANY = 30                                      # a numeric attribute with more values than this is shown as a range per group
NO_ID_NOTE = "the data has no annotator id: the raters are known only by the attributes that come with each rating"


def _tables(result: DatasetResult):
    """The raw table, the table after the prefilter, and the cleaned table with every column kept (the same steps as ``pipeline.clean``)."""
    spec, raw = result.spec, result.raw
    if raw is None:
        raise ValueError("this result has no raw table: use run_dataset, which keeps it")
    prefiltered = spec.prefilter(raw) if spec.prefilter else raw
    table = prefiltered.copy()
    for dimension, (column, mapping) in spec.dimensions.items():
        table[dimension] = mapping(prefiltered[column]) if callable(mapping) else prefiltered[column].map(mapping)
    clean = table.dropna(subset=list(spec.dimensions))
    if len(clean) != len(result.clean):
        raise ValueError(f"the cleaning gives {len(clean)} ratings here and {len(result.clean)} in the result: the steps differ")
    return raw, prefiltered, clean


def _count(table: pd.DataFrame, column) -> object:
    """Distinct annotators, or "not available" when the data has no annotator id."""
    return NA if column is None else int(table[column].nunique())


def _annotators(result: DatasetResult, raw, prefiltered, clean) -> pd.DataFrame:
    spec, column = result.spec, result.spec.annotator_column
    rows = {"annotators": [_count(t, column) for t in (raw, prefiltered, clean)], "ratings": [len(raw), len(prefiltered), len(clean)]}
    if column is None:
        rows["ratings per annotator (min / median / max)"] = [NA, NA, NA]
    else:
        per = clean.groupby(column).size()
        rows["ratings per annotator (min / median / max)"] = ["", "", f"{per.min()} / {per.median():g} / {per.max()}"]
        for dimension, (attribute, _) in spec.dimensions.items():
            changing = int((raw.groupby(column)[attribute].nunique() > 1).sum())
            rows[f"annotators with more than one value of {dimension} (raw)"] = [changing, "", ""]
        rows["repeated (annotator, text) pairs (raw)"] = [int(raw.duplicated([column, "item_id"]).sum()), "", ""]
    table = pd.DataFrame(rows, index=["raw", "after prefilter", "clean"]).T.rename_axis("metric")
    if column is None:
        table.loc["note"] = [NO_ID_NOTE, "", ""]
    return table


def _groups_annotators(result: DatasetResult, clean) -> pd.DataFrame:
    column, rows = result.spec.annotator_column, []
    for dimension, pr in result.pruned.items():
        for group, part in clean.groupby(dimension):
            rows.append({"dimension": dimension, "group": group, "ratings": len(part), "annotators": _count(part, column),
                         "kept in AU analysis": group in pr.counts})
    return pd.DataFrame(rows).set_index(["dimension", "group"])


def _why_excluded(value) -> str:
    text = str(value).lower()
    if text in ("nan", "none", ""):
        return "missing"
    if "prefer" in text:
        return "prefer not to say"
    if text in ("na", "n/a", "unknown", "no_opinion", "self-describe") or text.startswith("other"):
        return "other, unknown or no opinion"
    return "not part of the group definition of the dataset"


def _group_mapping(result: DatasetResult, raw) -> pd.DataFrame:
    spec, column_id, rows = result.spec, result.spec.annotator_column, []
    for dimension, (column, mapping) in spec.dimensions.items():
        values, removed = raw[column], result.pruned[dimension].removed
        present = pd.Series(pd.unique(values.dropna()))
        group = mapping(present) if callable(mapping) else present.map(mapping)
        pairs = pd.DataFrame({"value": present.to_numpy(), "group": pd.Series(group).to_numpy()})
        if pd.api.types.is_numeric_dtype(values) and len(present) > MANY:                # a range per group, not 64 ages one by one
            units = [(f"{part['value'].min():g} to {part['value'].max():g}", g, values.isin(part["value"]))
                     for g, part in pairs.groupby(pairs["group"].fillna("(no group)"))]
            units = [(label, None if g == "(no group)" else g, mask) for label, g, mask in units]
        else:
            units = [(str(v), None if pd.isna(g) else g, values == v) for v, g in zip(pairs["value"], pairs["group"])]
        if values.isna().any():
            units.append(("(missing)", None, values.isna()))
        for label, group_name, mask in units:
            if group_name is None:
                status, reason = "excluded", _why_excluded(None if label == "(missing)" else label)
            elif group_name in removed:
                status, reason = "kept in the cleaning, removed from the AU analysis", f"in fewer than {spec.min_au_texts} AU texts"
            else:
                status, reason = "kept", ""
            rows.append({"dimension": dimension, "raw value": label, "mapped group": group_name or "", "status": status, "reason": reason,
                         "ratings (raw)": int(mask.sum()), "annotators (raw)": _count(raw[mask], column_id)})
    return pd.DataFrame(rows)


def _dataset_info(result: DatasetResult, raw) -> pd.DataFrame:
    spec = result.spec
    observed = raw["rating"].value_counts().sort_index()
    rows = [("dataset", spec.name, "dataset definition"),
            ("description", spec.description or TODO, "dataset definition"),
            ("source of the texts", TODO, "to write"),
            ("task", TODO, "to write"),
            ("rating column used", spec.rating_column, "dataset definition"),
            ("observed rating values (analyzed scale), with their counts", ", ".join(f"{v:g}: {n:,}" for v, n in observed.items()), "read from the data"),
            ("scale analyzed", f"1 to {spec.scale_max}", "dataset definition"),
            ("meaning of each value of the scale", TODO, "to write"),
            ("ratings counted as toxic", f"rating >= {spec.toxic_threshold:g}", "dataset definition"),
            ("annotator id", spec.annotator_column or NA, "dataset definition"),
            ("dimensions", ", ".join(spec.dimensions), "dataset definition"),
            ("a group is removed from the AU analysis when it is in fewer AU texts than", spec.min_au_texts, "dataset definition"),
            ("DFU tolerance for polarization and unimodality", spec.tolerance, "dataset definition")]
    return pd.DataFrame(rows, columns=["field", "value", "where it comes from"]).set_index("field")


def dataset_tables(result: DatasetResult) -> Dict[str, pd.DataFrame]:
    """The four tables that describe the dataset itself (see the module docstring). Needs the raw table: use ``run_dataset``."""
    raw, prefiltered, clean = _tables(result)
    return {"annotators": _annotators(result, raw, prefiltered, clean), "groups_annotators": _groups_annotators(result, clean),
            "group_mapping": _group_mapping(result, raw), "dataset_info": _dataset_info(result, raw)}


def save_dataset_eda(result: DatasetResult, folder: Path) -> Dict[str, Path]:
    """Write the tables of ``dataset_tables`` as ``<name>.csv`` into ``folder``. Returns the files written, by file name."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in dataset_tables(result).items():
        table.to_csv(folder / f"{name}.csv", index=name != "group_mapping", encoding="utf-8")
        written[f"{name}.csv"] = folder / f"{name}.csv"
    return written


def eda_overview_all(results: Dict[str, DatasetResult]) -> pd.DataFrame:
    """One row per dataset: texts, ratings and annotators (raw / clean), ratings per text (median), the scale, and the dimensions, split into
    demographic (gender, race, age) and non-demographic (every other: politics, religion, education, occupation, traditionalism, ideology)."""
    rows = []
    for name, result in results.items():
        raw, _, clean = _tables(result)
        report, column = result.report, result.spec.annotator_column
        annotators = f"{_count(raw, column):,} / {_count(clean, column):,}" if column else NA
        dimensions = list(result.spec.dimensions)
        rows.append({"dataset": name, "texts (raw / clean)": f"{report['items raw']:,} / {report['items clean']:,}",
                     "ratings (raw / clean)": f"{report['ratings raw']:,} / {report['ratings clean']:,}",
                     "annotators (raw / clean)": annotators,
                     "ratings per text (median)": float(result.clean.groupby("item_id").size().median()),
                     "scale": f"1 to {result.spec.scale_max}",
                     "demographic dimensions": ", ".join(d for d in dimensions if d in DEMOGRAPHIC),
                     "non-demographic dimensions": ", ".join(d for d in dimensions if d not in DEMOGRAPHIC)})
    return pd.DataFrame(rows).set_index("dataset")
