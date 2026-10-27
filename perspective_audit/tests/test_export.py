"""Tests of the saving of AU tables, with small made-up tables."""
import pandas as pd
import pytest

from perspective_audit import read_au_csv, save_au_tables


def au_table(dataset, n_per_dimension, dimensions=("gender", "age")):
    rows = []
    for dim in dimensions:
        for i in range(n_per_dimension):
            rows.append({"dataset": dataset, "dimension": dim, "item_id": i, "text": f"{dataset} {dim} {i}", "n_ratings": 5,
                         "dfu": 0.5, "toxic_pole": "{a:civil, b:toxic}", "kept_groups": ["a", "b"],
                         "group_sizes": {"a": 2, "b": 3}, "group_label": {"a": "civil", "b": "toxic"}})
    return pd.DataFrame(rows)


def test_models_file_has_the_sample_of_the_excluded_dataset(tmp_path):
    tables = {"small": au_table("small", 3), "big": au_table("big", 40)}
    saved = save_au_tables(tables, tmp_path, exclude_for_models=("big",),
                           sample={"big": {"n": 5, "seed": 1, "dimensions": ["gender", "age"]}})
    models = read_au_csv(tmp_path / "aposteriori_for_models.csv")
    assert (models["dataset"] == "small").sum() == 6                       # the whole small dataset
    assert (models["dataset"] == "big").sum() == 10                        # only the sample of the big one: 5 per dimension
    assert set(models.loc[models["dataset"] == "big", "row_id"]) <= set(read_au_csv(tmp_path / "big" / "big_au.csv")["row_id"])
    assert len(read_au_csv(tmp_path / "aposteriori_all.csv")) == 6 + 80   # everything, not the sample
    assert len(read_au_csv(tmp_path / "big" / "big_au_sample.csv")) == 10
    assert {"aposteriori_all.csv", "aposteriori_for_models.csv", "big/big_au.csv", "big/big_au_sample.csv"} <= set(saved.index)


def test_unknown_sample_settings_are_refused_not_ignored(tmp_path):
    tables = {"big": au_table("big", 40)}
    for options in ({"method": "covr", "dimensions": ["gender"]},                     # misspelt method
                    {"method": "cover", "n": 20, "dimensions": ["gender"]},            # 'n' belongs to the random method
                    {"k": 10, "dimensions": ["gender"]}):                              # 'k' without method 'cover' (the old code would ignore it)
        with pytest.raises(ValueError):
            save_au_tables(tables, tmp_path, exclude_for_models=("big",), sample={"big": options})


def test_sample_is_reproducible_and_json_columns_come_back(tmp_path):
    tables = {"big": au_table("big", 40)}
    options = {"big": {"n": 5, "seed": 42, "dimensions": ["gender", "age"]}}
    save_au_tables(tables, tmp_path / "one", exclude_for_models=("big",), sample=options)
    save_au_tables(tables, tmp_path / "two", exclude_for_models=("big",), sample=options)
    assert (tmp_path / "one" / "big" / "big_au_sample.csv").read_bytes() == (tmp_path / "two" / "big" / "big_au_sample.csv").read_bytes()
    back = read_au_csv(tmp_path / "one" / "big" / "big_au_sample.csv")
    assert back.loc[0, "group_label"] == {"a": "civil", "b": "toxic"} and back.loc[0, "kept_groups"] == ["a", "b"]
