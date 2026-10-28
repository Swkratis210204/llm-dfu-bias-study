"""Tests of the description of the dataset itself (annotators, group mapping, dataset info), on a tiny made-up dataset."""
import pandas as pd

from perspective_audit import DatasetSpec, dataset_tables, eda_overview_all, run_dataset, save_dataset_eda


def toy(annotator_column="rater"):
    """Text a: r1-r3 (m) say 1, r4-r6 (f) say 5, r7 (x, an excluded value) says 3. Text b: everybody says 3, but r1 is recorded as f this time."""
    rows = [{"rater": f"r{i}", "item_id": "a", "text": "one two three", "rating": 1, "sex": "m"} for i in (1, 2, 3)]
    rows += [{"rater": f"r{i}", "item_id": "a", "text": "one two three", "rating": 5, "sex": "f"} for i in (4, 5, 6)]
    rows += [{"rater": "r7", "item_id": "a", "text": "one two three", "rating": 3, "sex": "x"}]
    rows += [{"rater": f"r{i}", "item_id": "b", "text": "four five", "rating": 3, "sex": "m"} for i in (2, 3)]
    rows += [{"rater": f"r{i}", "item_id": "b", "text": "four five", "rating": 3, "sex": "f"} for i in (1, 4, 5, 6)]
    raw = pd.DataFrame(rows)
    spec = DatasetSpec(name="toy", load=lambda root: raw, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1,
                       annotator_column=annotator_column, rating_column="score", description="a toy")
    return run_dataset(spec, raw=raw)


def test_annotators():
    table = dataset_tables(toy())["annotators"]
    assert table.loc["annotators"].to_dict() == {"raw": 7, "after prefilter": 7, "clean": 6}
    assert table.loc["ratings"].to_dict() == {"raw": 13, "after prefilter": 13, "clean": 12}
    assert table.loc["ratings per annotator (min / median / max)", "clean"] == "2 / 2 / 2"
    assert table.loc["annotators with more than one value of gender (raw)", "raw"] == 1          # r1 is m in one text and f in the other
    assert table.loc["repeated (annotator, text) pairs (raw)", "raw"] == 0


def test_without_an_annotator_id_nothing_is_invented():
    tables = dataset_tables(toy(annotator_column=None))
    assert set(tables["annotators"].loc["annotators"]) == {"not available"}
    assert "no annotator id" in tables["annotators"].loc["note", "raw"]
    assert set(tables["groups_annotators"]["annotators"]) == {"not available"}
    assert tables["groups_annotators"]["ratings"].to_dict() == {("gender", "man"): 5, ("gender", "woman"): 7}


def test_groups_annotators_has_the_rows_of_groups_csv():
    from perspective_audit import eda_tables
    result = toy()
    both = dataset_tables(result)["groups_annotators"]
    assert both["annotators"].to_dict() == {("gender", "man"): 3, ("gender", "woman"): 4}       # r1 counts in both groups
    assert list(both.index) == list(eda_tables(result)["groups"].index)
    assert both["ratings"].to_dict() == eda_tables(result)["groups"]["ratings"].to_dict()
    assert list(both["kept in AU analysis"]) == list(eda_tables(result)["groups"]["kept in AU analysis"])


def test_group_mapping_shows_every_raw_value():
    mapping = dataset_tables(toy())["group_mapping"].set_index("raw value")
    assert mapping.loc["m", "mapped group"] == "man" and mapping.loc["m", "status"] == "kept" and mapping.loc["m", "ratings (raw)"] == 5
    assert mapping.loc["f", "mapped group"] == "woman" and mapping.loc["f", "annotators (raw)"] == 4
    assert mapping.loc["x", "status"] == "excluded" and mapping.loc["x", "mapped group"] == "" and mapping.loc["x", "ratings (raw)"] == 1
    assert mapping.loc["x", "reason"] == "not part of the group definition of the dataset"


def test_a_numeric_attribute_with_many_values_is_shown_as_a_range_per_group():
    ages = list(range(18, 70))                                                    # 52 different ages
    raw = pd.DataFrame({"item_id": ["a"] * len(ages), "text": "t", "rating": 1, "age": ages})
    from perspective_audit.datasets.groups import age_generations
    spec = DatasetSpec(name="toy", load=lambda root: raw, dimensions={"age": ("age", age_generations)}, min_au_texts=1)
    mapping = dataset_tables(run_dataset(spec, raw=raw))["group_mapping"].set_index("mapped group")
    assert len(mapping) == 3 and mapping.loc["gen z", "raw value"] == "18 to 24" and mapping.loc["gen x+", "raw value"] == "45 to 69"
    assert mapping["ratings (raw)"].sum() == len(ages)


def test_missing_values_and_reasons():
    raw = pd.DataFrame({"item_id": ["a", "a", "a"], "text": "t", "rating": 1, "pol": ["liberal", "Prefer not to say", None]})
    spec = DatasetSpec(name="toy", load=lambda root: raw, dimensions={"politics": ("pol", {"liberal": "liberal"})}, min_au_texts=1)
    mapping = dataset_tables(run_dataset(spec, raw=raw))["group_mapping"].set_index("raw value")
    assert mapping.loc["Prefer not to say", "reason"] == "prefer not to say" and mapping.loc["(missing)", "reason"] == "missing"
    assert mapping.loc["(missing)", "ratings (raw)"] == 1


def test_dataset_info_fills_what_the_data_tells_and_leaves_the_rest_as_todo():
    info = dataset_tables(toy())["dataset_info"]["value"]
    assert info["rating column used"] == "score" and info["scale analyzed"] == "1 to 5" and info["ratings counted as toxic"] == "rating >= 3"
    assert info["observed rating values (analyzed scale), with their counts"] == "1: 3, 3: 7, 5: 3"
    assert info["source of the texts"] == "TODO" and info["task"] == "TODO" and info["meaning of each value of the scale"] == "TODO"


def test_save_and_the_overview_of_all_datasets(tmp_path):
    written = save_dataset_eda(toy(), tmp_path)
    assert set(written) == {"annotators.csv", "groups_annotators.csv", "group_mapping.csv", "dataset_info.csv"} and all(p.exists() for p in written.values())
    overview = eda_overview_all({"toy": toy(), "toy2": toy(annotator_column=None)})
    assert overview.loc["toy", "annotators (raw / clean)"] == "7 / 6" and overview.loc["toy2", "annotators (raw / clean)"] == "not available"
    assert overview.loc["toy", "texts (raw / clean)"] == "2 / 2" and overview.loc["toy", "ratings (raw / clean)"] == "13 / 12"
    assert overview.loc["toy", "ratings per text (median)"] == 6.0 and overview.loc["toy", "scale"] == "1 to 5"
    assert overview.loc["toy", "demographic dimensions"] == "gender" and overview.loc["toy", "non-demographic dimensions"] == ""
