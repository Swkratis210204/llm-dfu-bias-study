"""Tests of the DFU exploration across datasets, on tiny made-up datasets: no files, no models."""
import pandas as pd
import pytest

from perspective_audit import DatasetSpec, run_dataset, save_total_dfu
from perspective_audit.dfu_total import (dfu_by_ratings_per_text, dfu_group_level, dfu_histogram, dfu_pairwise_ks, dfu_summary, dfu_text_table)


def toy(name):
    """Text a: 3 men say 1 and 3 women say 5 (polarized, AU). Text b: 2 men and 1 woman say 3 (3 ratings, not polarized). Text c: 2 men and 2 women say 2."""
    rows = [{"item_id": "a", "text": "one two", "rating": r, "sex": s} for r, s in [(1, "m")] * 3 + [(5, "f")] * 3]
    rows += [{"item_id": "b", "text": "three", "rating": 3, "sex": s} for s in "mmf"]
    rows += [{"item_id": "c", "text": "four", "rating": 2, "sex": s} for s in "mmff"]
    raw = pd.DataFrame(rows)
    spec = DatasetSpec(name=name, load=lambda root: raw, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1)
    return run_dataset(spec, raw=raw)


def results():
    return {"mhs": toy("mhs"), "attitudes": toy("attitudes")}


def test_text_table_has_one_row_per_dataset_and_text_in_the_order_of_the_figures():
    table = dfu_text_table(results())
    assert list(table["dataset"].unique()) == ["attitudes", "mhs"] and len(table) == 6                       # attitudes comes first
    a = table[(table["dataset"] == "mhs") & (table["item_id"] == "a")].iloc[0]
    assert a["polarized"] and a["AU"] and a["AU dimensions"] == 1 and a["ratings"] == 6 and a["dfu"] > 0
    others = table[table["item_id"] != "a"]
    assert (others["dfu"] == 0).all() and not others["polarized"].any() and not others["AU"].any()


def test_summary_and_histogram():
    table = dfu_text_table(results())
    summary = dfu_summary(table)
    row = summary.loc["mhs"]
    assert row["texts"] == 3 and row["polarized texts"] == 1 and row["polarized %"] == 33.3 and row["DFU = 0 %"] == 66.7
    assert row["AU texts"] == 1 and row["distinct DFU values"] == 2 and row["max DFU"] == row["mean DFU (AU texts)"]
    histogram = dfu_histogram(table)
    mhs = histogram[histogram["dataset"] == "mhs"]
    assert mhs["all texts"].sum() == 3 and mhs["polarized texts"].sum() == 1 and mhs["AU texts"].sum() == 1
    assert mhs.iloc[0]["all texts"] == 2 and mhs.iloc[-1]["all texts"] == 1                                    # the two with DFU 0 and the one with DFU 1


def test_by_ratings_per_text():
    by = dfu_by_ratings_per_text(dfu_text_table(results())).set_index(["dataset", "ratings per text"])
    assert by.loc[("mhs", "6"), "polarized %"] == 100.0 and by.loc[("mhs", "3"), "polarized %"] == 0.0 and by.loc[("mhs", "4"), "texts"] == 1


def test_group_level_counts_the_texts_where_a_group_has_enough_ratings():
    level = dfu_group_level(results()).set_index(["dataset", "dimension", "group"])
    man, woman = level.loc[("mhs", "gender", "man")], level.loc[("mhs", "gender", "woman")]
    assert man["texts with enough ratings"] == 3 and woman["texts with enough ratings"] == 2                  # b has only one woman
    assert man["unimodal %"] == 100.0 and man["polarized texts"] == 1 and man["mean group DFU"] == 0.0


def test_pairwise_ks_of_identical_distributions_is_zero():
    ks = dfu_pairwise_ks(dfu_text_table(results()))
    assert len(ks) == 1 and ks.iloc[0]["dataset A"] == "attitudes" and ks.iloc[0]["KS distance (all texts)"] == 0.0


def test_save_writes_every_table_figure_and_the_readme(tmp_path):
    pytest.importorskip("matplotlib")
    written = save_total_dfu(results(), tmp_path / "total_dfu")
    assert set(written) == {"dfu_text_level.csv", "dfu_summary.csv", "dfu_histogram.csv", "dfu_by_ratings_per_text.csv", "dfu_group_level.csv",
                            "dfu_pairwise_ks.csv", "dfu_funnel.csv", "dfu_all_texts.png", "dfu_polarized_texts.png", "dfu_ecdf.png",
                            "dfu_by_ratings_per_text.png", "dfu_boxplot_all_texts.png", "dfu_boxplot_au_texts.png", "README.md"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
    assert "DFU is the normalized distance" in (tmp_path / "total_dfu" / "README.md").read_text(encoding="utf-8")
