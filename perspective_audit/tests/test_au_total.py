"""Tests of the AU exploration across datasets, on tiny made-up datasets: no files, no models."""
import json

import pandas as pd
import pytest

from perspective_audit import DatasetSpec, run_dataset, save_total_au
from perspective_audit.au_total import (au_dimension_overlap, au_groups, au_label_patterns, au_models_input, au_summary)


def toy(name):
    """Text a: 3 men say 1 and 3 women say 5 (polarized, AU). Text b: 2 men and 1 woman say 3 (not polarized). Text c: 2 men and 2 women say 2."""
    rows = [{"item_id": "a", "text": "one two", "rating": r, "sex": s} for r, s in [(1, "m")] * 3 + [(5, "f")] * 3]
    rows += [{"item_id": "b", "text": "three", "rating": 3, "sex": s} for s in "mmf"]
    rows += [{"item_id": "c", "text": "four", "rating": 2, "sex": s} for s in "mmff"]
    raw = pd.DataFrame(rows)
    spec = DatasetSpec(name=name, load=lambda root: raw, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1)
    return run_dataset(spec, raw=raw)


def results():
    return {"kumar": toy("kumar"), "attitudes": toy("attitudes")}


def models():
    label = json.dumps({"man": "civil", "woman": "toxic"})
    return pd.DataFrame({"dataset": ["kumar", "attitudes"], "dimension": ["gender", "gender"], "item_id": ["a", "a"], "group_label": [label, label]})


def test_summary():
    summary = au_summary(results(), models())
    assert list(summary.index) == ["attitudes", "kumar"]                                                      # the order of the figures
    row = summary.loc["kumar"]
    assert row["texts"] == 3 and row["AU texts"] == 1 and row["AU texts %"] == 33.3 and row["AU rows (text x dimension)"] == 1
    assert row["AU in 1 dimension"] == 1 and row["AU in 2 dimensions"] == 0 and row["rows given to the models"] == 1
    assert row["ratings in the AU texts"] == 6 and row["ratings per AU text (median)"] == 6 and row["ratings called toxic % (AU texts)"] == 50.0
    assert row["mean DFU (AU texts)"] > 0


def test_groups_status_and_au_numbers():
    table = au_groups(results()).set_index(["dataset", "dimension", "group"])
    man = table.loc[("kumar", "gender", "man")]
    assert man["AU texts at the start"] == 1 and man["AU texts at the end"] == 1 and man["status"] == "kept"
    assert man["ratings (AU texts)"] == 3 and man["toxic % (AU texts)"] == 0.0 and table.loc[("kumar", "gender", "woman"), "toxic % (AU texts)"] == 100.0


def test_a_group_below_the_minimum_is_labelled():
    result = toy("kumar")
    spec = DatasetSpec(name="kumar", load=result.spec.load, dimensions=result.spec.dimensions, min_au_texts=5)
    thin = run_dataset(spec, raw=result.raw)                                                                    # every group is below 5: nothing is removed
    table = au_groups({"kumar": thin}).set_index("group")
    assert set(table["status"]) == {"kept, below 5 (the rule could not remove it)"}


def test_label_patterns_overlap_and_models_input():
    patterns = au_label_patterns(results()).set_index(["dataset", "dimension"]).loc[("kumar", "gender")]
    assert patterns["AU texts"] == 1 and patterns["groups disagree"] == 1 and patterns["all toxic"] == 0 and patterns["group labels"] == 2
    assert patterns["decided by the mean"] == 0 and patterns["groups disagree %"] == 100.0
    assert len(au_dimension_overlap(results())) == 0                                                           # one dimension: no pair
    given = au_models_input(results(), models()).set_index(["dataset", "group"])
    row = given.loc[("kumar", "man")]
    assert row["AU rows with the group (all)"] == 1 and row["AU rows with the group (given to the models)"] == 1
    assert row["share of the group appearances % (all)"] == 50.0 and row["share of the group appearances % (given to the models)"] == 50.0


def test_save_writes_every_table_figure_and_the_readme(tmp_path):
    pytest.importorskip("matplotlib")
    written = save_total_au(results(), models(), tmp_path / "total_au")
    assert set(written) == {"au_summary.csv", "au_funnel.csv", "au_groups.csv", "au_label_patterns.csv", "au_dimension_overlap.csv",
                            "au_rating_distribution.csv", "au_models_input.csv", "au_funnel.png", "au_texts_per_group.png", "au_label_patterns.png",
                            "au_dimension_overlap.png", "au_rating_distribution.pdf", "au_rating_distribution.png", "au_toxicity_by_dimension.pdf",
                            "au_toxicity_by_dimension.png", "au_kumar_sample.png", "au_kumar_sample.pdf", "README.md"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
    from perspective_audit.au_total import save_kumar_sample_figure
    (tmp_path / "total_au" / "au_kumar_sample.pdf").unlink()
    assert set(save_kumar_sample_figure(tmp_path / "total_au")) == {"au_kumar_sample.pdf", "au_kumar_sample.png"}          # redrawn from the csv alone
