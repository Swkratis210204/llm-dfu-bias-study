"""Tests of the description of a dataset, on a tiny made-up dataset: no files, no models."""
import pandas as pd
import pytest

from perspective_audit import DatasetSpec, eda_figures, eda_tables, run_dataset, save_eda


def toy():
    """Text a: men say 1, women say 5 (polarised, each group unimodal: AU, the groups disagree). Text b: everybody says 3 (not polarised)."""
    rows = []
    for item, text, men, women in [("a", "one two three", [1, 1, 1], [5, 5, 5]), ("b", "four five", [3, 3, 3], [3, 3, 3])]:
        rows += [{"item_id": item, "text": text, "rating": r, "sex": "m"} for r in men]
        rows += [{"item_id": item, "text": text, "rating": r, "sex": "f"} for r in women]
    raw = pd.DataFrame(rows)
    spec = DatasetSpec(name="toy", load=lambda root: raw, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1)
    return run_dataset(spec, raw=raw)


def test_tables():
    tables = eda_tables(toy())
    pre = tables["preprocessing"]["count"]
    assert pre["ratings (raw)"] == 12 and pre["ratings (clean)"] == 12 and pre["removed by the prefilter"] == 0
    assert pre["texts (raw)"] == 2 and pre["texts lost"] == 0 and pre["ratings with an excluded gender value (after the prefilter)"] == 0
    overview = tables["overview"]
    assert overview.loc["texts with > 3x the median ratings (% of the ratings)", "all texts"] == "0 (0.0%)"
    assert overview.loc["texts"].to_dict() == {"all texts": 2, "AU texts": 1} and overview.loc["ratings"].to_dict() == {"all texts": 12, "AU texts": 6}
    assert overview.loc["polarised texts %"].to_dict() == {"all texts": 50.0, "AU texts": 100.0}
    assert overview.loc["ratings called toxic %"].to_dict() == {"all texts": 75.0, "AU texts": 50.0}
    assert overview.loc["ratings per text (min / median / max)", "all texts"] == "6 / 6 / 6"
    distribution = tables["rating_distribution"]
    assert distribution["all texts: ratings"].to_dict() == {1: 3, 2: 0, 3: 6, 4: 0, 5: 3}
    assert distribution["AU texts: ratings"].to_dict() == {1: 3, 2: 0, 3: 0, 4: 0, 5: 3} and distribution.loc[1, "AU texts: %"] == 50.0
    funnel = tables["funnel"].loc["gender"]
    assert funnel["texts"] == 2 and funnel["AU"] == 1 and funnel["not polarised"] == 1 and funnel["AU %"] == 50.0
    groups = tables["groups"]
    assert groups.loc[("gender", "man"), "mean rating"] == 2.0 and groups.loc[("gender", "man"), "toxic %"] == 50.0 and groups.loc[("gender", "woman"), "toxic %"] == 100.0
    assert groups.loc[("gender", "man"), "AU ratings"] == 3 and groups.loc[("gender", "man"), "AU toxic %"] == 0.0
    assert groups.loc[("gender", "woman"), "AU mean rating"] == 5.0 and groups.loc[("gender", "woman"), "AU toxic %"] == 100.0
    assert bool(groups.loc[("gender", "man"), "kept in AU analysis"])
    labels = tables["au_labels"].loc["gender"]
    assert labels["AU texts"] == 1 and labels["groups disagree"] == 1 and labels["all toxic"] == 0 and labels["groups disagree %"] == 100.0
    assert tables["text_length"].loc["count"].to_dict() == {"all texts": 2.0, "AU texts": 1.0}
    assert tables["text_length"].loc["mean"].to_dict() == {"all texts": 2.5, "AU texts": 3.0}


def test_save_writes_every_table_and_figure(tmp_path):
    pytest.importorskip("matplotlib")
    written = save_eda(toy(), tmp_path / "eda")
    assert set(written) == {"preprocessing.csv", "overview.csv", "rating_distribution.csv", "funnel.csv", "groups.csv", "au_labels.csv", "text_length.csv",
                            "ratings.png", "groups.png", "funnel.png", "au_labels.png"}
    assert all(path.exists() for path in written.values())
    assert len(eda_figures(toy(), tmp_path / "again")) == 4
