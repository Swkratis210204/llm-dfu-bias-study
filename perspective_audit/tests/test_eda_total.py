"""Tests of the description of all the datasets together, on two tiny made-up datasets (one with a 1-5 scale, one with 1-3): no files, no models."""
import pandas as pd
import pytest

from perspective_audit import DatasetSpec, ratings_per_text, run_dataset, save_total_eda
from perspective_audit.eda_total import ratings_per_text_lines, rating_shares, toxicity_shares


def toy(name, scale_max, high, middle, low):
    """Text a: 3 men say 1 and 3 women say ``high``; text b: 2 men and 1 woman say ``middle`` (only 3 ratings); text c: 2 men and 2 women say ``low``."""
    rows = [{"item_id": "a", "text": "one two", "rating": r, "sex": s} for r, s in [(1, "m")] * 3 + [(high, "f")] * 3]
    rows += [{"item_id": "b", "text": "three", "rating": middle, "sex": s} for s in "mmf"]
    rows += [{"item_id": "c", "text": "four", "rating": low, "sex": s} for s in "mmff"]
    raw = pd.DataFrame(rows)
    spec = DatasetSpec(name=name, load=lambda root: raw, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, scale_max=scale_max, min_au_texts=1)
    return run_dataset(spec, raw=raw)


def results():
    return {"five": toy("five", 5, high=5, middle=3, low=2), "three": toy("three", 3, high=3, middle=2, low=1)}


def test_rating_shares_are_per_dataset_and_cover_the_scale():
    shares = rating_shares(results())
    five = shares[shares["dataset"] == "five"].set_index("rating")
    assert list(five.index) == [1, 2, 3, 4, 5] and five["ratings"].to_dict() == {1: 3, 2: 4, 3: 3, 4: 0, 5: 3}
    assert five.loc[2, "%"] == 30.8 and abs(five["%"].sum() - 100) < 0.2
    assert list(shares[shares["dataset"] == "three"]["rating"]) == [1, 2, 3]


def test_toxicity_shares_use_the_threshold():
    shares = toxicity_shares(results()).set_index(["dataset", "label"])
    assert shares.loc[("five", "toxic"), "ratings"] == 6 and shares.loc[("five", "civil"), "ratings"] == 7      # 3, 3, 3 and 5, 5, 5 are toxic
    assert shares.loc[("five", "toxic"), "%"] == 46.2 and shares.loc[("five", "toxic"), "toxic is a rating >="] == 3


def test_ratings_per_text_and_its_lines():
    table = ratings_per_text(results())
    row = table.loc["five"]
    assert (row["texts"], row["min"], row["median"], row["max"]) == (3, 3, 4, 6)
    assert row["texts with fewer than 4 ratings"] == 1 and row["% of texts"] == 33.3
    assert [str(table[c].dtype) for c in ("texts", "min", "max", "texts with fewer than 4 ratings")] == ["int64"] * 4     # whole numbers stay whole
    assert "five,3,3,4,6,1,33.3" in table.to_csv().splitlines()
    lines = ratings_per_text_lines(table).splitlines()
    assert lines[0] == "five: ratings per text min 3, median 4, max 6; 1 of 3 texts (33.3%) have fewer than 4 ratings" and len(lines) == 2


def groups_tables():
    """The groups tables of the per-dataset EDA, as saved (dimension and group as the index)."""
    columns = ["ratings", "toxic %", "AU ratings", "AU toxic %", "kept in AU analysis"]
    one = pd.DataFrame([[100, 20.0, 10, 40.0, True], [120, 30.0, 12, 50.0, True]], columns=columns,
                       index=pd.MultiIndex.from_tuples([("gender", "Man"), ("gender", "Woman")], names=["dimension", "group"]))
    two = pd.DataFrame([[200, 25.0, 20, 45.0, True], [210, 35.0, 0, float("nan"), False], [50, 10.0, 5, 15.0, True]], columns=columns,
                       index=pd.MultiIndex.from_tuples([("gender", "male"), ("gender", "female"), ("ideology", "liberal")], names=["dimension", "group"]))
    return {"one": one, "two": two}


def test_toxicity_by_group_harmonizes_names_and_keeps_the_originals():
    from perspective_audit import toxicity_by_group
    table = toxicity_by_group(groups_tables())
    gender = table[table["dimension"] == "gender"]
    assert sorted(gender["group"].unique()) == ["man", "woman"] and len(gender) == 4                       # Man / male -> man, Woman / female -> woman
    assert set(gender["group (as in the dataset)"]) == {"Man", "Woman", "male", "female"}
    politics = table[table["dimension"] == "politics"].iloc[0]
    assert politics["dimension (as in the dataset)"] == "ideology" and politics["group"] == "left / liberal" and politics["dataset"] == "two"
    assert list(table["dimension"].unique()) == ["gender", "politics"]                                     # demographic first


def test_save_toxicity_by_dimension(tmp_path):
    pytest.importorskip("matplotlib")
    from perspective_audit import save_toxicity_by_dimension
    written = save_toxicity_by_dimension(groups_tables(), tmp_path / "t")
    assert set(written) == {"toxicity_by_group.csv", "toxicity_by_dimension_all_texts.png", "toxicity_by_dimension_all_texts.pdf",
                            "toxicity_by_dimension_au_texts.png", "toxicity_by_dimension_au_texts.pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())


def test_the_rating_figure_names_orders_and_colors(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from perspective_audit.eda_total import _rating_figure, BLUE, RED
    shares = pd.concat([rating_shares({"kumar": toy("kumar", 5, 5, 3, 2)}), rating_shares({"mhs": toy("mhs", 3, 3, 2, 1)}),
                        rating_shares({"attitudes": toy("attitudes", 5, 5, 3, 2)}), rating_shares({"popquorn": toy("popquorn", 5, 5, 3, 2)})], ignore_index=True)
    fig = _rating_figure(plt, shares, {"kumar": 3, "mhs": 3, "attitudes": 3, "popquorn": 3})
    panels = [ax for ax in fig.axes if ax.get_title()]
    assert [ax.get_title().split(" (")[0] for ax in panels] == ["Attitudes", "POPQUORN", "MHS", "Kumar"]               # names and order
    assert panels[0].get_title() == "Attitudes (13 ratings)"
    colors = {name: [to_hex(bar.get_facecolor()).lower() for bar in ax.patches] for name, ax in zip(["Attitudes", "POPQUORN", "MHS", "Kumar"], panels)}
    blue, red = BLUE.lower(), RED.lower()
    assert colors["Attitudes"] == [blue, blue, red, red, red] and colors["MHS"] == [blue, blue, red]               # only a rating of 3 or more is red
    assert [t.get_text() for t in panels[1].get_legend().get_texts()] == ["civil", "toxic"] and fig._suptitle is None
    plt.close(fig)


def test_the_dimension_figure_names_orders_and_group_labels():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from perspective_audit import toxicity_by_group
    from perspective_audit.eda_total import _by_dimension_figure
    columns = ["ratings", "toxic %", "AU ratings", "AU toxic %", "kept in AU analysis"]

    def table(groups):
        index = pd.MultiIndex.from_tuples(groups, names=["dimension", "group"])
        return pd.DataFrame([[10, 20.0, 5, 30.0, True]] * len(groups), columns=columns, index=index)

    tables = {"kumar": table([("age", "millenial"), ("politics", "liberal"), ("politics", "independent"), ("politics", "conservative")]),
              "mhs": table([("age", "millenial"), ("ideology", "neutral")]), "attitudes": table([("age", "millenial"), ("politics", "left"), ("politics", "right")]),
              "popquorn": table([("age", "millenial")])}
    fig = _by_dimension_figure(plt, toxicity_by_group(tables), "toxic %", False)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Attitudes", "POPQUORN", "MHS", "Kumar"] and fig._suptitle is None
    labels = {ax.get_title().split(" (")[0]: [t.get_text() for t in ax.get_xticklabels()] for ax in fig.axes if ax.get_title()}
    assert labels["age"] == ["millennial"] and labels["politics"] == ["left", "center", "right"]
    plt.close(fig)


def test_the_appendix_figures_are_redrawn_from_the_saved_csv_files(tmp_path):
    pytest.importorskip("matplotlib")
    from perspective_audit import save_appendix_figures, save_toxicity_by_dimension
    save_total_eda(results(), tmp_path)
    save_toxicity_by_dimension(groups_tables(), tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("rating_distribution.csv", "toxicity.csv", "toxicity_by_group.csv")}
    (tmp_path / "rating_distribution.pdf").unlink()
    (tmp_path / "toxicity_by_dimension_all_texts.png").unlink()
    written = save_appendix_figures(tmp_path)
    assert set(written) == {"rating_distribution.pdf", "rating_distribution.png", "toxicity_by_dimension_all_texts.pdf", "toxicity_by_dimension_all_texts.png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
    assert before == {name: (tmp_path / name).read_bytes() for name in before}                  # no number was touched


def test_save_writes_the_figures_tables_and_lines(tmp_path):
    pytest.importorskip("matplotlib")
    written = save_total_eda(results(), tmp_path / "total_eda")
    assert set(written) == {"rating_distribution.png", "rating_distribution.pdf", "rating_distribution.csv", "toxicity.png", "toxicity.csv",
                            "ratings_per_text.csv", "ratings_per_text.txt"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
