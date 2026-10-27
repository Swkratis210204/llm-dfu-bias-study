"""Tests of the minimal cover sample."""
import pandas as pd

from perspective_audit import save_au_tables
from perspective_audit.sampling import appearances, cover_per_dimension, minimal_cover


def table(counts, dimension="race"):
    """AU rows: ``counts`` maps a combination of groups to how many rows have it."""
    rows = []
    for combo, n in counts.items():
        for i in range(n):
            rows.append({"dataset": "d", "dimension": dimension, "item_id": f"{'-'.join(combo)}{i}", "text": "t", "n_ratings": 5, "dfu": 0.5,
                         "toxic_pole": "{}", "kept_groups": list(combo), "group_sizes": {g: 2 for g in combo},
                         "group_label": {g: "civil" for g in combo}})
    return pd.DataFrame(rows)


def test_two_groups_need_k_rows():
    chosen = minimal_cover(table({("a", "b"): 50}), k=10)
    assert len(chosen) == 10 and appearances(chosen) == {"a": 10, "b": 10}


def test_three_groups_compared_two_at_a_time_need_one_and_a_half_k():
    chosen = minimal_cover(table({("a", "b"): 80, ("a", "c"): 80, ("b", "c"): 80}), k=10)
    assert len(chosen) == 15 and min(appearances(chosen).values()) >= 10


def test_a_row_with_all_groups_covers_them_all():
    chosen = minimal_cover(table({("a", "b", "c"): 40, ("a", "b"): 100}), k=10)
    assert len(chosen) == 10 and appearances(chosen) == {"a": 10, "b": 10, "c": 10}


def test_a_rare_group_shapes_the_answer():
    data = table({("a", "b"): 500, ("a", "c"): 12, ("b", "c"): 3})
    chosen = minimal_cover(data, k=10)
    counts = appearances(chosen)
    # c needs 10 of the 15 rows that contain it, and only 3 of those also contain b: the best is 3 (b,c) + 7 (a,c) + 7 (a,b)
    assert len(chosen) == 17 and min(counts.values()) >= 10


def test_a_group_with_fewer_than_k_rows_is_taken_in_all_of_them():
    data = table({("a", "b"): 100, ("a", "c"): 4})
    chosen = minimal_cover(data, k=10)
    assert appearances(chosen)["c"] == 4 and appearances(chosen)["a"] >= 10 and appearances(chosen)["b"] >= 10


def test_same_seed_same_rows_and_no_row_twice():
    data = table({("a", "b"): 60, ("a", "c"): 60, ("b", "c"): 60})
    first, second = minimal_cover(data, k=10, seed=7), minimal_cover(data, k=10, seed=7)
    assert list(first["item_id"]) == list(second["item_id"]) and first["item_id"].is_unique
    assert list(minimal_cover(data, k=10, seed=8)["item_id"]) != list(first["item_id"])


def test_per_dimension_and_saving(tmp_path):
    data = pd.concat([table({("a", "b"): 40}, "gender"), table({("a", "b"): 40, ("a", "c"): 40, ("b", "c"): 40}, "race")], ignore_index=True)
    data["dataset"] = "big"
    out = cover_per_dimension(data, ["gender", "race"], k=10)
    assert out["dimension"].value_counts().to_dict() == {"race": 15, "gender": 10}
    saved = save_au_tables({"big": data}, tmp_path, exclude_for_models=("big",),
                           sample={"big": {"method": "cover", "k": 10, "seed": 42, "dimensions": ["gender", "race"]}})
    assert saved.loc["big/big_au_sample.csv", "rows"] == 25 and saved.loc["aposteriori_for_models.csv", "rows"] == 25
