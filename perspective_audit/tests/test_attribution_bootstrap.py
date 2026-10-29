"""Tests of the bootstrap of the attribution gap (section 7.1), on tiny made-up data."""
import numpy as np
import pandas as pd

from perspective_audit import DatasetSpec, attribution_gap, attribution_table, bootstrap_gap, run_dataset


def toy(name, extra_texts=0):
    """A polarized text a: men say 1, women say 5, and the 'old' say 1 while the 'young' say 5 (AU for gender and for age); the groups of 'tradition' split
    inside (not AU). Text b: not polarized. Text c: only 3 ratings. ``extra_texts`` adds more AU texts."""
    rows = []
    for item in ["a"] + [f"x{i}" for i in range(extra_texts)]:
        rows += [{"item_id": item, "text": "one two", "rating": r, "sex": s, "when": w, "view": v}
                 for r, s, w, v in [(1, "m", "old", "p"), (1, "m", "old", "q"), (5, "f", "young", "p"), (5, "f", "young", "q")]]
    rows += [{"item_id": "b", "text": "three", "rating": 3, "sex": s, "when": w, "view": v} for s, w, v in [("m", "old", "p"), ("f", "young", "q")] * 2]
    rows += [{"item_id": "c", "text": "four", "rating": r, "sex": s, "when": w, "view": v} for r, s, w, v in [(1, "m", "old", "p"), (5, "f", "young", "q"), (5, "f", "young", "q")]]
    raw = pd.DataFrame(rows)
    dimensions = {"gender": ("sex", {"m": "man", "f": "woman"}), "age": ("when", {"old": "gen x+", "young": "gen z"}), "traditionalism": ("view", {"p": "high", "q": "not_high"})}
    spec = DatasetSpec(name=name, load=lambda root: raw, dimensions=dimensions, min_au_texts=1)
    return run_dataset(spec, raw=raw)


def test_table_counts_testable_and_attributed_per_type():
    table = attribution_table(toy("attitudes"))
    assert list(table.index) == ["a"]                                          # b is not polarized, c has only 3 ratings
    row = table.loc["a"]
    assert row["testable D"] == 2 and row["attributed D"] == 2                 # gender and age: testable and AU
    assert row["testable N"] == 1 and row["attributed N"] == 0                 # traditionalism: two groups with two ratings, but each group splits: not AU


def test_the_bootstrap_of_a_constant_gap_has_no_spread():
    table = pd.DataFrame({"testable D": [1] * 20, "attributed D": [1] * 20, "testable N": [1] * 20, "attributed N": [0] * 20})
    result = bootstrap_gap(table, reps=500, seed=1)
    assert result["rate D %"] == 100.0 and result["rate N %"] == 0.0 and result["N - D (pp)"] == -100.0
    assert result["95% interval low"] == result["95% interval high"] == -100.0 and result["share of resamples with N - D <= 0 (%)"] == 100.0


def test_a_positive_gap_with_noise_has_an_interval_around_it_and_it_is_reproducible():
    rng = np.random.default_rng(0)
    n = 400
    table = pd.DataFrame({"testable D": np.ones(n, int), "attributed D": (rng.random(n) < 0.05).astype(int),
                          "testable N": np.ones(n, int), "attributed N": (rng.random(n) < 0.15).astype(int)})
    first, second = bootstrap_gap(table, reps=2000, seed=42), bootstrap_gap(table, reps=2000, seed=42)
    assert first == second                                                     # the same seed gives the same result
    assert first["95% interval low"] < first["N - D (pp)"] < first["95% interval high"] and first["N - D (pp)"] > 0
    assert first["share of resamples with N - D <= 0 (%)"] < 5


def test_attribution_gap_has_one_row_per_dataset_in_the_order_of_the_figures():
    gap = attribution_gap({"kumar": toy("kumar", 5), "attitudes": toy("attitudes", 5)}, reps=200, seed=42)
    assert list(gap.index) == ["attitudes", "kumar"] and (gap["texts"] == 6).all() and (gap["resamples"] == 200).all()
    assert gap.loc["kumar", "rate D %"] == 100.0 and gap.loc["kumar", "rate N %"] == 0.0 and gap.loc["kumar", "N - D (pp)"] == -100.0
