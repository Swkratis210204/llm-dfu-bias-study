"""Tests of the persona-by-group kappa (section 7.3), on the small made-up dataset of test_kappa_groups where the answers are known."""
import numpy as np
import pandas as pd
import pytest

from perspective_audit import compute_persona_kappa, save_persona_kappa
from test_kappa_groups import build, toy_result


def tables(tmp_path, reps=500):
    results_dir, given = build(tmp_path)
    return compute_persona_kappa({"attitudes": toy_result()}, results_dir, given, reps=reps, seed=42, models=["chatgpt", "deepseek"])


def test_matrix_of_known_cases(tmp_path):
    t = tables(tmp_path)
    m = t["matrix"].set_index(["model", "persona", "group"])
    assert len(m) == 8 and list(t["matrix"].columns) == ["dataset", "model", "characteristic", "persona", "group", "n_texts", "kappa", "se", "ci_low", "ci_high", "constant_label"]
    assert m.loc[("chatgpt", "man", "man"), "kappa"] == 1.0 and m.loc[("chatgpt", "woman", "woman"), "kappa"] == 1.0            # right as every persona
    assert m.loc[("chatgpt", "man", "woman"), "kappa"] == -1.0 and m.loc[("chatgpt", "woman", "man"), "kappa"] == -1.0       # the groups take opposite sides
    assert m.loc[("deepseek", "man", "man"), "kappa"] == pytest.approx(0.3333, abs=1e-4)                                      # the SCD kappa of kappa_by_group
    cell = m.loc[("deepseek", "man", "man")]
    assert cell["n_texts"] == 12 and cell["ci_low"] <= cell["kappa"] <= cell["ci_high"] and cell["se"] > 0 and not cell["constant_label"]


def test_pairs_diagonal_and_high_pairs(tmp_path):
    t = tables(tmp_path)
    pair = t["pairs"].set_index("model")
    assert pair.loc["chatgpt", "kappa"] == -1.0 and pair.loc["chatgpt", "persona_a"] == "man" and pair.loc["chatgpt", "persona_b"] == "woman" and pair.loc["chatgpt", "n_texts"] == 12
    assert pair.loc["deepseek", "kappa"] == pytest.approx(-0.3333, abs=1e-4) and t["high_pairs"].empty
    d = t["diagonal"].set_index(["model"])
    assert d.loc["chatgpt", "comparisons"] == 2 and d.loc["chatgpt", "diagonal_higher"] == 2 and d.loc["all", "comparisons"] == 4 and d.loc["all", "diagonal_lower"] == 0


def test_a_persona_is_compared_only_where_its_group_is(tmp_path):
    results_dir, given = build(tmp_path)
    from test_kappa_groups import toy_result as toy
    result = toy()
    keep = [i for i, labels in enumerate(result.au["group_label"])]
    assert len(keep) == 12                                                                                                       # every toy text has both groups: n = 12 for every cell
    t = compute_persona_kappa({"attitudes": result}, results_dir, given, reps=50, models=["chatgpt"])
    assert (t["matrix"]["n_texts"] == 12).all()


def test_save_writes_the_three_csv_files(tmp_path):
    written = save_persona_kappa(tables(tmp_path / "runs", reps=50), tmp_path / "out")
    assert {"persona_group_matrix.csv", "persona_pairs.csv", "persona_diagonal_summary.csv"} <= set(written)
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
