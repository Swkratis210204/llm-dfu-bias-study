"""Tests of the agreement between the models (section 7.4), on made-up ratings where the answers are known."""
import numpy as np
import pandas as pd
import pytest

from perspective_audit import compare_models, save_model_comparison

IDS = [f"t{i}" for i in range(8)]
SCORES = {"chatgpt": [1, 2, 3, 4, 5, 1, 2, 5], "deepseek": [1, 2, 3, 4, 5, 1, 2, 5], "gemini": [5, 4, 3, 2, 1, 5, 4, 1], "mistral": [4] * 8}


def build(tmp_path, datasets=("attitudes", "mhs")):
    for dataset in datasets:
        (tmp_path / dataset / "baseline").mkdir(parents=True)
        for model, scores in SCORES.items():
            pd.DataFrame({"dataset": dataset, "item_id": IDS, "text": "x", "model": model, "run": "baseline", "score": scores}).to_csv(tmp_path / dataset / "baseline" / f"{model}.csv", index=False)
    return tmp_path


def test_pairs_and_matrices_of_known_cases(tmp_path):
    t = compare_models(build(tmp_path), datasets=["attitudes"])
    pair = t["pairwise"].set_index(["model_a", "model_b"])
    assert pair.loc[("chatgpt", "deepseek"), "spearman"] == 1.0 and pair.loc[("chatgpt", "deepseek"), "kappa"] == 1.0 and pair.loc[("chatgpt", "deepseek"), "n_texts"] == 8
    assert pair.loc[("chatgpt", "gemini"), "spearman"] == pytest.approx(-0.9, abs=0.1) and pair.loc[("chatgpt", "gemini"), "kappa"] < 0
    assert np.isnan(pair.loc[("chatgpt", "mistral"), "spearman"]) and pair.loc[("chatgpt", "mistral"), "kappa"] == 0.0     # a constant model: no correlation, kappa 0
    assert pair.loc[("chatgpt", "mistral"), "constant_b"] and not pair.loc[("chatgpt", "mistral"), "constant_a"] and len(pair) == 6
    m = t["matrices"].query("measure == 'spearman'").set_index("model")
    assert all(m.loc[x, x] == 1.0 for x in SCORES) and m.loc["deepseek", "chatgpt"] == m.loc["chatgpt", "deepseek"] == 1.0 and np.isnan(m.loc["mistral", "gemini"])


def test_a_missing_model_is_reported(tmp_path):
    build(tmp_path)
    (tmp_path / "mhs" / "baseline" / "gemini.csv").unlink()
    with pytest.raises(FileNotFoundError, match="gemini"):
        compare_models(tmp_path, datasets=["attitudes", "mhs"])


def test_save_writes_the_tables_the_figure_and_the_readme(tmp_path):
    t = compare_models(build(tmp_path / "runs"), datasets=["attitudes", "mhs"])
    written = save_model_comparison(t, tmp_path / "model_comparison")
    assert set(written) == {"pairwise.csv", "matrices.csv", "model_agreement.pdf", "model_agreement.png", "README.md"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
    assert len(pd.read_csv(tmp_path / "model_comparison" / "pairwise.csv")) == 12 and len(pd.read_csv(tmp_path / "model_comparison" / "matrices.csv")) == 2 * 2 * 4
