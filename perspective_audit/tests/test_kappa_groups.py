"""Tests of the kappa between the models and the groups (section 7.2), on a small made-up dataset where the answers are known."""
import json

import numpy as np
import pandas as pd
import pytest

from perspective_audit import DatasetSpec, compute_kappa, run_dataset, save_kappa
from perspective_audit.kappa_groups import _kappa

N = 12                                    # texts t0..t5: men civil, women toxic; texts t6..t11: men toxic, women civil


def toy_result():
    rows = []
    for i in range(N):
        men, women = (1, 5) if i < 6 else (5, 1)
        rows += [{"item_id": f"t{i}", "text": f"text {i}", "rating": men, "sex": "m"} for _ in range(2)]
        rows += [{"item_id": f"t{i}", "text": f"text {i}", "rating": women, "sex": "f"} for _ in range(2)]
    raw = pd.DataFrame(rows)
    spec = DatasetSpec(name="attitudes", load=lambda root: raw, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1)
    return run_dataset(spec, raw=raw)


def man_toxic(i):
    return i >= 6


def build(tmp_path, with_deepseek=True):
    """chatgpt: baseline always toxic (constant), scd always right. deepseek: baseline right for women (= toxic on t0-t5), scd wrong on 4 texts as a man."""
    folder = tmp_path / "attitudes"
    (folder / "baseline").mkdir(parents=True)
    (folder / "scd").mkdir(parents=True)
    ids = [f"t{i}" for i in range(N)]
    def persona(model, i, wrong_as_man):
        man = man_toxic(i) != (i in wrong_as_man)
        return json.dumps([{"as": "man", "score": 5 if man else 1, "label": "x"}, {"as": "woman", "score": 1 if man_toxic(i) else 5, "label": "x"}])
    for model, base, wrong in (("chatgpt", [5] * N, []), ("deepseek", [5 if i < 6 else 1 for i in range(N)], [0, 1, 6, 7])):
        if model == "deepseek" and not with_deepseek:
            continue
        pd.DataFrame({"dataset": "attitudes", "item_id": ids, "text": "x", "model": model, "run": "baseline", "score": base,
                      "label": ["toxic" if s >= 3 else "civil" for s in base]}).to_csv(folder / "baseline" / f"{model}.csv", index=False)
        pd.DataFrame({"dataset": "attitudes", "item_id": ids, "dimension": "gender", "text": "x", "model": model, "run": "scd",
                      "personas": [persona(model, i, wrong) for i in range(N)]}).to_csv(folder / "scd" / f"{model}.csv", index=False)
    given = pd.DataFrame({"dataset": "attitudes", "dimension": "gender", "item_id": ids})
    return tmp_path, given


def tables(tmp_path, reps=2000, models=("chatgpt", "deepseek")):
    results_dir, given = build(tmp_path)
    return compute_kappa({"attitudes": toy_result()}, results_dir, given, reps=reps, seed=42, models=list(models))


def test_kappa_of_known_cases(tmp_path):
    by_group = tables(tmp_path)["by_group"].set_index(["model", "group", "prompt"])
    constant = by_group.loc[("chatgpt", "man", "baseline")]
    assert constant["kappa"] == 0.0 and constant["constant_model"] and constant["pct_toxic_model"] == 100.0 and constant["pct_toxic_group"] == 50.0
    assert not constant["constant_group"] and constant["n_texts"] == N and not constant["small_n"]
    assert by_group.loc[("chatgpt", "man", "scd"), "kappa"] == 1.0 and by_group.loc[("chatgpt", "woman", "scd"), "kappa"] == 1.0
    assert by_group.loc[("deepseek", "man", "scd"), "kappa"] == pytest.approx(0.3333, abs=1e-4)             # 4 wrong texts of 12, balanced labels
    assert by_group.loc[("deepseek", "woman", "baseline"), "kappa"] == 1.0 and by_group.loc[("deepseek", "man", "baseline"), "kappa"] == -1.0
    row = by_group.loc[("deepseek", "man", "scd")]
    assert row["ci_low"] <= row["kappa"] <= row["ci_high"] and row["resamples_without_value"] == 0


def test_gap_and_prompt_change_from_the_same_resamples(tmp_path):
    t = tables(tmp_path)
    gap = t["gaps"].set_index(["model", "prompt"]).loc[("chatgpt", "scd")]
    assert gap["gap"] == 0.0 and gap["share_le0"] == 100.0 and (gap["group_a"], gap["group_b"]) == ("man", "woman")
    gap = t["gaps"].set_index(["model", "prompt"]).loc[("deepseek", "scd")]
    assert gap["gap"] == pytest.approx(0.3333 - 1.0, abs=1e-3) and gap["ci_low"] <= gap["gap"] <= gap["ci_high"]
    change = t["prompt_change"].set_index(["model", "group"])
    assert change.loc[("chatgpt", "man"), "change"] == 1.0 and change.loc[("chatgpt", "man"), "share_le0"] < 1.0
    assert change.loc[("deepseek", "woman"), "kappa_baseline"] == 1.0 and change.loc[("deepseek", "woman"), "change"] == 0.0


def test_constant_summary_and_checks(tmp_path):
    t = tables(tmp_path)
    constant = t["constant"].set_index(["dataset", "model", "prompt"])
    assert constant.loc[("attitudes", "chatgpt", "baseline"), "constant_model"] == 2 and constant.loc[("attitudes", "chatgpt", "scd"), "constant_model"] == 0
    assert constant.loc[("all", "all", "all"), "combinations"] == 8 and constant.loc[("all", "all", "all"), "constant_model"] == 2          # 2 models x 2 groups x 2 prompts
    check = t["checks"].iloc[0]
    assert check["attributed rows used (text x characteristic)"] == N and check["distinct texts used"] == N and check["AU rows (all)"] == N


def test_a_group_without_texts_or_a_constant_label_in_the_kappa_function():
    position, label = np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    assert np.isnan(_kappa(np.zeros((1, 3)), position, label)[0])                                    # the group has no text in the resample: no value
    assert _kappa(np.ones((1, 3)), position, np.ones(3))[0] == 0.0                                  # constant model label: 0
    assert _kappa(np.ones((1, 3)), position, label)[0] == pytest.approx(0.4)                        # po = 2/3, model 1/3 toxic, group 2/3: kappa (2/3 - 4/9) / (5/9)


def test_a_missing_model_rating_is_reported_not_guessed(tmp_path):
    results_dir, given = build(tmp_path, with_deepseek=False)
    with pytest.raises(FileNotFoundError):
        compute_kappa({"attitudes": toy_result()}, results_dir, given, reps=50, models=["chatgpt", "deepseek"])
    broken = pd.read_csv(results_dir / "attitudes" / "baseline" / "chatgpt.csv")
    broken.iloc[:-1].to_csv(results_dir / "attitudes" / "baseline" / "chatgpt.csv", index=False)       # the rating of the last text is gone
    with pytest.raises(ValueError, match="missing"):
        compute_kappa({"attitudes": toy_result()}, results_dir, given, reps=50, models=["chatgpt"])


def test_save_writes_the_four_csv_files_and_the_readme(tmp_path):
    t = tables(tmp_path / "runs", reps=100)
    written = save_kappa(t, tmp_path / "kappa")
    assert set(written) == {"kappa_by_group.csv", "kappa_gaps.csv", "kappa_prompt_change.csv", "constant_label_summary.csv", "README.md"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
    assert len(pd.read_csv(tmp_path / "kappa" / "kappa_by_group.csv")) == 8


def test_latex_tables_one_per_dataset(tmp_path):
    from perspective_audit import save_kappa_tables
    t = tables(tmp_path / "runs", reps=200)
    assert "se" in t["by_group"].columns and (t["by_group"]["se"] >= 0).all()
    written = save_kappa_tables(t["by_group"], tmp_path / "tables")
    text = written["attitudes"].read_text(encoding="utf-8")
    assert set(written) == {"attitudes"} and r"\label{tab:kappa_attitudes}" in text and r"\toprule" in text and "Toxic labels" in text
    assert text.count("\makecell") == 8 and "$^\dagger$" in text and "{[}" in text and "(D)" in text


def test_shares_excluded_zero_and_mirrored(tmp_path):
    t = tables(tmp_path, reps=500)
    for name in ("gaps", "prompt_change"):
        table = t[name]
        assert "share_gap_le0" not in table and "share_change_le0" not in table
        assert (table["share_lt0"] + table["share_eq0"] - table["share_le0"]).abs().max() < 0.02
        assert (table[["share_lt0", "share_eq0", "share_gt0"]].sum(axis=1) - 100).abs().max() < 0.03
        assert (table["interval_excludes_zero"] == ((table["ci_low"] > 0) | (table["ci_high"] < 0))).all()
    assert t["share_problems"].empty
    assert t["by_group"]["mirrored"].all() and list(t["mirrored"]["characteristic"]) == ["gender"]      # men and women take opposite sides on every text
