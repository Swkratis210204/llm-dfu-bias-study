"""Tests of the prompt-wording ablation, on small made-up runs: no models, no files of the project."""
import numpy as np
import pandas as pd
import pytest

from perspective_audit import save_ablation_prompt
from perspective_audit.ablation_prompt import RUNS, agreement_by_model, agreement_pooled, load_samples, wording_vs_model

BASE = [1, 1, 1, 1, 1, 5, 5, 5, 5, 5]


def write(results_dir, dataset, prompt, model, scores, ids=None):
    ids = ids or [f"t{i}" for i in range(len(scores))]
    folder = results_dir / dataset / RUNS[prompt]
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dataset": dataset, "item_id": ids, "text": "x", "model": model, "run": RUNS[prompt], "score": scores,
                  "label": ["toxic" if s >= 3 else "civil" for s in scores]}).to_csv(folder / f"{model}.csv", index=False)


def build(tmp_path, datasets=("attitudes", "kumar"), models=("chatgpt", "deepseek")):
    """R1 differs from the baseline by one score that keeps the label (1 -> 2 on the first text); R2 by one score that flips it (1 -> 4)."""
    for dataset in datasets:
        for model in models:
            write(tmp_path, dataset, "baseline", model, BASE)
            write(tmp_path, dataset, "R1", model, [2] + BASE[1:])
            write(tmp_path, dataset, "R2", model, [4] + BASE[1:])
    return tmp_path


def test_agreement_by_model_matches_hand_calculation(tmp_path):
    samples = load_samples(build(tmp_path), datasets=("attitudes", "kumar"), models=("chatgpt", "deepseek"))
    table = agreement_by_model(samples, reps=200).set_index(["dataset", "model", "comparison"])
    r1 = table.loc[("attitudes", "chatgpt", "baseline vs R1")]
    assert r1["same label %"] == 100.0 and r1["Cohen's kappa (labels)"] == 1.0 and r1["same score %"] == 90.0 and 0.9 < r1["weighted kappa (scores)"] < 1.0
    r2 = table.loc[("attitudes", "chatgpt", "baseline vs R2")]
    assert r2["same label %"] == 90.0 and r2["Cohen's kappa (labels)"] == 0.8 and r2["mean diff (second - first)"] == 0.3        # (4 - 1) / 10 texts
    assert r2["same label % low"] <= r2["same label %"] <= r2["same label % high"] and 0 <= r2["same label % low"]
    both = table.loc[("attitudes", "chatgpt", "R1 vs R2")]
    assert both["same label %"] == 90.0 and both["texts"] == 10


def test_pooled_has_one_row_per_dataset_and_comparison_in_the_order_of_the_figures(tmp_path):
    samples = load_samples(build(tmp_path), datasets=("kumar", "attitudes"), models=("chatgpt", "deepseek"))
    pooled = agreement_pooled(samples, reps=200)
    assert list(pooled["dataset"].unique()) == ["attitudes", "kumar"] and len(pooled) == 6                       # attitudes first
    row = pooled[(pooled["dataset"] == "kumar") & (pooled["comparison"] == "baseline vs R2")].iloc[0]
    assert row["texts"] == 10 and row["pairs"] == 20 and row["same label %"] == 90.0                               # two models of ten texts


def test_a_constant_label_gives_an_undefined_kappa_not_an_error(tmp_path):
    for prompt in RUNS:
        write(tmp_path, "mhs", prompt, "chatgpt", [4] * 10)                                                        # every text toxic in every prompt
    samples = load_samples(tmp_path, datasets=("mhs",), models=("chatgpt",))
    row = agreement_by_model(samples, reps=50).iloc[0]
    assert row["same label %"] == 100.0 and np.isnan(row["Cohen's kappa (labels)"])


def test_only_the_texts_answered_everywhere_are_in_the_sample(tmp_path):
    ids = [f"t{i}" for i in range(10)]
    write(tmp_path, "attitudes", "baseline", "chatgpt", BASE, ids)
    write(tmp_path, "attitudes", "R1", "chatgpt", BASE[:9], ids[:9])                                                  # one text missing in R1
    write(tmp_path, "attitudes", "R2", "chatgpt", BASE, ids)
    assert load_samples(tmp_path, datasets=("attitudes",), models=("chatgpt",))["attitudes"]["ids"] == ids[:9]
    assert load_samples(tmp_path, datasets=("attitudes", "kumar"), models=("chatgpt",)).keys() == {"attitudes"}       # kumar has no runs here


def test_wording_against_model(tmp_path):
    samples = load_samples(build(tmp_path, models=("chatgpt", "deepseek", "mistral")), datasets=("attitudes", "kumar"), models=("chatgpt", "deepseek", "mistral"))
    table = wording_vs_model(samples).set_index("dataset")
    assert 0.9 < table.loc["attitudes", "same model, baseline vs R1"] < 1.0 and table.loc["attitudes", "different models, baseline"] == 1.0    # the models agree


def test_save_writes_every_table_figure_and_the_readme(tmp_path):
    pytest.importorskip("matplotlib")
    results_dir = build(tmp_path / "runs", datasets=("attitudes", "popquorn", "mhs", "kumar"), models=("chatgpt", "deepseek", "mistral", "gemini"))
    written = save_ablation_prompt(results_dir, tmp_path / "ablation_prompt", reps=100)
    assert set(written) == {"agreement_pooled.csv", "agreement_by_model.csv", "wording_vs_model.csv", "ablation_sample.csv", "ablation_weighted_kappa.pdf",
                            "ablation_weighted_kappa.png", "ablation_label_agreement.pdf", "ablation_label_agreement.png", "ablation_wording_vs_model.pdf",
                            "ablation_wording_vs_model.png", "README.md"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())
    readme = (tmp_path / "ablation_prompt" / "README.md").read_text(encoding="utf-8")
    assert "Prompt-wording ablation" in readme and "Attitudes 10" in readme
    assert len(pd.read_csv(tmp_path / "ablation_prompt" / "ablation_sample.csv")) == 40
