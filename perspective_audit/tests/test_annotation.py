"""Tests of annotate with stand-in models: no key, network or credits."""
import json

import pandas as pd
import pytest

from perspective_audit import Env, Model, ModelConfig, ModelError
from perspective_audit.annotation import AnnotationError, annotate as _annotate, collect_runs, load_model_inputs, progress


class Scorer:
    """A provider that gives every text of the batch the score 4 (and counts the calls); ``fail_on_call`` raises on that call."""

    def __init__(self, fail_on_call=None, reply=None, score=4):
        self.calls, self.fail_on_call, self.reply, self.score = [], fail_on_call, reply, score

    def complete(self, prompt, max_tokens, temperature):
        self.calls.append(prompt)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("api down")
        if self.reply is not None:
            return self.reply
        n = sum(1 for line in prompt.splitlines() if line.strip()[:1].isdigit() and ". " in line)       # the first item is indented in the prompt
        return "\n".join(f"score={self.score}, label=civil" for _ in range(n))          # the label is wrong on purpose


def thesis_prompt(texts, item):
    return f"Analyze the following {item}s for toxicity.\nReturn ONLY the score (1-5) and label (toxic/civil) for each {item}:\n    {texts}"


def annotate(*args, **kwargs):
    kwargs.setdefault("prompt", thesis_prompt)
    return _annotate(*args, **kwargs)


def model(provider, name="stub"):
    m = Model(ModelConfig(name=name, retries=1, model_id="stub-1"), env=Env({}), provider=provider)
    m.sleep = lambda s: None
    return m


def inputs(n_a=25, n_b=7):
    return {"popquorn": pd.DataFrame({"item_id": [str(i) for i in range(n_a)], "text": [f"text {i}" for i in range(n_a)]}),
            "kumar": pd.DataFrame({"item_id": [f"k{i}" for i in range(n_b)], "text": [f"comment {i}" for i in range(n_b)]})}


def test_files_folders_and_rows(tmp_path):
    provider = Scorer()
    summary = annotate(model(provider), inputs(), tmp_path)
    assert summary["texts"].to_dict() == {"popquorn": 25, "kumar": 7} and summary["batches"].to_dict() == {"popquorn": 3, "kumar": 1}
    assert (tmp_path / "popquorn" / "baseline" / "stub.csv").exists() and (tmp_path / "kumar" / "baseline" / "raw" / "stub.raw.jsonl").exists()
    saved = pd.read_csv(tmp_path / "kumar" / "baseline" / "stub.csv", dtype={"item_id": str})
    assert list(saved["item_id"]) == [f"k{i}" for i in range(7)] and set(saved["label"]) == {"toxic"}      # label follows the score (4 -> toxic)
    assert len(collect_runs(tmp_path)) == 32


def test_prompt_uses_the_word_of_the_dataset_and_batches_do_not_mix_datasets(tmp_path):
    provider = Scorer()
    annotate(model(provider), inputs(), tmp_path)
    assert "following texts for toxicity" in provider.calls[0] and "following comments for toxicity" in provider.calls[-1]
    assert not any("comment 0" in c and "text 0" in c for c in provider.calls)
    assert len(provider.calls) == 4


def test_a_user_function_is_the_prompt(tmp_path):
    provider = Scorer()

    def strict_moderator(texts, item):
        return f"Rate each {item} from 1 to 5, one line each as score=<n>, label=<toxic/civil>.\n{texts}"

    annotate(model(provider), inputs(), tmp_path, run="mine", prompt=strict_moderator)
    assert provider.calls[0].startswith("Rate each text from 1 to 5") and provider.calls[-1].startswith("Rate each comment from 1 to 5")
    assert "1. text 0" in provider.calls[0]
    assert set(pd.read_csv(tmp_path / "run_log.csv")["prompt_version"]) == {"strict_moderator"}
    with pytest.raises(TypeError):
        annotate(model(Scorer()), inputs(), tmp_path, run="bad", prompt="v1")      # a name is not a prompt: pass a function


def test_scale_and_threshold_are_numbers_of_annotate(tmp_path):
    annotate(model(Scorer(score=2)), inputs(), tmp_path, run="t2", toxic_threshold=2)
    assert set(pd.read_csv(tmp_path / "kumar" / "t2" / "stub.csv")["label"]) == {"toxic"}


def test_a_stopped_run_resumes_at_the_first_missing_batch(tmp_path):
    first = Scorer(fail_on_call=2)
    with pytest.raises(ModelError):
        annotate(model(first), inputs(), tmp_path)
    assert len(pd.read_csv(tmp_path / "popquorn" / "baseline" / "stub.csv")) == 10                  # batch 1 was kept
    assert pd.read_csv(tmp_path / "run_log.csv")["status"].iloc[-1].startswith("failed")
    second = Scorer()
    annotate(model(second), inputs(), tmp_path)
    assert len(second.calls) == 3                                                                     # 2 popquorn batches + 1 kumar, not 4
    assert len(collect_runs(tmp_path)) == 32 and not pd.read_csv(tmp_path / "popquorn" / "baseline" / "stub.csv")["item_id"].duplicated().any()


def test_a_reply_that_cannot_be_parsed_raises_and_is_kept(tmp_path):
    with pytest.raises(AnnotationError, match="expected 10 scores, received 0"):
        annotate(model(Scorer(reply="I cannot help with that.")), inputs(), tmp_path, parse_retries=1)
    last = json.loads((tmp_path / "popquorn" / "baseline" / "raw" / "stub.raw.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "unparseable" and last["attempts"] == 2 and "cannot help" in last["reply"]


def test_changed_inputs_are_not_mixed_with_a_saved_run(tmp_path):
    annotate(model(Scorer()), inputs(), tmp_path)
    changed = inputs()
    changed["popquorn"].loc[0, "item_id"] = "other"
    with pytest.raises(AnnotationError, match="another run name"):
        annotate(model(Scorer()), changed, tmp_path)


def test_limit_run_name_log_and_progress(tmp_path):
    annotate(model(Scorer(), "a"), inputs(), tmp_path, run="pilot", limit=4)
    annotate(model(Scorer(), "b"), inputs(), tmp_path, run="pilot", limit=4)
    log = pd.read_csv(tmp_path / "run_log.csv")
    assert list(log["model"]) == ["a", "a", "b", "b"] and set(log["run"]) == {"pilot"} and set(log["model_id"]) == {"stub-1"}
    table = progress({k: v.head(4) for k, v in inputs().items()}, tmp_path, ["a", "b", "c"], run="pilot")
    assert table.loc["popquorn", ["a", "b", "c"]].tolist() == [4, 4, 0]


def test_inspect_run_shows_scores_and_retried_batches(tmp_path):
    from perspective_audit.annotation import inspect_run

    class Flaky(Scorer):                                      # the first reply of the first batch has too few scores
        def complete(self, prompt, max_tokens, temperature):
            if not self.calls:
                self.calls.append(prompt)
                return "score=4, label=toxic"
            return super().complete(prompt, max_tokens, temperature)

    annotate(model(Flaky()), inputs(), tmp_path)
    look = inspect_run(tmp_path, "stub")
    assert look.loc["popquorn", "texts"] == 25 and look.loc["kumar", "texts"] == 7
    assert look.loc["popquorn", "score 4"] == 25 and look.loc["popquorn", "toxic %"] == 100 and look.loc["popquorn", "mean score"] == 4.0
    assert look.loc["popquorn", "batches asked again"] == 1 and look.loc["kumar", "batches asked again"] == 0
    assert inspect_run(tmp_path, "nobody").empty


def test_load_model_inputs_keeps_each_text_once(tmp_path):
    pd.DataFrame({"dataset": ["x", "x", "y"], "item_id": ["1", "1", "2"], "text": ["t1", "t1", "t2"]}).to_csv(tmp_path / "m.csv", index=False)
    loaded = load_model_inputs(tmp_path / "m.csv")
    assert list(loaded) == ["x", "y"] and len(loaded["x"]) == 1


def test_each_dataset_folder_has_one_file_per_model_and_one_with_all_of_them(tmp_path):
    from perspective_audit.annotation import ensemble

    annotate(model(Scorer(score=5), "chatgpt"), inputs(), tmp_path)
    annotate(model(Scorer(score=2), "mistral"), inputs(), tmp_path)
    annotate(model(Scorer(score=3), "llama"), inputs(), tmp_path, datasets=["kumar"])            # only one dataset for this model
    folder = tmp_path / "popquorn" / "baseline"
    assert sorted(p.name for p in folder.glob("*.csv")) == ["all_models.csv", "chatgpt.csv", "mistral.csv"]
    assert sorted(p.name for p in (folder / "raw").iterdir()) == ["chatgpt.raw.jsonl", "mistral.raw.jsonl"]
    assert sorted(p.name for p in (tmp_path / "kumar" / "baseline").glob("*.csv")) == ["all_models.csv", "chatgpt.csv", "llama.csv", "mistral.csv"]

    both = pd.read_csv(folder / "all_models.csv", dtype={"item_id": str})
    assert list(both.columns) == ["dataset", "item_id", "text", "chatgpt_score", "chatgpt_label", "mistral_score", "mistral_label"]
    assert len(both) == 25 and set(both["chatgpt_label"]) == {"toxic"} and set(both["mistral_label"]) == {"civil"}
    kumar = pd.read_csv(tmp_path / "kumar" / "baseline" / "all_models.csv")
    assert [c for c in kumar.columns if c.endswith("_score")] == ["chatgpt_score", "llama_score", "mistral_score"]       # fixed order of the models
    assert len(collect_runs(tmp_path)) == 25 * 2 + 7 * 3                                                              # all_models.csv is not counted twice
    assert ensemble(tmp_path, "popquorn", "baseline").shape == both.shape and ensemble(tmp_path, "nothing").empty
