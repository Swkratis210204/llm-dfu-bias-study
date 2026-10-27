"""Tests of annotate_personas with stand-in models: no key, network or credits."""
import json

import pandas as pd
import pytest

from perspective_audit import Env, Model, ModelConfig
from perspective_audit.annotation import AnnotationError
from perspective_audit.persona_annotation import (annotate_personas, collect_persona_runs, group_order, inspect_personas,
                                                  load_persona_inputs, persona_long)


class Raters:
    """A provider that answers as a JSON with a score for every group named in the prompt (man 2, everyone else 4), and counts the calls."""

    def __init__(self, replies=None):
        self.calls, self.replies = [], list(replies or [])

    def complete(self, prompt, max_tokens, temperature):
        self.calls.append((prompt, max_tokens))
        if self.replies:
            return self.replies.pop(0)
        groups = [line[3:-1] for line in prompt.splitlines() if line.startswith('- "')]
        return json.dumps({g: {"score": 2 if g.lower() == "man" else 4} for g in groups})


def prompt(text, item, dimension, groups):
    return f"Rate the {item} as each {dimension} group:\n" + "\n".join(f'- "{g}"' for g in groups) + f"\n{text}"


def model(provider):
    m = Model(ModelConfig(name="stub", retries=1, model_id="stub-1"), env=Env({}), provider=provider)
    m.sleep = lambda s: None
    return m


def inputs():
    return {"popquorn": pd.DataFrame({"item_id": ["1", "1", "2"], "text": ["a", "a", "b"], "dimension": ["gender", "age", "gender"],
                                      "groups": [["Man", "Woman"], ["Gen Z", "Millennial", "Boomer"], ["Man", "Woman"]]})}


def test_one_call_per_text_and_dimension_and_the_saved_form(tmp_path):
    provider = Raters()
    summary = annotate_personas(model(provider), inputs(), tmp_path, prompt, show_progress=False)
    assert len(provider.calls) == 3 and summary.loc["popquorn", "model calls"] == 3
    saved = pd.read_csv(tmp_path / "popquorn" / "scd" / "stub.csv", dtype={"item_id": str})
    assert list(saved.columns) == ["dataset", "item_id", "dimension", "text", "model", "run", "personas"] and len(saved) == 3
    long = persona_long(saved)
    assert len(long) == 2 + 3 + 2 and set(long["run"]) == {"scd"}
    man = long[long["group"] == "Man"]
    assert set(man["score"]) == {2} and set(man["label"]) == {"civil"}          # the label follows the score
    assert set(long[long["group"] == "Woman"]["label"]) == {"toxic"}
    assert sorted(long[(long["item_id"] == "1") & (long["dimension"] == "gender")]["position"]) == [1, 2]
    assert len(collect_persona_runs(tmp_path)) == 7
    assert inspect_personas(tmp_path, "stub").loc["popquorn", "texts"] == 3


def test_the_order_is_seeded_and_saved(tmp_path):
    assert group_order("d", "1", "gender", ["Man", "Woman"], 1) == group_order("d", "1", "gender", ["Woman", "Man"], 1)
    orders = {tuple(group_order("d", str(i), "gender", ["Man", "Woman"], 1)) for i in range(40)}
    assert orders == {("Man", "Woman"), ("Woman", "Man")}                        # both orders occur
    annotate_personas(model(Raters()), inputs(), tmp_path, prompt, show_progress=False)
    raw = [json.loads(line) for line in (tmp_path / "popquorn" / "scd" / "raw" / "stub.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert raw[0]["order"] == group_order("popquorn", "1", "gender", ["Man", "Woman"], 42)


def test_resume_asks_only_what_is_missing_and_another_seed_is_refused(tmp_path):
    first = Raters()
    annotate_personas(model(first), inputs(), tmp_path, prompt, limit=2, show_progress=False)
    second = Raters()
    summary = annotate_personas(model(second), inputs(), tmp_path, prompt, show_progress=False)
    assert len(first.calls) == 2 and len(second.calls) == 1 and summary.loc["popquorn", "resumed"] == 2
    other_seed = next(s for s in range(100) if group_order("popquorn", "1", "gender", ["Man", "Woman"], s) != group_order("popquorn", "1", "gender", ["Man", "Woman"], 42))
    with pytest.raises(AnnotationError):
        annotate_personas(model(Raters()), inputs(), tmp_path, prompt, seed=other_seed, show_progress=False)


def test_an_incomplete_reply_is_asked_again_with_more_tokens_and_then_fails_loudly(tmp_path):
    provider = Raters(replies=['{"man": {"score": 2}}', '{"Man": {"score": 2}, "Woman": {"score": 5}}'])
    annotate_personas(model(provider), inputs(), tmp_path, prompt, limit=1, show_progress=False)
    assert [tokens for _, tokens in provider.calls] == [200, 400]
    long = collect_persona_runs(tmp_path)
    assert dict(zip(long["group"], long["score"])) == {"Man": 2, "Woman": 5}
    with pytest.raises(AnnotationError):
        annotate_personas(model(Raters(replies=["nothing"] * 3)), inputs(), tmp_path, prompt, run="other", limit=1, show_progress=False)
    raw = json.loads((tmp_path / "popquorn" / "other" / "raw" / "stub.raw.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert raw["status"] == "incomplete" and raw["reply"] == "nothing"


def test_load_persona_inputs(tmp_path):
    path = tmp_path / "in.csv"
    pd.DataFrame({"dataset": ["popquorn", "popquorn", "kumar"], "item_id": ["1", "1", "9_5"], "text": ["a", "a", "k"],
                  "dimension": ["gender", "age", "gender"], "kept_groups": ['["Man", "Woman"]', '["Gen Z", "Boomer"]', '["Male", "Female"]']}).to_csv(path, index=False)
    loaded = load_persona_inputs(path)
    assert list(loaded) == ["popquorn", "kumar"] and len(loaded["popquorn"]) == 2          # the same text under two dimensions stays two rows
    assert loaded["kumar"]["groups"].iloc[0] == ["Male", "Female"] and loaded["kumar"]["item_id"].iloc[0] == "9_5"
