"""Tests of env, models, prompts and scoring. A stub provider replaces every API: no key, network or credits."""
import pandas as pd
import pytest

from aposteriori import (Env, Model, ModelConfig, ModelError, ModelHub, PersonaPrompt, ScoringPrompt, label_from_score,
                         score_personas, score_texts)
from aposteriori.env import parse_dotenv


class Stub:
    """A provider that returns the given replies in turn (the last one repeats) and records the calls."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, prompt, max_tokens, temperature):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


def model(*replies, **config):
    m = Model(ModelConfig(name="stub", **config), env=Env({}), provider=Stub(*replies))
    m.sleep = lambda s: None
    return m


# ---- env --------------------------------------------------------------------------------------------------
def test_dotenv_format(tmp_path):
    (tmp_path / "secret.txt").write_text("from a file\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "# comment\nchat_api=abc\nmistral_key = 'quoted'\nfile_key=@secret.txt\n"
        "OCI_KEY=-----BEGIN-----\nline two\n-----END-----\n\nnext=1\n", encoding="utf-8")
    values = parse_dotenv(tmp_path / ".env")
    assert values["chat_api"] == "abc" and values["mistral_key"] == "quoted" and values["file_key"] == "from a file"
    assert values["OCI_KEY"] == "-----BEGIN-----\nline two\n-----END-----" and values["next"] == "1"


def test_env_prefers_the_real_environment(tmp_path, monkeypatch):
    # names that cannot exist in a real environment, so the test does not depend on the keys of the machine it runs on
    (tmp_path / ".env").write_text("zz_test_key_a=from_file\nzz_test_key_b=from_file\n", encoding="utf-8")
    monkeypatch.delenv("zz_test_key_b", raising=False)
    monkeypatch.setenv("zz_test_key_a", "from_environment")
    env = Env.load(tmp_path / ".env")
    assert env.get("zz_test_key_a") == "from_environment" and env.get("zz_test_key_b") == "from_file"
    with pytest.raises(KeyError):
        env.require("does_not_exist_anywhere")


# ---- models -----------------------------------------------------------------------------------------------
def test_complete_returns_the_reply_and_uses_the_config():
    m = model("hello", temperature=0.3, max_tokens=77)
    assert m.complete("hi") == "hello"
    assert m.provider.calls == [{"prompt": "hi", "max_tokens": 77, "temperature": 0.3}]


def test_failing_calls_are_retried_then_raise_model_error():
    ok = model(RuntimeError("boom"), "fine", retries=3)
    assert ok.complete("x") == "fine" and len(ok.provider.calls) == 2
    bad = model(RuntimeError("boom"), retries=2)
    with pytest.raises(ModelError, match="stub: call failed after 2 attempts"):
        bad.complete("x")


def test_reply_without_the_pattern_is_asked_again_at_the_next_temperature():
    m = model("I can't help with that.", "score=3, label=toxic", retry_temperatures=(0.0, 0.9), reply_pattern=r"score\s*=")
    assert m.complete("x") == "score=3, label=toxic"
    assert [c["temperature"] for c in m.provider.calls] == [0.0, 0.9]


def test_availability_depends_on_the_key_only():
    needs_key = Model(ModelConfig("a", api_key_env="some_key"), env=Env({}))
    assert not needs_key.available()
    assert Model(ModelConfig("a", api_key_env="some_key"), env=Env({"some_key": "x"})).available()
    assert Model(ModelConfig("llama", api_key_env=None), env=Env({})).available()


def test_hub_from_env_selects_and_overrides():
    env = Env({"chat_api": "k", "MISTRAL_MODEL": "mistral-large", "MISTRAL_MAX_TOKENS": "300", "LLAMA_BASE_URL": "http://other:1/v1"})
    hub = ModelHub.from_env(env=env)
    assert hub.names == ["chatgpt", "llama", "gemini", "mistral", "deepseek"]            # grok is off by default
    assert hub["mistral"].config.model_id == "mistral-large" and hub["mistral"].config.max_tokens == 300
    assert hub["llama"].config.base_url == "http://other:1/v1"
    assert hub.available() == ["chatgpt", "llama"] and "gemini" in hub.missing() and "gemini_key" in hub.missing()["gemini"]
    assert ModelHub.from_env(env=Env({"APOSTERIORI_MODELS": "llama, grok"})).names == ["llama", "grok"]
    with pytest.raises(KeyError):
        ModelHub.from_env(["nope"], env=env)


def test_hub_asks_every_model():
    hub = ModelHub([model("a"), Model(ModelConfig("other"), env=Env({}), provider=Stub("b"))])
    assert hub.complete_all("q") == {"stub": "a", "other": "b"}


# ---- prompts ----------------------------------------------------------------------------------------------
def test_label_always_follows_the_score():
    assert [label_from_score(s) for s in (1, 2, 3, 4, 5)] == ["civil", "civil", "toxic", "toxic", "toxic"]
    parsed = ScoringPrompt().parse("score=3, label=civil\nscore=2, label=toxic\nnoise\nscore=5, label=civil")
    assert parsed == [{"score": 3, "label": "toxic"}, {"score": 2, "label": "civil"}, {"score": 5, "label": "toxic"}]


def test_scoring_prompt_wording():
    assert ScoringPrompt().build(["a", "b"]).endswith("1. a\n2. b\n")
    assert "3 or above, label it as toxic" in ScoringPrompt("v2").build(["a"])
    assert "comments" in ScoringPrompt(item="comment").build(["a"])


def test_persona_parse_handles_fences_truncation_and_bad_scores():
    p = PersonaPrompt()
    good = p.parse('```json\n{"man": {"label": "civil", "score": 3}, "woman": {"label": "toxic", "score": 2}}\n```', ["man", "woman"])
    assert good == {"man": {"label": "toxic", "score": 3}, "woman": {"label": "civil", "score": 2}}
    cut = p.parse('{"man": {"label": "toxic", "score": 4}, "woman": {"label": "civ', ["man", "woman"])
    assert cut["man"] == {"label": "toxic", "score": 4} and cut["woman"] == {"label": None, "score": None}
    assert p.parse('{"man": {"score": 9}}', ["man", "woman"]) == {"man": {"label": None, "score": None}, "woman": {"label": None, "score": None}}
    assert p.parse('{"a": {"score": 5}, "b": {"score": 1}}', ["man", "woman"])["woman"]["label"] == "civil"      # renamed keys


# ---- scoring ----------------------------------------------------------------------------------------------
def test_score_texts_batches_and_asks_a_bad_batch_again():
    good = "score=5, label=toxic" + chr(10) + "score=1, label=civil"
    bad = "score=2, label=civil"                                          # only one score for a batch of two
    m = model(good, bad, "score=2, label=civil" + chr(10) + "score=4, label=toxic")
    out = score_texts(m, ["t0", "t1", "t2", "t3"], batch_size=2)
    assert len(m.provider.calls) == 3                                   # the second batch needed two calls
    assert list(out.columns) == ["batch", "text_index", "text", "score", "label"]
    assert out[["batch", "text_index", "score", "label"]].values.tolist() == [
        [1, 0, 5, "toxic"], [1, 1, 1, "civil"], [2, 2, 2, "civil"], [2, 3, 4, "toxic"]]


def test_score_texts_raises_when_a_reply_cannot_be_parsed():
    with pytest.raises(RuntimeError, match="could not be parsed"):
        score_texts(model("no scores here"), ["a", "b"], batch_size=2, parse_retries=1)


def test_score_personas_retries_with_more_tokens_and_fails_loudly():
    items = pd.DataFrame({"text": ["x"], "dimension": ["gender"]})
    truncated = '{"man": {"label": "toxic", "score": 4}, "woman": {"label": "civ'
    full = '{"man": {"label": "toxic", "score": 4}, "woman": {"label": "civil", "score": 2}}'
    m = model(truncated, full)
    out = score_personas(m, items, {"gender": ["man", "woman"]})
    assert [c["max_tokens"] for c in m.provider.calls] == [200, 400]
    assert out[["persona", "label", "score"]].values.tolist() == [["man", "toxic", 4], ["woman", "civil", 2]]
    with pytest.raises(RuntimeError, match="still incomplete after 3 attempts"):
        score_personas(model("nothing useful"), items, {"gender": ["man", "woman"]})
