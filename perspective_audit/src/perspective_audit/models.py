"""Language models behind one interface.

``ModelConfig`` holds everything needed to call one model (provider, model id, name of the environment variable that
holds its key, URL, temperature, token limit, retry rules). ``Model`` calls it: ``model.complete(prompt) -> str``.
``ModelHub`` holds several models and builds them from the environment.

Clients are created on first use, so building a hub needs no key, no network and no provider package.
"""
from __future__ import annotations

import configparser
import random
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from .env import Env

__all__ = ["ModelConfig", "Model", "ModelHub", "ModelError", "DEFAULT_CONFIGS", "PROVIDERS"]


class ModelError(RuntimeError):
    """A model call failed after all retries."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str = "openai_compatible"        # "openai_compatible", "gemini" or "oci"
    model_id: str = ""
    api_key_env: Optional[str] = None          # NAME of the variable holding the key (None: no key needed)
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256
    retries: int = 3                           # attempts when the call raises
    retry_temperatures: Optional[tuple] = None  # try these temperatures in turn until the reply matches reply_pattern
    reply_pattern: Optional[str] = None
    request_extra: dict = field(default_factory=dict)   # provider specific, e.g. {"thinking_budget": 0}
    enabled: bool = True                       # part of the default hub


# ---- providers ---------------------------------------------------------------------------------------------
class Provider:
    def __init__(self, config: ModelConfig, env: Env):
        self.config, self.env = config, env

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatible(Provider):
    """OpenAI, Mistral, DeepSeek, Ollama, Gemini's OpenAI endpoint, ... (anything with chat.completions)."""

    def __init__(self, config, env):
        super().__init__(config, env)
        from openai import OpenAI

        key = env.require(config.api_key_env) if config.api_key_env else "not-needed"
        self.client = OpenAI(api_key=key, base_url=config.base_url)

    def complete(self, prompt, max_tokens, temperature):
        resp = self.client.chat.completions.create(
            model=self.config.model_id, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, stream=False, **self.config.request_extra)
        return (resp.choices[0].message.content or "").strip()


class Gemini(Provider):
    """Google's own client. ``request_extra={"thinking_budget": 0}`` stops 2.5 models from spending the token limit thinking."""

    def __init__(self, config, env):
        super().__init__(config, env)
        from google import genai

        self.client = genai.Client(api_key=env.require(config.api_key_env))

    def complete(self, prompt, max_tokens, temperature):
        settings = {"temperature": temperature, "max_output_tokens": max_tokens}
        if "thinking_budget" in self.config.request_extra:
            settings["thinking_config"] = {"thinking_budget": self.config.request_extra["thinking_budget"]}
        resp = self.client.models.generate_content(model=self.config.model_id, contents=prompt, config=settings)
        return (resp.text or "").strip()


class OCI(Provider):
    """Oracle OCI generative AI (Grok). Needs OCI_KEY, OCI_CONFIG, COMPARTMENT_ID and MODEL_ID. Not tested against the service."""

    ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

    def __init__(self, config, env):
        super().__init__(config, env)
        import oci

        self.oci = oci
        parser = configparser.ConfigParser()
        parser.read_string(env.require("OCI_CONFIG"))
        settings = dict(parser["DEFAULT"]) if "DEFAULT" in parser else dict(parser[parser.sections()[0]])
        settings.pop("key_file", None)
        settings["key_content"] = env.require("OCI_KEY")          # the key is passed directly, nothing is written to ~/.oci
        oci.config.validate_config(settings)
        self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=settings, service_endpoint=config.request_extra.get("endpoint", self.ENDPOINT),
            retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10, 240))
        self.compartment_id = env.require("COMPARTMENT_ID")
        self.model_id = config.model_id or env.require("MODEL_ID")

    def complete(self, prompt, max_tokens, temperature):
        m = self.oci.generative_ai_inference.models
        message = m.Message(role="USER", content=[m.TextContent(text=prompt)])
        request = m.GenericChatRequest(api_format=m.BaseChatRequest.API_FORMAT_GENERIC, messages=[message],
                                       max_tokens=max_tokens, temperature=temperature, top_p=1, top_k=0)
        detail = m.ChatDetails(serving_mode=m.OnDemandServingMode(model_id=self.model_id), chat_request=request,
                               compartment_id=self.compartment_id)
        resp = self.client.chat(detail)
        return resp.data.chat_response.choices[0].message.content[0].text.strip()


PROVIDERS: Dict[str, type] = {"openai_compatible": OpenAICompatible, "gemini": Gemini, "oci": OCI}


# ---- the model ---------------------------------------------------------------------------------------------
class Model:
    """One model: ``complete(prompt)`` with retries. ``provider`` can be passed in (tests, custom backends)."""

    sleep: Callable[[float], None] = staticmethod(time.sleep)

    def __init__(self, config: ModelConfig, env: Optional[Env] = None, provider: Optional[Provider] = None):
        self.config = config
        self.env = env if env is not None else Env.load()
        self._provider = provider

    @property
    def name(self) -> str:
        return self.config.name

    def available(self) -> bool:
        """True when everything the model needs from the environment is set (the model is not called)."""
        if self._provider is not None:
            return True
        if self.config.provider == "oci":
            return all(self.env.has(k) for k in ("OCI_KEY", "OCI_CONFIG", "COMPARTMENT_ID"))
        return self.config.api_key_env is None or self.env.has(self.config.api_key_env)

    @property
    def provider(self) -> Provider:
        if self._provider is None:
            if self.config.provider not in PROVIDERS:
                raise ValueError(f"unknown provider {self.config.provider!r}; known: {sorted(PROVIDERS)}")
            self._provider = PROVIDERS[self.config.provider](self.config, self.env)
        return self._provider

    def _one_call(self, prompt: str, max_tokens: int, temperature: Optional[float]) -> str:
        temperatures = self.config.retry_temperatures or (self.config.temperature if temperature is None else temperature,)
        text = ""
        for t in temperatures:
            text = self.provider.complete(prompt, max_tokens, t)
            if not self.config.reply_pattern or re.search(self.config.reply_pattern, text, re.IGNORECASE):
                return text
        return text                              # no reply matched: the last one is returned, the caller decides

    def complete(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        max_tokens = max_tokens or self.config.max_tokens
        for attempt in range(self.config.retries):
            try:
                return self._one_call(prompt, max_tokens, temperature)
            except Exception as exc:
                if attempt == self.config.retries - 1:
                    raise ModelError(f"{self.name}: call failed after {self.config.retries} attempts: {exc!r}") from exc
                limited = any(k in str(exc).lower() for k in ("429", "rate limit", "resource_exhausted", "503", "overloaded"))
                self.sleep(min(60, 10 + 5 * attempt) if limited else 2 ** attempt + random.random())
        raise AssertionError("unreachable")    # pragma: no cover

    def __repr__(self) -> str:
        return f"Model({self.name!r}, {self.config.provider}, {self.config.model_id!r}, available={self.available()})"


# ---- the models used in the thesis -------------------------------------------------------------------------
DEFAULT_CONFIGS: Dict[str, ModelConfig] = {
    "chatgpt": ModelConfig("chatgpt", "openai_compatible", "gpt-4o-mini", api_key_env="chat_api"),
    "llama": ModelConfig("llama", "openai_compatible", "llama3:8b", api_key_env=None, base_url="http://localhost:11434/v1",
                         # Llama 3 sometimes refuses a batch of offensive texts; a refusal repeats at temperature 0,
                         # so a reply without a score or label is retried at a higher temperature.
                         retry_temperatures=(0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9), reply_pattern=r'(score|label)\s*[=:"]'),
    "gemini": ModelConfig("gemini", "gemini", "gemini-2.5-flash", api_key_env="gemini_key", max_tokens=512,
                          request_extra={"thinking_budget": 0}),
    "mistral": ModelConfig("mistral", "openai_compatible", "open-mistral-7b", api_key_env="mistral_key",
                           base_url="https://api.mistral.ai/v1"),
    "deepseek": ModelConfig("deepseek", "openai_compatible", "deepseek-chat", api_key_env="deepseek_key",
                            base_url="https://api.deepseek.com"),
    "grok": ModelConfig("grok", "oci", "", max_tokens=512, enabled=False),     # off while there is no Grok quota
}


def apply_env_overrides(config: ModelConfig, env: Env) -> ModelConfig:
    """``<NAME>_MODEL``, ``<NAME>_BASE_URL``, ``<NAME>_API_KEY_ENV``, ``<NAME>_TEMPERATURE``, ``<NAME>_MAX_TOKENS`` (NAME in capitals)."""
    p = config.name.upper()
    changes = {}
    if env.has(f"{p}_MODEL"):
        changes["model_id"] = env.get(f"{p}_MODEL")
    if env.has(f"{p}_BASE_URL"):
        changes["base_url"] = env.get(f"{p}_BASE_URL")
    if env.has(f"{p}_API_KEY_ENV"):
        changes["api_key_env"] = env.get(f"{p}_API_KEY_ENV")
    if env.has(f"{p}_TEMPERATURE"):
        changes["temperature"] = float(env.get(f"{p}_TEMPERATURE"))
    if env.has(f"{p}_MAX_TOKENS"):
        changes["max_tokens"] = int(env.get(f"{p}_MAX_TOKENS"))
    return replace(config, **changes) if changes else config


# ---- all the models ----------------------------------------------------------------------------------------
class ModelHub:
    """Several models by name. ``ModelHub.from_env()`` builds the default ones; ``PERSPECTIVE_AUDIT_MODELS=chatgpt,llama`` selects."""

    def __init__(self, models: Iterable[Model]):
        self._models: Dict[str, Model] = {m.name: m for m in models}

    @classmethod
    def from_env(cls, names: Optional[Sequence[str]] = None, env: Optional[Env] = None,
                 configs: Optional[Dict[str, ModelConfig]] = None) -> "ModelHub":
        env = env if env is not None else Env.load()
        configs = configs or DEFAULT_CONFIGS
        if names is None:
            chosen = env.get("PERSPECTIVE_AUDIT_MODELS")
            names = [n.strip() for n in chosen.split(",") if n.strip()] if chosen else [n for n, c in configs.items() if c.enabled]
        unknown = [n for n in names if n not in configs]
        if unknown:
            raise KeyError(f"unknown model(s) {unknown}; known: {sorted(configs)}")
        return cls(Model(apply_env_overrides(configs[n], env), env) for n in names)

    @property
    def names(self) -> List[str]:
        return list(self._models)

    def __getitem__(self, name: str) -> Model:
        return self._models[name]

    def __iter__(self) -> Iterator[Model]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    def available(self) -> List[str]:
        return [m.name for m in self if m.available()]

    def missing(self) -> Dict[str, str]:
        """Models that cannot be called yet, and why."""
        out = {}
        for m in self:
            if not m.available():
                out[m.name] = ("OCI_KEY / OCI_CONFIG / COMPARTMENT_ID" if m.config.provider == "oci" else m.config.api_key_env) + " is not set"
        return out

    def complete_all(self, prompt: str, names: Optional[Sequence[str]] = None, **kwargs) -> Dict[str, str]:
        """The same prompt to every (or the named) model, one after the other."""
        return {n: self[n].complete(prompt, **kwargs) for n in (names or self.names)}
