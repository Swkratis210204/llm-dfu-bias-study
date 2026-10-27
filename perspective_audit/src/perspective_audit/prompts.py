"""Prompts and the parsers of their answers. The wording is the one used in the thesis notebook.

The label always follows from the score: a score below ``toxic_threshold`` is civil, at or above it is toxic, whatever
label the model wrote. A reply without a valid score counts as unusable.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

__all__ = ["label_from_score", "ScoringPrompt", "PersonaPrompt"]


def label_from_score(score: int, toxic_threshold: float = 3) -> str:
    """1-2 -> civil, 3-5 -> toxic (with the default threshold)."""
    return "toxic" if score >= toxic_threshold else "civil"


class ScoringPrompt:
    """One score and one label for each text of a batch.

    ``version="v1"`` is the prompt of the thesis runs; ``"v2"`` also states the rule "3 or above is toxic".
    ``item`` is the word used for a text ("tweet", "comment", ...).
    """

    def __init__(self, version: str = "v1", item: str = "tweet", scale_max: int = 5, toxic_threshold: float = 3):
        if version not in ("v1", "v2"):
            raise ValueError("version must be 'v1' or 'v2'")
        self.version, self.item, self.scale_max, self.toxic_threshold = version, item, scale_max, toxic_threshold

    def build(self, texts: Sequence[str]) -> str:
        i, k = self.item, self.scale_max
        if self.version == "v1":
            head = (f"\n    Analyze the following {i}s for toxicity.\n"
                    f"    Return ONLY the score (1-{k}) and label (toxic/civil) for each {i} in this format:\n"
                    f"    score=<1-{k}>, label=<toxic/civil>\n\n    {i.capitalize()}s:\n    ")
        else:
            head = (f"I'm giving you some texts for annotation.\n"
                    f"For each text, provide a toxicity score from 1 to {k}.\n"
                    f"If the score is {self.toxic_threshold:g} or above, label it as toxic, otherwise label it as civil.\n"
                    f"Return ONLY the score and label for each text in this format:\n"
                    f"score=<1-{k}>, label=<toxic/civil>\n\nTexts:\n")
        return head + "".join(f"{n}. {t}\n" for n, t in enumerate(texts, start=1))

    def parse(self, raw: str) -> List[Dict]:
        """One ``{"score", "label"}`` per line of the reply that contains a score."""
        results = []
        pattern = rf"score[=:<]\s*([1-{self.scale_max}])"
        for line in (raw or "").splitlines():
            m = re.search(pattern, line.strip(), re.IGNORECASE)
            if m:
                score = int(m.group(1))
                results.append({"score": score, "label": label_from_score(score, self.toxic_threshold)})
        return results


class PersonaPrompt:
    """A label and a score from each persona (group) of one dimension, as JSON.

    ``variant="v1"`` is the longer wording, ``"compact"`` the shorter one used for Mistral and the later models.
    """

    def __init__(self, variant: str = "compact", item: str = "tweet", scale_max: int = 5, toxic_threshold: float = 3):
        if variant not in ("v1", "compact"):
            raise ValueError("variant must be 'v1' or 'compact'")
        self.variant, self.item, self.scale_max, self.toxic_threshold = variant, item, scale_max, toxic_threshold

    def build(self, text: str, dimension: str, personas: Sequence[str]) -> str:
        names = "\n".join(f'- "{p}"' for p in personas)
        item = self.item.capitalize()
        if self.variant == "v1":
            return f"""
You are an annotation system. Output ONLY valid JSON. Do not add explanations.

{item}: "{text}"

Personas in the {dimension} dimension:
{names}

For each persona, independently classify the {self.item} as "toxic" or "civil" and assign a score (1–{self.scale_max}).
Each persona’s judgment must be made without considering the judgments of the other personas.

IMPORTANT:
- Treat each persona as a separate and independent annotation task.
- Do NOT ensure consistency or comparison across personas.
- Use the persona names EXACTLY as shown above.
- JSON keys MUST match these personas exactly.
- Do NOT substitute or rename personas.

Return ONLY valid JSON, for example:
{{
  "{personas[0]}": {{"label": "toxic", "score": 3}},
  "{personas[1]}": {{"label": "civil", "score": 1}}
}}
"""
        example = ",\n  ".join(f'"{p}": {{"label": "civil", "score": 1}}' for p in personas)
        return f"""
You are an annotation system. DO NOT output anything except valid JSON.

{item}: "{text}"

Personas in the {dimension} dimension:
{names}

For each persona, classify the {self.item} as "toxic" or "civil" and assign a score (1–{self.scale_max}).

Return EXACTLY valid JSON like this:
{{
  {example}
}}
"""

    def parse(self, raw: str, personas: Sequence[str]) -> Dict[str, Dict]:
        """``{persona: {"label", "score"}}``. A persona without a valid score gets ``None`` for both."""
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        parsed: dict = {}
        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = {}
        parsed = {str(k).lower(): v for k, v in parsed.items()} if isinstance(parsed, dict) else {}

        if parsed and set(parsed) != set(personas) and len(parsed) == len(personas):
            parsed = {expected: parsed[returned] for returned, expected in zip(parsed, personas)}     # renamed keys

        results = {}
        for p in personas:
            entry, score = parsed.get(p), None
            if isinstance(entry, dict):
                score = entry.get("score")
            if score is None:
                m = re.search(rf'"?{re.escape(p)}"?.*?([1-{self.scale_max}])', raw)
                score = m.group(1) if m else None
            try:
                score = int(float(score))
            except (TypeError, ValueError):
                score = None
            if score is not None and not 1 <= score <= self.scale_max:
                score = None
            results[p] = {"label": label_from_score(score, self.toxic_threshold) if score is not None else None, "score": score}
        return results
