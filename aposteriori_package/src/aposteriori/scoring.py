"""Run a model over many texts: plain scoring in batches, and persona prompting per dimension.

Both functions fail loudly. A failing API call raises ``ModelError`` (the model already retried); a reply that cannot be
parsed is retried and then raises, instead of leaving an empty row.
"""
from __future__ import annotations

from itertools import islice
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from .models import Model
from .prompts import PersonaPrompt, ScoringPrompt

__all__ = ["score_texts", "score_personas", "batched"]


def batched(items, size: int):
    """Yield lists of ``(index, item)``, ``size`` at a time."""
    it = iter(enumerate(items))
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def score_texts(model: Model, texts: Sequence[str], prompt: Optional[ScoringPrompt] = None, batch_size: int = 10,
                parse_retries: int = 2, on_batch: Optional[Callable[[int, list], None]] = None) -> pd.DataFrame:
    """A score (1-5) and a label for every text. Columns: ``batch``, ``text_index``, ``text``, ``score``, ``label``.

    A batch whose reply has the wrong number of scores is asked again ``parse_retries`` more times, then an error is raised.
    """
    prompt = prompt or ScoringPrompt()
    rows = []
    for number, batch in enumerate(batched(list(texts), batch_size), start=1):
        question = prompt.build([t for _, t in batch])
        last_problem = None
        for _ in range(parse_retries + 1):
            parsed = prompt.parse(model.complete(question))
            if len(parsed) == len(batch):
                break
            last_problem = f"expected {len(batch)} scores, received {len(parsed)}"
        else:
            raise RuntimeError(f"{model.name}: batch {number} could not be parsed after {parse_retries + 1} attempts: {last_problem}")
        for (index, text), result in zip(batch, parsed):
            rows.append({"batch": number, "text_index": index, "text": text, **result})
        if on_batch:
            on_batch(number, batch)
    return pd.DataFrame(rows, columns=["batch", "text_index", "text", "score", "label"])


def score_personas(model: Model, items: pd.DataFrame, persona_map: Dict[str, Sequence[str]],
                   prompt: Optional[PersonaPrompt] = None, max_attempts: int = 3, first_max_tokens: int = 200,
                   fail_on_missing: bool = True) -> pd.DataFrame:
    """Persona prompting. ``items`` has the columns ``text`` and ``dimension``; ``persona_map`` gives the personas of each dimension.

    Returns one row per item and persona: ``row``, ``dimension``, ``persona``, ``label``, ``score``. A reply is accepted only
    when every persona has a valid score; otherwise the call is repeated with a larger token budget (truncation is the usual
    failure). Items still incomplete after ``max_attempts`` raise an error that lists them (after everything else was done).
    """
    prompt = prompt or PersonaPrompt()
    rows, failed = [], []
    for row_id, item in items.iterrows():
        personas = persona_map.get(item["dimension"])
        if not personas:
            continue
        question = prompt.build(item["text"], item["dimension"], personas)
        for attempt in range(1, max_attempts + 1):
            results = prompt.parse(model.complete(question, max_tokens=first_max_tokens * attempt), personas)
            if all(r["score"] is not None for r in results.values()):
                break
        else:
            failed.append((row_id, item["dimension"], str(item["text"])[:70]))
        rows.extend({"row": row_id, "dimension": item["dimension"], "persona": p, **r} for p, r in results.items())
    out = pd.DataFrame(rows, columns=["row", "dimension", "persona", "label", "score"])
    if failed and fail_on_missing:
        listing = "\n".join(f"  row {i} [{d}] {t!r}" for i, d, t in failed)
        raise RuntimeError(f"{model.name}: {len(failed)} item(s) still incomplete after {max_attempts} attempts:\n{listing}")
    return out
