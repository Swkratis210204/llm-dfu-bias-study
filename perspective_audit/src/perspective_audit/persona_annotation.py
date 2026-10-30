"""Run ONE model over the texts as a person of each group (sociodemographic prompting, "SCD") and save every reply.

A text comes with the groups of one dimension (e.g. gender: man, woman). ``annotate_personas(model, inputs, out_dir, prompt)`` asks
the model ONCE per (text, dimension), to rate the text as each of those groups, and keeps the answer together. The groups are
listed in a random order that is fixed by ``seed`` (so the same text always gets the same order, and the order is saved). Files,
in ``out_dir/<dataset>/<run>/`` (``run`` is the condition, e.g. ``scd``; a group is a column, not a folder)::

    <model>.csv             one row per (text, dimension): dataset, item_id, dimension, text, model, run, personas
    raw/<model>.raw.jsonl   one line per call: the prompt, the order of the groups, the reply, the parsed answer (the source of truth)

``personas`` is a JSON list in the order the groups were shown, e.g.
``[{"as": "woman", "score": 4, "label": "toxic"}, {"as": "man", "score": 2, "label": "civil"}]``. ``persona_long`` turns it into one row
per (text, dimension, group), the form to join with the baseline and with the human group labels. The label always follows from the
score. Nothing from the human annotators is written here: the files hold the model's answers only.

If a run stops, everything finished is kept and calling ``annotate_personas`` again carries on at the first missing text.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .annotation import AnnotationError, _log, _paths
from .datasets import SPECS
from .models import Model
from .prompts import PersonaPrompt

__all__ = ["load_persona_inputs", "annotate_personas", "persona_long", "collect_persona_runs", "inspect_personas", "group_order"]

SAVED_COLUMNS = ["dataset", "item_id", "dimension", "text", "model", "run", "personas"]
LONG_COLUMNS = ["dataset", "item_id", "dimension", "text", "model", "run", "group", "position", "score", "label"]


def load_persona_inputs(path: Path) -> Dict[str, pd.DataFrame]:
    """The texts to annotate, per dataset: one row per (text, dimension) with the groups of that dimension
    (``item_id``, ``text``, ``dimension``, ``groups``), in the order of the file."""
    table = pd.read_csv(path, dtype={"item_id": str}).drop_duplicates(["dataset", "item_id", "dimension"])
    table = table.assign(groups=table["kept_groups"].map(json.loads))
    return {dataset: part[["item_id", "text", "dimension", "groups"]].reset_index(drop=True)
            for dataset, part in table.groupby("dataset", sort=False)}


def group_order(dataset: str, item_id: str, dimension: str, groups: Sequence[str], seed: int = 42) -> List[str]:
    """The groups of a text in a random order that only depends on ``seed`` and the text (not on the order of the input)."""
    order = sorted(groups)
    random.Random(f"{seed}|{dataset}|{item_id}|{dimension}").shuffle(order)
    return order


def _parse(reply: str, groups: Sequence[str], scale_max: int, toxic_threshold: float) -> List[dict]:
    """``[{"as", "score", "label"}]`` in the order of ``groups``; a group without a valid score has ``None`` for both."""
    lowered = [g.lower() for g in groups]
    if len(set(lowered)) != len(lowered):
        raise ValueError(f"the groups {list(groups)} are the same once the case is ignored")
    found = PersonaPrompt(scale_max=scale_max, toxic_threshold=toxic_threshold).parse(reply, lowered)
    return [{"as": g, "score": found[g.lower()]["score"], "label": found[g.lower()]["label"]} for g in groups]


def _read_calls(raw_path: Path) -> Dict[tuple, dict]:
    done: Dict[tuple, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] == "ok":
                done[(record["item_id"], record["dimension"])] = record
    return done


def _write_csv(dataset: str, run: str, model: str, done: Dict[tuple, dict], csv_path: Path) -> None:
    """The csv is rebuilt from the saved calls only (each keeps its own text), never from the current input."""
    rows = [{"dataset": dataset, "item_id": r["item_id"], "dimension": r["dimension"], "text": r["text"], "model": model, "run": run,
             "personas": json.dumps(r["personas"], ensure_ascii=False)} for r in done.values()]
    pd.DataFrame(rows, columns=SAVED_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8")


def annotate_personas(model: Model, inputs: Dict[str, pd.DataFrame], out_dir: Path, prompt: Callable[[str, str, str, List[str]], str],
                      run: str = "scd", seed: int = 42, limit: Optional[int] = None, datasets: Optional[Iterable[str]] = None,
                      item_words: Optional[Dict[str, str]] = None, max_attempts: int = 3, first_max_tokens: int = 200,
                      scale_max: int = 5, toxic_threshold: float = 3, show_progress: bool = True) -> pd.DataFrame:
    """Rate every (text, dimension) of every dataset as each of its groups with one model, one call each. Returns one row per dataset.

    ``prompt`` is your function ``prompt(text, item, dimension, groups) -> str``: ``item`` is the word for a text (e.g. "comment"),
    ``dimension`` the dimension (e.g. "gender") and ``groups`` the groups in the order they must be shown. The reply must be JSON with
    one key per group, written exactly as given, and a score 1-5 for each, e.g. ``{"woman": {"score": 4}, "man": {"score": 2}}``
    (a label in the reply is ignored: the label follows from the score). A reply without a valid score for every group is asked again
    (with a larger token budget, as truncation is the usual failure), ``max_attempts`` times in all, and then raises ``AnnotationError``
    (the reply is kept in the raw file). ``limit`` keeps only the first rows of each dataset (a pilot). Use another ``run`` name when
    you change the prompt or the seed.
    """
    if not callable(prompt):
        raise TypeError("prompt must be a function prompt(text, item, dimension, groups) -> str")
    words = {name: spec.item_word for name, spec in SPECS.items()}
    words.update(item_words or {})
    summary = []
    for dataset in (list(datasets) if datasets is not None else list(inputs)):
        table = inputs[dataset].head(limit) if limit else inputs[dataset]
        table = table.reset_index(drop=True)
        item = words.get(dataset, "text")
        folder, raw_path, csv_path = _paths(out_dir, dataset, run, model.name)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        done = _read_calls(raw_path)
        started, calls, resumed, status = time.time(), 0, 0, "complete"
        bar = tqdm(total=len(table), desc=f"{dataset} / {model.name}", unit="text", disable=not show_progress)
        try:
            for row in table.itertuples(index=False):
                key = (row.item_id, row.dimension)
                order = group_order(dataset, row.item_id, row.dimension, row.groups, seed)
                if key in done:
                    if done[key]["order"] != order:
                        raise AnnotationError(f"{model.name}/{dataset}: the saved groups of {row.item_id} [{row.dimension}] are in another order "
                                              f"(another seed or other groups); use another run name")
                    resumed += 1
                    bar.update(1)
                    continue
                bar.set_postfix_str(f"{row.dimension}: {str(row.text).replace(chr(10), ' ')[:40]}...", refresh=True)
                question = prompt(str(row.text), item, row.dimension, order)
                for attempt in range(1, max_attempts + 1):
                    reply = model.complete(question, max_tokens=first_max_tokens * attempt)
                    calls += 1
                    personas = _parse(reply, order, scale_max, toxic_threshold)
                    if all(p["score"] is not None for p in personas):
                        break
                else:
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"item_id": row.item_id, "dimension": row.dimension, "order": order, "prompt": question, "reply": reply,
                                            "attempts": attempt, "status": "incomplete"}, ensure_ascii=False) + "\n")
                    raise AnnotationError(f"{model.name}/{dataset}: {row.item_id} [{row.dimension}]: no valid score for every group of {order} "
                                          f"after {attempt} attempts; the reply is in {raw_path.name}")
                record = {"item_id": row.item_id, "dimension": row.dimension, "text": str(row.text), "order": order, "prompt": question,
                          "reply": reply, "personas": personas, "attempts": attempt, "status": "ok",
                          "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[key] = record
                bar.update(1)
        except Exception as exc:
            status = f"failed: {type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            bar.close()
            _write_csv(dataset, run, model.name, done, csv_path)
            cfg = model.config
            _log(out_dir, time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), run=run, dataset=dataset, model=model.name,
                 provider=cfg.provider, model_id=cfg.model_id, temperature=cfg.temperature, max_tokens=first_max_tokens,
                 prompt_version=getattr(prompt, "__name__", "custom"), item_word=item, batch_size=1, texts=len(table), batches=len(table),
                 batches_resumed=resumed, model_calls=calls, seconds=round(time.time() - started, 1), status=status)
        summary.append({"dataset": dataset, "texts": len(table), "resumed": resumed, "model calls": calls, "seconds": round(time.time() - started, 1)})
    return pd.DataFrame(summary).set_index("dataset")


def persona_long(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (text, dimension, group) from the saved rows (``personas`` as JSON text or as a list). ``position`` is the place of
    the group in the prompt (1 = shown first)."""
    rows = []
    for saved in table.to_dict("records"):
        personas = json.loads(saved["personas"]) if isinstance(saved["personas"], str) else saved["personas"]
        for position, persona in enumerate(personas, start=1):
            rows.append({"dataset": saved["dataset"], "item_id": saved["item_id"], "dimension": saved["dimension"], "text": saved["text"],
                         "model": saved["model"], "run": saved["run"], "group": persona["as"], "position": position,
                         "score": persona["score"], "label": persona["label"]})
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def collect_persona_runs(out_dir: Path, run: str = "scd") -> pd.DataFrame:
    """All saved answers of a run in the long form: one row per dataset, model, text, dimension and group."""
    tables = [pd.read_csv(f, dtype={"item_id": str}) for f in sorted(Path(out_dir).glob(f"*/{run}/*.csv"))]
    tables = [t for t in tables if "personas" in t.columns]               # other results can sit in a folder of the same name (new_results/kappa/scd holds kappa tables)
    if not tables:
        return pd.DataFrame(columns=LONG_COLUMNS)
    return persona_long(pd.concat(tables, ignore_index=True))


def inspect_personas(out_dir: Path, model: str, run: str = "scd") -> pd.DataFrame:
    """A first look at what one model returned, per dataset: texts (calls), groups rated, mean score, share called toxic, and how many
    calls had to be asked again. Meant for checking a model before running the next one."""
    rows = {}
    for csv_path in sorted(Path(out_dir).glob(f"*/{run}/{model}.csv")):
        long = persona_long(pd.read_csv(csv_path, dtype={"item_id": str}))
        raw_path = csv_path.parent / "raw" / f"{model}.raw.jsonl"
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()] if raw_path.exists() else []
        rows[csv_path.parent.parent.name] = {
            "texts": long[["item_id", "dimension"]].drop_duplicates().shape[0], "groups rated": len(long),
            "mean score": round(float(long["score"].mean()), 2) if len(long) else None,
            "toxic %": round(100 * float((long["label"] == "toxic").mean())) if len(long) else None,
            "asked again": sum(1 for r in records if r["status"] == "ok" and r["attempts"] > 1),
            "not complete": sum(1 for r in records if r["status"] == "incomplete")}
    return pd.DataFrame(rows).T if rows else pd.DataFrame()
