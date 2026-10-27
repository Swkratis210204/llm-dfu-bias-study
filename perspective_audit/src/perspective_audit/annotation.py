"""Run ONE model over the texts of the datasets and save every reply (the baseline: no persona).

``annotate(model, inputs, out_dir, datasets=[...])`` is called once per model (and, if wanted, once per dataset), from the
notebook, so the order is controlled there and a problem shows up where it happens. Everything of a dataset is kept in its
own folder, ``out_dir/<dataset>/<run>/``::

    <model>.csv         one row per text: dataset, item_id, text, model, run, batch, score, label   (one file per model)
    all_models.csv      the models side by side, one row per text: <model>_score, <model>_label for every model run so far
    raw/<model>.raw.jsonl   one line per batch: the prompt, the reply, the parsed scores (the source of truth)

and one line per (model, dataset) pass is added to ``out_dir/run_log.csv``. ``all_models.csv`` is refreshed after every pass.
If a run stops (an API error, a reply that cannot be parsed) everything finished so far is kept, and calling ``annotate``
again carries on at the first missing batch.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .datasets import SPECS
from .models import DEFAULT_CONFIGS, Model
from .prompts import ScoringPrompt

__all__ = ["annotate", "ensemble", "load_model_inputs", "progress", "inspect_run", "collect_runs", "sample_inputs", "AnnotationError"]

ENSEMBLE_FILE = "all_models.csv"
MODEL_ORDER = list(DEFAULT_CONFIGS)                  # the order of the models in all_models.csv; other names follow alphabetically

LOG_COLUMNS = ["time_utc", "run", "dataset", "model", "provider", "model_id", "temperature", "max_tokens", "prompt_version",
               "item_word", "batch_size", "texts", "batches", "batches_resumed", "model_calls", "seconds", "status"]


class AnnotationError(RuntimeError):
    """A batch could not be annotated; what was finished before it is saved."""


def load_model_inputs(path: Path) -> Dict[str, pd.DataFrame]:
    """The texts to annotate, per dataset: each text once, in the order of the file (``item_id``, ``text``)."""
    table = pd.read_csv(path, dtype={"item_id": str})
    return {dataset: part.drop_duplicates("item_id")[["item_id", "text"]].reset_index(drop=True)
            for dataset, part in table.groupby("dataset", sort=False)}


def _paths(out_dir: Path, dataset: str, run: str, model: str):
    folder = Path(out_dir) / dataset / run
    return folder, folder / "raw" / f"{model}.raw.jsonl", folder / f"{model}.csv"


def ensemble(out_dir: Path, dataset: str, run: str = "baseline", models: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """The models of one dataset side by side: one row per text, ``<model>_score`` and ``<model>_label`` for each model that has
    a saved file (or for ``models``). Written to ``<dataset>/<run>/all_models.csv`` and returned."""
    folder = Path(out_dir) / dataset / run
    found = [p.stem for p in folder.glob("*.csv") if p.name != ENSEMBLE_FILE]
    names = list(models) if models is not None else sorted(found, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))
    merged = None
    for name in names:
        part = pd.read_csv(folder / f"{name}.csv", dtype={"item_id": str})
        part = part[["dataset", "item_id", "text", "score", "label"]].rename(columns={"score": f"{name}_score", "label": f"{name}_label"})
        merged = part if merged is None else merged.merge(part.drop(columns=["dataset", "text"]), on="item_id", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged.to_csv(folder / ENSEMBLE_FILE, index=False, encoding="utf-8")
    return merged


def _read_batches(raw_path: Path) -> Dict[int, dict]:
    done: Dict[int, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] == "ok":
                done[record["batch"]] = record
    return done


def _write_csv(dataset: str, run: str, model: str, done: Dict[int, dict], csv_path: Path) -> None:
    """The csv is rebuilt from the saved batches only (each keeps its own texts), never from the current input."""
    rows = [{"dataset": dataset, "item_id": item_id, "text": text, "model": model, "run": run,
             "score": parsed["score"], "label": parsed["label"]}
            for _, record in sorted(done.items())
            for item_id, text, parsed in zip(record["item_ids"], record["texts"], record["parsed"])]
    pd.DataFrame(rows, columns=["dataset", "item_id", "text", "model", "run", "score", "label"]).to_csv(csv_path, index=False, encoding="utf-8")


def _log(out_dir: Path, **fields) -> None:
    path = Path(out_dir) / "run_log.csv"
    line = pd.DataFrame([{c: fields.get(c) for c in LOG_COLUMNS}], columns=LOG_COLUMNS)
    line.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def annotate(model: Model, inputs: Dict[str, pd.DataFrame], out_dir: Path, prompt: Callable[[str, str], str], run: str = "baseline",
             batch_size: int = 10, parse_retries: int = 2, limit: Optional[int] = None, datasets: Optional[Iterable[str]] = None,
             item_words: Optional[Dict[str, str]] = None, scale_max: int = 5, toxic_threshold: float = 3,
             show_progress: bool = True) -> pd.DataFrame:
    """Score (1-5) and label every text of every dataset with one model, batch by batch. Returns one row per dataset.

    ``limit`` keeps only the first texts of each dataset (a pilot). ``item_words`` overrides the word used for a text in the
    prompt (default: the word of the dataset, e.g. "comment"). A batch whose reply does not have exactly one score per text is
    asked again ``parse_retries`` times and then raises ``AnnotationError`` (the reply is kept in the raw file).
    ``prompt`` is your function ``prompt(texts, item) -> str``: ``texts`` is the batch already numbered ("1. ...\n2. ..."), ``item``
    the word for a text, and the string it returns is sent to the model. The reply must have a line ``score=<n>, label=<toxic/civil>``
    per text. Use another ``run`` name when you change the prompt. ``show_progress`` shows a bar per dataset and model (texts done,
    time elapsed and left, and the batch being asked).
    """
    if not callable(prompt):
        raise TypeError("prompt must be a function prompt(texts, item) -> str")
    words = {name: spec.item_word for name, spec in SPECS.items()}
    words.update(item_words or {})
    summary = []
    for dataset in (list(datasets) if datasets is not None else list(inputs)):
        table = inputs[dataset].head(limit) if limit else inputs[dataset]
        table = table.reset_index(drop=True)
        scoring = ScoringPrompt(item=words.get(dataset, "text"), scale_max=scale_max, toxic_threshold=toxic_threshold, write=prompt)
        folder, raw_path, csv_path = _paths(out_dir, dataset, run, model.name)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        done = _read_batches(raw_path)
        batches = [table.iloc[i:i + batch_size] for i in range(0, len(table), batch_size)]
        started, calls, resumed, status = time.time(), 0, 0, "complete"
        bar = tqdm(total=len(table), desc=f"{dataset} / {model.name}", unit="text", disable=not show_progress)
        try:
            for number, batch in enumerate(batches, start=1):
                ids = batch["item_id"].tolist()
                if number in done:
                    if done[number]["item_ids"] != ids:
                        raise AnnotationError(f"{model.name}/{dataset}: saved batch {number} holds other texts than the input; use another run name")
                    resumed += 1
                    bar.update(len(batch))
                    continue
                first = str(batch["text"].iloc[0]).replace("\n", " ")
                bar.set_postfix_str(f"batch {number}/{len(batches)}: {first[:40]}...", refresh=True)
                question = scoring.build(batch["text"].tolist())
                for attempt in range(1, parse_retries + 2):
                    reply = model.complete(question)
                    calls += 1
                    parsed = scoring.parse(reply)
                    if len(parsed) == len(batch):
                        break
                else:
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"batch": number, "item_ids": ids, "prompt": question, "reply": reply, "attempts": attempt,
                                            "status": "unparseable"}, ensure_ascii=False) + "\n")
                    raise AnnotationError(f"{model.name}/{dataset}: batch {number}: expected {len(batch)} scores, received {len(parsed)} "
                                          f"after {attempt} attempts; the reply is in {raw_path.name}")
                record = {"batch": number, "item_ids": ids, "texts": batch["text"].tolist(), "prompt": question, "reply": reply,
                          "parsed": parsed, "attempts": attempt,
                          "status": "ok", "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[number] = record
                bar.update(len(batch))
        except Exception as exc:
            status = f"failed: {type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            bar.close()
            _write_csv(dataset, run, model.name, done, csv_path)
            try:
                ensemble(out_dir, dataset, run)                          # all_models.csv follows the model that was just run
            except Exception as exc:                                     # never hide the error that stopped the run
                print(f"warning: all_models.csv of {dataset}/{run} could not be refreshed: {exc!r}")
            cfg = model.config
            _log(out_dir, time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), run=run, dataset=dataset, model=model.name,
                 provider=cfg.provider, model_id=cfg.model_id, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                 prompt_version=getattr(prompt, "__name__", "custom"), item_word=words.get(dataset, "text"), batch_size=batch_size, texts=len(table),
                 batches=len(batches), batches_resumed=resumed, model_calls=calls, seconds=round(time.time() - started, 1), status=status)
        summary.append({"dataset": dataset, "texts": len(table), "batches": len(batches), "resumed": resumed, "model calls": calls,
                        "seconds": round(time.time() - started, 1)})
    return pd.DataFrame(summary).set_index("dataset")


def progress(inputs: Dict[str, pd.DataFrame], out_dir: Path, models: Iterable[str], run: str = "baseline") -> pd.DataFrame:
    """For each dataset and model: how many of the texts have a saved score."""
    rows = {}
    for dataset, table in inputs.items():
        row = {"texts": len(table)}
        for name in models:
            _, raw_path, _ = _paths(out_dir, dataset, run, name)
            row[name] = sum(len(r["item_ids"]) for r in _read_batches(raw_path).values())
        rows[dataset] = row
    return pd.DataFrame(rows).T


def inspect_run(out_dir: Path, model: str, run: str = "baseline") -> pd.DataFrame:
    """A first look at what one model returned, per dataset: how many texts, the scores it gave, the share it called toxic, and
    whether any batch had to be asked again or could not be parsed. Meant for checking a model before running the next one."""
    rows = {}
    for csv_path in sorted(Path(out_dir).glob(f"*/{run}/{model}.csv")):
        table = pd.read_csv(csv_path)
        raw_path = csv_path.parent / "raw" / f"{model}.raw.jsonl"
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()] if raw_path.exists() else []
        row = {"texts": len(table), "mean score": round(float(table["score"].mean()), 2) if len(table) else None,
               "toxic %": round(100 * float((table["label"] == "toxic").mean())) if len(table) else None}
        row.update({f"score {s}": int((table["score"] == s).sum()) for s in range(1, 6)})
        row["batches asked again"] = sum(1 for r in records if r["status"] == "ok" and r["attempts"] > 1)
        row["batches not parsed"] = sum(1 for r in records if r["status"] == "unparseable")
        rows[csv_path.parent.parent.name] = row
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def sample_inputs(inputs: Dict[str, pd.DataFrame], n: int, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """``n`` random texts of every dataset (all of them when a dataset has fewer), in the order of the input. The same ``n`` and
    ``seed`` always give the same texts, so every prompt and every model sees the same subset."""
    return {name: table.sample(n=min(n, len(table)), random_state=seed).sort_index().reset_index(drop=True)
            for name, table in inputs.items()}


def collect_runs(out_dir: Path, run: str = "baseline") -> pd.DataFrame:
    """All saved scores of a run: one row per dataset, model and text."""
    files = sorted(p for p in Path(out_dir).glob(f"*/{run}/*.csv") if p.name != ENSEMBLE_FILE)
    if not files:
        return pd.DataFrame(columns=["dataset", "item_id", "text", "model", "run", "score", "label"])
    return pd.concat([pd.read_csv(f, dtype={"item_id": str}) for f in files], ignore_index=True)
