"""Alignment of the models with the two poles of a polarized text (section 7.6): does the model side with the toxic pole or the civil pole when the groups split,
and which group holds each pole.

It reads the attributed (AU) texts with the position of each group and the default-prompt labels the models already gave, and asks nothing of a model.

* **split text**: an attributed text (text x characteristic) where at least one group is toxic and at least one is civil; a text where all groups hold the same
  position is *same*, and is not split;
* **toxic pole**: the group(s) whose position is toxic on a split text; **civil pole**: the group(s) whose position is civil;
* a model sides with the toxic pole when its default-prompt label is toxic (a score of 3 or more), with the civil pole when it is civil.

Uses the attributed rows, the model labels and the resamples of ``kappa_groups`` (10,000 resamples of the texts of a characteristic, seed 42), so the numbers
refer to the same texts as the kappa tables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .eda_total import _ordered
from .kappa_groups import MODELS, THRESHOLD, _attributed_rows, _interval, _model_labels, _r, _se
from .kappa_tables import MODEL_NAME
from .persona_annotation import collect_persona_runs

__all__ = ["compute_poles", "save_poles"]

EXPECTED_ATTRIBUTIONS = {"attitudes": 51, "popquorn": 183, "mhs": 105, "kumar": 65}          # attributed rows (text x characteristic) used in the paper
SMALL = 5                                                                                  # a direction cell with fewer texts is flagged
DISTINCT = "all (distinct texts)"
KUMAR_GROUP_ORDER = ["politics", "gender", "race", "age", "religion"]


def _share(weights: np.ndarray, mask: np.ndarray, label: np.ndarray) -> np.ndarray:
    """The share of toxic labels among the masked texts for every weighting of the texts (``weights`` is resamples x texts); NaN where no masked text is drawn."""
    den, num = weights @ mask, weights @ (mask * label)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _summary(point: float, boot: np.ndarray) -> dict:
    low, high, _ = _interval(boot)
    return {"share": _r(point), "sd": _r(_se(boot)), "ci_low": _r(low), "ci_high": _r(high)}


def _alignment_rows(dataset, dimension, model, label, split, weights, n_attr) -> dict:
    """One row of ``poles_alignment`` from the 0/1 arrays ``split`` and ``label`` over the texts, and the resample weights (``counts``)."""
    same = 1.0 - split
    point_w = np.ones((1, len(label)))
    s = _summary(_share(point_w, split, label)[0], _share(weights, split, label))
    row_same = _share(point_w, same, label)[0]
    return {"dataset": dataset, "characteristic": dimension, "model": model, "n_attributed": n_attr, "n_split": int(split.sum()), "n_same": int((same).sum()),
            "n_model_toxic": int((split * label).sum()), "n_model_civil": int((split * (1 - label)).sum()),
            "share_model_toxic": s["share"], "sd": s["sd"], "ci_low": s["ci_low"], "ci_high": s["ci_high"],
            "share_model_toxic_same": _r(row_same) if np.isfinite(row_same) else np.nan, "constant_model": bool(len(set(label)) == 1)}


def compute_poles(results, results_dir: Path, given: pd.DataFrame, reps: int = 10_000, seed: int = 42, models: Sequence[str] = MODELS) -> Dict[str, pd.DataFrame]:
    """``alignment``, ``groups``, ``direction`` and ``checks``. ``results`` are the AU results, ``results_dir`` holds ``<dataset>/baseline/<model>.csv``, ``given`` is
    ``aposteriori_for_models.csv`` (for Kumar the sample of 65 attributions). Raises if the number of attributed rows differs from the paper's."""
    scd = collect_persona_runs(results_dir, "scd")                      # _model_labels reads both prompts; only the baseline is used here
    alignment, groups_rows, direction, checks = [], [], [], []
    for dataset in _ordered(results):
        result = results[dataset]
        attributed = _attributed_rows(result, given, dataset)
        if len(attributed) != EXPECTED_ATTRIBUTIONS.get(dataset, len(attributed)):
            raise ValueError(f"{dataset}: {len(attributed)} attributed rows, the paper has {EXPECTED_ATTRIBUTIONS[dataset]}")
        base = _model_labels(results_dir, dataset, models, scd)["baseline"]
        distinct = {}                                                   # item -> split under at least one characteristic
        for dimension in sorted(attributed["dimension"].unique(), key=list(result.spec.dimensions).index):
            part = attributed[attributed["dimension"] == dimension].reset_index(drop=True)
            n = len(part)
            names = sorted({str(g) for lab in part["group_label"] for g in lab})
            present, toxic = np.zeros((n, len(names))), np.zeros((n, len(names)))
            for i, lab in enumerate(part["group_label"]):
                for g, value in lab.items():
                    present[i, names.index(str(g))], toxic[i, names.index(str(g))] = 1.0, 1.0 if value == "toxic" else 0.0
            split = ((present * toxic).sum(axis=1) > 0) & ((present * (1 - toxic)).sum(axis=1) > 0)
            split = split.astype(float)
            for item, flag in zip(part["item_id"], split):
                distinct[item] = max(distinct.get(item, 0.0), flag)
            counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=reps).astype(float)
            labels = {m: (base[m].loc[part["item_id"]].to_numpy() >= THRESHOLD).astype(float) for m in models}
            for m in models:
                alignment.append(_alignment_rows(dataset, dimension, m, labels[m], split, counts, n))
            for j, name in enumerate(names):
                shown = present[:, j] * split                           # split texts where the group has a position
                at_toxic, at_civil = shown * toxic[:, j], shown * (1 - toxic[:, j])
                share_pole = _share(counts, shown, toxic[:, j])
                s = _summary(_share(np.ones((1, n)), shown, toxic[:, j])[0], share_pole)
                groups_rows.append({"dataset": dataset, "characteristic": dimension, "group": name, "n_texts": int(shown.sum()),
                                    "share_toxic_pole": s["share"], "share_civil_pole": _r(1 - s["share"]) if shown.sum() else np.nan,
                                    "sd": s["sd"], "ci_low": s["ci_low"], "ci_high": s["ci_high"]})
                for m in models:
                    for pole, mask, agree in (("toxic", at_toxic, labels[m]), ("civil", at_civil, 1 - labels[m])):
                        k = int(mask.sum())
                        direction.append({"dataset": dataset, "characteristic": dimension, "group": name, "model": m, "group_pole": pole, "n_texts": k,
                                          "agreement": _r((mask * agree).sum() / k) if k else np.nan, "small_n": k < SMALL})
        # the same per dataset, each distinct text once (split under at least one characteristic)
        items = list(distinct)
        flag = np.array([distinct[i] for i in items])
        counts = np.random.default_rng(seed).multinomial(len(items), np.full(len(items), 1.0 / len(items)), size=reps).astype(float)
        for m in models:
            label = (base[m].loc[items].to_numpy() >= THRESHOLD).astype(float)
            alignment.append(_alignment_rows(dataset, DISTINCT, m, label, flag, counts, len(items)))
        checks.append({"dataset": dataset, "attributed rows": len(attributed), "paper": EXPECTED_ATTRIBUTIONS.get(dataset), "distinct texts": len(items), "split texts (any characteristic)": int(flag.sum())})
    return {"alignment": pd.DataFrame(alignment), "groups": pd.DataFrame(groups_rows), "direction": pd.DataFrame(direction), "checks": pd.DataFrame(checks)}


# ---------------------------------------------------------------------------------------------------------------------------------------------
def _pct(value) -> str:
    return "–" if pd.isna(value) else f"{100 * value:.0f}%"


def _alignment_table(a: pd.DataFrame) -> list:
    lines = ["| characteristic | model | n attributed | n split | n same | model toxic / civil on split | share toxic on split (SD) [2.5–97.5] | share toxic on same |", "|---|---|---|---|---|---|---|---|"]
    for _, r in a.iterrows():
        dagger = "†" if r["constant_model"] else ""
        lines.append(f"| {r['characteristic']} | {MODEL_NAME[r['model']]} | {r['n_attributed']} | {r['n_split']} | {r['n_same']} | {r['n_model_toxic']} / {r['n_model_civil']} | "
                     f"{_pct(r['share_model_toxic'])}{dagger} ({100 * r['sd']:.0f}) [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}] | {_pct(r['share_model_toxic_same'])} |")
    return lines


def _groups_table(g: pd.DataFrame) -> list:
    lines = ["| characteristic | group | n split texts | at toxic pole | at civil pole | toxic-pole share SD | [2.5–97.5] |", "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r['characteristic']} | {r['group']} | {r['n_texts']} | {_pct(r['share_toxic_pole'])} | {_pct(r['share_civil_pole'])} | {100 * r['sd']:.0f} | "
                     f"[{_pct(r['ci_low'])}, {_pct(r['ci_high'])}] |" if r["n_texts"] else f"| {r['characteristic']} | {r['group']} | 0 | – | – | – | – |")
    return lines


def _direction_table(d: pd.DataFrame) -> list:
    lines = ["| characteristic | group | model | model toxic when group at toxic pole (n) | model civil when group at civil pole (n) |", "|---|---|---|---|---|"]
    for (dimension, group, model), p in d.groupby(["characteristic", "group", "model"], sort=False):
        cells = []
        for pole in ("toxic", "civil"):
            r = p[p["group_pole"] == pole].iloc[0]
            cells.append(f"{_pct(r['agreement'])} ({r['n_texts']}){'*' if r['small_n'] else ''}")
        lines.append(f"| {dimension} | {group} | {MODEL_NAME[model]} | {cells[0]} | {cells[1]} |")
    return lines


def _dataset_section(tables: Dict[str, pd.DataFrame], dataset: str, direction_for: Sequence[str]) -> list:
    a = tables["alignment"][tables["alignment"]["dataset"] == dataset]
    g = tables["groups"][tables["groups"]["dataset"] == dataset]
    d = tables["direction"][(tables["direction"]["dataset"] == dataset) & tables["direction"]["characteristic"].isin(direction_for)]
    return (["**Alignment with the poles (default prompt)**", ""] + _alignment_table(a) + ["", "**Which group holds each pole** (split texts where the group has a position)", ""]
            + _groups_table(g) + ["", f"**Direction by group** ({', '.join(direction_for)})", ""] + _direction_table(d) + [""])


def summary_text(tables: Dict[str, pd.DataFrame], reps: int, seed: int) -> str:
    L = ["# Alignment with the toxic and the civil pole (default prompt)", "",
         f"Split text = an attributed text where at least one group is toxic and at least one is civil. Bootstrap: {reps:,} resamples of the texts, seed {seed}. "
         "† = the model gave one label to every attributed text of the characteristic. \\* = fewer than 5 texts. Kumar: the 65-attribution sample. "
         f"The row `{DISTINCT}` counts each distinct text once, split if it is split under at least one characteristic.", "",
         "## 1. Kumar", ""]
    kumar = tables["alignment"][tables["alignment"]["dataset"] == "kumar"]
    L += _dataset_section(tables, "kumar", ["politics"])
    L += ["## 2. Other datasets", ""]
    for dataset in [d for d in _ordered(tables["alignment"]["dataset"].unique()) if d != "kumar"]:
        political = [c for c in tables["alignment"][tables["alignment"]["dataset"] == dataset]["characteristic"].unique() if c in ("politics", "ideology")]
        directions = political or [c for c in tables["alignment"][tables["alignment"]["dataset"] == dataset]["characteristic"].unique() if c != DISTINCT]
        L += [f"### {dataset}", ""] + _dataset_section(tables, dataset, directions)
    a, d = tables["alignment"], tables["direction"]
    L += ["## 3. Caveats", "", "n split per characteristic (attributed texts in brackets):", "", "| dataset | characteristic | n split | n attributed |", "|---|---|---|---|"]
    for (dataset, dimension), p in a[a["model"] == MODELS[0]].groupby(["dataset", "characteristic"], sort=False):
        L.append(f"| {dataset} | {dimension} | {p['n_split'].iloc[0]} | {p['n_attributed'].iloc[0]} |")
    small = d[d["small_n"]]
    L += ["", f"- Direction cells with fewer than {SMALL} texts: {len(small)} of {len(d)} ({int(small['n_texts'].eq(0).sum())} with no text).",
          "- Models that gave one label to every attributed text of a characteristic (share toxic on split is then 0% or 100%):"]
    const = a[a["constant_model"]]
    L += [f"  - {r['dataset']}/{r['characteristic']}: {MODEL_NAME[r['model']]} ({'toxic' if r['n_model_civil'] == 0 and r['n_split'] else 'civil' if r['n_model_toxic'] == 0 and r['n_split'] else 'no split texts'})"
          for _, r in const.iterrows()] or ["  - none"]
    L += [f"- Characteristics with no split text: {', '.join(f'{x.dataset}/{x.characteristic}' for x in a[(a['model'] == MODELS[0]) & (a['n_split'] == 0)].itertuples()) or 'none'}.",
          "- Attributed rows per dataset equal the paper's (Attitudes 51, POPQUORN 183, MHS 105, Kumar 65): checked when the tables are computed.", ""]
    return "\n".join(L)


def save_poles(tables: Dict[str, pd.DataFrame], folder: Path, reps: int = 10_000, seed: int = 42) -> Dict[str, Path]:
    """Write ``poles_alignment.csv``, ``poles_groups.csv``, ``poles_direction.csv`` and ``summary.md`` into ``folder`` (created if needed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for file, key in (("poles_alignment.csv", "alignment"), ("poles_groups.csv", "groups"), ("poles_direction.csv", "direction")):
        tables[key].to_csv(folder / file, index=False, encoding="utf-8")
        written[file] = folder / file
    (folder / "summary.md").write_text(summary_text(tables, reps, seed), encoding="utf-8")
    written["summary.md"] = folder / "summary.md"
    return written
