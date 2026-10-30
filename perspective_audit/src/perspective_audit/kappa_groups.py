"""Section 7.2: Cohen's kappa between each model's labels and each annotator group's position on the attributed (AU) texts, with bootstrap intervals.
Nothing here asks a model for anything: it reads the AU results and the labels the models already gave (the baseline prompt and the SCD prompt).

For every dataset, characteristic, group, model and prompt:

* **group position** of a text: the label of the group in the final AU results (``group_label``): the majority of the binary labels of its ratings (a rating
  of 3 or more is toxic), an exact tie decided by the group's mean rating (``Unimodality.label_info``). Groups removed by the rule of ten AU texts are not
  in it, and characteristics kept with fewer than ten texts stay in (``small_n``);
* **model label**: the model's 1-5 rating binarized the same way (3 or more is toxic). ``baseline``: its rating of the text; ``scd``: its rating of the text
  as a person of that group;
* **kappa**: Cohen's kappa of the two labels over the attributed texts in which the group is compared. A **constant model label** gives kappa 0 (and
  ``constant_model``), also inside a resample;
* **bootstrap**: the attributed texts of the characteristic are resampled with replacement (``reps`` times, fixed ``seed``); every group, model and prompt is
  recomputed on the same resamples. A resample in which the group has no text has no kappa: it is counted (``resamples_without_value``) and left out of
  the percentiles.

``compute_kappa`` also gives the gap between two groups of a characteristic (``kappa(A) - kappa(B)``) and the change from the baseline prompt to the SCD
prompt (``kappa(scd) - kappa(baseline)``), each with an interval and the share of resamples with a value of zero or less, from the same resamples.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .eda_total import _ordered
from .persona_annotation import collect_persona_runs
from .pipeline import DatasetResult

__all__ = ["compute_kappa", "save_kappa"]

MODELS = ["chatgpt", "deepseek", "gemini", "mistral"]
PROMPTS = ["baseline", "scd"]
DEMOGRAPHIC = ("gender", "race", "age")
THRESHOLD = 3                    # a rating at or above this is toxic (the humans' and the models')
SMALL = 10                       # a characteristic with fewer attributed texts than this is flagged small_n
EXPECTED_DISTINCT = {"attitudes": 41, "popquorn": 154, "mhs": 95, "kumar": 5209}          # distinct attributed texts of the full AU results
EXPECTED_SAMPLE_ROWS = {"kumar": 65}


def _kappa(weights: np.ndarray, position: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Cohen's kappa of the 0/1 ``label`` against the 0/1 ``position`` for every weighting of the texts at once (``weights`` is resamples x texts; a weight is
    how often a text is drawn, 0 where the group is not compared). A constant model label gives 0; no text at all gives NaN."""
    tt, tc = weights @ (position * label), weights @ (position * (1 - label))
    ct, cc = weights @ ((1 - position) * label), weights @ ((1 - position) * (1 - label))
    total = tt + tc + ct + cc
    with np.errstate(divide="ignore", invalid="ignore"):
        model_rate, group_rate = (tt + ct) / total, (tt + tc) / total
        chance = model_rate * group_rate + (1 - model_rate) * (1 - group_rate)
        kappa = ((tt + cc) / total - chance) / (1 - chance)
    kappa = np.where((model_rate == 0) | (model_rate == 1), 0.0, kappa)
    return np.where(total > 0, kappa, np.nan)


def _r(value: float) -> float:
    """Rounded to four decimals, without a negative zero."""
    return round(float(value), 4) + 0.0


def _se(values: np.ndarray) -> float:
    """The standard error: the standard deviation (ddof=1) of the bootstrap values that exist (the same resamples as the percentiles)."""
    valid = values[np.isfinite(values)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan


def _interval(values: np.ndarray):
    """The 2.5th and 97.5th percentile of the values that exist, and how many resamples had none."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan, np.nan, len(values)
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)), int(len(values) - len(valid))


def _shares(values: np.ndarray) -> dict:
    """The share (%) of the valid resamples with a value below 0, exactly 0, above 0, and 0 or less."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {"share_lt0": np.nan, "share_eq0": np.nan, "share_gt0": np.nan, "share_le0": np.nan}
    return {"share_lt0": round(100 * float(np.mean(valid < 0)), 2), "share_eq0": round(100 * float(np.mean(valid == 0)), 2),
            "share_gt0": round(100 * float(np.mean(valid > 0)), 2), "share_le0": round(100 * float(np.mean(valid <= 0)), 2)}


def _excludes_zero(low: float, high: float) -> bool:
    """True when the interval (as written in the csv) lies strictly on one side of 0."""
    return bool(low > 0 or high < 0)


def _attributed_rows(result: DatasetResult, given: Optional[pd.DataFrame], dataset: str) -> pd.DataFrame:
    """The AU rows (text x characteristic) of the dataset, restricted to the rows given to the models (all of them except for Kumar, which got a sample)."""
    rows = pd.DataFrame({"dimension": result.au["dimension"].to_numpy(), "item_id": result.au["item_id"].astype(str).to_numpy(),
                         "group_label": result.au["group_label"].to_numpy()})
    if given is not None:
        part = given[given["dataset"] == dataset]
        keys = set(zip(part["dimension"], part["item_id"].astype(str)))
        rows = rows[[key in keys for key in zip(rows["dimension"], rows["item_id"])]]
    return rows.sort_values(["dimension", "item_id"], kind="stable").reset_index(drop=True)


def _model_labels(results_dir: Path, dataset: str, models: Sequence[str], scd: pd.DataFrame) -> dict:
    """The saved 1-5 ratings: ``baseline[model]`` is a Series by item id, ``scd[model]`` a Series by (dimension, item id, group)."""
    out = {"baseline": {}, "scd": {}}
    for model in models:
        path = Path(results_dir) / dataset / "baseline" / f"{model}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no baseline file for {dataset}/{model}: {path}")
        out["baseline"][model] = pd.read_csv(path, dtype={"item_id": str}).drop_duplicates("item_id").set_index("item_id")["score"]
        part = scd[(scd["dataset"] == dataset) & (scd["model"] == model)]
        if part.empty:
            raise FileNotFoundError(f"no scd answers for {dataset}/{model}")
        out["scd"][model] = part.set_index(["dimension", "item_id", "group"])["score"]
    return out


def _block(rows: pd.DataFrame, dimension: str, labels: dict, models: Sequence[str], dataset: str) -> dict:
    """The arrays of one characteristic: which group is compared in which text, its position, and every model's label for every prompt."""
    part = rows[rows["dimension"] == dimension].reset_index(drop=True)
    groups = sorted({str(g) for labels_of_text in part["group_label"] for g in labels_of_text})
    n = len(part)
    present, position = np.zeros((n, len(groups))), np.zeros((n, len(groups)))
    for i, labels_of_text in enumerate(part["group_label"]):
        for g, label in labels_of_text.items():
            present[i, groups.index(str(g))], position[i, groups.index(str(g))] = 1.0, 1.0 if label == "toxic" else 0.0
    block = {"n": n, "groups": groups, "present": present, "position": position, "baseline": {}, "scd": {}}
    for model in models:
        base = labels["baseline"][model]
        missing = [i for i in part["item_id"] if i not in base.index]
        if missing:
            raise ValueError(f"{dataset}/{dimension}: the baseline rating of {model} is missing for {len(missing)} attributed texts, e.g. {missing[0]}")
        block["baseline"][model] = (base.loc[part["item_id"]].to_numpy() >= THRESHOLD).astype(float)
        persona = np.zeros((n, len(groups)))
        for i, item in enumerate(part["item_id"]):
            for j, g in enumerate(groups):
                if present[i, j]:
                    try:
                        persona[i, j] = float(labels["scd"][model].loc[(dimension, item, g)] >= THRESHOLD)
                    except KeyError:
                        raise ValueError(f"{dataset}/{dimension}: no scd answer of {model} for text {item} as {g}") from None
        block["scd"][model] = persona
    return block


def _tables_of_block(dataset: str, dimension: str, block: dict, models: Sequence[str], reps: int, seed: int) -> tuple:
    n, groups = block["n"], block["groups"]
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=reps).astype(float)        # how often each text is drawn, per resample
    small = n < SMALL
    kappas, by_group = {}, []
    for j, group in enumerate(groups):
        present, position = block["present"][:, j], block["position"][:, j]
        point_weights, boot_weights = present[None, :], counts * present[None, :]
        shown = present.astype(bool)
        for model in models:
            for prompt in PROMPTS:
                label = block["baseline"][model] if prompt == "baseline" else block["scd"][model][:, j]
                point, boot = float(_kappa(point_weights, position, label)[0]), _kappa(boot_weights, position, label)
                kappas[(prompt, model, j)] = (point, boot)
                low, high, without = _interval(boot)
                by_group.append({"dataset": dataset, "model": model, "characteristic": dimension, "type": "demographic" if dimension in DEMOGRAPHIC else "non-demographic",
                                 "group": group, "prompt": prompt, "n_texts": int(present.sum()), "kappa": _r(point), "ci_low": _r(low), "ci_high": _r(high),
                                 "resamples_without_value": without, "constant_model": bool(len(set(label[shown])) == 1), "constant_group": bool(len(set(position[shown])) == 1),
                                 "pct_toxic_model": round(100 * float(label[shown].mean()), 2), "pct_toxic_group": round(100 * float(position[shown].mean()), 2), "small_n": small,
                                 "se": _r(_se(boot))})
    gaps, change = [], []
    for model in models:
        for prompt in PROMPTS:
            for a, b in combinations(range(len(groups)), 2):
                (pa, ba), (pb, bb) = kappas[(prompt, model, a)], kappas[(prompt, model, b)]
                low, high, without = _interval(ba - bb)
                gaps.append({"dataset": dataset, "model": model, "characteristic": dimension, "prompt": prompt, "group_a": groups[a], "group_b": groups[b],
                             "n_texts_a": int(block["present"][:, a].sum()), "n_texts_b": int(block["present"][:, b].sum()), "gap": _r(pa - pb),
                             "ci_low": _r(low), "ci_high": _r(high), **_shares(ba - bb),
                             "resamples_without_value": without, "small_n": small, "se": _r(_se(ba - bb)),
                             "interval_excludes_zero": _excludes_zero(_r(low), _r(high))})
        for j, group in enumerate(groups):
            (p0, b0), (p1, b1) = kappas[("baseline", model, j)], kappas[("scd", model, j)]
            low, high, without = _interval(b1 - b0)
            change.append({"dataset": dataset, "model": model, "characteristic": dimension, "group": group, "n_texts": int(block["present"][:, j].sum()),
                           "kappa_baseline": _r(p0), "kappa_scd": _r(p1), "change": _r(p1 - p0), "ci_low": _r(low), "ci_high": _r(high),
                           **_shares(b1 - b0), "resamples_without_value": without,
                           "small_n": small, "se": _r(_se(b1 - b0)), "interval_excludes_zero": _excludes_zero(_r(low), _r(high))})
    mirrored = len(groups) == 2 and bool(block["present"].all() and (block["position"][:, 0] != block["position"][:, 1]).all())          # opposite sides on every text
    for row in by_group:
        row["mirrored"] = mirrored
    return by_group, gaps, change


def _constant_summary(by_group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in by_group.groupby(["dataset", "model", "prompt"], sort=False):
        rows.append({"dataset": keys[0], "model": keys[1], "prompt": keys[2], "combinations": len(part), "constant_model": int(part["constant_model"].sum())})
    table = pd.DataFrame(rows)
    totals = [table.assign(**{"model": "all", "prompt": "all"}).groupby("dataset", sort=False)[["combinations", "constant_model"]].sum().reset_index().assign(model="all", prompt="all"),
              pd.DataFrame([{"dataset": "all", "model": "all", "prompt": "all", "combinations": int(table["combinations"].sum()), "constant_model": int(table["constant_model"].sum())}])]
    table = pd.concat([table] + totals, ignore_index=True)
    table["share_constant_model_pct"] = (100 * table["constant_model"] / table["combinations"]).round(2)
    return table


def share_problems(gaps: pd.DataFrame, change: Optional[pd.DataFrame], tolerance: float = 0.015) -> pd.DataFrame:
    """The rows of the gap and prompt-change tables in which share_lt0 + share_eq0 differs from share_le0, or the three shares do not sum to 100 (rounding allowed)."""
    rows = []
    for name, table in (("gaps", gaps), ("prompt_change", change)):
        if table is None:
            continue
        bad = table[((table["share_lt0"] + table["share_eq0"] - table["share_le0"]).abs() > tolerance)
                    | ((table[["share_lt0", "share_eq0", "share_gt0"]].sum(axis=1) - 100).abs() > 2 * tolerance)]
        rows.append(bad.assign(table=name))
    return pd.concat(rows, ignore_index=True)


def excludes_zero_counts(gaps: pd.DataFrame, change: Optional[pd.DataFrame]) -> pd.DataFrame:
    """How many gaps (per prompt) and prompt changes have an interval that excludes 0, per dataset, out of all."""
    rows = []
    for name, table, by in (("gaps", gaps, ["dataset", "prompt"]), ("prompt_change", change, ["dataset"])):
        if table is None:
            continue
        count = table.groupby(by, sort=False)["interval_excludes_zero"].agg(["sum", "size"]).reset_index()
        count.insert(0, "table", name)
        rows.append(count.rename(columns={"sum": "excluding_zero", "size": "rows"}))
    return pd.concat(rows, ignore_index=True).fillna({"prompt": "baseline to scd"})


def mirrored_pairs(by_group: pd.DataFrame) -> pd.DataFrame:
    """The (dataset, characteristic) pairs with two groups that take opposite sides on every attributed text."""
    return by_group[by_group["mirrored"]][["dataset", "characteristic"]].drop_duplicates().reset_index(drop=True)


def compute_kappa(results: Dict[str, DatasetResult], results_dir: Path, given: pd.DataFrame, reps: int = 10_000, seed: int = 42,
                  models: Sequence[str] = MODELS, prompt: str = "baseline") -> Dict[str, pd.DataFrame]:
    """The tables of one prompt and a table of checks: ``by_group``, ``gaps``, ``constant``, ``checks`` and the checks of the shares, and for ``prompt="scd"`` also
    ``prompt_change`` (the change from the default prompt to the SCD prompt). ``prompt="baseline"`` is the default prompt (section 7.2), ``"scd"`` section 7.3; both
    use the same resamples (same seed), so their numbers can be compared.
    ``results`` are the AU results of the datasets, ``results_dir`` holds ``<dataset>/baseline/<model>.csv`` and ``<dataset>/scd/<model>.csv`` (the
    labels the models gave), and ``given`` is ``aposteriori_for_models.csv`` (the rows given to the models: for Kumar the sample)."""
    scd = collect_persona_runs(results_dir, "scd")
    by_group, gaps, change, checks = [], [], [], []
    for dataset in _ordered(results):
        result = results[dataset]
        rows = _attributed_rows(result, given, dataset)
        labels = _model_labels(results_dir, dataset, models, scd)
        for dimension in sorted(rows["dimension"].unique(), key=list(result.spec.dimensions).index):
            block = _block(rows, dimension, labels, models, dataset)
            parts = _tables_of_block(dataset, dimension, block, models, reps, seed)
            for target, part in zip((by_group, gaps, change), parts):
                target.extend(part)
        full_distinct, full_rows = result.au["item_id"].nunique(), len(result.au)
        checks.append({"dataset": dataset, "attributed rows used (text x characteristic)": len(rows), "distinct texts used": rows["item_id"].nunique(),
                       "AU rows (all)": full_rows, "distinct AU texts (all)": full_distinct, "expected distinct AU texts (all)": EXPECTED_DISTINCT.get(dataset),
                       "expected rows used": EXPECTED_SAMPLE_ROWS.get(dataset, full_rows),
                       "matches": bool(full_distinct == EXPECTED_DISTINCT.get(dataset, full_distinct) and len(rows) == EXPECTED_SAMPLE_ROWS.get(dataset, full_rows))})
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {PROMPTS}, not {prompt!r}")
    by_group, gaps, change = pd.DataFrame(by_group), pd.DataFrame(gaps), pd.DataFrame(change)
    by_group, gaps = (table[table["prompt"] == prompt].reset_index(drop=True) for table in (by_group, gaps))
    change = change if prompt == "scd" else None                          # the change compares the two prompts: it belongs to the SCD tables
    out = {"by_group": by_group, "gaps": gaps, "constant": _constant_summary(by_group), "checks": pd.DataFrame(checks),
           "share_problems": share_problems(gaps, change), "excludes_zero": excludes_zero_counts(gaps, change), "mirrored": mirrored_pairs(by_group)}
    if change is not None:
        out["prompt_change"] = change
    return out


README = """# Cohen's kappa between the models and the annotator groups

{which}

Nothing here asks a model for anything: it reads the AU results and the labels the models already gave.

## Definitions

- **Attributed texts**: the AU rows (text x characteristic) of the final AU results, after the removal of groups in fewer than 10 AU texts (those groups stay
  removed). **Kumar** was evaluated on a sample of 65 rows, so only those rows are used for Kumar (13 groups x 10).
- **Group position** of a text: the group's label in the AU results: the majority of the binary labels of its ratings (a rating of 3 or more is toxic), an exact
  tie decided by the group's mean rating (3 or more is toxic). For **MHS** the humans' scale is 1-3, so only a rating of 3 ("Yes") is toxic, and the tie rule uses the
  mean on that scale; the models rate 1-5 and are binarized at 3 like every other dataset.
- **Model label**: the model's 1-5 rating binarized (3 or more is toxic). `baseline`: its rating of the text; `scd`: its rating of the text as a person of that group.
- **kappa**: Cohen's kappa between the model's label and the group's position over the texts in which the group is compared. A **constant model label gives kappa 0**
  and `constant_model = True` (also inside a resample). `constant_group` flags a group whose position is the same in all its texts.
- **small_n**: the characteristic has fewer than 10 attributed texts (Attitudes age and traditionalism; these characteristics stay in).
- **se**: the standard error, the standard deviation (ddof = 1) of the bootstrap values of that quantity over the valid resamples (the same resamples as the percentiles).
- **Bootstrap**: the attributed texts of a characteristic are resampled with replacement 10,000 times (seed 42); every group, model and prompt is recomputed on the same
  resamples. `ci_low` and `ci_high` are the 2.5th and 97.5th percentiles. A resample in which a group has no text has no kappa: it is counted in
  `resamples_without_value` and left out of the percentiles.

## Files

| file | content |
|---|---|
| `kappa_by_group.csv` | one row per dataset, model, characteristic, group and prompt: `n_texts`, `kappa`, `ci_low`, `ci_high`, `resamples_without_value`, `constant_model`, `constant_group`, `pct_toxic_model`, `pct_toxic_group`, `small_n`, `se`, `mirrored` (a characteristic with two groups that take opposite sides on every text) |
| `kappa_gaps.csv` | per dataset, model, characteristic, prompt and pair of groups: `gap` = kappa(A) - kappa(B), its interval from the same resamples, `share_lt0`, `share_eq0`, `share_gt0`, `share_le0` (% of resamples with a gap below 0, exactly 0, above 0, 0 or less), `interval_excludes_zero` |
| `kappa_prompt_change.csv` | per dataset, model, characteristic and group: `change` = kappa(scd) - kappa(baseline), its interval, `share_lt0`, `share_eq0`, `share_gt0`, `share_le0`, `interval_excludes_zero` |
| `constant_label_summary.csv` | how many combinations have a constant model label, per dataset, model and prompt, and in total |
"""


def save_kappa(tables: Dict[str, pd.DataFrame], folder: Path) -> Dict[str, Path]:
    """Write the csv files and the README of one prompt (section 7.2: default, 7.3: SCD) into ``folder`` (created if needed). Returns the files written, by file name."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prompt = tables["by_group"]["prompt"].iloc[0]
    which = ("**This folder (section 7.2): the default prompt only** (the rating of the text as it is). The SCD prompt is in the folder `scd`." if prompt == "baseline" else
             "**This folder (section 7.3): the SCD prompt only** (the model rates the text as a person of each group). `kappa_prompt_change.csv` compares it with the default prompt "
             "(folder `default`) on the same resamples.")
    written = {}
    for file, key in (("kappa_by_group.csv", "by_group"), ("kappa_gaps.csv", "gaps"), ("kappa_prompt_change.csv", "prompt_change"), ("constant_label_summary.csv", "constant")):
        if key in tables:
            tables[key].to_csv(folder / file, index=False, encoding="utf-8")
            written[file] = folder / file
    (folder / "README.md").write_text(README.replace("{which}", which), encoding="utf-8")
    written["README.md"] = folder / "README.md"
    return written
