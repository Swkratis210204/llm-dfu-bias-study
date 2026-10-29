"""The prompt-wording ablation (section 6.2): does rewording the question change the ratings? Nothing here asks a model for anything: it reads the
saved runs of the baseline and of the two rewordings, which were run on the same random sample of texts of every dataset.

``save_ablation_prompt(results_dir, folder)`` writes into ``folder``:

    agreement_pooled.csv       per dataset and pair of prompts, the four models together: the share of identical labels, Cohen's kappa of the labels,
                               the weighted kappa of the scores, Spearman, the share of identical scores and the mean difference, with 95% bootstrap
                               intervals (texts are resampled, the four models of a text move together)
    agreement_by_model.csv     the same, per dataset and model
    wording_vs_model.csv       the same model under another wording against different models under the same prompt (mean weighted kappa)
    ablation_sample.csv        the texts of the sample (the same for every model and prompt)
    ablation_weighted_kappa    figure (pdf, png): the weighted kappa per dataset and model, with intervals
    ablation_label_agreement   figure (pdf, png): the share of identical labels per dataset and model, with intervals
    ablation_wording_vs_model  figure (pdf, png): a change of wording against a change of model
    README.md                  the design (what changed, what was fixed), the definitions, the caveats and the files

The prompts are ``baseline`` (run ``baseline``), ``R1`` (run ``prompt_sensitivity``) and ``R2`` (run ``prompt_sensitivity_2``); ``R1`` and ``R2`` are two
rewordings of the baseline that keep the task, the 1-5 scale, the rule and the reply format.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .agreement import cohen_kappa, spearman, weighted_kappa
from .eda_total import DATASET_LABEL, _ordered, _save_figure

__all__ = ["load_samples", "agreement_pooled", "agreement_by_model", "wording_vs_model", "save_ablation_prompt"]

RUNS = {"baseline": "baseline", "R1": "prompt_sensitivity", "R2": "prompt_sensitivity_2"}          # prompt -> name of the run (folder)
COMPARISONS = [("baseline", "R1"), ("baseline", "R2"), ("R1", "R2")]
MODELS = ["chatgpt", "deepseek", "mistral", "gemini"]
MODEL_LABEL = {"chatgpt": "ChatGPT", "deepseek": "DeepSeek", "mistral": "Mistral", "gemini": "Gemini"}
COMPARISON_COLOR = {("baseline", "R1"): "#4C72B0", ("baseline", "R2"): "#DD8452", ("R1", "R2"): "#888888"}
DATASETS = ["attitudes", "popquorn", "mhs", "kumar"]
REPS = 2000


def load_samples(results_dir: Path, datasets: Sequence[str] = DATASETS, models: Sequence[str] = MODELS) -> Dict[str, dict]:
    """The saved runs of the three prompts for every dataset: ``{dataset: {"ids", "models", "score": {prompt: {model: array}}, "label": {...}}}`` on the
    texts that every model has answered in every prompt (the sample)."""
    samples = {}
    for dataset in datasets:
        frames = {}
        for prompt, run in RUNS.items():
            for model in models:
                path = Path(results_dir) / dataset / run / f"{model}.csv"
                if not path.exists():
                    break
                frames[(prompt, model)] = pd.read_csv(path, dtype={"item_id": str}).drop_duplicates("item_id").set_index("item_id")
            else:
                continue
            break
        else:
            ids = sorted(set.intersection(*[set(frame.index) for frame in frames.values()]))
            samples[dataset] = {"ids": ids, "models": list(models),
                                "score": {p: {m: frames[(p, m)].loc[ids, "score"].to_numpy().astype(int) for m in models} for p in RUNS},
                                "label": {p: {m: frames[(p, m)].loc[ids, "label"].to_numpy() for m in models} for p in RUNS}}
    return samples


def _core(a_score, b_score, a_label, b_label) -> dict:
    return {"label %": 100 * float(np.mean(a_label == b_label)), "cohen": cohen_kappa(a_label, b_label), "wk": weighted_kappa(a_score, b_score)}


def _full(a_score, b_score, a_label, b_label) -> dict:
    return {**_core(a_score, b_score, a_label, b_label), "same score %": 100 * float(np.mean(a_score == b_score)), "spearman": spearman(a_score, b_score),
            "diff": float(np.mean(b_score - a_score))}


def _interval(values, q) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanpercentile(values, q)) if np.isfinite(values).any() else np.nan


def _bootstrap(statistic, n_items: int, reps: int, seed: int) -> Dict[str, tuple]:
    """95% percentile intervals of ``statistic(indices)`` (a dict of numbers) over ``reps`` resamples of the ``n_items`` texts."""
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            for key, value in statistic(rng.integers(0, n_items, n_items)).items():
                out[key].append(value)
    return {key: (_interval(v, 2.5), _interval(v, 97.5)) for key, v in out.items()}


def _row(prefix: dict, point: dict, interval: dict, texts: int, pairs: int) -> dict:
    row = {**prefix, "texts": texts, "pairs": pairs}
    for key, name, digits in (("label %", "same label %", 1), ("cohen", "Cohen's kappa (labels)", 3), ("wk", "weighted kappa (scores)", 3)):
        row[name] = round(point[key], digits) if np.isfinite(point[key]) else np.nan
        row[f"{name} low"], row[f"{name} high"] = (round(interval[key][0], digits), round(interval[key][1], digits))
    row.update({"same score %": round(point["same score %"], 1), "Spearman": round(point["spearman"], 3), "mean diff (second - first)": round(point["diff"], 3)})
    return row


def agreement_pooled(samples: Dict[str, dict], reps: int = REPS, seed: int = 42) -> pd.DataFrame:
    """Per dataset and pair of prompts, the models together: the agreement of every (text, model) pair, with intervals from resampling the texts."""
    rows = []
    for dataset in _ordered(samples):
        s = samples[dataset]
        n, models = len(s["ids"]), s["models"]
        for x, y in COMPARISONS:
            a_score, b_score = (np.concatenate([s["score"][p][m] for m in models]) for p in (x, y))
            a_label, b_label = (np.concatenate([s["label"][p][m] for m in models]) for p in (x, y))
            interval = _bootstrap(lambda i: _core(*(v[np.concatenate([i + k * n for k in range(len(models))])] for v in (a_score, b_score, a_label, b_label))),
                                  n, reps, seed)
            rows.append(_row({"dataset": dataset, "comparison": f"{x} vs {y}"}, _full(a_score, b_score, a_label, b_label), interval, n, len(a_score)))
    return pd.DataFrame(rows)


def agreement_by_model(samples: Dict[str, dict], reps: int = REPS, seed: int = 42) -> pd.DataFrame:
    """Per dataset, model and pair of prompts: the same measures, with intervals from resampling the texts."""
    rows = []
    for dataset in _ordered(samples):
        s = samples[dataset]
        n = len(s["ids"])
        for model in s["models"]:
            for x, y in COMPARISONS:
                a_score, b_score, a_label, b_label = s["score"][x][model], s["score"][y][model], s["label"][x][model], s["label"][y][model]
                interval = _bootstrap(lambda i: _core(a_score[i], b_score[i], a_label[i], b_label[i]), n, reps, seed)
                rows.append(_row({"dataset": dataset, "model": model, "comparison": f"{x} vs {y}"}, _full(a_score, b_score, a_label, b_label), interval, n, n))
    return pd.DataFrame(rows)


def wording_vs_model(samples: Dict[str, dict]) -> pd.DataFrame:
    """Per dataset: the mean weighted kappa of a model with itself under another wording (over the models), and of two different models under the same
    prompt (over the pairs of models). If a change of wording matters less than a change of model, the first columns are higher than the last."""
    rows = []
    for dataset in _ordered(samples):
        s = samples[dataset]
        row = {"dataset": dataset}
        for x, y in COMPARISONS:
            row[f"same model, {x} vs {y}"] = round(float(np.nanmean([weighted_kappa(s["score"][x][m], s["score"][y][m]) for m in s["models"]])), 3)
        for prompt in RUNS:
            row[f"different models, {prompt}"] = round(float(np.nanmean([weighted_kappa(s["score"][prompt][a], s["score"][prompt][b])
                                                                          for a, b in combinations(s["models"], 2)])), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------------------------------
def _interval_figure(plt, by_model: pd.DataFrame, pooled: pd.DataFrame, measure: str, ylabel: str, ylim):
    """One panel per dataset: the measure for every model and for the four together, one bar per pair of prompts, with the 95% interval."""
    datasets = _ordered(by_model["dataset"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), squeeze=False)
    for ax, dataset in zip(axes.flatten(), datasets):
        models = list(dict.fromkeys(by_model[by_model["dataset"] == dataset]["model"]))
        groups = [(m, by_model[(by_model["dataset"] == dataset) & (by_model["model"] == m)]) for m in models] + [("all models", pooled[pooled["dataset"] == dataset])]
        x, width = np.arange(len(groups)), 0.26
        for i, (x_prompt, y_prompt) in enumerate(COMPARISONS):
            part = [g[g["comparison"] == f"{x_prompt} vs {y_prompt}"].iloc[0] for _, g in groups]
            value = np.array([p[measure] for p in part], dtype=float)
            low, high = np.array([p[f"{measure} low"] for p in part], dtype=float), np.array([p[f"{measure} high"] for p in part], dtype=float)
            ax.bar(x + (i - 1) * width, value, width=width, color=COMPARISON_COLOR[(x_prompt, y_prompt)],
                   yerr=[np.clip(value - low, 0, None), np.clip(high - value, 0, None)], error_kw={"elinewidth": 0.8, "capsize": 2})
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL.get(name, name) if name != "all models" else "all models" for name, _ in groups], fontsize=9)
        ax.set(title=DATASET_LABEL.get(dataset, dataset), ylabel=ylabel, ylim=ylim)
        ax.axhline(0, color="black", linewidth=0.6)
    for ax in axes.flatten()[len(datasets):]:
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COMPARISON_COLOR[c]) for c in COMPARISONS], [f"{x} vs {y}" for x, y in COMPARISONS], loc="lower center", ncol=3)
    return fig


def _wording_figure(plt, table: pd.DataFrame):
    """For every dataset, the mean weighted kappa of a model with itself under another wording (blue, orange) against two different models (grey)."""
    datasets = _ordered(table["dataset"])
    table = table.set_index("dataset").loc[datasets]
    columns = [("same model, baseline vs R1", "#4C72B0", "same model, baseline vs R1"), ("same model, baseline vs R2", "#DD8452", "same model, baseline vs R2"),
               ("different models, baseline", "#888888", "different models, same prompt (baseline)")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x, width = np.arange(len(datasets)), 0.26
    for i, (column, color, _) in enumerate(columns):
        ax.bar(x + (i - 1) * width, table[column], width=width, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets])
    ax.set(ylabel="mean weighted kappa", ylim=(0, 1))
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in columns], [t for _, _, t in columns], loc="lower center", ncol=3, fontsize=8)
    return fig


def _readme(pooled: pd.DataFrame, by_model: pd.DataFrame, wording: pd.DataFrame, sizes: Dict[str, int]) -> str:
    label, kappa = pooled["same label %"], pooled["weighted kappa (scores)"]
    baseline_rows = pooled[pooled["comparison"].str.startswith("baseline")]
    return f"""# Prompt-wording ablation (6.2)

**Question.** Does rewording the question change the ratings of the models? The baseline prompt and two rewordings of it (R1 and R2) were run on the same random
sample of texts of every dataset; this folder compares the answers. The prompts are in `new_results/prompts.md`.

## Design

| | |
|---|---|
| What changed | the wording of the question only |
| What stayed fixed | the task, the 1-5 scale, the rule (1-2 civil, 3-5 toxic), the reply format, the texts, the four models, temperature 0; no definition, example or role was added |
| Prompts | `baseline` (run `baseline`), R1 (run `prompt_sensitivity`), R2 (run `prompt_sensitivity_2`); ten numbered texts per call |
| Sample | a random sample per dataset with seed 42, the same texts for every model and prompt: {", ".join(f"{DATASET_LABEL.get(d, d)} {n}" for d, n in sizes.items())} texts (Attitudes has only 41) |
| Models | ChatGPT `gpt-4o-mini`, DeepSeek `deepseek-chat`, Mistral `open-mistral-7b`, Gemini `gemini-2.5-flash`; all runs on 2026-10-27, the exact snapshot was not recorded |
| Runs | `new_results/run_log.csv`; the replies are in `new_results/<dataset>/<run>/raw/` |
| Each run once | no repetition of the same prompt, so the differences between R1 and R2 are the closest measure of the normal variation between wordings |

## Measures

- **same label %**: the share of (text, model) pairs with the same label (toxic, civil) under the two prompts.
- **Cohen's kappa (labels)**: the agreement of the labels corrected for chance. It is **unreliable when almost every label is the same** (MHS): it can be near 0
  with 95% of labels equal, and it is undefined when both runs give one and the same label to every text.
- **weighted kappa (scores)**: the agreement of the 1-5 scores with quadratic weights (a big disagreement counts more than a small one). 1 is perfect, 0 is chance.
- **Spearman**, **same score %** and **mean diff** (second prompt minus first; above 0 means the second prompt scores higher).
- **Intervals**: 95% percentile bootstrap, 2000 resamples of the texts (the models of a text move together in the pooled rows).

## Files

| file | content |
|---|---|
| `agreement_pooled.csv` | per dataset and pair of prompts, the four models together ({len(pooled)} rows) |
| `agreement_by_model.csv` | per dataset, model and pair of prompts ({len(by_model)} rows) |
| `wording_vs_model.csv` | a model against itself under another wording, against different models under the same prompt |
| `ablation_sample.csv` | the item ids of the sample |
| `ablation_weighted_kappa.pdf/.png`, `ablation_label_agreement.pdf/.png` | the agreement per dataset and model, with intervals |
| `ablation_wording_vs_model.pdf/.png` | a change of wording against a change of model |

## Summary of the numbers

- Pooled over the four models, the share of identical labels is {label.min():.1f}% to {label.max():.1f}% over the {len(pooled)} rows of `agreement_pooled.csv`,
  and the weighted kappa of the scores {kappa.min():.2f} to {kappa.max():.2f}. Against the baseline (R1 and R2 together) the labels agree
  {baseline_rows["same label %"].mean():.1f}% on average and the weighted kappa is {baseline_rows["weighted kappa (scores)"].mean():.2f} on average.
- Mean weighted kappa of a model with itself under another wording (baseline vs R1 and vs R2): {wording[["same model, baseline vs R1", "same model, baseline vs R2"]].to_numpy().min():.2f}
  to {wording[["same model, baseline vs R1", "same model, baseline vs R2"]].to_numpy().max():.2f}; of different models under the baseline prompt: {wording["different models, baseline"].min():.2f}
  to {wording["different models, baseline"].max():.2f}.

## Be careful

- The samples are small (41-50 texts per dataset), so the intervals are wide, especially per model.
- Only rewordings that add no information were tested; a definition of toxicity, examples or a role were not.
- Each wording was run once. Hosted models can differ between days even at temperature 0, and the exact model snapshot was not recorded.
- **MHS:** almost every label is the same class, so the share of equal labels is high (94-100%) while Cohen's kappa is near 0 or undefined for some models; read the
  weighted kappa and the share of equal labels there.
- R2 gives slightly higher scores than the baseline in some datasets (see `mean diff` in the tables): a systematic shift, not only noise.
"""


def save_ablation_prompt(results_dir: Path, folder: Path, reps: int = REPS, seed: int = 42,
                         datasets: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Write the tables, the figures and the README described in the module docstring into ``folder``, from the saved runs in ``results_dir``
    (``<results_dir>/<dataset>/<run>/<model>.csv``). Returns the files written, by file name."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    samples = load_samples(results_dir, datasets or DATASETS)
    samples = {name: samples[name] for name in _ordered(samples)}
    pooled, by_model, wording = agreement_pooled(samples, reps, seed), agreement_by_model(samples, reps, seed), wording_vs_model(samples)
    sample_table = pd.DataFrame([{"dataset": d, "item_id": i} for d, s in samples.items() for i in s["ids"]])
    written = {}
    for file, table in (("agreement_pooled.csv", pooled), ("agreement_by_model.csv", by_model), ("wording_vs_model.csv", wording), ("ablation_sample.csv", sample_table)):
        table.to_csv(folder / file, index=False, encoding="utf-8")
        written[file] = folder / file
    _save_figure(plt, _interval_figure(plt, by_model, pooled, "weighted kappa (scores)", "weighted kappa", (0, 1.05)), folder, "ablation_weighted_kappa", written,
                 formats=("pdf", "png"), rect=(0, 0.05, 1, 1))
    _save_figure(plt, _interval_figure(plt, by_model, pooled, "same label %", "% of identical labels", (0, 105)), folder, "ablation_label_agreement", written,
                 formats=("pdf", "png"), rect=(0, 0.05, 1, 1))
    _save_figure(plt, _wording_figure(plt, wording), folder, "ablation_wording_vs_model", written, formats=("pdf", "png"), rect=(0, 0.07, 1, 1))
    (folder / "README.md").write_text(_readme(pooled, by_model, wording, {d: len(s["ids"]) for d, s in samples.items()}), encoding="utf-8")
    written["README.md"] = folder / "README.md"
    return written
