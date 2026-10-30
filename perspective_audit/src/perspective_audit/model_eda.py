"""A simple description of the models' default ratings (section 7.5): how each model uses the 1-5 scale, and how often it calls a text civil or toxic.

It reads the ratings the models already gave with the default prompt (``<dataset>/baseline/<model>.csv``), all datasets together, and asks nothing of a model.
Two figures: the histogram of the scores of every model (four panels in one figure) and the share of civil and toxic labels of every model (one panel).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from .eda_total import DATASET_ORDER
from .kappa_groups import MODELS, THRESHOLD
from .kappa_tables import MODEL_NAME

SCORES = [1, 2, 3, 4, 5]
CIVIL_COLOR, TOXIC_COLOR = "#0072B2", "#D55E00"
MODEL_COLOR = {"chatgpt": "#0072B2", "deepseek": "#D55E00", "gemini": "#009E73", "mistral": "#CC79A7"}          # Okabe-Ito
WIDTH = 6.3                                   # inches: one text column of the ACM manuscript format


def model_eda(results_dir: Path, datasets: Sequence[str] = DATASET_ORDER, models: Sequence[str] = MODELS, threshold: int = THRESHOLD) -> Dict[str, pd.DataFrame]:
    """``scores``: ratings per model and score (1-5) with the share (%); ``labels``: texts per model that are civil or toxic (a score of ``threshold`` or more is
    toxic) with the share (%). All ``datasets`` together; one rating per text and model (a text given twice counts once)."""
    ratings = []
    for dataset in datasets:
        for model in models:
            path = Path(results_dir) / dataset / "baseline" / f"{model}.csv"
            if not path.exists():
                raise FileNotFoundError(f"no baseline file for {dataset}/{model}: {path}")
            part = pd.read_csv(path, dtype={"item_id": str}).drop_duplicates("item_id")
            ratings.append(pd.DataFrame({"dataset": dataset, "model": model, "score": part["score"].astype(int)}))
    ratings = pd.concat(ratings, ignore_index=True)
    counts = ratings.groupby(["model", "score"]).size().unstack(fill_value=0).reindex(index=list(models), columns=SCORES, fill_value=0)
    scores = counts.stack().rename("ratings").reset_index()
    scores["share_pct"] = (100 * scores["ratings"] / scores.groupby("model")["ratings"].transform("sum")).round(2)
    toxic = counts[[s for s in SCORES if s >= threshold]].sum(axis=1)
    civil = counts.sum(axis=1) - toxic
    labels = pd.DataFrame({"model": list(models), "civil": civil.to_numpy(), "toxic": toxic.to_numpy()})
    labels["civil_pct"] = (100 * labels["civil"] / (labels["civil"] + labels["toxic"])).round(2)
    labels["toxic_pct"] = (100 - labels["civil_pct"]).round(2)
    return {"scores": scores, "labels": labels}


def _common_n(totals: Sequence[int]) -> int:
    """The number of attributed texts every model scored; the figures state it, so it must be the same for all models."""
    assert len(set(int(t) for t in totals)) == 1, f"the models did not score the same number of texts: {list(totals)}"
    return int(totals[0])


def score_histogram_figure(scores: pd.DataFrame, models: Sequence[str] = MODELS):
    """The share of ratings at every score (1-5) for each model: four panels in a 2 x 2 figure with the same axes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.4), sharex=True, sharey=True)
    top = float(scores["share_pct"].max())
    n_texts = _common_n(scores.groupby("model")["ratings"].sum().reindex(list(models)).tolist())
    for ax, model in zip(axes.flatten(), models):
        part = scores[scores["model"] == model].set_index("score").reindex(SCORES)
        ax.bar(SCORES, part["share_pct"], color=MODEL_COLOR.get(model, "#888888"), width=0.7)
        for score, share in zip(SCORES, part["share_pct"]):
            ax.text(score, share + 0.02 * top, f"{share:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(MODEL_NAME.get(model, model), fontsize=9)
        ax.set_xticks(SCORES)
        ax.set_ylim(0, 1.12 * top)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("score", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"% of the {n_texts} texts", fontsize=8)
    fig.suptitle(f"Scores on the {n_texts} attributed texts (default prompt)", fontsize=10)
    fig.tight_layout()
    return fig


def label_share_figure(labels: pd.DataFrame, models: Sequence[str] = MODELS):
    """One panel: for every model a horizontal bar split into the share of civil and toxic labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    names = [MODEL_NAME.get(m, m) for m in models]
    part = labels.set_index("model").reindex(list(models))
    ax.barh(names, part["civil_pct"], color=CIVIL_COLOR, label="civil", height=0.6)
    ax.barh(names, part["toxic_pct"], left=part["civil_pct"], color=TOXIC_COLOR, label="toxic", height=0.6)
    for i, (civil, toxic) in enumerate(zip(part["civil_pct"], part["toxic_pct"])):
        if civil >= 6:
            ax.text(civil / 2, i, f"{civil:.0f}%", ha="center", va="center", color="white", fontsize=8)
        if toxic >= 6:
            ax.text(civil + toxic / 2, i, f"{toxic:.0f}%", ha="center", va="center", color="white", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    n_texts = _common_n((part["civil"] + part["toxic"]).tolist())
    ax.set_xlabel(f"% of the {n_texts} texts labeled (score of 3 or more is toxic)", fontsize=8)
    ax.set_title(f"Labels on the {n_texts} attributed texts (default prompt)", fontsize=10, pad=24)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def save_model_eda(tables: Dict[str, pd.DataFrame], folder: Path, models: Sequence[str] = MODELS) -> Dict[str, Path]:
    """Write ``scores.csv``, ``labels.csv`` and the two figures (``score_histograms``, ``label_shares``; pdf and png) into ``folder`` (created if needed)."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("scores", "labels"):
        tables[name].to_csv(folder / f"{name}.csv", index=False, encoding="utf-8")
        written[f"{name}.csv"] = folder / f"{name}.csv"
    for name, fig in (("score_histograms", score_histogram_figure(tables["scores"], models)), ("label_shares", label_share_figure(tables["labels"], models))):
        for extension in ("pdf", "png"):
            fig.savefig(folder / f"{name}.{extension}", dpi=300, bbox_inches="tight")
            written[f"{name}.{extension}"] = folder / f"{name}.{extension}"
        plt.close(fig)
    return written
