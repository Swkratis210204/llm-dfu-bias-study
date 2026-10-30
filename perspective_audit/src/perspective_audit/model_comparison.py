"""How much the models agree with each other (section 7.4), as a diagnostic: Spearman correlation of their scores and Cohen's kappa of their labels.

It reads the ratings the models already gave with the default prompt (``<dataset>/baseline/<model>.csv``) and asks nothing of a model. For every dataset and every pair
of models the two sets of ratings of the texts that both rated are compared; the result is a table of pairs and a heatmap per dataset and measure.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .agreement import cohen_kappa, spearman
from .eda_total import DATASET_LABEL, DATASET_ORDER
from .kappa_groups import MODELS, THRESHOLD, _r
from .kappa_tables import MODEL_NAME

MEASURES = {"spearman": "Spearman correlation", "kappa": "Cohen's kappa"}

README = """# Agreement between the models (section 7.4)

A diagnostic: do the models rate the same texts alike? Nothing here asks a model for anything: it reads the ratings they gave with the default prompt
(`<dataset>/baseline/<model>.csv`, the texts given to the models: for Kumar the sample).

| file | content |
|---|---|
| `pairwise.csv` | one row per dataset and pair of models: `n_texts` (rated by both), `spearman`, `kappa`, `agreement_pct`, `pct_toxic_a`, `pct_toxic_b`, `constant_a`, `constant_b` |
| `matrices.csv` | the same numbers as one matrix per dataset and measure (the heatmaps): rows and columns are the models, the diagonal is 1 |
| `model_agreement.pdf`, `.png` | the heatmaps: one column per dataset, Spearman on the top row and Cohen's kappa on the bottom row |

- **Spearman**: the rank correlation of the 1-5 scores of the two models (do they put the texts in the same order?). It has no value when a model gives the same score to every text.
- **Cohen's kappa**: the agreement of the two binary labels (a score of 3 or more is toxic) beyond chance. It is 0 when only one of the models gives one label to every text,
  and has no value when both give the same single label to every text.
- **agreement_pct**: the share of texts with the same label.
- The diagonal of the heatmaps is 1 by definition. A cell without a value is shown as `n/a`.
- No confidence intervals: this is a diagnostic of the models, not a test.
"""


def _scores(results_dir: Path, dataset: str, models: Sequence[str]) -> Dict[str, pd.Series]:
    out = {}
    for model in models:
        path = Path(results_dir) / dataset / "baseline" / f"{model}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no default ratings of {model} for {dataset}: {path}")
        out[model] = pd.read_csv(path, dtype={"item_id": str}).drop_duplicates("item_id").set_index("item_id")["score"]
    return out


def compare_models(results_dir: Path, datasets: Sequence[str] = DATASET_ORDER, models: Sequence[str] = MODELS, threshold: int = THRESHOLD) -> Dict[str, pd.DataFrame]:
    """The tables of section 7.4: ``pairwise`` (one row per dataset and pair of models) and ``matrices`` (the same numbers as a matrix per dataset and measure, diagonal 1).
    ``results_dir`` holds ``<dataset>/baseline/<model>.csv``; a score at or above ``threshold`` is toxic."""
    pairs, matrices = [], []
    for dataset in datasets:
        scores = _scores(results_dir, dataset, models)
        values = {m: {measure: pd.Series(np.nan, index=models) for measure in MEASURES} for m in models}
        for m in models:
            for measure in MEASURES:
                values[m][measure][m] = 1.0
        for a, b in combinations(models, 2):
            both = scores[a].index.intersection(scores[b].index)
            x, y = scores[a].loc[both], scores[b].loc[both]
            label_x, label_y = (x >= threshold).to_numpy(), (y >= threshold).to_numpy()
            spear, kappa = spearman(x, y), cohen_kappa(label_x, label_y)
            pairs.append({"dataset": dataset, "model_a": a, "model_b": b, "n_texts": int(len(both)), "spearman": _r(spear), "kappa": _r(kappa),
                          "agreement_pct": round(100 * float(np.mean(label_x == label_y)), 2), "pct_toxic_a": round(100 * float(label_x.mean()), 2),
                          "pct_toxic_b": round(100 * float(label_y.mean()), 2), "constant_a": bool(len(set(label_x)) == 1), "constant_b": bool(len(set(label_y)) == 1)})
            for measure, value in (("spearman", spear), ("kappa", kappa)):
                values[a][measure][b] = values[b][measure][a] = _r(value)
        for measure in MEASURES:
            for m in models:
                matrices.append({"dataset": dataset, "measure": measure, "model": m, **values[m][measure].to_dict()})
    return {"pairwise": pd.DataFrame(pairs), "matrices": pd.DataFrame(matrices)}


def model_comparison_figure(tables: Dict[str, pd.DataFrame], models: Sequence[str] = MODELS):
    """One heatmap per dataset (columns) and measure (rows: Spearman on top, Cohen's kappa below), the diagonal 1, a shared colorbar (viridis, from 0 to 1, or from -1 when a value is negative)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from .kappa_figures import FONT, MINUS, WIDTH, _text_color

    matrices, pairwise = tables["matrices"], tables["pairwise"]
    datasets = list(dict.fromkeys(matrices["dataset"]))
    cmap = plt.get_cmap("viridis")                                       # purple - blue - green - yellow: distinct hues, colorblind-safe, readable in grey
    values = matrices[list(models)].to_numpy(dtype=float)
    low = -1.0 if np.nanmin(values) < 0 else 0.0                         # the scale starts at 0 unless a value is negative
    norm = Normalize(low, 1)
    names = [MODEL_NAME[m] for m in models]
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": FONT, "axes.unicode_minus": True}):
        fig, axes = plt.subplots(len(MEASURES), len(datasets), figsize=(WIDTH, 1.55 * len(MEASURES) + 0.9), squeeze=False)
        for r, measure in enumerate(MEASURES):
            for c, dataset in enumerate(datasets):
                ax = axes[r][c]
                part = matrices[(matrices["dataset"] == dataset) & (matrices["measure"] == measure)].set_index("model").loc[list(models), list(models)]
                for i in range(len(models)):
                    for j in range(len(models)):
                        value = part.iloc[i, j]
                        color = (0.88, 0.88, 0.88, 1.0) if pd.isna(value) else cmap(norm(value))
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=0.6))
                        text = "n/a" if pd.isna(value) else ("1" if i == j else (MINUS + f"{abs(round(value, 2)):.2f}" if round(value, 2) < 0 else f"{round(value, 2):.2f}"))
                        ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=FONT - 1, color=_text_color(color))
                ax.set_xlim(0, len(models))
                ax.set_ylim(len(models), 0)
                ax.set_aspect("equal")
                ax.set_xticks([j + 0.5 for j in range(len(models))])
                ax.set_yticks([i + 0.5 for i in range(len(models))])
                ax.set_xticklabels(names if r == len(MEASURES) - 1 else [], rotation=45, ha="right", fontsize=FONT - 1)
                ax.set_yticklabels(names if c == 0 else [], fontsize=FONT - 1)
                ax.tick_params(length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    n = int(pairwise[pairwise["dataset"] == dataset]["n_texts"].min())
                    ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)}\n(n = {n})", fontsize=FONT)
                if c == 0:
                    ax.annotate(MEASURES[measure], xy=(0, 0.5), xycoords="axes fraction", xytext=(-52, 0), textcoords="offset points", rotation=90, ha="center", va="center",
                                fontweight="bold", fontsize=FONT)
        fig.subplots_adjust(left=0.12, right=0.9, top=0.85, bottom=0.2, wspace=0.08, hspace=0.12)
        cax = fig.add_axes([0.92, 0.2, 0.015, 0.65])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[low, 0, 1] if low < 0 else [0, 0.5, 1])
        bar.ax.set_yticklabels([MINUS + "1", "0", "1"] if low < 0 else ["0", "0.5", "1"])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, labelsize=FONT - 1)
    return fig


def save_model_comparison(tables: Dict[str, pd.DataFrame], folder: Path, models: Sequence[str] = MODELS) -> Dict[str, Path]:
    """Write ``pairwise.csv``, ``matrices.csv``, ``model_agreement.pdf``/``.png`` (300 dpi) and the README into ``folder`` (created if needed). Returns the files, by name."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for file, key in (("pairwise.csv", "pairwise"), ("matrices.csv", "matrices")):
        tables[key].to_csv(folder / file, index=False, encoding="utf-8")
        written[file] = folder / file
    fig = model_comparison_figure(tables, models)
    for extension in ("pdf", "png"):
        fig.savefig(folder / f"model_agreement.{extension}", dpi=300, bbox_inches="tight")
        written[f"model_agreement.{extension}"] = folder / f"model_agreement.{extension}"
    plt.close(fig)
    (folder / "README.md").write_text(README, encoding="utf-8")
    written["README.md"] = folder / "README.md"
    return written
