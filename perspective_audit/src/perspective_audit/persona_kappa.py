"""Kappa between the personas of the SCD prompt and the groups (section 7.3), the full persona-by-group matrix and the agreement between two personas.

In the SCD prompt a model rates a text as a person of each group that is compared in that text, so for every attributed text it has one label per persona.
``compute_persona_kappa`` reads those answers (nothing is asked of a model) and the group positions of ``kappa_groups`` and gives, per dataset, model and characteristic:

* ``matrix``: the kappa of the label of persona P against the position of group G for every pair (P, G). The diagonal (P = G) is the SCD kappa of ``kappa_by_group``;
* ``pairs``: the kappa between the labels of two personas on the same texts;
* ``diagonal``: for every group, whether the diagonal cell is higher than the off-diagonal cells of its column, and how often.

A persona only has a label on the texts where its group is compared, so a cell uses the attributed texts where both its persona and its group are compared
(for the diagonal: the texts of the group, as in ``kappa_by_group``); ``n_texts`` says how many. The resamples are those of ``kappa_groups`` (same seed).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .eda_total import _ordered
from .kappa_groups import MODELS, _attributed_rows, _block, _interval, _kappa, _model_labels, _r, _se
from .persona_annotation import collect_persona_runs
from .pipeline import DatasetResult

HIGH = 0.8                                      # two personas with a kappa above this give (almost) the same labels

README = """# Personas and groups (section 7.3)

Nothing here asks a model for anything: it reads the SCD answers (the model rates a text as a person of each group compared in it) and the group positions of `kappa_by_group.csv`.

| file | content |
|---|---|
| `persona_group_matrix.csv` | per dataset, model, characteristic, persona (the group the model answered as) and group (the group whose position is compared): `n_texts`, `kappa`, `se`, `ci_low`, `ci_high`, `constant_label`. The cells with persona = group are the SCD kappas of `kappa_by_group.csv` |
| `persona_pairs.csv` | per dataset, model, characteristic and pair of personas: the kappa between their labels on the same texts, with the same columns |
| `persona_diagonal_summary.csv` | per dataset, characteristic and model (and `all` models): in how many (persona, group) comparisons the diagonal cell is higher than the off-diagonal cell of its column |

- A persona has a label only on the texts where its group is compared, so a cell uses the attributed texts where both its persona and its group are compared (`n_texts`; 0 gives no value).
- A constant label (the persona gives one label to every text of the cell, `constant_label = True`) gives kappa 0; in `persona_pairs.csv` the flag is set when either persona is constant.
- `se` and the 95% range (`ci_low`, `ci_high`) come from 10,000 resamples of the attributed texts of the characteristic (seed 42, the resamples of `kappa_by_group.csv`).
- `persona_diagonal_summary.csv` compares the point estimates (rounded to four decimals); `diagonal_higher`, `tie` and `diagonal_lower` count the pairs of one diagonal cell and one off-diagonal cell of the same column.
"""


def _cell(boot_weights: np.ndarray, shown: np.ndarray, position: np.ndarray, label: np.ndarray, constant: bool) -> dict:
    """The kappa of ``label`` against ``position`` on the texts ``shown`` with its bootstrap standard error and 95% range (empty cell: no value)."""
    n = int(shown.sum())
    if n == 0:
        return {"n_texts": 0, "kappa": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "constant_label": False}
    point = float(_kappa(shown.astype(float)[None, :], position, label)[0])
    boot = _kappa(boot_weights * shown.astype(float)[None, :], position, label)
    low, high, _ = _interval(boot)
    return {"n_texts": n, "kappa": _r(point), "se": _r(_se(boot)), "ci_low": _r(low), "ci_high": _r(high), "constant_label": constant}


def _constant(label: np.ndarray, shown: np.ndarray) -> bool:
    return bool(shown.any() and len(set(label[shown])) == 1)


def _tables_of_block(dataset: str, dimension: str, block: dict, models: Sequence[str], reps: int, seed: int) -> tuple:
    n, groups, present, position = block["n"], block["groups"], block["present"], block["position"]
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=reps).astype(float)       # the resamples of kappa_groups
    matrix, pairs = [], []
    for model in models:
        persona = block["scd"][model]
        for p, who in enumerate(groups):
            for g, group in enumerate(groups):
                shown = (present[:, p] * present[:, g]).astype(bool)
                matrix.append({"dataset": dataset, "model": model, "characteristic": dimension, "persona": who, "group": group,
                               **_cell(counts, shown, position[:, g], persona[:, p], _constant(persona[:, p], shown))})
        for a, b in combinations(range(len(groups)), 2):
            shown = (present[:, a] * present[:, b]).astype(bool)
            pairs.append({"dataset": dataset, "model": model, "characteristic": dimension, "persona_a": groups[a], "persona_b": groups[b],
                          **_cell(counts, shown, persona[:, a], persona[:, b], _constant(persona[:, a], shown) or _constant(persona[:, b], shown))})
    return matrix, pairs


def diagonal_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, characteristic and model (and ``all`` models): how often the diagonal cell (persona = group) is higher than, equal to or lower than an off-diagonal cell
    of the same column, counted over every group and every other persona with a value, and the mean of the diagonal and of the off-diagonal cells."""
    rows = []
    for (dataset, characteristic), part in matrix.groupby(["dataset", "characteristic"], sort=False):
        cells = {}
        for model, sub in part.groupby("model", sort=False):
            high = tie = low = 0
            diagonal, off = [], []
            for group, column in sub.groupby("group", sort=False):
                d = column[(column["persona"] == group) & column["kappa"].notna()]["kappa"]
                o = column[(column["persona"] != group) & column["kappa"].notna()]["kappa"]
                if d.empty:
                    continue
                diagonal.append(float(d.iloc[0]))
                off += [float(v) for v in o]
                high += int((d.iloc[0] > o).sum())
                tie += int((d.iloc[0] == o).sum())
                low += int((d.iloc[0] < o).sum())
            cells[model] = (high, tie, low, diagonal, off)
        total = tuple(sum(cells[m][i] for m in cells) if i < 3 else sum((cells[m][i] for m in cells), []) for i in range(5))
        for model, (high, tie, low, diagonal, off) in list(cells.items()) + [("all", total)]:
            comparisons = high + tie + low
            rows.append({"dataset": dataset, "characteristic": characteristic, "model": model, "comparisons": comparisons, "diagonal_higher": high, "tie": tie,
                         "diagonal_lower": low, "share_higher_pct": round(100 * high / comparisons, 1) if comparisons else np.nan,
                         "mean_diagonal": _r(np.mean(diagonal)) if diagonal else np.nan, "mean_off_diagonal": _r(np.mean(off)) if off else np.nan})
    return pd.DataFrame(rows)


def compute_persona_kappa(results: Dict[str, DatasetResult], results_dir: Path, given: Optional[pd.DataFrame], reps: int = 10_000, seed: int = 42,
                          models: Sequence[str] = MODELS) -> Dict[str, pd.DataFrame]:
    """``matrix``, ``pairs``, ``diagonal`` (see the module) and ``high_pairs`` (the persona pairs with a kappa above 0.8). ``results`` are the AU results of the datasets,
    ``results_dir`` holds ``<dataset>/scd/<model>.csv`` and ``<dataset>/baseline/<model>.csv``, ``given`` is ``aposteriori_for_models.csv``."""
    scd = collect_persona_runs(results_dir, "scd")
    matrix, pairs = [], []
    for dataset in _ordered(results):
        result = results[dataset]
        rows = _attributed_rows(result, given, dataset)
        labels = _model_labels(results_dir, dataset, models, scd)
        for dimension in sorted(rows["dimension"].unique(), key=list(result.spec.dimensions).index):
            block = _block(rows, dimension, labels, models, dataset)
            for target, part in zip((matrix, pairs), _tables_of_block(dataset, dimension, block, models, reps, seed)):
                target.extend(part)
    matrix, pairs = pd.DataFrame(matrix), pd.DataFrame(pairs)
    return {"matrix": matrix, "pairs": pairs, "diagonal": diagonal_summary(matrix), "high_pairs": pairs[pairs["kappa"] > HIGH].reset_index(drop=True)}


def save_persona_kappa(tables: Dict[str, pd.DataFrame], folder: Path) -> Dict[str, Path]:
    """Write ``persona_group_matrix.csv``, ``persona_pairs.csv`` and ``persona_diagonal_summary.csv`` into ``folder`` (created if needed). Returns the files, by name."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for file, key in (("persona_group_matrix.csv", "matrix"), ("persona_pairs.csv", "pairs"), ("persona_diagonal_summary.csv", "diagonal")):
        tables[key].to_csv(folder / file, index=False, encoding="utf-8")
        written[file] = folder / file
    (folder / "README_personas.md").write_text(README, encoding="utf-8")
    written["README_personas.md"] = folder / "README_personas.md"
    return written
