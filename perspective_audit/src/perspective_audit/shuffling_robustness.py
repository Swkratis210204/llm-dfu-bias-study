"""Permutation null for the attribution rates (section 7.7): how much of the AU attribution rate of a characteristic (Table 2) and of the gap between non-demographic
and demographic characteristics (Table 3) is chance?

A characteristic with more groups gives smaller groups per text, and a small group passes the unimodality test more easily. For every characteristic the null keeps
everything but the link between a rating and the group of its rater:

* the texts are the polarized texts with at least four ratings (``attribution_bootstrap.attribution_table``);
* for every such text the ratings of the **retained groups** of the characteristic (the groups left after the removal of thin groups, Appendix D.2) are shuffled
  among the raters of those groups, so the ratings of the text and the size of every group on the text stay the same; the removal rule is not run again;
* the AU test is run again with the rules of ``Unimodality.analyse_text`` (a group needs at least two ratings, at least two such groups, every such group has DFU = 0;
  the text is polarized overall by construction). The test is vectorised: DFU = 0 of a histogram is "no rise after the first maximum and no fall before it", which is
  what ``ndfu.dfu`` computes, and the vectorised test is checked against the observed AU texts before any permutation;
* every characteristic is shuffled independently of the others (its own random stream from ``seed``).

The group sizes do not change, so the testable texts of a permutation are the observed ones (asserted). The chance probability of a (text, characteristic) pair is the
share of permutations in which it is attributed; the chance rate of a characteristic is the mean over permutations of attributed / testable.

Nothing here asks a model for anything and nothing changes the AU results: it reads them.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .attribution_bootstrap import COLUMNS, DEMOGRAPHIC, attribution_gap, attribution_table
from .eda_total import DATASET_LABEL, _ordered
from .pipeline import DatasetResult

__all__ = ["compute_null", "save_null"]

PERMUTATIONS = 1_000
KUMAR_PERMUTATIONS = 200                       # used for Kumar only when 1,000 would take too long (see ``slow_seconds``)
SLOW_SECONDS = 1_800                           # Kumar uses KUMAR_PERMUTATIONS when the estimated time of its 1,000 permutations exceeds this
MIN_RATINGS, MIN_GROUPS = 2, 2


def _unimodal(counts: np.ndarray) -> np.ndarray:
    """DFU == 0 for histograms ``counts`` (..., V), as ``ndfu.dfu``: no rise after the first maximum and no fall before it (equal counts are not a rise or a fall)."""
    diff = np.diff(counts, axis=-1)
    peak = np.argmax(counts, axis=-1)[..., None]
    steps = np.arange(counts.shape[-1] - 1)
    rise_after = ((steps >= peak) & (diff > 0)).any(axis=-1)
    fall_before = ((steps < peak) & (diff < 0)).any(axis=-1)
    return ~(rise_after | fall_before)


def _au(sorted_ratings: np.ndarray, group: np.ndarray, n_groups: int, scale: int) -> tuple:
    """For every text: (testable, AU) from the ratings (T x M, in the order of ``group``, the local group index of every slot, -1 for padding)."""
    member = group[:, :, None] == np.arange(n_groups)[None, None, :]                                   # T x M x G
    onehot = sorted_ratings[:, :, None] == np.arange(1, scale + 1)[None, None, :]                      # T x M x V
    counts = np.einsum("tmg,tmv->tgv", member.astype(np.float32), onehot.astype(np.float32)).astype(np.int16)
    eligible = counts.sum(axis=2) >= MIN_RATINGS
    testable = eligible.sum(axis=1) >= MIN_GROUPS
    return testable, testable & (~eligible | _unimodal(counts)).all(axis=1)                          # the text is polarized overall by construction


def _arrays(result: DatasetResult, dimension: str, items: pd.Index) -> dict:
    """The ratings of the retained groups of ``dimension`` of the texts ``items``, one row per text: ratings, the local group index of every slot (sorted), padding."""
    pr = result.pruned[dimension]
    ratings = result.clean[["item_id", "rating", dimension]]
    ratings = ratings[ratings["item_id"].isin(items) & ~ratings[dimension].isin(pr.removed)].copy()
    codes = {g: i for i, g in enumerate(sorted(pr.counts, key=str))}
    ratings["group"] = ratings[dimension].map(codes)
    assert ratings["group"].notna().all(), f"{result.spec.name}/{dimension}: a rating of an unknown group"
    ratings = ratings.sort_values(["item_id", "group"], kind="stable")
    ratings["slot"] = ratings.groupby("item_id").cumcount()
    row = pd.Series(np.arange(len(items)), index=items)
    r, m = row.loc[ratings["item_id"]].to_numpy(), ratings["slot"].to_numpy()
    values = np.zeros((len(items), m.max() + 1), dtype=np.int16)
    group = np.full(values.shape, -1, dtype=np.int16)
    scores = ratings["rating"].to_numpy()
    assert np.array_equal(scores, scores.astype(int)), "the ratings are not whole numbers"
    values[r, m], group[r, m] = scores.astype(int), ratings["group"].to_numpy()
    return {"values": values, "group": group, "n_groups": len(codes)}


def _observed_check(result: DatasetResult, dimension: str, items: pd.Index, arrays: dict) -> None:
    """The vectorised test on the observed assignment must give the AU texts of the results, and the sizes the observed testable texts."""
    testable, au = _au(arrays["values"], arrays["group"], arrays["n_groups"], result.spec.scale_max)
    observed = result.pruned[dimension].result.set_index("item_id").reindex(items)
    assert testable.all(), f"{result.spec.name}/{dimension}: a testable text is not testable in the arrays"
    assert np.array_equal(au, observed["au"].to_numpy(dtype=bool)), f"{result.spec.name}/{dimension}: the vectorised AU test differs from the AU results"


def _permute(arrays: dict, rng: np.random.Generator) -> np.ndarray:
    """The ratings of every text shuffled among its slots: valid slots stay valid, so the group sizes are unchanged."""
    values, group = arrays["values"], arrays["group"]
    keys = rng.random(values.shape)
    keys[group < 0] = np.inf                                       # padding goes last
    return np.take_along_axis(values, np.argsort(keys, axis=1), axis=1)


def _null_one(result: DatasetResult, dimension: str, items: pd.Index, permutations: int, rng: np.random.Generator) -> dict:
    arrays = _arrays(result, dimension, items)
    _observed_check(result, dimension, items, arrays)
    scale, n_groups = result.spec.scale_max, arrays["n_groups"]
    attributed_pairs = np.zeros(len(items))
    totals = np.zeros(permutations, dtype=int)
    for p in range(permutations):
        testable, au = _au(_permute(arrays, rng), arrays["group"], n_groups, scale)
        assert testable.all(), f"{result.spec.name}/{dimension}: the testable texts changed in a permutation"
        attributed_pairs += au
        totals[p] = au.sum()
    return {"chance_probability": attributed_pairs / permutations, "attributed_per_permutation": totals}


def _slow_estimate(result: DatasetResult, items_by_dim: dict, rng_seed: int) -> float:
    """Seconds that 1,000 permutations of all characteristics would take, from 5 permutations of each."""
    start = time.perf_counter()
    for dimension, items in items_by_dim.items():
        _null_one(result, dimension, items, 5, np.random.default_rng(rng_seed))
    return (time.perf_counter() - start) / 5 * PERMUTATIONS


def compute_null(results: Dict[str, DatasetResult], permutations: int = PERMUTATIONS, reps: int = 10_000, seed: int = 42, slow_seconds: float = SLOW_SECONDS,
                 log=print) -> Dict[str, pd.DataFrame]:
    """``per_characteristic``, ``gap``, ``per_pair``, ``correlation`` and ``checks``. Reproduces the observed rates and gap of ``attribution_gap`` first and stops when
    they differ. A dataset whose estimated time for ``permutations`` permutations exceeds ``slow_seconds`` uses ``KUMAR_PERMUTATIONS`` instead (Kumar only)."""
    reference_gap = attribution_gap(results, reps=reps, seed=seed)
    per_char, pairs, gap_rows, checks, used = [], [], [], [], {}
    for d_index, dataset in enumerate(_ordered(results)):
        result = results[dataset]
        table = attribution_table(result)
        dims = list(result.pruned)
        items_by_dim = {}
        for dimension in dims:
            pr_rows = result.pruned[dimension].result.set_index("item_id")
            testable = (pr_rows["polarised"] & (pr_rows["kept_groups"].map(len) >= MIN_GROUPS)).reindex(table.index, fill_value=False)
            items_by_dim[dimension] = table.index[testable.to_numpy()]
        n_perm = permutations
        if dataset == "kumar" and permutations > KUMAR_PERMUTATIONS:
            estimate = _slow_estimate(result, items_by_dim, seed)
            log(f"{dataset}: estimated {estimate / 60:.1f} min for {permutations} permutations")
            if estimate > slow_seconds:
                n_perm = KUMAR_PERMUTATIONS
        used[dataset] = n_perm
        chance = {"D": np.zeros(len(table)), "N": np.zeros(len(table))}
        for dimension in dims:
            items = items_by_dim[dimension]
            kind = "D" if dimension in DEMOGRAPHIC else "N"
            started = time.perf_counter()
            null = _null_one(result, dimension, items, n_perm, np.random.default_rng([seed, d_index, dims.index(dimension)]))
            log(f"{dataset}/{dimension}: {len(items)} testable texts, {n_perm} permutations, {time.perf_counter() - started:.1f} s")
            chance[kind][table.index.get_indexer(items)] += null["chance_probability"]
            observed_au = result.pruned[dimension].result.set_index("item_id")["au"].reindex(items).to_numpy(dtype=bool)
            observed = int(observed_au.sum())
            percent = 100 * null["attributed_per_permutation"] / len(items)
            mean = float(percent.mean())
            per_char.append({"dataset": dataset, "characteristic": dimension, "type": kind, "n_groups": len(result.pruned[dimension].counts), "testable": len(items),
                             "observed_attributed": observed, "observed_pct": round(100 * observed / len(items), 2), "chance_mean_pct": round(mean, 2),
                             "chance_ci_low_pct": round(float(np.percentile(percent, 2.5)), 2), "chance_ci_high_pct": round(float(np.percentile(percent, 97.5)), 2),
                             "excess_pct": round(100 * observed / len(items) - mean, 2), "ratio": round(100 * observed / len(items) / mean, 3) if mean > 0 else np.nan,
                             "p": (1 + int((null["attributed_per_permutation"] >= observed).sum())) / (1 + n_perm), "permutations": n_perm})
            pairs.append(pd.DataFrame({"dataset": dataset, "characteristic": dimension, "item_id": items, "chance_probability": null["chance_probability"],
                                       "observed_attributed": observed_au}))
        mine = pd.DataFrame([r for r in per_char if r["dataset"] == dataset]).groupby("type")[["testable", "observed_attributed"]].sum()
        for kind in ("D", "N"):                                    # Table 2 per characteristic adds up to the per-text table behind Table 3
            assert int(mine.loc[kind, "testable"]) == int(table[f"testable {kind}"].sum()) and int(mine.loc[kind, "observed_attributed"]) == int(table[f"attributed {kind}"].sum()),                 f"{dataset}: the testable or attributed counts of type {kind} differ from the table behind Table 3"
        gap_rows.append(_gap(dataset, table, chance, n_perm, reps, seed, reference_gap.loc[dataset]))
        checks.append({"dataset": dataset, "polarized texts with >= 4 ratings": len(table), "permutations": n_perm})
    per_char = pd.DataFrame(per_char)
    corr = pd.DataFrame([{"measure": f"n_groups vs {name}", "spearman": round(float(spearmanr(per_char["n_groups"], per_char[column]).statistic), 3), "characteristics": len(per_char)}
                         for name, column in (("observed %", "observed_pct"), ("chance %", "chance_mean_pct"))])
    return {"per_characteristic": per_char, "gap": pd.DataFrame(gap_rows), "per_pair": pd.concat(pairs, ignore_index=True), "correlation": corr, "checks": pd.DataFrame(checks)}


def _gap(dataset: str, table: pd.DataFrame, chance: dict, n_perm: int, reps: int, seed: int, reference: pd.Series) -> dict:
    """The observed gap and the gap in excess of chance, with the bootstrap of the texts of Table 3 (``bootstrap_gap``: the same generator and draws)."""
    values = np.column_stack([table[COLUMNS].to_numpy(dtype=float), chance["D"], chance["N"]])        # testable D, attributed D, testable N, attributed N, chance D, chance N
    n = len(values)
    rng = np.random.default_rng(seed)
    gaps, excess = np.empty(reps), np.empty(reps)
    for b in range(reps):
        s = np.bincount(rng.integers(0, n, n), minlength=n) @ values
        with np.errstate(divide="ignore", invalid="ignore"):
            obs_d, obs_n, ch_d, ch_n = s[1] / s[0], s[3] / s[2], s[4] / s[0], s[5] / s[2]
        gaps[b], excess[b] = 100 * (obs_n - obs_d), 100 * ((obs_n - ch_n) - (obs_d - ch_d))
    total = values.sum(axis=0)
    obs_d, obs_n, ch_d, ch_n = total[1] / total[0], total[3] / total[2], total[4] / total[0], total[5] / total[2]
    finite, finite_excess = gaps[np.isfinite(gaps)], excess[np.isfinite(excess)]
    row = {"dataset": dataset, "texts": n, "testable_D": int(total[0]), "attributed_D": int(total[1]), "testable_N": int(total[2]), "attributed_N": int(total[3]),
           "observed_rate_D_pct": round(100 * obs_d, 2), "observed_rate_N_pct": round(100 * obs_n, 2), "chance_rate_D_pct": round(100 * ch_d, 2), "chance_rate_N_pct": round(100 * ch_n, 2),
           "observed_gap_pp": round(100 * (obs_n - obs_d), 2), "gap_ci_low": round(float(np.percentile(finite, 2.5)), 2), "gap_ci_high": round(float(np.percentile(finite, 97.5)), 2),
           "gap_share_le0_pct": round(100 * float(np.mean(finite <= 0)), 2),
           "excess_gap_pp": round(100 * ((obs_n - ch_n) - (obs_d - ch_d)), 2), "excess_ci_low": round(float(np.percentile(finite_excess, 2.5)), 2),
           "excess_ci_high": round(float(np.percentile(finite_excess, 97.5)), 2), "excess_share_le0_pct": round(100 * float(np.mean(finite_excess <= 0)), 2), "permutations": n_perm}
    same = (row["observed_rate_D_pct"], row["observed_rate_N_pct"], row["observed_gap_pp"], row["gap_ci_low"], row["gap_ci_high"], round(row["gap_share_le0_pct"], 2))
    paper = (reference["rate D %"], reference["rate N %"], reference["N - D (pp)"], reference["95% interval low"], reference["95% interval high"],
             reference["share of resamples with N - D <= 0 (%)"])
    if not np.allclose(same, paper, atol=1e-9):
        raise ValueError(f"{dataset}: Table 3 is not reproduced: {same} against {tuple(paper)}")
    return row


# ---------------------------------------------------------------------------------------------------------------------------------------------
def _p(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else "<0.0001"


def summary_text(tables: Dict[str, pd.DataFrame], reps: int, seed: int) -> str:
    c, g = tables["per_characteristic"], tables["gap"]
    L = ["# Permutation null for the attribution rates", "",
         f"Chance = the AU test rerun after shuffling, on every polarized text with at least four ratings, the group labels among the ratings of the retained groups of the "
         f"characteristic (group sizes and the text's ratings unchanged; thin groups not removed again; every characteristic shuffled independently, seed {seed}). "
         "p = (1 + permutations with attributed >= observed) / (1 + permutations). Ranges are 2.5–97.5 percentiles. D = demographic (gender, race, age), N = non-demographic. "
         "The observed testable counts, attributed counts and rates, the pooled rates and the gap with its range reproduce Tables 2 and 3 (checked against `attribution_gap`; the "
         "testable texts of every permutation equal the observed ones, asserted).", ""]
    for dataset in _ordered(c["dataset"].unique()):
        part = c[c["dataset"] == dataset]
        L += [f"## {DATASET_LABEL.get(dataset, dataset)} ({int(part['permutations'].iloc[0])} permutations)", "",
              "| characteristic | type | n groups | testable | observed attributed | observed % | chance % | chance range | excess (pp) | ratio | p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            L.append(f"| {r['characteristic']} | {r['type']} | {r['n_groups']} | {r['testable']} | {r['observed_attributed']} | {r['observed_pct']:.1f} | {r['chance_mean_pct']:.1f} | "
                     f"[{r['chance_ci_low_pct']:.1f}, {r['chance_ci_high_pct']:.1f}] | {r['excess_pct']:+.1f} | {r['ratio']:.2f} | {_p(r['p'])} |")
        L.append("")
    L += ["## The gap (non-demographic minus demographic, percentage points)", "",
          f"Bootstrap: {reps:,} resamples of the polarized texts (seed {seed}, as Table 3). In every resample the observed rates are recomputed and the chance rates are the mean of the "
          "per-pair chance probabilities of the resampled pairs.", "",
          "| dataset | observed D % | observed N % | observed gap [range] | share <= 0 | chance D % | chance N % | excess gap [range] | share <= 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {DATASET_LABEL.get(r['dataset'], r['dataset'])} | {r['observed_rate_D_pct']:.1f} | {r['observed_rate_N_pct']:.1f} | {r['observed_gap_pp']:+.1f} "
                 f"[{r['gap_ci_low']:+.1f}, {r['gap_ci_high']:+.1f}] | {r['gap_share_le0_pct']:.1f}% | {r['chance_rate_D_pct']:.1f} | {r['chance_rate_N_pct']:.1f} | "
                 f"{r['excess_gap_pp']:+.1f} [{r['excess_ci_low']:+.1f}, {r['excess_ci_high']:+.1f}] | {r['excess_share_le0_pct']:.1f}% |")
    L += ["", "## Spearman correlation over all characteristics and datasets", "", "| measure | Spearman | characteristics |", "|---|---|---|"]
    L += [f"| {r['measure']} | {r['spearman']:+.3f} | {r['characteristics']} |" for _, r in tables["correlation"].iterrows()]
    L += ["", "## Permutations used", "", "| dataset | permutations |", "|---|---|"] + [f"| {r['dataset']} | {r['permutations']} |" for _, r in tables["checks"].iterrows()]
    low = c[c["testable"] < 15]
    L += ["", "## Caveats", "",
          "- Chance is computed with the groups retained by the paper's 10-text removal rule; the rule is not rerun inside the permutations.",
          "- A permutation keeps every group's size on every text, so a text with a group of one rating is never made testable or untestable; only which ratings fall in which group changes.",
          f"- Characteristics with fewer than 15 testable texts (their chance range and p are coarse): {', '.join(f'{r.dataset}/{r.characteristic} ({r.testable})' for r in low.itertuples()) or 'none'}.",
          "- The per-pair chance probabilities are in `per_pair_chance_probability.csv` (one row per testable text and characteristic).", ""]
    return "\n".join(L)


def save_null(tables: Dict[str, pd.DataFrame], folder: Path, reps: int = 10_000, seed: int = 42) -> Dict[str, Path]:
    """Write the three csv files and ``summary.md`` into ``folder`` (created if needed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for file, key in (("attribution_null_per_characteristic.csv", "per_characteristic"), ("attribution_null_gap.csv", "gap"), ("per_pair_chance_probability.csv", "per_pair")):
        tables[key].to_csv(folder / file, index=False, encoding="utf-8")
        written[file] = folder / file
    (folder / "summary.md").write_text(summary_text(tables, reps, seed), encoding="utf-8")
    written["summary.md"] = folder / "summary.md"
    return written
