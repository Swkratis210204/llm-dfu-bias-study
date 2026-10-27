"""DFU (distance from unimodality) and a-posteriori unimodality (AU), independent of the dataset.

One text has several ratings on an ordinal scale, and every rating comes from a rater who belongs to one group of
some dimension (gender, race, ...).  A text is

* polarised            when the DFU of all its ratings is > 0;
* a-posteriori unimodal (AU) when it is polarised, at least ``min_groups`` groups have at least ``min_ratings``
  ratings each, and every one of those groups is unimodal (DFU == 0) on its own.

Groups with fewer than ``min_ratings`` ratings are removed from the comparison, the others are kept.  With five
ratings split 2 / 2 / 1 the group of one is removed and the two groups of two are compared.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = ["Unimodality", "PrunedAU", "dfu_of_hist"]


class PrunedAU(NamedTuple):
    """Result of :meth:`Unimodality.au_pruned`."""
    result: pd.DataFrame      # one row per item, from the last round
    history: pd.DataFrame     # one row per round
    removed: list             # groups removed, in order
    counts: dict              # AU texts per remaining group (last round)
    still_thin: list          # remaining groups that are still below the minimum (only if no group could be removed)


def dfu_of_hist(hist) -> float:
    """Normalised DFU of a histogram of relative frequencies (same computation as the thesis notebook)."""
    hist = np.asarray(hist, dtype=float)
    max_value = hist.max()
    pos_max = int(np.where(hist == max_value)[0][0])
    max_diff = 0
    for i in range(pos_max, len(hist) - 1):          # to the right of the mode
        diff = hist[i + 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    for i in range(pos_max, 0, -1):                  # to the left of the mode
        diff = hist[i - 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    return max_diff / max_value


class Unimodality:
    """DFU and AU for one rating scale and one set of rules.

    Parameters
    ----------
    scale : the ordinal rating categories, e.g. ``range(1, 6)``.
    min_ratings : a group needs at least this many ratings to take part in the comparison (default 2).
    min_groups : at least this many groups must remain after the small ones are removed (default 2).
    toxic_threshold : a rating >= threshold counts as toxic.
    label_rule : how a group gets its toxic / civil label.
        ``"majority_then_mean"`` (default): the majority of its ratings; when exactly half are toxic and half
        civil, the group's mean rating decides (mean >= threshold is toxic). Every group gets a label and the
        groups decided by the mean are listed in ``tie_broken``.
        ``"majority"``: the majority of its ratings, ties are labelled ``"tie"`` (the thesis notebook's
        ``extract_examples_with_means``). ``"mean"``: mean rating >= threshold only.
    tolerance : a set of ratings counts as unimodal when its DFU is at most this value (default 0 = exactly
        unimodal). The same value is used for a text to count as polarised (overall DFU > tolerance) and for a
        group to count as unimodal (group DFU <= tolerance). Useful for large groups, where a tiny dip is noise.
    """

    LABEL_RULES = ("majority_then_mean", "majority", "mean")

    def __init__(self, scale=range(1, 6), min_ratings: int = 2, min_groups: int = 2,
                 toxic_threshold: float = 3, label_rule: str = "majority_then_mean", tolerance: float = 0.0):
        if label_rule not in self.LABEL_RULES:
            raise ValueError(f"label_rule must be one of {self.LABEL_RULES}")
        self.tolerance = tolerance
        self.scale = list(scale)
        self.min_ratings = min_ratings
        self.min_groups = min_groups
        self.toxic_threshold = toxic_threshold
        self.label_rule = label_rule

    # ---- DFU -------------------------------------------------------------------------------------
    def pdf(self, ratings) -> np.ndarray:
        """Relative frequency of every category of the scale."""
        ratings = np.asarray(list(ratings))
        if ratings.size == 0:
            raise ValueError("no ratings")
        return np.array([np.count_nonzero(ratings == s) / ratings.size for s in self.scale])

    def dfu(self, ratings) -> float:
        """Distance from unimodality of a set of ratings (0 = unimodal, larger = more polarised)."""
        return dfu_of_hist(self.pdf(ratings))

    def is_polarised(self, ratings) -> bool:
        return self.dfu(ratings) > self.tolerance

    # ---- labels ----------------------------------------------------------------------------------
    def label_info(self, ratings) -> tuple[str, bool]:
        """(label, decided_by_mean) of one group."""
        ratings = list(ratings)
        by_mean = "toxic" if np.mean(ratings) >= self.toxic_threshold else "civil"
        if self.label_rule == "mean":
            return by_mean, False
        toxic = sum(r >= self.toxic_threshold for r in ratings)
        civil = len(ratings) - toxic
        if toxic == civil:                                   # exactly half and half
            return (by_mean, True) if self.label_rule == "majority_then_mean" else ("tie", False)
        return ("toxic" if toxic > civil else "civil"), False

    def label(self, ratings) -> str:
        return self.label_info(ratings)[0]

    # ---- AU of one text --------------------------------------------------------------------------
    def analyse_text(self, ratings, groups) -> dict:
        """AU analysis of one text.

        ``ratings`` and ``groups`` are parallel: ``groups[i]`` is the group of the rater who gave ``ratings[i]``.
        A rating whose group is missing is ignored for the group comparison but still counts in the overall DFU.
        """
        ratings = list(ratings)
        groups = list(groups)
        if len(ratings) != len(groups):
            raise ValueError("ratings and groups differ in length")

        by_group: dict = {}
        for r, g in zip(ratings, groups):
            if g is None or (isinstance(g, float) and np.isnan(g)):
                continue
            by_group.setdefault(g, []).append(r)

        overall = self.dfu(ratings)
        kept = {g: v for g, v in by_group.items() if len(v) >= self.min_ratings}
        dropped = {g: len(v) for g, v in by_group.items() if len(v) < self.min_ratings}
        group_dfu = {g: self.dfu(v) for g, v in kept.items()}

        if overall <= self.tolerance:
            reason = "not polarised"
        elif len(kept) < self.min_groups:
            reason = f"fewer than {self.min_groups} groups with >= {self.min_ratings} ratings"
        elif any(d > self.tolerance for d in group_dfu.values()):
            reason = "a group is not unimodal"
        else:
            reason = "AU"

        keys = sorted(kept, key=str)
        info = {g: self.label_info(kept[g]) for g in keys}
        return {
            "n_ratings": len(ratings),
            "dfu": overall,
            "polarised": overall > self.tolerance,
            "group_sizes": {g: len(v) for g, v in by_group.items()},
            "kept_groups": keys,
            "dropped_groups": dropped,
            "group_dfu": {g: group_dfu[g] for g in keys},
            "group_mean": {g: float(np.mean(kept[g])) for g in keys},
            "group_label": {g: info[g][0] for g in keys},
            "tie_broken": [g for g in keys if info[g][1]],
            "au": reason == "AU",
            "reason": reason,
            "toxic_pole": "{" + ", ".join(f"{g}:{info[g][0]}" for g in keys) + "}",
        }

    # ---- AU of a whole long table ----------------------------------------------------------------
    def analyse(self, df: pd.DataFrame, item_col: str, rating_col: str, group_col: str,
                keep_cols: list[str] | None = None) -> pd.DataFrame:
        """One row per item for one dimension.  ``df`` is a long table: one row per rating.

        ``keep_cols`` are item-level columns (e.g. the text) copied from the first rating of the item.
        """
        rows = []
        for item, sub in df.groupby(item_col, sort=False):
            res = self.analyse_text(sub[rating_col].tolist(), sub[group_col].tolist())
            res[item_col] = item
            for c in keep_cols or []:
                res[c] = sub[c].iloc[0]
            rows.append(res)
        out = pd.DataFrame(rows)
        first = [item_col] + (keep_cols or [])
        return out[first + [c for c in out.columns if c not in first]]

    def au_table(self, df: pd.DataFrame, item_col: str, rating_col: str, group_col: str,
                 keep_cols: list[str] | None = None) -> pd.DataFrame:
        """Only the AU texts of :meth:`analyse`."""
        res = self.analyse(df, item_col, rating_col, group_col, keep_cols)
        return res[res["au"]].reset_index(drop=True)

    def au_pruned(self, df: pd.DataFrame, item_col: str, rating_col: str, group_col: str,
                  min_au_texts: int = 10, keep_cols: list[str] | None = None) -> "PrunedAU":
        """AU analysis of one dimension, removing groups that take part in too few AU texts.

        Round 0 is the full analysis. While a group takes part in fewer than ``min_au_texts`` AU texts, the
        smallest such group is removed (its raters get a missing group, so their ratings still count in the overall
        DFU but not in the group comparison) and the analysis is repeated: removing a group can turn other texts
        into AU texts or take them out, so the counts are recomputed every round. Stops when every remaining group
        reaches ``min_au_texts``, when only ``min_groups`` groups are left, or when all groups are below the minimum
        (nothing is removed then; ``still_thin`` lists them).
        """
        work = df
        removed: list = []
        history: list[dict] = []
        while True:
            result = self.analyse(work, item_col, rating_col, group_col, keep_cols)
            au = result[result["au"]]
            counts = {g: 0 for g in work[group_col].dropna().unique()}
            for labels in au["group_label"]:
                for g in labels:
                    counts[g] += 1
            history.append({"round": len(history), "group removed before this round": removed[-1] if removed else None,
                            "AU texts": len(au), "groups": len(counts), "AU texts per group": dict(sorted(counts.items(), key=str))})
            thin = sorted((c, str(g), g) for g, c in counts.items() if c < min_au_texts)
            # Stop when nothing is thin, when only min_groups groups are left, or when EVERY group is thin:
            # then no group is a better reference than another and removing one would be arbitrary.
            if not thin or len(counts) <= self.min_groups or len(thin) == len(counts):
                break
            worst = thin[0][2]
            removed.append(worst)
            work = work.assign(**{group_col: work[group_col].where(work[group_col] != worst)})
        return PrunedAU(result=result, history=pd.DataFrame(history), removed=removed, counts=counts,
                        still_thin=[g for g, c in counts.items() if c < min_au_texts])

    @staticmethod
    def reason_counts(result: pd.DataFrame) -> pd.Series:
        """How many items ended in each state (AU, not polarised, ...)."""
        return result["reason"].value_counts()
