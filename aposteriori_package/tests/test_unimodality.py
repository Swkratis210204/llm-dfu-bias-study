"""Tests of the AU tool. They need no dataset."""
from collections import Counter

import numpy as np
import pytest

from aposteriori import Unimodality


def dfu_thesis(scores, scale):
    """The DFU function of ThesisCodef.ipynb, copied only to compare."""
    freqs = Counter(scores)
    hist = np.array([freqs[s] / len(scores) for s in scale])
    max_value = max(hist)
    pos_max = np.where(hist == max_value)[0][0]
    max_diff = 0
    for i in range(pos_max, len(hist) - 1):
        max_diff = max(max_diff, hist[i + 1] - hist[i])
    for i in range(pos_max, 0, -1):
        max_diff = max(max_diff, hist[i - 1] - hist[i])
    return max_diff / max_value


@pytest.mark.parametrize("scale_max", [3, 5])
def test_dfu_equals_the_thesis_function(scale_max):
    rng = np.random.default_rng(0)
    u = Unimodality(scale=range(1, scale_max + 1))
    for _ in range(3000):
        ratings = rng.integers(1, scale_max + 1, size=int(rng.integers(2, 40))).tolist()
        assert u.dfu(ratings) == dfu_thesis(ratings, range(1, scale_max + 1))


def test_two_two_one_removes_the_single_rating_and_keeps_the_text():
    r = Unimodality().analyse_text([1, 1, 5, 5, 3], ["a", "a", "b", "b", "c"])
    assert r["au"] and r["kept_groups"] == ["a", "b"] and r["dropped_groups"] == {"c": 1}


def test_three_one_is_not_au():
    r = Unimodality().analyse_text([1, 1, 1, 5], ["a", "a", "a", "b"])
    assert not r["au"] and r["reason"].startswith("fewer than 2 groups")


def test_not_polarised_text_is_not_au():
    assert Unimodality().analyse_text([3, 3, 3, 3], ["a", "a", "b", "b"])["reason"] == "not polarised"


def test_a_group_that_is_not_unimodal_blocks_au():
    r = Unimodality().analyse_text([1, 5, 5, 5, 1, 5, 1], ["a", "a", "a", "b", "b", "b", "b"])
    assert r["reason"] == "a group is not unimodal"


def test_missing_group_counts_in_the_overall_dfu_only():
    u = Unimodality()
    r = u.analyse_text([1, 1, 5, 5, 5], ["a", "a", "b", "b", None])
    assert r["au"] and r["dfu"] == u.dfu([1, 1, 5, 5, 5])


def test_tie_is_decided_by_the_mean():
    u = Unimodality()
    assert u.label_info([2, 3]) == ("civil", True)
    assert u.label_info([3, 3, 2]) == ("toxic", False)
    assert Unimodality(label_rule="majority").label([2, 3]) == "tie"


def test_tolerance_applies_to_polarisation_and_to_groups():
    ratings = [1] * 18 + [2] + [3, 3]                 # a tiny dip at 2: DFU = (2/21 - 1/21) / (18/21) = 0.056
    u0, u1 = Unimodality(scale=range(1, 4)), Unimodality(scale=range(1, 4), tolerance=0.1)
    assert u0.dfu(ratings) > 0 and u0.is_polarised(ratings)
    assert u1.dfu(ratings) < 0.1 and not u1.is_polarised(ratings)


def test_pruning_does_not_remove_anything_when_every_group_is_thin():
    import pandas as pd
    rows = []
    for i in range(3):                                # 3 AU items, three groups, all groups far below 10
        for g, v in (("a", 1), ("a", 1), ("b", 5), ("b", 5), ("c", 3), ("c", 3)):
            rows.append((i, v, g))
    df = pd.DataFrame(rows, columns=["item", "rating", "group"])
    pr = Unimodality().au_pruned(df, "item", "rating", "group", min_au_texts=10)
    assert pr.removed == [] and set(pr.still_thin) == {"a", "b", "c"}
