"""Tests of the agreement measures. The expected values were computed with scikit-learn and scipy (not dependencies of the package)."""
import math

import pytest

from perspective_audit import cohen_kappa, spearman, weighted_kappa


def test_cohen_kappa_of_labels():
    assert cohen_kappa(["toxic", "civil", "toxic"], ["toxic", "civil", "toxic"]) == 1
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)               # 75% agree, 50% would by chance
    assert cohen_kappa([True, False, True, False], [False, True, False, True]) == pytest.approx(-1)
    assert math.isnan(cohen_kappa(["toxic"] * 4, ["toxic"] * 4))                       # both always say the same: undefined


def test_weighted_kappa_of_scores():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 3, 3, 5, 4]) == pytest.approx(0.842105, abs=1e-6)
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 3, 3, 5, 4]) == pytest.approx(weighted_kappa([1, 3, 3, 5, 4], [1, 2, 3, 4, 5]))   # symmetric
    near, far = weighted_kappa([1, 2, 3, 4, 5, 3], [2, 2, 3, 4, 5, 3]), weighted_kappa([1, 2, 3, 4, 5, 3], [5, 2, 3, 4, 5, 3])
    assert near > far                                                                  # a big disagreement counts more than a small one
    assert weighted_kappa([1, 2, 3], [1, 2, 3], scale=range(1, 4)) == 1                # another scale
    with pytest.raises(ValueError):
        weighted_kappa([1, 2, 6], [1, 2, 3])                                           # 6 is not on the 1-5 scale


def test_spearman_rank_correlation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert math.isnan(spearman([3, 3, 3], [1, 2, 3]))                                  # one side never changes


def test_missing_values_are_left_out_and_lengths_must_match():
    nan = float("nan")
    assert weighted_kappa([1, 2, nan, 4], [1, 2, 3, 4]) == 1                           # the pair with a missing value is dropped
    assert cohen_kappa([nan, nan], [1, 2]) != cohen_kappa([nan, nan], [1, 2])          # nothing left: NaN
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])
