"""run_dataset(cache=...) reads a saved result instead of running the AU analysis again."""
import pandas as pd
import pytest

from perspective_audit import DatasetSpec, run_dataset

RAW = pd.DataFrame([{"item_id": "a", "text": "one two", "rating": r, "sex": s} for r, s in [(1, "m")] * 3 + [(5, "f")] * 3]
                   + [{"item_id": "b", "text": "three", "rating": 3, "sex": s} for s in "mmfff"])


def _load(root):
    return RAW


def _boom(root):
    raise AssertionError("the data must not be loaded when the result is cached")


def spec(load):
    return DatasetSpec(name="toy", load=load, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1)


def test_the_second_call_reads_the_cache(tmp_path):
    cache = tmp_path / "temp" / "toy.pkl"
    first = run_dataset(spec(_load), tmp_path, cache=cache)
    assert cache.exists()
    second = run_dataset(spec(_boom), tmp_path, cache=cache)                    # would raise if it loaded the data
    assert second.report == first.report and second.au.equals(first.au) and len(second.raw) == len(RAW)


def test_another_threshold_is_not_served_from_the_cache(tmp_path):
    cache = tmp_path / "toy.pkl"
    run_dataset(spec(_load), tmp_path, cache=cache)
    with pytest.raises(AssertionError):
        run_dataset(spec(_boom), tmp_path, min_au_texts=5, cache=cache)


def test_without_a_cache_nothing_is_written(tmp_path):
    run_dataset(spec(_load), tmp_path)
    assert list(tmp_path.iterdir()) == []
