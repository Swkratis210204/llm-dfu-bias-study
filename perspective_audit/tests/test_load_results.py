"""load_results reads the AU results from a temporary cache and keeps them without the raw ratings."""
import pandas as pd
import pytest

from perspective_audit import DatasetSpec, load_results

RAW = pd.DataFrame([{"item_id": "a", "text": "one two", "rating": r, "sex": s} for r, s in [(1, "m")] * 3 + [(5, "f")] * 3]
                   + [{"item_id": "b", "text": "three", "rating": 3, "sex": s} for s in "mmf"])


def _load(root):
    return RAW


def _boom(root):
    raise AssertionError("the data must not be loaded when the results are cached")


def spec(load):
    return DatasetSpec(name="toy", load=load, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=1)


def test_the_first_call_computes_and_the_second_reads_the_cache(tmp_path, monkeypatch):
    import perspective_audit.datasets as datasets
    monkeypatch.setitem(datasets.SPECS, "toy", spec(_load))
    first = load_results(["toy"], root=tmp_path, cache_dir=tmp_path / "cache")["toy"]
    assert first.raw is None and len(first.au) == 1                                      # kept without the raw ratings
    assert (tmp_path / "cache" / "toy_result.pkl").exists() and (tmp_path / "cache" / "toy_result_slim.pkl").exists()
    monkeypatch.setitem(datasets.SPECS, "toy", spec(_boom))                              # would raise if it loaded the data
    second = load_results(["toy"], root=tmp_path, cache_dir=tmp_path / "cache")["toy"]
    assert second.raw is None and second.report == first.report and second.au.equals(first.au)


def test_keep_raw_returns_the_full_result(tmp_path, monkeypatch):
    import perspective_audit.datasets as datasets
    monkeypatch.setitem(datasets.SPECS, "toy", spec(_load))
    assert len(load_results(["toy"], root=tmp_path, cache_dir=tmp_path / "cache", keep_raw=True)["toy"].raw) == len(RAW)


def test_a_changed_threshold_is_not_served_from_the_cache(tmp_path, monkeypatch):
    import perspective_audit.datasets as datasets
    monkeypatch.setitem(datasets.SPECS, "toy", spec(_load))
    load_results(["toy"], root=tmp_path, cache_dir=tmp_path / "cache")
    changed = DatasetSpec(name="toy", load=_boom, dimensions={"gender": ("sex", {"m": "man", "f": "woman"})}, min_au_texts=5)
    monkeypatch.setitem(datasets.SPECS, "toy", changed)
    with pytest.raises(AssertionError):
        load_results(["toy"], root=tmp_path, cache_dir=tmp_path / "cache")
