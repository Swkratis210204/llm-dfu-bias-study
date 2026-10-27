"""The datasets: one module each, with its loader and the definition of its annotator groups."""
from . import attitudes, kumar, mhs, popquorn

SPECS = {spec.name: spec for spec in (popquorn.SPEC, attitudes.SPEC, mhs.SPEC, kumar.SPEC)}


def get_spec(name: str):
    if name not in SPECS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


__all__ = ["SPECS", "get_spec", "popquorn", "attitudes", "mhs", "kumar"]
