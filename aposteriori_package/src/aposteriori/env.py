"""Settings and API keys, read from the environment and from a ``.env`` file.

Format of ``.env``: one ``KEY=value`` per line, ``#`` for comments. ``KEY=@relative/path`` is replaced by the content
of that file. ``OCI_KEY`` and ``OCI_CONFIG`` may span several lines; a blank line ends them. Variables that are
really set in the environment win over the file. The file is searched from the working directory upwards.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Optional

MULTILINE_KEYS = frozenset({"OCI_KEY", "OCI_CONFIG"})
_KEY_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """The first ``.env`` in ``start`` (default: the working directory) or one of its parents."""
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(path: Path) -> dict:
    """Read a ``.env`` file into a dict (see the module docstring for the format)."""
    values: dict = {}
    multiline_key = None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if multiline_key is not None:
            if not line.strip():
                multiline_key = None
            else:
                values[multiline_key] += "\n" + line
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _KEY_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value[:1] in "\"'" and value[-1:] == value[:1] and len(value) > 1:
            value = value[1:-1]
        if value.startswith("@"):
            value = (Path(path).parent / value[1:]).read_text(encoding="utf-8").strip()
        values[key] = value
        if key in MULTILINE_KEYS:
            multiline_key = key
    return values


class Env:
    """A read-only view of the settings: the real environment first, then the ``.env`` file."""

    def __init__(self, values: Optional[Mapping[str, str]] = None, source: Optional[Path] = None,
                 use_os_environ: bool = False):
        self._values = dict(values or {})
        self.source = source
        self._use_os_environ = use_os_environ

    @classmethod
    def load(cls, path: Optional[Path] = None, use_os_environ: bool = True) -> "Env":
        dotenv = Path(path) if path else find_dotenv()
        return cls(parse_dotenv(dotenv) if dotenv else {}, dotenv, use_os_environ)

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if self._use_os_environ:                       # looked up directly: os.environ ignores capitals on Windows
            value = os.environ.get(name)
            if value:
                return value
        value = self._values.get(name)
        return value if value not in (None, "") else default

    def has(self, name: Optional[str]) -> bool:
        return bool(name) and self.get(name) is not None

    def require(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            where = f" (looked in the environment and in {self.source})" if self.source else " (no .env file found)"
            raise KeyError(f"{name} is not set{where}")
        return value

    def __contains__(self, name: str) -> bool:
        return self.has(name)
