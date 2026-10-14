"""Collect every deck text the test suite hands momwire's NEC-5 reader.

A pytest plugin: it wraps `parse_nec5`, `parse_nec5_shared`,
`split_structures`, `_serve.split_runs` and `_shell.render` BEFORE any test
module imports them (load it with `-p`, so it is imported ahead of collection),
records each distinct text they are called with, and writes them as JSON
strings, one per line, to `$NEC5_HARVEST_OUT` at the end of the session
(`$NEC5_HARVEST_OUT.<worker>` per xdist worker).
`deck_compare.py --texts` then reads the C++ reader against the Python one on
all of them: every malformed card, refusal and edge case the suite's authors
wrote by hand, not only the decks on disk.

    NEC5_HARVEST_OUT=harvest.jsonl PYTHONPATH=scripts/core_split \\
        python -m pytest tests/ -p nec5_harvest_plugin -q -m "not slow ..."
"""

from __future__ import annotations

import functools
import json
import os

import momwire.deck
from momwire.deck import _nec5
from momwire.eznec import _serve, _shell

_SEEN: dict[str, None] = {}


def _record(text) -> None:
    if isinstance(text, str):
        _SEEN.setdefault(text, None)


def _wrap(fn, many=False):
    @functools.wraps(fn)
    def wrapper(arg, *args, **kwargs):
        if many:
            arg = list(arg)
            for text in arg:
                _record(text)
        else:
            _record(arg)
        return fn(arg, *args, **kwargs)

    return wrapper


_nec5.parse_nec5 = _wrap(_nec5.parse_nec5)
_nec5.parse_nec5_shared = _wrap(_nec5.parse_nec5_shared, many=True)
_nec5.split_structures = _wrap(_nec5.split_structures)
_serve.split_runs = _wrap(_serve.split_runs)
momwire.deck.parse_nec5 = _nec5.parse_nec5
_shell.parse_nec5_shared = _nec5.parse_nec5_shared
_shell.split_structures = _nec5.split_structures
_shell.render = _wrap(_shell.render)


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("NEC5_HARVEST_OUT")
    if not out:
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:  # each xdist worker writes its own file; concatenate them
        out = f"{out}.{worker}"
    with open(out, "w", encoding="utf-8") as handle:
        for text in _SEEN:
            handle.write(json.dumps(text) + "\n")
    print(f"\nnec5_harvest_plugin: {len(_SEEN)} distinct deck texts -> {out}")
