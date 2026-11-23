"""Per-structure constructor work, shared by IDENTITY of frozen arrays
(momwire#1943).

The nec2 launchers construct a solver per FR step from one
:class:`~momwire.deck._solver.PreparedMesh`, and every construction re-ran
the same geometry-only checks over the same polylines: the duplicate-wire
search, the junction coincidence check, the short-segment census and the
sector map's digest — about half a second of the sym4 ELT whip's thirteen
steps. Hashing the arrays to recognise them again costs about as much as the
work (momwire#1929), so the memo here recognises them by WHO they are.

Identity is only a sound key for an array nothing can write to, so a memo
applies only to :func:`frozen` arrays: read-only, with every base below them
read-only too (a read-only view of a writable buffer is not frozen).
:func:`freeze` is what ``prepare_mesh`` does to the polylines it hands every
solver, which is the contract its docstring already stated ("no solver
writes to them") made into a fact numpy enforces. Any other caller's arrays
are writable and simply miss: the memo never answers for them.

An entry holds strong references to its arrays, so while it lives no other
object can take one of their ``id``\\ s; that is what makes the ``id`` tuple a
key. The memos are small LRUs (a sweep holds one structure at a time).

Everything else a result depends on (segment counts, junction lists, radii)
is in the key BY VALUE, passed as ``extra``; those are small and cheap to
compare, unlike the coordinates.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable, Sequence

import numpy as np


def frozen(array: Any) -> bool:
    """True when ``array`` is an ndarray that nothing can write through: it
    and every base under it are read-only ndarrays."""
    a = array
    while True:
        if not isinstance(a, np.ndarray) or a.flags.writeable:
            return False
        base = a.base
        if base is None:
            return True
        a = base


def freeze(arrays: Sequence[np.ndarray]) -> None:
    """Make each array read-only, in place (see the module docstring)."""
    for a in arrays:
        a.setflags(write=False)


def frozen_tuple(arrays: Sequence[Any]) -> tuple | None:
    """``tuple(arrays)`` when every one is :func:`frozen`, else None."""
    out = tuple(arrays)
    for a in out:
        if not frozen(a):
            return None
    return out


class IdentityMemo:
    """An LRU of ``fn``'s results keyed on (the identity of some frozen
    arrays, a hashable ``extra``)."""

    def __init__(self, size: int = 4):
        self._size = size
        self._entries: OrderedDict = OrderedDict()
        # The resident server solves on several threads, and a lookup that
        # moves an entry to the end races an eviction that removes it. The
        # lock covers the table only, never the computation.
        self._lock = threading.Lock()
        # Counted for the tests: a memo whose hit count never moves measured
        # nothing (the global instructions' "a green gate can measure nothing").
        self.hits = 0
        self.misses = 0

    def get(
        self,
        arrays: tuple | None,
        extra: Hashable,
        compute: Callable[[], Any],
    ) -> Any:
        """``compute()``, or the result it gave for the same frozen ``arrays``
        and an equal ``extra``. ``arrays`` None (not frozen) always computes
        and stores nothing."""
        if arrays is None:
            return compute()
        key = (tuple(map(id, arrays)), extra)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                held, value = entry
                if len(held) == len(arrays) and all(
                    x is y for x, y in zip(held, arrays)
                ):
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return value
            self.misses += 1
        value = compute()
        with self._lock:
            self._entries[key] = (arrays, value)
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
