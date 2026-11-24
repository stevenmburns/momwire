"""The basis names a caller may spell, and the deprecated spellings of them.

A SHORT name is the basis alone (``bs2``, ``bs1``, ``sg``, ``razor2p``,
``sinusoidal``); a LONG name is dialect, then basis (``nec5-bs2``,
``nec42-sinusoidal``, ``nec2c-bs2``). Short names carry no dash (momwire#2091),
so a long name splits at its first dash with no ambiguity. The dialect parts
are exactly ``nec5``, ``nec42``, ``nec2c`` and ``nec2d``: SimNEC identifies an
engine's dialect by the first three of those substrings in its path, and
``nec2d`` is the NEC-2 dialect printed in LLNL's NEC-2D layout for 4nec2
(momwire#2138). A name that folds the dialect into the basis (``bs42``,
``sin42``) is not a basis name.

`BASIS_ALIASES` maps each spelling that shipped before the rename to the name
it now answers as. An alias answers exactly as its canonical name does: every
reader canonicalises at its entry, so nothing past it ever sees an alias, and
the printout stamps the canonical name. The aliases are removed over time, no
earlier than two minor releases after 0.97.0, announced in the release notes.

Stdlib only: `momwire._features` imports this, and it stays light. The
nec2 thin client (`momwire_nec2c_client`) restates the table rather than
importing momwire (`tests/test_basis_names_2091.py` pins the copy equal).
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from collections.abc import Iterator
from types import MappingProxyType

__all__ = [
    "BASIS_ALIASES",
    "DIALECTS",
    "RETIRED_BASES",
    "canonical_basis",
    "canonical_engine",
    "deprecation_note",
    "retired_refusal",
    "spelled",
    "spelling",
]

BASIS_ALIASES = MappingProxyType(
    {
        "bspline": "bs2",
        "bspline-d1": "bs1",
        "sinusoidal-galerkin": "sg",
        "razor-2p": "razor2p",
    }
)

# Spellings that no longer answer at all, each with the name to use: refused
# BY NAME wherever a basis name is read, so a host still pointed at one learns
# what to change rather than reading "unknown basis". `razor-nec5` was
# deprecated at momwire#785/#794 (it names another vendor's product rather
# than the quadrature rule) and left the launcher names at momwire#1583;
# momwire#2091 retires it rather than carry it forward as an alias.
RETIRED_BASES = MappingProxyType({"razor-nec5": "razor2p"})

# The dialect prefixes of a long name, longest first so `nec42-` is not read
# as `nec4` + `2-...`.
DIALECTS = ("nec42", "nec2c", "nec2d", "nec5", "nec4")


def _warn(name: str, canonical: str, stacklevel: int) -> None:
    warnings.warn(
        f"basis name {name!r} is deprecated; use {canonical!r} (momwire#2091)",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def canonical_basis(name: str, *, warn: bool = True, stacklevel: int = 2) -> str:
    """The canonical short name `name` answers as: `name` itself unless it is a
    deprecated alias, which (with `warn`) raises a ``DeprecationWarning``
    naming the canonical name. Unknown names pass through unchanged, so each
    front end still refuses them in its own words."""
    canonical = BASIS_ALIASES.get(name)
    if canonical is None:
        return name
    if warn:
        _warn(name, canonical, stacklevel)
    return canonical


def canonical_engine(name: str, *, warn: bool = True, stacklevel: int = 2) -> str:
    """A long engine name with its basis part canonicalised
    (``nec5-razor-2p`` -> ``nec5-razor2p``); a short name as
    :func:`canonical_basis` reads it."""
    for dialect in DIALECTS:
        if name.startswith(f"{dialect}-"):
            basis = name[len(dialect) + 1 :]
            canonical = BASIS_ALIASES.get(basis)
            if canonical is None:
                return name
            if warn:
                _warn(name, f"{dialect}-{canonical}", stacklevel)
            return f"{dialect}-{canonical}"
    return canonical_basis(name, warn=warn, stacklevel=stacklevel + 1)


def retired_refusal(name: str | None) -> str | None:
    """The sentence refusing a retired spelling, or None for any other name."""
    replacement = RETIRED_BASES.get(name or "")
    if replacement is None:
        return None
    return (
        f"basis {name!r} was renamed and is no longer accepted; "
        f"use {replacement!r} (momwire#2091)"
    )


# The deprecated spelling a drop-in entry was handed, for the printout's
# line 2. Everything past the entry sees only the canonical name (so every
# refusal and advisory names it), and line 2 is rendered far from the entry,
# on every path including refusals: a context variable carries the one fact
# line 2 needs without threading a second name through each of them.
_SPELLED: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "momwire_basis_spelled", default=None
)


@contextlib.contextmanager
def spelled(name: str) -> Iterator[str]:
    """Canonicalise a drop-in entry's basis: yields the canonical name and,
    while the block runs, records a deprecated spelling for
    :func:`spelling`. A canonical name leaves an outer record in place, so a
    nested entry handed the canonical name keeps the note."""
    canonical = canonical_basis(name, stacklevel=4)
    if canonical == name:
        yield name
        return
    token = _SPELLED.set(name)
    try:
        yield canonical
    finally:
        _SPELLED.reset(token)


def spelling() -> str | None:
    """The deprecated spelling the running drop-in entry was handed, or None."""
    return _SPELLED.get()


def deprecation_note(name: str | None) -> str:
    """The printout's note that `name` is a deprecated spelling, or ``""``.

    It rides at the end of printout line 2, the free-text field EZNEC and
    SimNEC accept and every byte gate masks, so a deprecated launcher prints
    line for line what its canonical twin prints."""
    if name is None or name not in BASIS_ALIASES:
        return ""
    return f"(renamed from {name})"
