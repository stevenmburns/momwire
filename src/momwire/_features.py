"""Experimental features, how far each is certified, and what happens past that.

Each opt-in feature has one entry in :data:`FEATURES`. Its REACH is the
deepest level of the support hierarchy (``support/hierarchy.toml``, mirrored
in :data:`LEVELS`) at which it is certified; it is certified on levels
1..reach, and stable when its reach is the deepest level. Every feature is OFF
unless asked for, by its keyword argument (which wins) or by its environment
variable, ``MOMWIRE_X_<NAME>``.

On outside its reach, a feature is handled by its KIND:

* ``"python"``: a Python route over kernels that have already shipped. It
  runs, and a printout says so on line 2 (:func:`stamp`).
* ``"native"``: new C++ or a new numerical path. It refuses, in one sentence
  naming the feature and the platform (:class:`FeatureNotCertified`).

The standalone executables certify no feature (every reach is below their
levels): one started with any ``MOMWIRE_X_`` variable on refuses before it
reads a deck (``standalone/momwire_razor2p.cpp``, :data:`EXE_REFUSAL`).

With every feature off nothing here changes a byte of any printout.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import sys
from dataclasses import dataclass

__all__ = [
    "ENV_PREFIX",
    "EXE_REFUSAL",
    "FEATURES",
    "LEVELS",
    "Feature",
    "FeatureNotCertified",
    "Level",
    "active",
    "certified",
    "check",
    "context_scope",
    "in_context",
    "current_level",
    "enabled",
    "stamp",
]


@dataclass(frozen=True)
class Level:
    id: str
    description: str


# ``support/hierarchy.toml``'s levels, in its order (level 1 first). Kept here
# rather than read from the TOML at run time because an installed wheel ships
# no repo-root file and the Python floor (3.10) has no tomllib;
# tests/test_features_registry.py holds the two equal.
LEVELS: tuple[Level, ...] = (
    Level("ak-linux-avx2", "antennaknobs on Linux avx2: the hosted app"),
    Level("ak-windows", "antennaknobs on Windows, avx2 and sse2"),
    Level(
        "portal-linux-windows",
        "the drop-in portal on Linux and Windows, avx2 and sse2",
    ),
    Level("macos", "antennaknobs and the portal on macOS"),
    Level(
        "exe-bs2-razor-windows",
        "the bspline_nec5 and razor2p executables on Windows, avx2 and sse2",
    ),
    Level(
        "exe-bs2-razor-linux-macos",
        "the bspline_nec5 and razor2p executables on Linux and macOS",
    ),
    Level("exe-bs42", "the bspline_nec42 executable, all platforms"),
    Level("exe-sin42", "the sin42 executable, all platforms"),
)
DEEPEST = len(LEVELS)

ENV_PREFIX = "MOMWIRE_X_"
# Values that turn a variable on, case-insensitively; anything else (unset,
# empty, "0", ...) is off. The executables' scan uses the same set.
_ON = frozenset({"1", "true", "yes", "on"})

# The executables' refusal, word for word (`feature_refusal` in
# standalone/momwire_razor2p.cpp; the test holds the two equal).
EXE_REFUSAL = (
    "{name} is not certified for this executable; it runs only in the Python route"
)

KINDS = ("python", "native")


@dataclass(frozen=True)
class Feature:
    name: str
    reach: int
    env: str
    kind: str
    issue: int
    since: str  # ISO date the opt-in landed on main
    kwarg: str | None = None
    # Older spellings of the variable, still honoured. The canonical
    # ``env`` wins when both are set.
    aliases: tuple[str, ...] = ()

    @property
    def stable(self) -> bool:
        return self.reach == DEEPEST


FEATURES: dict[str, Feature] = {
    f.name: f
    for f in (
        # A front end's hint that edge pairs share a pose: one block filled
        # per pose, the rest copied (PRs #1810, #1811). Python's plan over
        # the copy the Sommerfeld remainder kernel shipped in 0.87.0. The
        # hint is DATA, so the variable alone reuses nothing: no front end
        # that reads the variable supplies a hint.
        Feature(
            name="pair_reuse",
            reach=1,
            env="MOMWIRE_X_PAIR_REUSE",
            kind="python",
            issue=1809,
            since="2026-11-04",
            kwarg="pair_reuse",
        ),
        # A uniform edge's same-edge reg moments filled by segment offset
        # (PR #1815): the shipped kernels, summed in another order.
        Feature(
            name="offset_reuse",
            reach=1,
            env="MOMWIRE_X_OFFSET_REUSE",
            kind="python",
            issue=1809,
            since="2026-11-05",
            kwarg="offset_reuse",
        ),
        # The exact-kernel point moments as a k-series (Python and its C++
        # port): a new numerical path that flips last printed digits.
        Feature(
            name="exact_kernel_series",
            reach=1,
            env="MOMWIRE_X_EXACT_KERNEL_SERIES",
            kind="native",
            issue=1454,
            since="2026-10-13",
            aliases=("MOMWIRE_EXACT_KERNEL_SERIES",),
        ),
        # razor-2p's solve in one C++ call from Python (the prototype, PR
        # #1479).
        Feature(
            name="razor_cxx",
            reach=1,
            env="MOMWIRE_X_RAZOR_CXX",
            kind="native",
            issue=1479,
            since="2026-10-14",
            aliases=("MOMWIRE_RAZOR_CXX",),
        ),
    )
}


# Which route this code is running under: "python" unless a portal entry
# (`momwire.eznec._shell.render`, `momwire.portal`'s `render_deck`) set
# "portal" around its deck. Read by `check` and `stamp` when no context is
# passed.
_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "momwire_feature_context", default="python"
)


@contextlib.contextmanager
def context_scope(context: str):
    """Run the body as `context` (``"python"`` or ``"portal"``)."""
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def in_context(context: str):
    """Decorate an entry point so its body runs as `context`."""

    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with context_scope(context):
                return fn(*args, **kwargs)

        return wrapper

    return wrap


class FeatureNotCertified(RuntimeError):
    """A native feature was asked for outside its reach."""


def _env_on(feature: Feature) -> bool:
    for var in (feature.env, *feature.aliases):
        value = os.environ.get(var)
        if value is not None and value.strip():
            return value.strip().lower() in _ON
    return False


def enabled(
    name: str, kwarg_value: object = None, *, context: str | None = None
) -> bool:
    """Whether `name` is on: the keyword's value when one was passed (not
    None), else its environment variable. Off by default. A native feature
    that is on outside its reach raises :class:`FeatureNotCertified`."""
    feature = FEATURES[name]
    on = bool(kwarg_value) if kwarg_value is not None else _env_on(feature)
    if on:
        check(name, context)
    return on


def active() -> list[str]:
    """The features the environment turns on, in registry order."""
    return [name for name, f in FEATURES.items() if _env_on(f)]


_EXE_BASES = {
    "razor2p": "bs2-razor",
    "razor-2p": "bs2-razor",
    "bspline_nec5": "bs2-razor",
    "bs2": "bs2-razor",
    "bspline": "bs2-razor",
    "bspline_nec42": "bs42",
    "bs42": "bs42",
    "sin42": "sin42",
}


def _os(platform: str | None) -> str:
    p = sys.platform if platform is None else platform
    if p.startswith("linux"):
        return "linux"
    if p.startswith(("win", "cygwin")):
        return "windows"
    if p == "darwin" or p == "macos":
        return "macos"
    raise ValueError(f"no support level for platform {p!r}")


def current_level(
    context: str = "python", *, basis: str | None = None, platform: str | None = None
) -> int:
    """The support level (1-based) of the running configuration.

    `context` is ``"python"`` (momwire imported by a Python program, as
    antennaknobs does), ``"portal"`` (the drop-in portal) or ``"exe"`` (a
    standalone executable, whose `basis` then names it). `platform` defaults
    to this process's ``sys.platform``."""
    os_name = _os(platform)
    if context == "python":
        return {"linux": 1, "windows": 2, "macos": 4}[os_name]
    if context == "portal":
        return 4 if os_name == "macos" else 3
    if context == "exe":
        family = _EXE_BASES.get(basis or "")
        if family is None:
            raise ValueError(f"no standalone executable for basis {basis!r}")
        if family == "bs2-razor":
            return 5 if os_name == "windows" else 6
        return 7 if family == "bs42" else 8
    raise ValueError(f"context must be python, portal or exe, not {context!r}")


def certified(
    name: str,
    context: str = "python",
    *,
    basis: str | None = None,
    platform: str | None = None,
) -> bool:
    """Whether `name` is certified where it is running."""
    level = current_level(context, basis=basis, platform=platform)
    return level <= FEATURES[name].reach


def check(name: str, context: str | None = None) -> None:
    """Refuse a NATIVE feature outside its reach; a Python one always runs.
    `context` defaults to the running one (:func:`context_scope`)."""
    context = _CONTEXT.get() if context is None else context
    feature = FEATURES[name]
    if feature.kind != "native" or certified(name, context):
        return
    level = current_level(context)
    raise FeatureNotCertified(
        f"{name} is not certified for the {context} route on {_os(None)} "
        f"(support level {level}, {LEVELS[level - 1].id}; it reaches level "
        f"{feature.reach}, {LEVELS[feature.reach - 1].id})"
    )


def stamp(context: str | None = None) -> str:
    """What a printout's line 2 adds for the features that are on: ``""``
    when none is (the line is then byte-identical to one without features),
    else ``"+a+b"``, followed by ``" uncertified on this platform"`` when
    any of them is outside its reach here (`context` defaulting to the
    running one)."""
    context = _CONTEXT.get() if context is None else context
    names = active()
    if not names:
        return ""
    text = "".join(f"+{n}" for n in names)
    if not all(certified(n, context) for n in names):
        text += " uncertified on this platform"
    return text
