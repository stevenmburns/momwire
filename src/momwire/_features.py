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
    Level("ak-linux-avx2", "antennaknobs on Linux, avx2 variant only: the hosted app"),
    Level(
        "ak-linux-sse2",
        "momwire Python on Linux with the sse2 variant (pre-AVX2 CPUs)",
    ),
    Level("ak-windows", "antennaknobs on Windows, avx2 and sse2"),
    Level(
        "dropin-linux-windows",
        "the Python drop-in engine (NEC-2 / NEC-4.2 / NEC-5 dialects) on Linux"
        " and Windows, avx2 and sse2",
    ),
    Level(
        "macos",
        "antennaknobs and the Python drop-in engine (NEC-2 / NEC-4.2 / NEC-5"
        " dialects) on macOS",
    ),
    Level(
        "exe-nec5-bspline-razor-2p-windows",
        "the nec5-bspline and nec5-razor-2p executables on Windows, avx2 and sse2",
    ),
    Level(
        "exe-nec5-bspline-razor-2p-linux-macos",
        "the nec5-bspline and nec5-razor-2p executables on Linux and macOS",
    ),
    Level("exe-nec42-bspline", "the nec42-bspline executable, all platforms"),
    Level("exe-nec42-sinusoidal", "the nec42-sinusoidal executable, all platforms"),
)
DEEPEST = len(LEVELS)


def _level(level_id: str) -> int:
    """The 1-based number of the level named `level_id`. Every number below
    comes from here, so a re-rank is an edit of the TOML and of LEVELS."""
    for number, level in enumerate(LEVELS, start=1):
        if level.id == level_id:
            return number
    raise KeyError(level_id)


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
    # The files that are this feature's alone (globs, from the repo root).
    # A pull request touching one is FEATURE work: it must pass the PR jobs
    # of levels 1..reach, and no more (scripts/support_matrix.py). A file
    # the feature shares with stable code (bspline.py, razor.py) does not
    # belong here: a change to it is a stable-path change.
    paths: tuple[str, ...] = ()
    # What a front end shows for it (momwire#1869): `label` is short checkbox
    # text, `summary` a sentence or two for the hover, and `offer` whether the
    # switch alone is worth exposing per request (False where it needs the
    # front end's data, or is folded into another feature). `drift` is the
    # author's measured upper bound on the relative change in input
    # impedance, |dZ|/|Z|, with the feature on against off on its own decks:
    # declared, never inferred at run time, so a front end draws its own line.
    label: str = ""
    summary: str = ""
    drift: float = 1.0
    offer: bool = False

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
            paths=("tests/test_pair_reuse_1809.py",),
            label="pair reuse",
            summary=(
                "Fills one block per repeated pose of edge pairs and copies the "
                "rest. It needs the front end's hint of which pairs share a pose, "
                "so the switch alone reuses nothing."
            ),
            drift=1e-13,
            offer=False,
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
            paths=("tests/test_offset_reuse_1809.py",),
            label="offset reuse",
            summary=(
                "Fills a uniform edge's same-edge moments once per segment offset. "
                "Folded into pair reuse."
            ),
            drift=1e-12,
            offer=False,
        ),
        # The transmitted (below->above) grid blended from frequency anchors
        # on a lattice fixed in metres (momwire#1847), so a sweep shares its
        # tables: the shipped evaluator, interpolated in ln f.
        Feature(
            name="transmitted_anchors",
            reach=1,
            env="MOMWIRE_X_TRANSMITTED_ANCHORS",
            kind="python",
            issue=1847,
            since="2026-11-07",
            kwarg="transmitted_anchors",
            paths=(
                "src/momwire/_transmitted_anchors.py",
                "tests/test_transmitted_anchors_1847.py",
            ),
            label="share buried tables across a sweep",
            summary=(
                "Over buried wires, build the table that carries fields up through "
                "the ground once for a band and blend it at each frequency, instead"
                " of rebuilding it at every point."
            ),
            drift=1e-8,
            offer=True,
        ),
        # The point-matched sinusoidal fill's complex-k field tensor on the
        # complex Galerkin fill's staged rows (momwire#1849): a new C++ path,
        # close to the scalar kernel and not bit-identical to it.
        Feature(
            name="point_cplx_staged",
            reach=1,
            env="MOMWIRE_X_POINT_CPLX_STAGED",
            kind="native",
            issue=1849,
            since="2026-11-07",
            kwarg="point_cplx_staged",
            paths=("tests/test_point_cplx_staged_1849.py",),
            label="vectorised buried fill",
            summary=(
                "Fill the entries between buried wires with the vectorised kernel "
                "the Galerkin solver uses, several times faster on a large radial "
                "screen."
            ),
            drift=1e-11,
            offer=True,
        ),
        # The near-interface column twin's factored route (momwire#1860 L1):
        # on a grid-shaped column each exponential factor once per distinct
        # z and z', the product per member. A new C++ path, a few ulp from
        # the member route and not bit-identical to it.
        Feature(
            name="factored_columns",
            reach=1,
            env="MOMWIRE_X_FACTORED_COLUMNS",
            kind="native",
            issue=1860,
            since="2026-11-08",
            kwarg="factored_columns",
            paths=("tests/test_factored_columns_1860.py",),
            label="factored rod columns",
            summary=(
                "Where a buried rod crosses the ground under a vertical (an EFHW "
                "with a ground rod), fill the rod's column of entries by shared "
                "factors instead of one by one."
            ),
            drift=1e-11,
            offer=True,
        ),
        # The near-interface height-sheet census recalibrated (momwire#1860
        # L2): a horizontal wire's rows on a height sheet at their measured
        # worth, and heights a few ulp apart taken as one sheet. The shipped
        # sheets and twin, a different split between them.
        Feature(
            name="height_sheet_recal",
            reach=1,
            env="MOMWIRE_X_HEIGHT_SHEET_RECAL",
            kind="python",
            issue=1860,
            since="2026-11-08",
            kwarg="height_sheet_recal",
            paths=("tests/test_height_sheet_recal_1860.py",),
            label="shared near-ground tables",
            summary=(
                "Build one table for a wire just above the ground (a low "
                "counterpoise over a ground rod) and read it from both parts of the"
                " fill, instead of computing those entries twice."
            ),
            drift=1e-9,
            offer=True,
        ),
        # The crossing table (momwire#1860 L5): the crossing rows inside a
        # fixed wavelength-scaled grid read from one table per frequency, the
        # singular part taken out in closed form. A new C++ path and a coarse
        # accuracy class (printed Zin within ~0.1 ohm on the EFHW decks).
        Feature(
            name="crossing_table",
            reach=1,
            env="MOMWIRE_X_CROSSING_TABLE",
            kind="native",
            issue=1860,
            since="2026-11-09",
            kwarg="crossing_table",
            paths=("tests/test_crossing_table_1860.py",),
            label="table for wires crossing the ground",
            summary=(
                "Where a wire crosses the ground (an EFHW over a ground rod), read "
                "the entries near the crossing from one table per frequency instead"
                " of computing each. Coarser: input impedance moves by about 0.1 "
                "ohm on an EFHW."
            ),
            drift=1e-3,
            offer=True,
        ),
        # nec5-bspline's and nec42-bspline's crossing route fills the above x
        # above Sommerfeld remainder at the no-rod deck's rule (momwire#1860
        # B1): the base order plus #1189's raised pairs, where the shipped
        # route takes the cross block's q = 6, on a grid no dearer than the
        # shipped one (else that fill declines). The shipped kernels, a
        # different quadrature.
        Feature(
            name="crossing_above_rule",
            reach=1,
            env="MOMWIRE_X_CROSSING_ABOVE_RULE",
            kind="python",
            issue=1860,
            since="2026-11-09",
            kwarg="crossing_above_rule",
            paths=("tests/test_crossing_above_rule_1860.py",),
            label="lighter above-ground rule at a crossing",
            summary=(
                "On a deck where a wire crosses the ground, fill the above-ground "
                "pairs with the rule used when nothing crosses, keeping the finer "
                "rule next to the crossing. Helps nec5-bspline and nec42-bspline "
                "when most of the antenna is above ground. Measured to move the "
                "input impedance by at most 3e-8 relative."
            ),
            drift=1e-7,
            offer=True,
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
            # Not _exact_kernel_series.py: razor's flags-off plan reads its
            # constants and group_reach, so that module is stable code.
            paths=(
                "src/momwire/_accel_exact_kernel_series.cpp",
                "tests/test_exact_kernel_series_1454.py",
            ),
            label="exact kernel series",
            summary=(
                "Computes the exact-kernel point moments as a series in k. Flips "
                "last printed digits. Not measured at Zin; the bound is "
                "conservative."
            ),
            drift=1e-9,
            offer=False,
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
            # Its C++ cores (_razor_solve_core, _razor_front_core) build the
            # razor2p executable too, so they are stable code; only the tests
            # of the opt-in route are its own.
            paths=(
                "tests/test_razor_cxx_front.py",
                "tests/test_razor_cxx_solve_proto.py",
            ),
            label="razor C++ solve",
            summary=(
                "Runs nec5-razor-2p's whole solve in one C++ call from Python. A "
                "prototype. Not measured at Zin; the bound is conservative."
            ),
            drift=1e-6,
            offer=False,
        ),
    )
}


# Which route this code is running under: "python" unless a drop-in entry
# (`momwire.dropin.nec5._shell.render`, `momwire.dropin.nec2c`'s `render_deck`)
# set "dropin" around its deck. Read by `check` and `stamp` when no context is
# passed.
_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "momwire_feature_context", default="python"
)


@contextlib.contextmanager
def context_scope(context: str):
    """Run the body as `context` (``"python"`` or ``"dropin"``)."""
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


# The executable a basis name selects, by its level family. Keys are the
# names an executable or a caller may spell: the long form, the build tokens
# (`MW_EXE_BASIS`) and the solver model values.
_EXE_BASES = {
    "nec5-razor-2p": "nec5-bspline-razor-2p",
    "razor-2p": "nec5-bspline-razor-2p",
    "razor2p": "nec5-bspline-razor-2p",
    "nec5-bspline": "nec5-bspline-razor-2p",
    "bspline_nec5": "nec5-bspline-razor-2p",
    "bspline": "nec5-bspline-razor-2p",
    "nec42-bspline": "nec42-bspline",
    "bspline_nec42": "nec42-bspline",
    "nec42-sinusoidal": "nec42-sinusoidal",
    "sin42": "nec42-sinusoidal",
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


_UNSET: object = object()


def _loaded_variant() -> str | None:
    """The accelerator variant this process loaded (``"avx2"``, ``"sse2"``,
    ``"legacy"``), or None when it runs pure Python. Read from `_accel`,
    whose import is deferred to the call so this module stays light."""
    from . import _accel

    return _accel.VARIANT


def current_level(
    context: str = "python",
    *,
    basis: str | None = None,
    platform: str | None = None,
    variant: str | None = _UNSET,  # type: ignore[assignment]
) -> int:
    """The support level (1-based) of the running configuration.

    `context` is ``"python"`` (momwire imported by a Python program, as
    antennaknobs does), ``"dropin"`` (the Python drop-in engine) or ``"exe"`` (a
    standalone executable, whose `basis` then names it). `platform` defaults
    to this process's ``sys.platform``.

    On Linux the Python context splits by accelerator `variant`: ``"avx2"``
    is level 1 (the hosted app), anything else is ``ak-linux-sse2``. That
    covers ``"sse2"``, the ``"legacy"`` build, and None (no accelerator
    loaded, pure Python), which runs on the CPUs the sse2 variant is for or
    on a failed load, and is no more certified than they are. `variant`
    defaults to the loaded one (:func:`_loaded_variant`); it is read only
    for Linux Python."""
    os_name = _os(platform)
    if context == "python":
        if os_name == "linux":
            if variant is _UNSET:
                variant = _loaded_variant()
            return _level("ak-linux-avx2" if variant == "avx2" else "ak-linux-sse2")
        return _level("ak-windows" if os_name == "windows" else "macos")
    if context == "dropin":
        return _level("macos" if os_name == "macos" else "dropin-linux-windows")
    if context == "exe":
        family = _EXE_BASES.get(basis or "")
        if family is None:
            raise ValueError(f"no standalone executable for basis {basis!r}")
        if family == "nec5-bspline-razor-2p":
            return _level(
                "exe-nec5-bspline-razor-2p-windows"
                if os_name == "windows"
                else "exe-nec5-bspline-razor-2p-linux-macos"
            )
        return _level(f"exe-{family}")
    raise ValueError(f"context must be python, dropin or exe, not {context!r}")


def certified(
    name: str,
    context: str = "python",
    *,
    basis: str | None = None,
    platform: str | None = None,
    variant: str | None = _UNSET,  # type: ignore[assignment]
) -> bool:
    """Whether `name` is certified where it is running."""
    level = current_level(context, basis=basis, platform=platform, variant=variant)
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
