"""Experimental features, how far each is certified, and what happens past that.

Each opt-in feature has one entry in :data:`FEATURES`. Its REACH is the
deepest level of the support hierarchy (``support/hierarchy.toml``, mirrored
in :data:`LEVELS`) at which it is certified; it is certified on levels
1..reach, and stable when its reach is the deepest level. A feature is OFF
unless asked for, by its keyword argument (which wins) or by its environment
variable, ``MOMWIRE_X_<NAME>`` -- except one PROMOTED to on by default
(``default=True``): that one is on wherever it is certified (levels
1..reach) unless the keyword is False or the variable is set to a value that
is not an on value ("0", say), and off by default outside its reach.
Default-on is a separate decision from reach (the maintainer's, recorded in
the entry), and its bar is :data:`DEFAULT_DRIFT_MAX`.

On outside its reach, a feature is handled by its KIND:

* ``"python"``: a Python route over kernels that have already shipped. It
  runs, and a printout says so on line 2 (:func:`stamp`).
* ``"native"``: new C++ or a new numerical path. It refuses, in one sentence
  naming the feature and the platform (:class:`FeatureNotCertified`).

The standalone executables run only the features their C++ route carries
and whose reach covers their level (``EXE_FEATURES`` in
``standalone/momwire_razor2p.cpp``; among them momwire#1860's four levers,
reach 9, so every executable but nec42-sinusoidal's, and the one fill-reuse
switch ``pair_reuse``, reach 11, which every executable accepts and maps to
its own route's reuse). One started with any other ``MOMWIRE_X_`` variable
on refuses before it reads a deck (:data:`EXE_REFUSAL`).

A feature that each solver family runs in its own way carries one
:class:`FamilyReach` per family (``families``): that family's reach, kind
and drift, as data on the one row. They are not separately switchable: the
feature's keyword and variable turn every family's on and off. Today that
is ``pair_reuse`` (momwire#2063 step 4), whose bspline and razor2p rows
carry the deprecated spellings those families took before it was one
switch (``offset_reuse``, ``razor_pair_reuse``: :attr:`FamilyReach.alias`).
On a family, a NATIVE row outside its reach is withheld (off, and
:func:`withheld` says so) rather than refused: the switch asks for "what
this solver supports here".

Each feature also names the solver FAMILIES that honour it (``solvers``,
keyed on each solver class's ``FEATURE_SOLVER``: :data:`SOLVERS`). A
constructor resolves the ones its family honours through :func:`bind`, and
line 2 names only the features the running solver honours (:func:`stamp`'s
`solver`, from the printout's basis by :func:`basis_solver`): a variable a
solver ignores is neither read by it nor stamped. A feature withheld on the
running family is not named either.

With every feature at its default nothing here changes a byte of line 2:
the stamp names only a feature that is on where its default is off. With every
feature off -- the default-on ones turned off explicitly -- the printout is
the one before they were promoted.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import sys
import warnings
from dataclasses import dataclass

from ._basis_names import canonical_engine

__all__ = [
    "DEFAULT_DRIFT_MAX",
    "ENV_PREFIX",
    "EXE_REFUSAL",
    "FEATURES",
    "LEVELS",
    "SOLVERS",
    "FamilyReach",
    "Feature",
    "FeatureNotCertified",
    "Level",
    "active",
    "basis_solver",
    "bind",
    "certified",
    "check",
    "context_scope",
    "in_context",
    "current_level",
    "pragma_note",
    "pragma_scope",
    "deck_pragmas",
    "default_on",
    "enabled",
    "family",
    "honours",
    "stamp",
    "withheld",
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
        "exe-nec5-bs2-razor2p-windows",
        "the nec5-bs2 and nec5-razor2p executables on Windows, avx2 and sse2",
    ),
    Level(
        "exe-nec5-bs2-razor2p-windows-mingw",
        "the nec5-bs2 and nec5-razor2p single-file MinGW-w64 (UCRT64)"
        " executables on Windows, avx2 and sse2 in one file (preview)",
    ),
    Level(
        "exe-nec5-bs2-razor2p-linux-macos",
        "the nec5-bs2 and nec5-razor2p executables on Linux and macOS",
    ),
    Level("exe-nec42-bs2", "the nec42-bs2 executable, all platforms"),
    Level("exe-nec42-sinusoidal", "the nec42-sinusoidal executable, all platforms"),
    Level(
        "exe-nec2c-sinusoidal",
        "the nec2c-sinusoidal executable (the NEC-2 portal), all platforms",
    ),
)
DEEPEST = len(LEVELS)

# Features an executable runs as its launcher's own default although the
# feature's reach is short of that executable's level: {feature: level id}.
# Each is a maintainer's named exception, recorded in the feature's entry
# with the gate that certifies it on that executable; the executables' main
# (`feature_runs_here`, under its dialect) carries exactly these.
EXE_LAUNCHER_DEFAULTS: dict[str, str] = {"nec2_end_cap": "exe-nec2c-sinusoidal"}


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

# The solver families a feature can name in `Feature.solvers`: each solver
# class's `FEATURE_SOLVER` (BSplineSolver and its H-matrix / array-block
# subclasses "bspline", RazorSolver "razor", SinusoidalSolver "sinusoidal",
# SinusoidalGalerkinSolver "sinusoidal_galerkin").
SOLVERS = ("bspline", "razor", "sinusoidal", "sinusoidal_galerkin")

# The most a default-on feature may move an answer: its declared `drift`
# (|dZ|/|Z|) at or under this, below which a five-figure printout reads the
# same (antennaknobs draws its "(faster)" line here too, LAST_DIGITS_DRIFT).
# A feature that moves the answer more stays opt-in, whatever its reach.
DEFAULT_DRIFT_MAX = 1e-6


# The deprecated per-family spellings of a feature (`FamilyReach.alias`) are
# read and warned in the release that made them aliases and removed in the
# next minor (Steve, 2026-11-27: one minor, not the engine aliases' two).
ALIAS_DEPRECATED_IN = "0.99.0"
ALIAS_REMOVED_IN = "0.100.0"


@dataclass(frozen=True)
class FamilyReach:
    """What one solver family runs of a feature: its reach (``<=`` the
    feature's), its kind, and the author's measured bound on |dZin|/|Zin|
    on that family (``<=`` the feature's ``drift``). Data only, never a
    switch of its own.

    `alias` is the keyword this family took for the feature before it was
    one switch, and ``MOMWIRE_X_<ALIAS>`` its variable: each still turns
    the feature on or off on this family alone, with a
    ``DeprecationWarning``, and the feature's own keyword or variable,
    when given, wins over it (:func:`enabled`)."""

    solver: str
    reach: int
    kind: str
    drift: float
    alias: str | None = None

    @property
    def alias_env(self) -> str | None:
        return None if self.alias is None else ENV_PREFIX + self.alias.upper()


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
    # On by default wherever it is certified (levels 1..reach): promoted by
    # the maintainer, in its own PR, with drift <= DEFAULT_DRIFT_MAX. The
    # keyword False, or the variable at a value that is not on ("0"),
    # still turns it off. Outside its reach the default is off.
    default: bool = False
    # One row per solver family that runs it in its own way (`FamilyReach`):
    # per-family reach, kind, drift and deprecated spelling, as data. A
    # family with no row runs the feature at the feature's own reach, kind
    # and drift.
    families: tuple[FamilyReach, ...] = ()
    # The solver families that honour it (`SOLVERS`; momwire#1980): the ones
    # whose constructor resolves it (`bind`, for a feature with a kwarg) or
    # whose fill reads it where it runs (one without). A solver outside this
    # set ignores the variable and the kwarg, and line 2 does not name it
    # (`stamp`). tests/test_feature_solvers_1980.py holds it to the code.
    solvers: tuple[str, ...] = ()
    # Whether a deck may ask for it in a comment card (`CM momwire: <name>`,
    # :func:`deck_pragmas`): the drop-ins and the executables then read the
    # pragma as if this feature's variable were set on for that deck alone.
    # Only for a feature that cannot change the printed answer (`drift` <=
    # DEFAULT_DRIFT_MAX) and that a route declines rather than refuses
    # where it cannot serve the deck; never a thread, path or approximate
    # one. The maintainer marks it, in its own PR.
    pragma: bool = False

    @property
    def stable(self) -> bool:
        return self.reach == DEEPEST


FEATURES: dict[str, Feature] = {
    f.name: f
    for f in (
        # Each congruent pair's fill once, the rest copied: the one fill-reuse
        # switch (momwire#1982; one name since momwire#2063 step 4). Every
        # solver takes `pair_reuse=`, and MOMWIRE_X_PAIR_REUSE turns on the
        # reuse it supports; False turns all of it off.
        #
        # BSplineSolver (momwire#2063): the Galerkin fill finds its own
        # classes of (observer segment, source segment) pairs on the common
        # core (`_bspline_pair_reuse_core.h`) and evaluates each class's
        # moments once with the shipped off-edge kernel; its same-edge pairs
        # are keyed apart, each class the same-edge routine's (step 3); its
        # Sommerfeld remainder has a plan of its own (step 2b), its segment
        # pairs keyed by the plane's invariants, each class evaluated once by
        # the fused remainder kernel; a front end's edge-pair hint (PRs #1810,
        # #1811) is read as on. A bs2 frequency sweep with a served plan runs
        # per frequency to read it, the plan built once for the sweep: the
        # batched sweep reads none (momwire#2172). The same switch fills a
        # uniform edge's same-edge regularised moments by segment offset, in
        # their Toeplitz form (PR #1815, what was `offset_reuse`): inert where
        # the census serves the direct fill (the class values' regularised
        # part is the dense routine's), it pays where no plan does: the
        # near-image fixup of a long grazing wire (a 40 m wire 2 cm over PEC,
        # 1500 segments: 32 % of the fill beside the census), a long wire's
        # same-edge blocks when the census declines, the buried fill.
        #
        # RazorSolver (momwire#1971, what was `razor_pair_reuse`): the fill
        # finds its own classes of (centroid observer, source segment) pairs,
        # the direct and image moments keyed on what the kernel reads and the
        # composing ground's remainder on the height-aware key, evaluates one
        # representative per class with the shipped kernels and gathers the
        # rest, in Python and in nec5-razor2p (one C++ core).
        #
        # SinusoidalSolver (momwire#1911): the point-matched fill finds its
        # own classes of (observer, source) pairs and evaluates each one's
        # field tensors once with the shipped kernel, free space and PEC
        # image: one C++ copy on the common reuse core
        # (`_sin_pair_reuse_core.h`, momwire#2063), which the Python route
        # and the nec42-sinusoidal and nec2c-sinusoidal executables both run.
        #
        # Reach 11, every level: every Python route and drop-in, and every
        # executable, each of which accepts MOMWIRE_X_PAIR_REUSE and runs its
        # own basis's reuse in its C++ fill, at the launcher's bytes. On a
        # family the row in `families` says how far that family's reuse is
        # certified, and line 2 says so (`stamp`). Still off by default and
        # not offered.
        Feature(
            name="pair_reuse",
            reach=11,
            env="MOMWIRE_X_PAIR_REUSE",
            solvers=("bspline", "razor", "sinusoidal"),
            kind="python",
            issue=1809,
            since="2026-11-04",
            kwarg="pair_reuse",
            paths=(
                "src/momwire/_sin_pair_reuse.py",
                "src/momwire/_bspline_pair_reuse_core.cpp",
                "src/momwire/_bspline_pair_reuse_core.h",
                "src/momwire/_razor_pair_reuse_core.cpp",
                "src/momwire/_razor_pair_reuse_core.h",
                "src/momwire/_pair_reuse_core.cpp",
                "src/momwire/_pair_reuse_core.h",
                "tests/test_pair_reuse_1809.py",
                "tests/test_offset_reuse_1809.py",
                "tests/test_razor_pair_reuse_1971.py",
                "tests/test_razor_reuse_rows_2144.py",
                "tests/test_one_reuse_switch_2063.py",
                "tests/test_sin_pair_reuse_1911.py",
                "tests/test_sin_pair_reuse_ground_1970.py",
                "tests/test_sin_pair_reuse_canonical_1969.py",
                "tests/test_sin_pair_reuse_small_deck_1969.py",
                "tests/test_pair_reuse_contract_1982.py",
                "tests/test_pair_reuse_chunked_2020.py",
                "tests/test_bs2_fused_assembly_2166.py",
                "tests/test_bs2_image_reuse_shared_2063.py",
                "tests/test_bs2_remainder_plane_reuse_2063.py",
                "tests/test_sin_pair_reuse_fold_census.py",
                "tests/test_sin_pair_reuse_cxx_1911.py",
                "tests/test_sin_pair_reuse_1916.py",
                "tests/test_sin_pair_reuse_shared_1916.py",
                "tests/test_sin_pair_reuse_allowance_2123.py",
                "tests/test_sin_pair_reuse_keys_1982.py",
            ),
            label="pair reuse",
            summary=(
                "Fills each repeated pose of a pair of segments once and copies "
                "the rest. Every solver finds its own classes, in free space and "
                "over a perfect or finite ground, on decks large enough to pay "
                "for the search; bs2's include a wire's own pairs, one class per "
                "segment offset."
            ),
            # The worst of the families' below (razor2p's and bs2's).
            drift=2e-9,
            offer=False,
            pragma=True,
            families=(
                # bspline: reach 9, the Python routes and drop-ins (levels 1
                # to 5, macOS included: Steve, 2026-11-25) and the nec5-bs2
                # and nec42-bs2 executables (levels 6 to 9), whose C++ fill
                # runs the census and the Toeplitz form at the launcher's
                # bytes (momwire#2063). Not 10 or 11: those executables have
                # no bs2 route.
                #
                # |dZin|/|Zin| against the flag-off fill, the census and the
                # Toeplitz form together, both routes assembling Z from the
                # class values tile by tile (momwire#2166; the tensor route
                # since the follow-up to momwire#2063 step 3, whose sum order
                # is the windowed assembler's, not `assemble_Z`'s): 1.0e-11
                # or less over the step-2a decks (Yagis of one and two edges,
                # a V-bent Yagi, a vertical stack), a 1200-segment Yagi, a
                # 36-edge loop and two collinear edges, free, PEC, refl-coef
                # and Sommerfeld; 1.0e-11 and 1.3e-13 on antennaknobs'
                # 10-element Yagi at 180 and 120 segments a wire, 2.8e-12 on
                # its fan dipole flat and drooped, free and Sommerfeld;
                # 4.5e-12, 3.3e-12 and 2.3e-13 on the 10-element Yagi,
                # 3.5-wavelength long wire and 32-radial vertical decks.
                # Outside 5e-10, the bound every other deck holds: the #1971
                # hairpin (cond(Z) ~7e9, its bridge's pairs all near singles
                # at their own bits) moves 1.5e-9 over refl-coef, 1.0e-9 over
                # PEC and 5.0e-10 free, of which the sum order alone is
                # 1.3e-9, 8.9e-10 and 7.0e-10 (its flag-off Z through the
                # windowed assembler), and its two flag-off routes differ by
                # 2.3e-8. 2e-9 is the hairpin's with 1.3x margin. Outside it,
                # recorded: a 40 m wire 2 cm over PEC (1500 segments, its
                # near-image block analytic) moves 2.8e-9, where its off fill
                # moves 1.05e-8 between its tensor and chunked routes; a
                # 0.3 m square loop at 137 kHz (440 segments, the tensor
                # route) moves 5.2e-7, of which the sum order alone is
                # 5.7e-7, where its off fill moves 5.8e-7 translated 3.7 m.
                FamilyReach(
                    "bspline",
                    reach=9,
                    kind="python",
                    drift=2e-9,
                    alias="offset_reuse",
                ),
                # razor: reach 8, every Python route and the nec5 executables
                # (levels 6 to 8; nec5-razor2p's route). Not 9 or 10: the
                # nec42 executables have no razor2p route. Native: outside its
                # reach the switch leaves it off (`withheld`).
                #
                # |dZin|/|Zin| against the flag-off fill, the worst over every
                # frequency, the extended kernel on (the NEC-5 shell's), each
                # deck with exact_kernel auto / True / False (Python route,
                # Rocketlake, momwire#1971):
                #
                #   the six benchmark decks (1097-2535 segs, 13 frequencies):
                #     gsky_m24 2.6e-13 (Sommerfeld), fd_h51 2.6e-13, sky_m24
                #     2.0e-13, p_m24 1.9e-13, gfd_h51 1.5e-13 (Sommerfeld),
                #     fd_h27 3.9e-14;
                #   tiny bridges, h < a (a 2 mm hairpin, 21-segment bridge):
                #     3.1e-12 free, 3.7e-12 Sommerfeld, 3.4e-12 translated
                #     3.3 m (each with exact_kernel False; 1.4e-12 auto /
                #     True); a folded dipole, a shorted stub, a thick
                #     monopole: <= 1.3e-13;
                #   close-spaced two-wire lines (10 m, 1 mm wire, 2.1 to 12 mm
                #     apart, 41 to 161 segments a side, free / PEC /
                #     Sommerfeld, at the origin and translated 3.3 m): 4.3e-10
                #     at the worst (2.1 mm, 41 segments, free, exact_kernel
                #     False), 4.2e-10 at 2.5 mm, 1e-10 at 6 mm, 4e-11 at 12 mm.
                #
                # The lines' drift is the fill's own rounding, not a wrong
                # class: the flag-off fill moves as much when the same deck is
                # translated 3.3 m (up to 9.4e-10 at 2.1 mm), the pairs differ
                # from their representatives only by coordinate rounding
                # (KEY_BITS), and `verify` passes. 2e-9 is about 5x the worst.
                FamilyReach(
                    "razor",
                    reach=8,
                    kind="native",
                    drift=2e-9,
                    alias="razor_pair_reuse",
                ),
                # sinusoidal: every level. The class map moves Z itself by
                # ~1e-12 relative to its largest entry, and the solve
                # amplifies that at the feed. Measured over every frequency of
                # the 11-16 reuse ladders (13 each; gsfd_* 101), dense:
                #   free space (Rocketlake, main 816ccd5b): p_m24 2.15e-10,
                #     sky_c24 7.6e-11, sky_m24 6.6e-11, fd_h51 9.6e-12;
                #   finite ground (Skylake, the momwire#1970 change): gsky_m24
                #     9.9e-11 (Sommerfeld), 8.2e-11 (reflection coefficient);
                #     gfd_h51 1.0e-11 / 4.1e-12; gsky_m12 7.9e-12; every other
                #     gfd_*, gsfd_*, gsky_m6 6.1e-12 or less.
                # 5e-10 is the worst (p_m24) with about 2x margin.
                FamilyReach("sinusoidal", reach=11, kind="python", drift=5e-10),
            ),
        ),
        # The transmitted (below->above) grid blended from frequency anchors
        # on a lattice fixed in metres (momwire#1847), so a sweep shares its
        # tables: the shipped evaluator, interpolated in ln f. Ported to the
        # C++ buried route (`somm_trans::anchored_grid`, in process only, no
        # disk store) so the executables can carry it once reach is raised;
        # it has no EXE_FEATURES row, so today they refuse it.
        #
        # `drift` is the survey's worst with 2.3x margin (momwire#2022): flag
        # on against off, |dZin|/|Zin|, 326 served sweep points on bs2,
        # sinusoidal and sinusoidal Galerkin, each counted as served.
        #
        #   wires 1-50 cm either side of the interface   1.3e-8  (sinusoidal)
        #     the same, bs2                              4.7e-9
        #   10 % lattice-phase sweep, m6_close's shape   1.2e-9
        #   band sweeps to +- 25 %                       4.1e-9
        #   repo fixtures (transmitted_mixed)            8.7e-10
        #   dipole over a buried wire                    8.6e-10
        #   long reach, soils 2/0.0001 to 15/0.1         1.2e-10
        #   vertical over a screen, two depths           2.8e-11
        #
        # Crossing decks never reach this grid. A long deck over a very
        # shallow source declines, and the exact grid serves it.
        Feature(
            name="transmitted_anchors",
            reach=1,
            env="MOMWIRE_X_TRANSMITTED_ANCHORS",
            solvers=("bspline", "sinusoidal", "sinusoidal_galerkin"),
            kind="python",
            issue=1847,
            since="2026-11-07",
            kwarg="transmitted_anchors",
            paths=(
                "src/momwire/_transmitted_anchors.py",
                "tests/test_transmitted_anchors_1847.py",
                "tests/test_transmitted_anchors_cxx_1847.py",
            ),
            label="share buried tables across a sweep",
            summary=(
                "Over buried wires, build the table that carries fields up through "
                "the ground once for a band and blend it at each frequency, instead"
                " of rebuilding it at every point."
            ),
            drift=3e-8,
            offer=True,
        ),
        # The point-matched sinusoidal fill's complex-k field tensor on the
        # complex Galerkin fill's staged rows (momwire#1849): a new C++ path,
        # close to the scalar kernel and not bit-identical to it.
        Feature(
            name="point_cplx_staged",
            reach=1,
            env="MOMWIRE_X_POINT_CPLX_STAGED",
            solvers=("sinusoidal",),
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
        # the member route and not bit-identical to it. Reach 9: the
        # nec5-razor2p, nec5-bs2 and nec42-bs2 executables carry it
        # (the ports of #1860's measurement build). On by default since
        # 2026-11-15 (Steve), as is height_sheet_recal: certified through
        # level 9, and level 10's nec42-sinusoidal route has no column twin
        # to factor.
        Feature(
            name="factored_columns",
            reach=9,
            env="MOMWIRE_X_FACTORED_COLUMNS",
            solvers=("bspline", "razor"),
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
            default=True,
        ),
        # The near-interface height-sheet census recalibrated (momwire#1860
        # L2): a horizontal wire's rows on a height sheet at their measured
        # worth, and heights a few ulp apart taken as one sheet. The shipped
        # sheets and twin, a different split between them. Reach 9, as
        # factored_columns.
        Feature(
            name="height_sheet_recal",
            reach=9,
            env="MOMWIRE_X_HEIGHT_SHEET_RECAL",
            solvers=("bspline", "razor"),
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
            default=True,
        ),
        # The crossing table (momwire#1860 L5): the crossing rows inside a
        # fixed wavelength-scaled grid read from one table per frequency, the
        # singular part taken out in closed form. A new C++ path and a coarse
        # accuracy class (printed Zin within ~0.1 ohm on the EFHW decks).
        # Reach 9, as factored_columns. Opt-in: its drift class is over
        # DEFAULT_DRIFT_MAX, and it stays off by default (Steve, 11-15).
        # drift 5e-3 (momwire#1955, was 1e-3): measured 3.6e-3 on fat_slant
        # and 2.5e-3 on fat_hub (a 19 mm vertical, Delta/a 2.2, over four short
        # buried radials), 4e-4 on aca, under 1e-3 on the EFHW decks. Set
        # against the ground itself: +/-10 % in eps_r or sigma moves those
        # same decks' Zin by 0.7-4.5 %, so 5e-3 stays well inside what a user
        # can know about their soil.
        Feature(
            name="crossing_table",
            reach=9,
            env="MOMWIRE_X_CROSSING_TABLE",
            solvers=("bspline", "razor"),
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
                "ohm on an EFHW and up to about 0.4 % on a fat vertical over a "
                "small buried screen, well under what a 10 % error in the ground's"
                " constants does."
            ),
            drift=5e-3,
            offer=True,
        ),
        # nec5-bs2's and nec42-bs2's crossing route fills the above x
        # above Sommerfeld remainder at the no-rod deck's rule (momwire#1860
        # B1): the base order plus #1189's raised pairs, where the shipped
        # route takes the cross block's q = 6, on a grid no dearer than the
        # shipped one (else that fill declines). The shipped kernels, a
        # different quadrature. Reach 9: the two bspline executables carry it;
        # nec5-razor2p's accepts the variable and, as Python's razor2p,
        # ignores it.
        Feature(
            name="crossing_above_rule",
            reach=9,
            env="MOMWIRE_X_CROSSING_ABOVE_RULE",
            solvers=("bspline",),
            kind="python",
            issue=1860,
            since="2026-11-09",
            kwarg="crossing_above_rule",
            paths=("tests/test_crossing_above_rule_1860.py",),
            label="lighter above-ground rule at a crossing",
            summary=(
                "On a deck where a wire crosses the ground, fill the above-ground "
                "pairs with the rule used when nothing crosses, keeping the finer "
                "rule next to the crossing. Helps nec5-bs2 and nec42-bs2 "
                "when most of the antenna is above ground. Measured to move the "
                "input impedance by at most 3e-8 relative."
            ),
            # Not default-on: approved for promotion 2026-11-15, held back
            # because the pre-tag sweep measured |dZin|/|Zin| = 1.28e-6 on
            # default|sommerfeld|ground_rod_efhw_wa7ark (nec5-bs2, Skylake),
            # over this declared 1e-7 and over DEFAULT_DRIFT_MAX.
            drift=1e-7,
            offer=True,
        ),
        # The near-interface sheets' build weights retuned (momwire#1956):
        # the plane and height census at a lower node weight than the 0.15
        # measured with every sheet built from scratch, so a crossing deck's
        # radial plane (and a low wire's height) is read from a sheet where
        # the shipped census computes each row. The shipped sheets and
        # twin, a different split between them; the weights are constants,
        # so the plan is still never decided on what is cached. Opt-in:
        # its cost is a cold process's first sheet build, which a small
        # deck pays without the warm gain.
        # Reach 8 (momwire#1956 promotion, 2026-11-17): the two nec5
        # executables carry it (their C++ census reads the same two
        # constants, `ni_eval::sheet_build_weight`); the nec42-bs2
        # executable (level 9) does not, and refuses it.
        Feature(
            name="crossing_sheet_retune",
            reach=8,
            env="MOMWIRE_X_CROSSING_SHEET_RETUNE",
            solvers=("bspline", "razor"),
            kind="python",
            issue=1956,
            since="2026-11-16",
            kwarg="crossing_sheet_retune",
            paths=("tests/test_crossing_sheet_retune_1956.py",),
            label="more near-ground tables",
            summary=(
                "Read more of the entries between wires above and below the "
                "ground from cached tables (a buried radial screen's, a low "
                "wire's), instead of computing each. Faster once the tables "
                "are cached; the first solve in a fresh process builds them."
            ),
            # |dZin|/|Zin| against off, every run (frequency) of every deck
            # in tests/fixtures with a ground card (168) and the momwire#1956
            # scoping decks, both executables' C++ twin
            # (Skylake, T4): at most 6.2e-13 (fat_hub, nec5-razor2p;
            # nec5-bs2 3.2e-13), the plane sheets' own interpolation error
            # where a row moves from the column twin to a sheet. Deeper and
            # higher in frequency the plan takes more planes (2 -> 51-75) and
            # the error grows with them: a 16-radial hub at 1.5 m, 28 MHz,
            # 2.5 m radials and vertical measures 9.77e-12 on razor2p
            # (1.2e-13 nec5-bs2), 3.0 m at 14 MHz 6.15e-12, 5.0 m at
            # 14 MHz 3.65e-12; the same to the bit on Haswell and Skylake.
            # (Those figures are the Python route's, on the dict-built decks
            # of tests/test_crossing_sheet_retune_1956.py: the 1.5 m / 28 MHz
            # one is `_deep_hub(28e6, 1.5, 2.5)`.)
            # Re-measured at the promotion to the nec5 executables (Rocketlake, T1,
            # MW_EXE_ARRAYS) through the nec5 executables, on a scan of the
            # same 16-radial hub written as a NEC-5 deck (`hub_nec` there:
            # 7-30 MHz, depth 0.15-3 m, radials and vertical 2.5-7.5 m), a
            # different geometry from `_deep_hub` (radials laid from the hub
            # out, the vertical and its 7th-of-15 feed counted from the base;
            # the worst case 30 MHz and 5 m, not 28 MHz and 2.5 m). The worst
            # measured: 1.93e-11 on
            # nec5-bs2 at 30 MHz, 1.5 m, 5 m (that module's
            # DECKS["deep_hub"], 1.40e-11 on nec5-razor2p there), and
            # 1.56e-11 on nec5-razor2p at 28 MHz, 0.5 m, 5 m; the sse2
            # builds the same on those two, and the launcher's own route on
            # DECKS["deep_hub"] the same digits (1.93e-11 / 1.40e-11).
            # The declared 6e-11 is NOT a measurement: it is margin, about 3x
            # over that measured worst case of 1.93e-11 (Steve, 2026-11-17;
            # 3e-11 before it, which left 1.6x). Still far under
            # DEFAULT_DRIFT_MAX, so no printout reads differently.
            drift=6e-11,
            offer=False,
        ),
        # BSplineSolver's junction constraints eliminated before the factor
        # (momwire#1898): Z reduced to the constraint null space in place,
        # where the Schur route back-substitutes n_c extra right-hand sides.
        # The shipped fill and LAPACK, on a smaller system; the same
        # Galerkin solution up to rounding.
        Feature(
            name="kcl_nullspace",
            reach=1,
            env="MOMWIRE_X_KCL_NULLSPACE",
            solvers=("bspline",),
            kind="python",
            issue=1898,
            since="2026-11-11",
            kwarg="kcl_nullspace",
            paths=(
                "src/momwire/_kcl_nullspace.py",
                "tests/test_kcl_nullspace_1898.py",
                "tests/test_kcl_nullspace_cxx_1898.py",
                "tests/test_kcl_nullspace_sector_1983.py",
            ),
            label="eliminate junction constraints",
            summary=(
                "Solve nec5-bs2's and nec42-bs2's junction conservation "
                "by eliminating one wire end per junction before the factor, "
                "instead of carrying the constraints through the solve. Fewer "
                "unknowns and no extra solves on junction-dense designs (coils, "
                "cages, grids)."
            ),
            # Measured on the ELT whip (2,029 rows), nec5-bs2 degrees 1
            # and 2, free space and Sommerfeld: at most 6.7e-13.
            drift=1e-12,
            offer=True,
        ),
        # The exact-kernel point moments as a k-series (Python and its C++
        # port): a new numerical path that flips last printed digits. The
        # executables' C++ route carries it too (`_exact_kernel_series_core`,
        # read from `ni_eval::Features`), but at reach 1 they have no
        # `EXE_FEATURES` row and refuse the variable by name.
        Feature(
            name="exact_kernel_series",
            reach=1,
            env="MOMWIRE_X_EXACT_KERNEL_SERIES",
            solvers=("razor",),
            kind="native",
            issue=1454,
            since="2026-10-13",
            aliases=("MOMWIRE_EXACT_KERNEL_SERIES",),
            # RazorSolver(exact_kernel_series=...); None defers to the
            # variable as read at import (`_exact_kernel_series.ENABLED`).
            kwarg="exact_kernel_series",
            # Not _exact_kernel_series.py: razor's flags-off plan reads its
            # constants and group_reach, so that module is stable code.
            paths=(
                "src/momwire/_accel_exact_kernel_series.cpp",
                "src/momwire/_exact_kernel_series_inline.h",
                "src/momwire/_exact_kernel_series_core.cpp",
                "src/momwire/_exact_kernel_series_core.h",
                "tests/test_exact_kernel_series_1454.py",
            ),
            label="exact kernel series",
            summary=(
                "Computes the exact-kernel point moments as a series in k. Flips "
                "last printed digits: at most 2.3e-14 of Zin as measured; the "
                "bound is conservative."
            ),
            # Measured on 105 synthetic razor2p dipoles (h/a 1.1 to 230; free
            # space, PEC and Sommerfeld ground; 54 of them at wavelengths just
            # inside the series' own bounds): |dZin|/|Zin| at most 2.3e-14.
            drift=1e-9,
            offer=False,
        ),
        # razor2p's solve in one C++ call from Python (the prototype, PR
        # #1479).
        Feature(
            name="razor_cxx",
            reach=1,
            env="MOMWIRE_X_RAZOR_CXX",
            solvers=("razor",),
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
                "Runs nec5-razor2p's whole solve in one C++ call from Python. A "
                "prototype. Not measured at Zin; the bound is conservative."
            ),
            drift=1e-6,
            offer=False,
        ),
        # razor2p's cross blocks across a frequency sweep from anchors
        # (`_razor_crossing_core.cpp`, `set_band_plan`): the above x below
        # blocks and the crossing node charges computed exactly at a few
        # wavenumbers spanning the sweep (Chebyshev-Lobatto, the plan's two
        # ends among them) and every other point's values interpolated
        # between them as they are (barycentric, no phase taken out: dividing
        # out exp(-j k R) measured worse). The above and below blocks stay
        # exact per frequency. The anchor count is chosen per deck from an
        # a-priori error bound (`anchors::bound`: the half band's phase turn
        # over the deck's above-to-below span, scaled by how fast k_m turns
        # with k), the fewest from 5 to 12 that meet a target a factor of
        # three under the drift below. Declines (the exact blocks, flag-off
        # bits) under too few frequencies, too wide a band, a deck too large
        # electrically for 12, a fill outside the plan or on another
        # geometry. One C++ core for the plan, the
        # entries and the interpolation (`_razor_crossing_core.h`'s
        # `anchors`): the nec5-razor2p executable's fill and RazorSolver's
        # (`_crossing_anchors_now`, which fills its anchors with its own
        # cross terms), so the two print the same bytes. It acts where a
        # seam hands the fill its sweep: the NEC-5 seam's stepping FR, in
        # C++ and in the drop-in. Reach 8: the Python routes and the nec5
        # executables (nec5-razor2p's alone, `EXE_FEATURE_BASES`).
        Feature(
            name="crossing_anchors",
            reach=8,
            env="MOMWIRE_X_CROSSING_ANCHORS",
            solvers=("razor",),
            kind="native",
            issue=2010,
            since="2026-11-17",
            kwarg="crossing_anchors",
            paths=("tests/test_crossing_anchors_1117.py",),
            label="shared ground-crossing entries across a sweep",
            summary=(
                "On a frequency sweep of a wire crossing the ground (a vertical "
                "over buried radials), compute the entries between the wires "
                "above and below the ground at a few frequencies and "
                "interpolate them at the rest, instead of computing them at "
                "every frequency."
            ),
            # |dZin|/|Zin| against off, full precision (the nec5-razor2p
            # executable's MW_EXE_ARRAYS, Rocketlake, the same threads both
            # sides), every point of a 21-point sweep, the anchor count the
            # error rule chose:
            #   the four-radial buried vertical (251 segments, 7 MHz, 3 %):
            #     5 anchors, 9.1e-12 at 1 thread, 9.9e-12 at 8;
            #   79 synthetic crossing decks (`_razor_crossing_core.cpp`'s
            #     calibration set, a held-out set and seven extremes: radials
            #     to 40 m at 28 MHz on sigma 1e-3, 1.5 m deep, sloped):
            #     5 to 12 anchors, worst 2.1e-9 (28 MHz, 6.3 m radials,
            #     sigma 1e-3, 9.5 %, 7 anchors).
            # A fixed five, the plan before the rule, reached 1.3e-5 on
            # those decks (12.7 m radials at 28 MHz, 9.5 %): the band alone
            # does not bound it. The rule's target is a third of this drift
            # and its estimate carries at least twice the calibration fit, so
            # the drift is the rule's bound with margin rather than a corpus
            # maximum. It is an a-priori estimate fitted on measurements, not
            # a proof: every stress point came in at most 0.46 of it, and the
            # worst measured anywhere is 24x under 5e-8. Composed with
            # crossing_table the anchors interpolate the table's values,
            # which are not smooth in k: 2.0e-8 against crossing_table alone
            # on the vertical, inside this drift but outside the rule, and
            # under the table's own 8.9e-4.
            drift=5e-8,
            offer=False,
        ),
        # NEC-2's free-end condition on the point-matched sinusoidal basis
        # (momwire#2033, Steve 2026-11-18): the current runs onto the end cap,
        # X = J1(ka)/J0(ka) in NEC Part I's Eqs. 23 and 54-64, where the
        # default takes it to zero. A compatibility option for comparing with
        # NEC-2 engines, not better physics (momwire#1327 kept no cap as the
        # default). Off by default and never seeded: `drift` is 1.0, no
        # bound, because it changes the answer on purpose. Offered (Steve,
        # 2026-11-19), so antennaknobs shows it as an unticked box wherever
        # it is certified (levels 1-3 of its reach: the hosted app, Linux sse2
        # and Windows installs; not macOS), labelled as changing the answer,
        # not faster, and never seeded on. momwire#2033 measured the end condition's effect at up to 58 % of Zin, stock nec2c
        # against this solver over 65 frequencies of a fat-wire log-periodic
        # Yagi.
        #
        # Verified against nec2c on the NEC-2 route only. The NEC-4.2 slot's
        # route honours it on a deck with no junction the NEC-4 rule solves,
        # with no reference compared there. A deck with such a junction and a
        # free end refuses, because that route has its own end cap and this
        # one could not act. A buried or crossing deck with a free end refuses
        # too.
        #
        # The NEC-2 drop-in launcher (`momwire-nec2c-sinusoidal`, or any
        # `momwire-nec2c*` running the sinusoidal basis) turns it ON by
        # default (Steve, 2026-11-19): that launcher exists to answer like
        # NEC-2. `default` here stays False, so the Python API, the NEC-4.2
        # slot, the NEC-5 drop-in and antennaknobs are unchanged. There
        # MOMWIRE_X_NEC2_END_CAP=0 gives the no-cap answer and =1 the
        # refusing one; unset, a deck the cap would refuse runs without it
        # and its printout says so (`_nec2_end_cap.launcher_default`).
        #
        # Reach 4, `dropin-linux-windows` (Steve, 2026-11-19), so that default
        # is certified on the route it runs on: certified on the drop-in on
        # the evidence of the nec2c agreement (the 71 nec_portal decks through
        # the launcher, =0 byte-identical to the uncapped printouts and the
        # default identical to =1; the synthetic decks above at nec2c's five
        # printed figures) and the W8IO whip and log-periodic Yagi (the
        # default's Z identical to =1, timing unchanged). Python-kind, so the
        # macOS drop-in (level 5) still runs it, uncertified.
        #
        # One executable carries it by NAMED EXCEPTION (Steve, 2026-11-20,
        # momwire#2065): the nec2c-sinusoidal executable (level 11) runs it as
        # its launcher's own default, as the Python launcher does, certified
        # by the same corpus gate run against that launcher (the 71 nec_portal
        # decks: 69 byte-identical with the cap at its default, =0 and =1, the
        # 2 GN 0 decks refused by name, 0 differing; tests/test_standalone_
        # nec2c.py). Reach stays 4: reach is one number, so raising it to 11
        # would certify levels 5..10 (the macOS drop-in among them) on no
        # evidence. :data:`EXE_LAUNCHER_DEFAULTS` records the exception.
        Feature(
            name="nec2_end_cap",
            reach=4,
            env="MOMWIRE_X_NEC2_END_CAP",
            solvers=("sinusoidal",),
            kind="python",
            issue=2033,
            since="2026-11-18",
            kwarg="nec2_end_cap",
            paths=(
                "src/momwire/_nec2_end_cap.py",
                "tests/test_nec2_end_cap_2033.py",
            ),
            label="NEC-2 free-end cap (changes the answer)",
            summary=(
                "Uses NEC-2's end condition at free wire ends, so the sinusoidal "
                "solver can be compared with NEC-2 engines. This deliberately "
                "changes the answer, most on fat wires, and is not a better "
                "model of a real wire end."
            ),
            drift=1.0,
            offer=True,
        ),
        # NEC-2's interaction approximation (`_nec2_far_approx`): a pair whose
        # segment centres are more than the range apart (1 wavelength, NEC-2's
        # default, or a deck's KH card, which the NEC-2 reader accepts only
        # with this on) takes the source segment's current lumped into a
        # dipole at its centre. APPROXIMATE, so never on by default anywhere:
        # it trades the answer's last digits for a cheaper fill
        # (`_sinusoidal_far_core.cpp`), and moves momwire-nec2c-sinusoidal
        # toward NEC-2 engines on the decks where they apply it. NEC-2
        # dialect only: the NEC-4 junction rule and buried or crossing wires
        # refuse. Native (a new C++ kernel), reach 1: certified on Linux avx2
        # Python only, so the drop-in route (level 4) refuses it.
        #
        # `drift` 1e-3, a declared bound above everything measured
        # (2026-11-20, Linux avx2, the port admittances on against off at
        # NEC-2's 1 wavelength): 1.3e-4 at most, on a 1049-segment raised
        # whip with radials over a Fresnel ground; 2e-5 to 9e-5 on the same
        # family at 1049-4387 segments in free space, over PEC and over
        # Sommerfeld; 5.4e-5 at most over the 71 nec_portal decks, 13 of
        # which have a pair beyond the range; 1.2e-6 on a 688-segment
        # log-periodic. OpenNEC's own approximation moved a 5,000-segment
        # whip's Z by 5.1e-4. On those decks, with the NEC-2 launcher's
        # settings, momwire then reads nec2c's five printed figures.
        #
        # Stage 1 of momwire#2082: the Python route at reach 1. Its follow-ups
        # (reach, the nec2c executable, the EK and folded speed-ups,
        # antennaknobs' importer) are listed there.
        Feature(
            name="nec2_far_approx",
            reach=1,
            env="MOMWIRE_X_NEC2_FAR_APPROX",
            solvers=("sinusoidal",),
            kind="native",
            issue=2082,
            since="2026-11-20",
            kwarg="nec2_far_approx",
            paths=(
                "src/momwire/_nec2_far_approx.py",
                "src/momwire/_sinusoidal_far_core.cpp",
                "src/momwire/_sinusoidal_far_core.h",
                "src/momwire/_accel_sinusoidal_far.cpp",
                "tests/test_nec2_far_approx.py",
            ),
            label="NEC-2 far-pair approximation",
            summary=(
                "Uses NEC-2's interaction approximation: segment pairs more "
                "than a wavelength apart (or a deck's KH range) are filled "
                "from a lumped current, as NEC-2 does. A faster fill on large "
                "decks, and closer to NEC-2 engines there, at the cost of the "
                "answer's last digits."
            ),
            drift=1e-3,
            offer=True,
        ),
        # The dense port solve by complex64 LU and complex128 iterative
        # refinement (momwire#2125): `_lapack.solve` / `solve_transposed` /
        # `factor_and_solve` under `mixed_lu_scope`, which each family sets
        # around its one-shot solve: sinusoidal (`compute_port_solution`, the
        # swept impedance), bs2 / bs1 (`_solve_with_kcl*`'s `_lu_solve` and
        # the KCL-reduced solve; hmatrix and arrayblock only on their dense
        # fallback), razor2p (its three `_lu_solve`s) and sg
        # (`_solve_constrained`). Not the factors a solver keeps for later
        # solves (`lu_factor_strict`), the stacked swept solves, or the
        # rotational-symmetry sectors.
        # Shipped LAPACK and BLAS routines only (cgetrf, cgetrs, the complex128
        # residual product), so python-kind: on the drop-in it runs,
        # uncertified, and the NEC-2 launcher's nec2c-layout printout does
        # not name it. The refinement must reach a normwise backward error of
        # u64 (2**-53), or the floor of a complex128 residual under zcgesv's
        # sqrt(n) u64, after at least two corrections, contracting
        # by 2**-6 each sweep, within five; anything else -- no contraction,
        # the cap, an inf or NaN, a singular complex64 factor, a matrix whose
        # range complex64 cannot hold -- falls back to the ordinary complex128
        # factor on the untouched Z and returns the flag-off bits. Memory:
        # Z stays intact for the residuals, plus its complex64 copy, 1.5 Z
        # (the allowance with the flag on is 1.5 * 16 N^2 + 512 MB). No host
        # LAPACK route: the executables' table carries no cgetrf, so they
        # refuse the variable (no EXE_FEATURES row).
        #
        # `drift`, |dZin|/|Zin| flag on against off (Rocketlake, 2026-11-23,
        # 8 threads). Sinusoidal, the NEC-2 launcher route:
        #   W8IO ELT whip, n = 4392, 13 frequencies              2.5e-13
        #     (the same off at 4 threads against 8: 2.1e-13)
        # bs2 / bs1 / sg (momwire#2125 follow-up), the size rule forced off so
        # every solve is tried: the 71 nec_portal decks and every deck of
        # scripts/nightly/corpus.txt (nec4 dialect), 628 per engine:
        #   bs2 5.2e-13, bs1 2.1e-12, sg 7.1e-12 (a buried sg deck whose
        #   second complex128 factor itself moves Zin 1.1e-11; sg's worst
        #   such scatter is 2.4e-11)
        # razor2p, on the solution, through the NEC-5 drop-in: 4.9e-12.
        # Against an 80-bit reference (n <= 700, about 800 solves per engine)
        # the refined answer's worst error is no larger than the ordinary
        # route's: bs2 3.0e-12 (ordinary 3.1e-12), bs1 1.7e-12 (6.0e-12), sg
        # 2.6e-13 (5.5e-12), razor2p 1.1e-12 (1.1e-12). 2e-11 covers the
        # worst with 2.8x margin, far under DEFAULT_DRIFT_MAX. Large decks
        # (n 2404 to 10442, bs2 / sg / razor2p / bs2 with kcl_nullspace)
        # print byte-identically bar timing and line 2.
        #
        # Speed (5 paired runs, medians, AVX-512 off; one frequency of the
        # NEC-5 whip): sinusoidal whip, 13 frequencies, 8 threads 1.19x, 1
        # thread 1.37x (with pair_reuse 1.23x / 1.54x); razor2p whip (n 6372)
        # 1.13x / 1.40x; bs2 whip with kcl_nullspace (n 10442) 1.15x / 1.42x;
        # a free-space bs2 array (n 4818) 1.05x / 1.28x; over Sommerfeld
        # ground (n 3614) and buried (n 2404), where the fill dominates,
        # 1.00x at 8 threads and 1.05x / 1.03x at one. The bs2 whip itself
        # declines: its Schur solve carries 2095 right-hand sides.
        # A stall fallback costs the complex64 factor and a sweep: 0.86x on
        # an over-segmented free-space pair (n 5204, kΔ 1e-3).
        # Peak RSS: sinusoidal whip 582 -> 737 MiB, razor2p whip 806 -> 1080
        # MiB, bs2 whip with kcl_nullspace 3252 -> 3635 MiB, each under its
        # 1.5 Z + 512 MB allowance.
        Feature(
            name="mixed_lu",
            reach=1,
            env="MOMWIRE_X_MIXED_LU",
            solvers=("bspline", "razor", "sinusoidal", "sinusoidal_galerkin"),
            kind="python",
            issue=2125,
            since="2026-11-23",
            kwarg="mixed_lu",
            paths=("tests/test_mixed_lu_2125.py",),
            label="mixed-precision solve",
            summary=(
                "Factors the moment matrix in single precision, about twice as "
                "fast, then refines the answer against the double-precision "
                "matrix until it is as accurate as the ordinary solve; any "
                "solve that does not converge cleanly is redone the ordinary "
                "way. Memory: the matrix plus a half-size copy."
            ),
            drift=2e-11,
            offer=True,
        ),
        # The sector (block-circulant) route for a deck that is N identical
        # sectors about one vertical axis (momwire#1029 bspline, #1897 off-
        # axis verticals, #1912 sinusoidal): one sector's rows filled, one
        # block solved per harmonic. Its keyword is older than this entry,
        # and keeps its contract: True refuses a deck the route does not
        # serve. The variable is the drop-ins' way in (Steve, 2026-11-24, on
        # momwire#1338: "Lets just do it for the dropin. We want it optin."):
        # with the keyword unset it TRIES the route and, on a deck or a
        # solver it does not serve, solves densely and says why
        # (`rotational_symmetry_declined`, the fill report's
        # `sector.declined`, a RotationalSymmetryDeclined warning). Never
        # auto-selected and off by default.
        #
        # bs2 / bs1 and the point-matched sinusoidal solver honour it. razor2p
        # has no sector route yet (momwire#1251), so it is not in `solvers`:
        # there the variable is accepted, unread and unstamped, and the deck
        # is solved as with it off. Nor is the Galerkin subclass, whose
        # keyword refuses by name.
        #
        # Reach 4, `dropin-linux-windows`, as pair_reuse: the drop-in on
        # Linux and Windows is the route asked for, and the levels above it
        # are every Python route but macOS. Python-kind (shipped fills and
        # LAPACK on one sector's rows), so the macOS drop-in (level 5) runs it
        # stamped uncertified. No executable carries it: there is no C++
        # sector route and no EXE_FEATURES row, so they refuse the variable.
        #
        # `drift`, |dZin|/|Zin| on against off through the drop-ins (laptop,
        # avx2, 2026-11-24; the port admittance at full precision), on
        # radial screens generated as each dialect writes them: a vertical on
        # the axis fed on the axis over N = 2 to 24 radials, flat or drooping,
        # rotated off the axes, free space, PEC and Sommerfeld (above the
        # ground, and buried through the portal and NEC-4.2), one to three
        # frequencies:
        #   NEC-5 bs2 / bs1, NEC-4.2 bs2, portal bs2 and sinusoidal, with
        #     N = 2-24 at 7-11 segments a wire: 1.4e-11 or less, except
        #     drooping and rotated N = 5 over Sommerfeld 2.3e-10 (sinusoidal)
        #     and N = 7 rotated 1.1 rad at 21-31 segments a wire 4.3e-10.
        #   The 80 EZNEC capture decks through NEC-5 bs2: 3 take the route
        #     (0013, 0033, 0034; 1.3e-12 at most), the other 77 decline and
        #     print the flag-off printout bar line 2.
        # The 4.3e-10 is the dense fill's own rounding, not the route: the
        # same deck solved densely moves 6.2e-10 when it is rotated 1.1 rad
        # about the axis. 1e-9, the route's own acceptance bar (its tests'
        # TOL), is 2.3x the worst; far under DEFAULT_DRIFT_MAX.
        Feature(
            name="rotational_symmetry",
            reach=4,
            env="MOMWIRE_X_ROTATIONAL_SYMMETRY",
            solvers=("bspline", "sinusoidal"),
            kind="python",
            issue=1338,
            since="2026-11-24",
            kwarg="rotational_symmetry",
            paths=("tests/test_rotational_symmetry_dropin_1338.py",),
            label="rotational symmetry (radial screens)",
            summary=(
                "Solves a deck that is N identical sectors about one vertical "
                "axis (a vertical over a radial screen) from one sector's rows, "
                "one block per harmonic, instead of the whole matrix. A deck "
                "that is not built that way is solved as usual."
            ),
            drift=1e-9,
            offer=False,
            pragma=True,
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


# --- per-deck pragmas --------------------------------------------------------
#
# A deck may turn on a `pragma` feature for itself in a comment card, one
# feature a line:
#
#     CM momwire: pair_reuse
#
# The card is CM (either case) after any blanks or tabs; its text, after
# blanks or tabs, starts with "momwire:" (any case), and the name is the run
# of ASCII letters, digits and underscores after it (blanks or tabs between),
# lowered; the rest of the line is ignored. Lines end at CR or LF. EZNEC
# copies a model's description into the deck's CM lines, one line of it a
# card, so a user types the pragma there. The deck readers never see this:
# a pragma card is an ordinary comment, echoed as before.
#
# A pragma acts as the feature's VARIABLE would for that deck, never as its
# keyword True: the variable set either way wins (MOMWIRE_X_<NAME>=0 is the
# user's kill switch), and a route that cannot serve the deck declines as it
# does under the variable. A name that is not a pragma feature here (unknown,
# not marked `pragma`, or one this process withholds, `withhold_pragmas`) is
# ignored and named on line 2 (`pragma_note`). The executables' copy of the
# reader and the rule is `deck_pragmas` in standalone/momwire_razor2p.cpp;
# tests/test_deck_pragma.py holds the two to one table.

PRAGMA_PREFIX = "momwire:"
_PRAGMA_BLANKS = " \t"
_PRAGMA_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def deck_pragmas(text: str) -> list[str]:
    """The feature names `text`'s ``CM momwire: <name>`` cards ask for, in
    the order they first appear, each once. Not checked against the
    registry (:func:`pragma_scope` sorts them)."""
    names: list[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        card = line.lstrip(_PRAGMA_BLANKS)
        if card[:2].upper() != "CM":
            continue
        rest = card[2:].lstrip(_PRAGMA_BLANKS)
        if rest[: len(PRAGMA_PREFIX)].lower() != PRAGMA_PREFIX:
            continue
        rest = rest[len(PRAGMA_PREFIX) :].lstrip(_PRAGMA_BLANKS)
        end = 0
        while end < len(rest) and rest[end] in _PRAGMA_NAME_CHARS:
            end += 1
        name = rest[:end].lower()
        if name and name not in names:
            names.append(name)
    return names


# The pragma features this process never takes from a deck: set by an
# executable's embedded launcher (standalone/momwire_razor2p.cpp's fallback)
# to the ones the executable does not carry, so its Python answer ignores
# them as the C++ route does.
_PRAGMA_WITHHELD: frozenset[str] = frozenset()


def withhold_pragmas(names) -> None:
    """Ignore a pragma for each of `names` from now on in this process."""
    global _PRAGMA_WITHHELD
    _PRAGMA_WITHHELD = frozenset(names)


def pragma_eligible(name: str) -> bool:
    """Whether a deck's pragma may turn `name` on here."""
    feature = FEATURES.get(name)
    return (
        feature is not None
        and feature.pragma
        and feature.drift <= DEFAULT_DRIFT_MAX
        and not feature.default
        and name not in _PRAGMA_WITHHELD
    )


# (the pragma features on, the names ignored) for the deck being answered,
# or None outside a deck's scope.
_PRAGMAS: contextvars.ContextVar[tuple[frozenset[str], tuple[str, ...]] | None] = (
    contextvars.ContextVar("momwire_feature_pragmas", default=None)
)


@contextlib.contextmanager
def pragma_scope(text: str):
    """Answer one deck under its own pragmas (:func:`deck_pragmas`): each
    eligible one reads as its variable set on (:func:`_env_value`) unless
    the variable is set; the rest are named by :func:`pragma_note`. A scope
    inside another is the outer one's (a chain's structures share their
    file's), and a deck with no pragma card changes nothing."""
    if _PRAGMAS.get() is not None:
        yield
        return
    names = deck_pragmas(text)
    on = frozenset(n for n in names if pragma_eligible(n))
    ignored = tuple(n for n in names if n not in on)
    token = _PRAGMAS.set((on, ignored))
    try:
        yield
    finally:
        _PRAGMAS.reset(token)


def pragma_scoped(fn):
    """Decorate a drop-in entry whose first argument is the deck's text so
    its body runs under :func:`pragma_scope`."""

    @functools.wraps(fn)
    def wrapper(text, *args, **kwargs):
        with pragma_scope(text):
            return fn(text, *args, **kwargs)

    return wrapper


def pragmas_on() -> frozenset[str]:
    """The pragma features the deck being answered turns on (before the
    variables, which win), or none."""
    scope = _PRAGMAS.get()
    return frozenset() if scope is None else scope[0]


def pragma_note() -> str:
    """Line 2's note for the deck's ignored pragmas: ``""`` when there are
    none, else ``"(pragma ignored: a, b)"``."""
    scope = _PRAGMAS.get()
    if scope is None or not scope[1]:
        return ""
    return f"(pragma ignored: {', '.join(scope[1])})"


def _set(var: str) -> str | None:
    value = os.environ.get(var)
    return value.strip() if value is not None and value.strip() else None


def _warn_alias(old: str, new: str, stacklevel: int) -> None:
    warnings.warn(
        f"{old} is deprecated; use {new} (momwire#2063)",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def _env_value(
    feature: Feature, solver: str | None = None, *, warn: bool = False
) -> bool | None:
    """What the environment says of `feature`: True (on), False (set to
    anything else, "0" say) or None (unset or empty: the default).

    The feature's own variable first, then its older spellings
    (``aliases``), then a family's deprecated variable
    (:attr:`FamilyReach.alias_env`: `solver`'s alone, or any family's when
    `solver` is None), each warned when `warn`. Unset, a pragma of the deck
    being answered (:func:`pragma_scope`) reads as on."""
    for var in (feature.env, *feature.aliases):
        value = _set(var)
        if value is not None:
            return value.lower() in _ON
    for fam in feature.families:
        if fam.alias_env is None or solver not in (None, fam.solver):
            continue
        value = _set(fam.alias_env)
        if value is not None:
            if warn:
                _warn_alias(f"variable {fam.alias_env}", feature.env, stacklevel=5)
            return value.lower() in _ON
    if feature.name in pragmas_on():
        return True
    return None


def requested(name: str, kwarg_value: object = None) -> bool | None:
    """What was asked of `name`: the keyword's value when one was passed (not
    None), else its variable's (:func:`_env_value`), else None, when nothing
    asked and the default decides. A route with a default of its own (the
    NEC-2 launcher's `nec2_end_cap`) reads this to apply it only then."""
    if kwarg_value is not None:
        return bool(kwarg_value)
    return _env_value(FEATURES[name])


def _env_on(feature: Feature, solver: str | None = None) -> bool:
    return _env_value(feature, solver) is True


def default_on(name: str, context: str | None = None) -> bool:
    """Whether `name` is on when nothing asks either way: a default-on
    feature where it is certified (`context` defaulting to the running
    one), else no."""
    feature = FEATURES[name]
    if not feature.default:
        return False
    return certified(name, _CONTEXT.get() if context is None else context)


def family(name: str, solver: str | None) -> FamilyReach | None:
    """Feature `name`'s row for solver family `solver` (``families``), or
    None when it has none (the feature's own reach, kind and drift apply)."""
    for fam in FEATURES[name].families:
        if fam.solver == solver:
            return fam
    return None


def _decide(
    name: str,
    kwarg_value: object,
    alias_value: object,
    solver: str | None,
    context: str | None,
    warn: bool,
) -> tuple[bool, bool]:
    """(asked on, withheld): what :func:`enabled` decides before refusing.

    The order: the keyword, then the family's deprecated keyword
    (`alias_value`), then the variable, then the family's deprecated
    variable, then a pragma, then the default. On, a native family row
    outside its reach is withheld."""
    if kwarg_value is not None:
        on = bool(kwarg_value)
    elif alias_value is not None:
        on = bool(alias_value)
    else:
        env = _env_value(FEATURES[name], solver, warn=warn)
        on = default_on(name, context) if env is None else env
    return on, on and _withheld_here(name, solver, context)


def enabled(
    name: str,
    kwarg_value: object = None,
    *,
    context: str | None = None,
    solver: str | None = None,
    alias_value: object = None,
) -> bool:
    """Whether `name` is on: the keyword's value when one was passed (not
    None), else its environment variable when set, else its default
    (:func:`default_on`: off, unless the feature is promoted and certified
    here). A native feature that is on outside its reach raises
    :class:`FeatureNotCertified`.

    On solver family `solver`, its row (:func:`family`) adds the family's
    deprecated keyword (`alias_value`, the value the caller was given for
    it) after the keyword, and its deprecated variable after the variable:
    the feature's own spelling, when given, wins. A native row outside its
    reach is off rather than refused (:func:`withheld`)."""
    on, held = _decide(name, kwarg_value, alias_value, solver, context, warn=False)
    if held:
        return False
    if on:
        check(name, context)
    return on


def _withheld_here(name: str, solver: str | None, context: str | None) -> bool:
    fam = family(name, solver)
    if fam is None or fam.kind != "native":
        return False
    context = _CONTEXT.get() if context is None else context
    return not certified(name, context, solver=solver)


def withheld(
    name: str,
    kwarg_value: object = None,
    *,
    context: str | None = None,
    solver: str | None = None,
    alias_value: object = None,
) -> bool:
    """Whether `name` was asked for on family `solver` and stayed off because
    that family's row is native and outside its reach here (:func:`enabled`'s
    soft refusal), so a solver can say why it did not reuse."""
    return _decide(name, kwarg_value, alias_value, solver, context, warn=False)[1]


def honours(name: str, solver: str) -> bool:
    """Whether the solver family `solver` (a :data:`SOLVERS` name, a solver
    class's ``FEATURE_SOLVER``) honours feature `name`."""
    return solver in FEATURES[name].solvers


def bind(
    solver: object,
    kind: str,
    *,
    deferred: tuple[str, ...] = (),
    **kwargs: object,
) -> dict[str, bool]:
    """Resolve the features a solver's constructor reads, for family `kind`.

    Each keyword is a feature's name and the constructor's value for its
    kwarg (None defers to the variable, then the default, as
    :func:`enabled`). A feature `kind` honours is resolved by
    :func:`enabled`, so a native one outside its reach refuses here; one the
    constructor reads that `kind` does not honour (a subclass's narrower
    set: SinusoidalGalerkinSolver runs SinusoidalSolver's constructor) is
    off, and its variable is not read.

    A keyword may also be a deprecated spelling of a feature on `kind`
    (:attr:`FamilyReach.alias`): it is not a feature, has no answer of its
    own, and reaches the feature in the order :func:`enabled` documents,
    with a ``DeprecationWarning`` when it is not None; so does the family's
    deprecated variable. A feature withheld on `kind` (a native row outside
    its reach) is off, and its name is kept on ``solver.features_withheld``
    (:func:`withheld`).

    Every feature with a kwarg that `kind` honours must be passed, so the
    registry cannot name a family whose constructor never reads the feature.
    The answers, ``{name: on}`` in the keywords' order, are returned and kept
    on ``solver.features_bound``.

    `deferred` names the features the constructor takes but resolves later,
    where the feature first acts, so that a solve it never acts on does not
    refuse it outside its reach (razor's `exact_kernel_series`). Each must be
    a feature with a kwarg that `kind` honours, and is not passed as a value
    too. It is not resolved here, is absent from ``solver.features_bound``,
    and is kept on ``solver.features_deferred``."""
    if kind not in SOLVERS:
        raise ValueError(f"no solver family {kind!r} in the feature registry")
    spellings = {
        fam.alias: name
        for name, f in FEATURES.items()
        for fam in f.families
        if fam.alias is not None and fam.solver == kind
    }
    alias_values: dict[str, object] = {}
    for alias, name in spellings.items():
        if alias in kwargs:
            value = kwargs.pop(alias)
            if value is not None:
                _warn_alias(
                    f"keyword {alias!r}", f"{FEATURES[name].kwarg!r}", stacklevel=3
                )
                alias_values[name] = value
    unknown = sorted((set(kwargs) | set(deferred)) - set(FEATURES))
    if unknown:
        raise TypeError(f"bind: no feature named {', '.join(unknown)}")
    for name in deferred:
        if FEATURES[name].kwarg is None or kind not in FEATURES[name].solvers:
            raise TypeError(
                f"bind: {kind} cannot defer {name}: it is not a constructor "
                "keyword of this family"
            )
        if name in kwargs:
            raise TypeError(f"bind: {name} is both deferred and passed a value")
    missing = [
        name
        for name, f in FEATURES.items()
        if f.kwarg is not None
        and kind in f.solvers
        and name not in kwargs
        and name not in deferred
    ]
    if missing:
        raise TypeError(
            f"bind: {kind} honours {', '.join(missing)} but its constructor "
            "does not read it"
        )
    bound: dict[str, bool] = {}
    held: list[str] = []
    for name, value in kwargs.items():
        if not honours(name, kind):
            bound[name] = False
            continue
        alias = alias_values.get(name)
        # once with the deprecation warning, then as `enabled` decides
        if _decide(name, value, alias, kind, None, warn=True)[1]:
            held.append(name)
            bound[name] = False
        else:
            bound[name] = enabled(name, value, solver=kind, alias_value=alias)
    solver.features_bound = bound
    solver.features_withheld = tuple(held)
    solver.features_deferred = tuple(deferred)
    return bound


def active(solver: str | None = None) -> list[str]:
    """The features the environment turns on, in registry order (a
    default-on one only when its variable says on, not by its default), for
    solver family `solver` (a family's deprecated variable counts only for
    its own family; None, any family's)."""
    return [name for name, f in FEATURES.items() if _env_on(f, solver)]


# The executable a basis name selects, by its level family. Keys are the
# names an executable or a caller may spell: the long form, the short form
# and the build tokens (`MW_EXE_BASIS` in standalone/CMakeLists.txt, which
# are CMake cache values, not names a user types). A deprecated spelling is
# canonicalised before the lookup (`_basis_names.canonical_engine`).
_EXE_BASES = {
    "nec5-razor2p": "nec5-bs2-razor2p",
    "razor2p": "nec5-bs2-razor2p",
    "nec5-bs2": "nec5-bs2-razor2p",
    "bs2": "nec5-bs2-razor2p",
    "bspline_nec5": "nec5-bs2-razor2p",
    "nec42-bs2": "nec42-bs2",
    "bspline_nec42": "nec42-bs2",
    "nec42-sinusoidal": "nec42-sinusoidal",
    "sin42": "nec42-sinusoidal",
    "nec2c-sinusoidal": "nec2c-sinusoidal",
    "nec2c_sin": "nec2c-sinusoidal",
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


# The solver family each basis spelling runs, for line 2 (`stamp`): a front
# end's basis names (`momwire.deck.BASES`), the executables' build tokens and
# names, and the long engine names. "" is a family that honours no feature
# (the pulse family's HarringtonSolver); a spelling not listed is unknown
# (`basis_solver` returns None). tests/test_feature_solvers_1980.py holds the
# roster's names to their classes' `FEATURE_SOLVER`. "hmatrix" and "arrayblock"
# run BSplineSolver's constructor, so they are that family; their accelerated
# solve reads none of its features and only their dense fallback does, which
# a table keyed on the basis cannot tell apart deck by deck.
_BASIS_SOLVERS = {
    "bs2": "bspline",
    "bs1": "bspline",
    "hmatrix": "bspline",
    "arrayblock": "bspline",
    "nec5-bs2": "bspline",
    "nec42-bs2": "bspline",
    "bspline_nec5": "bspline",
    "bspline_nec42": "bspline",
    "razor2p": "razor",
    "nec5-razor2p": "razor",
    "sinusoidal": "sinusoidal",
    "nec42-sinusoidal": "sinusoidal",
    "sin42": "sinusoidal",
    "nec2c-sinusoidal": "sinusoidal",
    "nec2c_sin": "sinusoidal",
    "sg": "sinusoidal_galerkin",
    "pulse": "",
}


def basis_solver(basis: str | None) -> str | None:
    """The solver family (:data:`SOLVERS`) basis name `basis` runs, ``""``
    for one that honours no feature, or None when the name is not known.
    A deprecated spelling reads as its canonical name, silently: this is
    asked while printing, where the front end has already warned."""
    return _BASIS_SOLVERS.get(canonical_engine(basis or "", warn=False))


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
    toolchain: str | None = None,
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
    for Linux Python.

    `toolchain` splits the Windows nec5 executables: ``"mingw"`` is the
    single-file MinGW-w64 build, a level of its own because its libm and
    vector math are not MSVC's; anything else is the MSVC build. Python never
    runs as the MinGW build, so only a caller asking about that executable
    passes it (`exe_level()` in standalone/momwire_razor2p.cpp answers for
    the executable itself)."""
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
        family = _EXE_BASES.get(canonical_engine(basis or "", warn=False))
        if family is None:
            raise ValueError(f"no standalone executable for basis {basis!r}")
        if family == "nec5-bs2-razor2p":
            if os_name != "windows":
                return _level("exe-nec5-bs2-razor2p-linux-macos")
            if toolchain == "mingw":
                return _level("exe-nec5-bs2-razor2p-windows-mingw")
            return _level("exe-nec5-bs2-razor2p-windows")
        return _level(f"exe-{family}")
    raise ValueError(f"context must be python, dropin or exe, not {context!r}")


def certified(
    name: str,
    context: str = "python",
    *,
    basis: str | None = None,
    platform: str | None = None,
    variant: str | None = _UNSET,  # type: ignore[assignment]
    toolchain: str | None = None,
    solver: str | None = None,
) -> bool:
    """Whether `name` is certified where it is running: on solver family
    `solver`, as far as that family's row reaches (:func:`family`), else as
    far as the feature does."""
    level = current_level(
        context, basis=basis, platform=platform, variant=variant, toolchain=toolchain
    )
    fam = None if solver is None else family(name, solver)
    return level <= (FEATURES[name].reach if fam is None else fam.reach)


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
        f"{feature.reach}, {LEVELS[feature.reach - 1].id}); unset "
        f"{feature.env} to run without it"
    )


def _runs_on(name: str, solver: str, context: str) -> bool:
    """Whether feature `name`, on in the environment, runs on family `solver`:
    the family honours it and does not withhold it (a native family row
    outside its reach, :func:`withheld`)."""
    return honours(name, solver) and not _withheld_here(name, solver, context)


def stamp(context: str | None = None, solver: str | None = None) -> str:
    """What a printout's line 2 adds for the features the environment turns
    on where their default is off: ``""`` when there are none (the line is
    then byte-identical to one without features), else ``"+a+b"``, followed
    by ``" uncertified on this platform"`` when any of them is outside its
    reach here as it runs (:func:`_certified_as_run`: its family row's reach
    on a family that has one), `context` defaulting to the running one. A
    default-on feature is never named: on, it is the stable path; turned
    off, the printout is the one before it was promoted. A feature turned
    on by a family's deprecated variable is named by its own name.

    `solver` is the running solver's family (:data:`SOLVERS`, or ``""`` for
    one that honours none; :func:`basis_solver`): only the features that run
    on it are named (:func:`_runs_on`), so a variable that solver ignores
    never reaches line 2 (momwire#1980). None, a caller that does not know
    the solver, names every feature on."""
    context = _CONTEXT.get() if context is None else context
    names = [
        n
        for n in active(solver or None)
        if not default_on(n, context)
        and (solver is None or _runs_on(n, solver, context))
    ]
    if not names:
        return ""
    text = "".join(f"+{n}" for n in names)
    if not all(_certified_as_run(n, solver, context) for n in names):
        text += " uncertified on this platform"
    return text


def _certified_as_run(name: str, solver: str | None, context: str) -> bool:
    """Whether `name`, on in the environment, runs certified here: it is
    certified, and so is its row for family `solver` when it has one (with
    `solver` None, every family's row the environment turns it on for that
    is not withheld). A feature reaches as far as its widest family; what
    runs on a family is that family's row."""
    if not certified(name, context):
        return False
    feature = FEATURES[name]
    for fam in feature.families:
        if solver is not None and fam.solver != solver:
            continue
        if solver is None and not _env_on(feature, fam.solver):
            continue
        if _withheld_here(name, fam.solver, context):
            continue
        if not certified(name, context, solver=fam.solver):
            return False
    return True
