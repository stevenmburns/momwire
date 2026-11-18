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
``standalone/momwire_razor2p.cpp``; today momwire#1860's four levers, reach
8, so every executable but nec42-sinusoidal's, and razor_pair_reuse, reach 7,
which only nec5-razor-2p's route carries, so nec5-bspline refuses it by
basis, ``EXE_FEATURE_BASES``). One started with any other ``MOMWIRE_X_``
variable on refuses before it reads a deck (:data:`EXE_REFUSAL`).

A feature may be a PART of another (``part_of``): one user-facing switch
that turns on what each solver supports of it. Today that is ``pair_reuse``,
whose parts are bspline's ``offset_reuse`` and razor-2p's
``razor_pair_reuse`` (momwire#1982). Asking for the whole turns each part
on; a part's own keyword or variable, its older spelling, still wins (see
:func:`enabled` for the order). A native part asked for only through the
whole stays off outside its own reach instead of refusing: the whole asks
for "what this solver supports here", and :func:`withheld` says when that
was nothing.

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
from dataclasses import dataclass

__all__ = [
    "DEFAULT_DRIFT_MAX",
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
    "default_on",
    "enabled",
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

# The most a default-on feature may move an answer: its declared `drift`
# (|dZ|/|Z|) at or under this, below which a five-figure printout reads the
# same (antennaknobs draws its "(faster)" line here too, LAST_DIGITS_DRIFT).
# A feature that moves the answer more stays opt-in, whatever its reach.
DEFAULT_DRIFT_MAX = 1e-6


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
    # The user-facing feature this one is a part of (momwire#1982), or None:
    # asking for that one (its keyword or its variable) turns this one on,
    # unless this one's own keyword or variable says otherwise.
    part_of: str | None = None

    @property
    def stable(self) -> bool:
        return self.reach == DEEPEST


FEATURES: dict[str, Feature] = {
    f.name: f
    for f in (
        # Each congruent pair's fill once, the rest copied. BSplineSolver: a
        # front end's hint that edge pairs share a pose, one block filled per
        # pose (PRs #1810, #1811), Python's plan over the copy the Sommerfeld
        # remainder kernel shipped in 0.87.0; the hint is DATA, so there the
        # variable alone reuses nothing. SinusoidalSolver (momwire#1911): the
        # point-matched fill finds its own classes of (observer, source)
        # pairs and evaluates each one's field tensors once with the shipped
        # kernel, free space and PEC image; there the variable alone is on.
        #
        # The one user-facing pair-reuse switch (momwire#1982): its parts,
        # `offset_reuse` (bspline, within one edge) and `razor_pair_reuse`
        # (razor-2p), follow this keyword and this variable, so every solver
        # takes `pair_reuse=` and MOMWIRE_X_PAIR_REUSE turns on the reuse it
        # supports. Their own keywords and variables are kept as aliases.
        Feature(
            name="pair_reuse",
            reach=1,
            env="MOMWIRE_X_PAIR_REUSE",
            kind="python",
            issue=1809,
            since="2026-11-04",
            kwarg="pair_reuse",
            paths=(
                "src/momwire/_sin_pair_reuse.py",
                "tests/test_pair_reuse_1809.py",
                "tests/test_sin_pair_reuse_1911.py",
                "tests/test_sin_pair_reuse_ground_1970.py",
                "tests/test_sin_pair_reuse_canonical_1969.py",
                "tests/test_sin_pair_reuse_small_deck_1969.py",
                "tests/test_pair_reuse_contract_1982.py",
            ),
            label="pair reuse",
            summary=(
                "Fills each repeated pose of a pair of wires or segments once and "
                "copies the rest. nec5-bspline and nec42-bspline need the front "
                "end's hint of which edge pairs share a pose, and also fill a "
                "uniform wire's own entries once per segment offset; the "
                "sinusoidal solver and nec5-razor-2p find their own classes, in "
                "free space and over a perfect or finite ground, on decks large "
                "enough to pay for the search."
            ),
            # |dZin|/|Zin|, the registry's measure, against the flag-off fill.
            # bspline: 1e-12. sinusoidal: the class map moves Z itself by
            # ~1e-12 relative to its largest entry, and the solve amplifies
            # that at the feed. Measured over every frequency of the 11-16
            # reuse ladders (13 each; gsfd_* 101), dense:
            #   free space (Rocketlake, main 816ccd5b): p_m24 2.15e-10, sky_c24 7.6e-11,
            #     sky_m24 6.6e-11, fd_h51 9.6e-12;
            #   finite ground (Skylake, the momwire#1970 change): gsky_m24 9.9e-11
            #     (Sommerfeld), 8.2e-11 (reflection coefficient); gfd_h51
            #     1.0e-11 / 4.1e-12; gsky_m12 7.9e-12; every other gfd_*,
            #     gsfd_*, gsky_m6 6.1e-12 or less.
            # 5e-10 is the worst (p_m24) with about 2x margin. The 3e-12 this
            # replaces was the matrix ratio, not |dZin|/|Zin|. Still far under
            # DEFAULT_DRIFT_MAX (antennaknobs' LAST_DIGITS_DRIFT, 1e-6).
            drift=5e-10,
            offer=False,
        ),
        # A uniform edge's same-edge reg moments filled by segment offset
        # (PR #1815): the shipped kernels, summed in another order. A part of
        # `pair_reuse` (momwire#1982): on with it, and its own keyword and
        # variable still turn it on or off alone. Not retired: within one
        # edge the moment depends only on the segment offset, which a whole-
        # block copy between edge pairs does not use, so the two compose.
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
                "Fills a uniform edge's same-edge moments once per segment offset;"
                " turned on with pair reuse, alongside its copies between edges."
            ),
            drift=1e-12,
            offer=False,
            part_of="pair_reuse",
        ),
        # The transmitted (below->above) grid blended from frequency anchors
        # on a lattice fixed in metres (momwire#1847), so a sweep shares its
        # tables: the shipped evaluator, interpolated in ln f.
        #
        # `drift` is the survey's worst with 2.3x margin (momwire#2022): flag
        # on against off, |dZin|/|Zin|, 323 served sweep points on bs2,
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
        # the member route and not bit-identical to it. Reach 8: the
        # nec5-razor-2p, nec5-bspline and nec42-bspline executables carry it
        # (the ports of #1860's measurement build). On by default since
        # 2026-11-15 (Steve), as is height_sheet_recal: certified through
        # level 8, and level 9's nec42-sinusoidal route has no column twin to
        # factor.
        Feature(
            name="factored_columns",
            reach=8,
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
            default=True,
        ),
        # The near-interface height-sheet census recalibrated (momwire#1860
        # L2): a horizontal wire's rows on a height sheet at their measured
        # worth, and heights a few ulp apart taken as one sheet. The shipped
        # sheets and twin, a different split between them. Reach 8, as
        # factored_columns.
        Feature(
            name="height_sheet_recal",
            reach=8,
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
            default=True,
        ),
        # The crossing table (momwire#1860 L5): the crossing rows inside a
        # fixed wavelength-scaled grid read from one table per frequency, the
        # singular part taken out in closed form. A new C++ path and a coarse
        # accuracy class (printed Zin within ~0.1 ohm on the EFHW decks).
        # Reach 8, as factored_columns. Opt-in: its drift class is over
        # DEFAULT_DRIFT_MAX, and it stays off by default (Steve, 11-15).
        # drift 5e-3 (momwire#1955, was 1e-3): measured 3.6e-3 on fat_slant
        # and 2.5e-3 on fat_hub (a 19 mm vertical, Delta/a 2.2, over four short
        # buried radials), 4e-4 on aca, under 1e-3 on the EFHW decks. Set
        # against the ground itself: +/-10 % in eps_r or sigma moves those
        # same decks' Zin by 0.7-4.5 %, so 5e-3 stays well inside what a user
        # can know about their soil.
        Feature(
            name="crossing_table",
            reach=8,
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
                "ohm on an EFHW and up to about 0.4 % on a fat vertical over a "
                "small buried screen, well under what a 10 % error in the ground's"
                " constants does."
            ),
            drift=5e-3,
            offer=True,
        ),
        # nec5-bspline's and nec42-bspline's crossing route fills the above x
        # above Sommerfeld remainder at the no-rod deck's rule (momwire#1860
        # B1): the base order plus #1189's raised pairs, where the shipped
        # route takes the cross block's q = 6, on a grid no dearer than the
        # shipped one (else that fill declines). The shipped kernels, a
        # different quadrature. Reach 8: the two bspline executables carry it;
        # nec5-razor-2p's accepts the variable and, as Python's razor-2p,
        # ignores it.
        Feature(
            name="crossing_above_rule",
            reach=8,
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
            # Not default-on: approved for promotion 2026-11-15, held back
            # because the pre-tag sweep measured |dZin|/|Zin| = 1.28e-6 on
            # default|sommerfeld|ground_rod_efhw_wa7ark (nec5-bspline, Skylake),
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
        # Reach 7 (momwire#1956 promotion, 2026-11-17): the two nec5
        # executables carry it (their C++ census reads the same two
        # constants, `ni_eval::sheet_build_weight`); the nec42-bspline
        # executable (level 8) does not, and refuses it.
        Feature(
            name="crossing_sheet_retune",
            reach=7,
            env="MOMWIRE_X_CROSSING_SHEET_RETUNE",
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
            # (Skylake, T4): at most 6.2e-13 (fat_hub, nec5-razor-2p;
            # nec5-bspline 3.2e-13), the plane sheets' own interpolation error
            # where a row moves from the column twin to a sheet. Deeper and
            # higher in frequency the plan takes more planes (2 -> 51-75) and
            # the error grows with them: a 16-radial hub at 1.5 m, 28 MHz,
            # 2.5 m radials and vertical measures 9.77e-12 on razor-2p
            # (1.2e-13 nec5-bspline), 3.0 m at 14 MHz 6.15e-12, 5.0 m at
            # 14 MHz 3.65e-12; the same to the bit on Haswell and Skylake.
            # (Those figures are the Python route's, on the dict-built decks
            # of tests/test_crossing_sheet_retune_1956.py: the 1.5 m / 28 MHz
            # one is `_deep_hub(28e6, 1.5, 2.5)`.)
            # Re-measured at the promotion to reach 7 (Rocketlake, T1,
            # MW_EXE_ARRAYS) through the nec5 executables, on a scan of the
            # same 16-radial hub written as a NEC-5 deck (`hub_nec` there:
            # 7-30 MHz, depth 0.15-3 m, radials and vertical 2.5-7.5 m), a
            # different geometry from `_deep_hub` (radials laid from the hub
            # out, the vertical and its 7th-of-15 feed counted from the base;
            # the worst case 30 MHz and 5 m, not 28 MHz and 2.5 m). The worst
            # measured: 1.93e-11 on
            # nec5-bspline at 30 MHz, 1.5 m, 5 m (that module's
            # DECKS["deep_hub"], 1.40e-11 on nec5-razor-2p there), and
            # 1.56e-11 on nec5-razor-2p at 28 MHz, 0.5 m, 5 m; the sse2
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
            kind="python",
            issue=1898,
            since="2026-11-11",
            kwarg="kcl_nullspace",
            paths=(
                "src/momwire/_kcl_nullspace.py",
                "tests/test_kcl_nullspace_1898.py",
                "tests/test_kcl_nullspace_cxx_1898.py",
            ),
            label="eliminate junction constraints",
            summary=(
                "Solve nec5-bspline's and nec42-bspline's junction conservation "
                "by eliminating one wire end per junction before the factor, "
                "instead of carrying the constraints through the solve. Fewer "
                "unknowns and no extra solves on junction-dense designs (coils, "
                "cages, grids)."
            ),
            # Measured on the ELT whip (2,029 rows), nec5-bspline degrees 1
            # and 2, free space and Sommerfeld: at most 6.7e-13.
            drift=1e-12,
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
        # razor-2p's fill-pair reuse (momwire#1971): the fill finds its own
        # classes of (centroid observer, source segment) pairs, the direct
        # and image moments keyed on what the kernel reads and the composing
        # ground's remainder on the height-aware key, evaluates one
        # representative per class with the shipped kernels and gathers the
        # rest, in Python and in the executables (one C++ core).
        Feature(
            name="razor_pair_reuse",
            # Reach 7: every Python route (the hosted app, Linux sse2,
            # Windows, the drop-in, macOS) and the nec5 executables (levels 6
            # and 7, nec5-razor-2p's `EXE_FEATURES` row; nec5-bspline, which
            # shares those levels, refuses it by basis). Not 8 or 9: those
            # are the nec42 executables, and neither has a razor-2p route to
            # run it in. Still off by default and not offered (Steve, 11-17;
            # it was reach 1, the hosted app only, from 11-16).
            reach=7,
            env="MOMWIRE_X_RAZOR_PAIR_REUSE",
            kind="native",
            issue=1971,
            since="2026-11-16",
            kwarg="razor_pair_reuse",
            paths=(
                "src/momwire/_razor_pair_reuse_core.cpp",
                "src/momwire/_razor_pair_reuse_core.h",
                "tests/test_razor_pair_reuse_1971.py",
            ),
            label="razor pair reuse",
            summary=(
                "nec5-razor-2p fills each class of congruent segment pairs once "
                "and copies the rest, in free space and over a perfect or "
                "Sommerfeld ground (the buried fill excepted)."
            ),
            # |dZin|/|Zin| against the flag-off fill, the worst over every
            # frequency, the extended kernel on (the NEC-5 shell's), each
            # deck with exact_kernel auto / True / False (Python route,
            # Rocketlake, momwire#1971):
            #
            #   the six benchmark decks (1097-2535 segs, 13 frequencies):
            #     gsky_m24 2.6e-13 (Sommerfeld), fd_h51 2.6e-13, sky_m24
            #     2.0e-13, p_m24 1.9e-13, gfd_h51 1.5e-13 (Sommerfeld),
            #     fd_h27 3.9e-14;
            #   tiny bridges, h < a (a 2 mm hairpin, 21-segment bridge): 3.1e-12
            #     free, 3.7e-12 Sommerfeld, 3.4e-12 translated 3.3 m (each
            #     with exact_kernel False; 1.4e-12 auto / True); a folded
            #     dipole, a shorted stub, a thick monopole: <= 1.3e-13;
            #   close-spaced two-wire lines (10 m, 1 mm wire, 2.1 to 12 mm
            #     apart, 41 to 161 segments a side, free / PEC / Sommerfeld,
            #     at the origin and translated 3.3 m): 4.3e-10 at the worst
            #     (2.1 mm, 41 segments, free, exact_kernel False), 4.2e-10 at
            #     2.5 mm, 1e-10 at 6 mm, 4e-11 at 12 mm.
            #
            # The lines' drift is the fill's own rounding, not a wrong class:
            # the flag-off fill moves as much when the same deck is translated
            # 3.3 m (up to 9.4e-10 at 2.1 mm), the pairs differ from their
            # representatives only by coordinate rounding (KEY_BITS), and
            # `verify` passes. 2e-9 is about 5x the worst.
            drift=2e-9,
            offer=False,
            # momwire#1982: on with `pair_reuse`; its own keyword and variable
            # are aliases. Outside its reach, asked for only that way, off.
            part_of="pair_reuse",
        ),
        # razor-2p's cross blocks across a frequency sweep from anchors
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
        # `anchors`): the nec5-razor-2p executable's fill and RazorSolver's
        # (`_crossing_anchors_now`, which fills its anchors with its own
        # cross terms), so the two print the same bytes. It acts where a
        # seam hands the fill its sweep: the NEC-5 seam's stepping FR, in
        # C++ and in the drop-in. Reach 7: the Python routes and the nec5
        # executables (nec5-razor-2p's alone, `EXE_FEATURE_BASES`).
        Feature(
            name="crossing_anchors",
            reach=7,
            env="MOMWIRE_X_CROSSING_ANCHORS",
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
            # |dZin|/|Zin| against off, full precision (the nec5-razor-2p
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


def _env_value(feature: Feature) -> bool | None:
    """What the environment says of `feature`: True (on), False (set to
    anything else, "0" say) or None (unset or empty: the default)."""
    for var in (feature.env, *feature.aliases):
        value = os.environ.get(var)
        if value is not None and value.strip():
            return value.strip().lower() in _ON
    return None


def _env_on(feature: Feature) -> bool:
    return _env_value(feature) is True


def default_on(name: str, context: str | None = None) -> bool:
    """Whether `name` is on when nothing asks either way: a default-on
    feature where it is certified (`context` defaulting to the running
    one), else no."""
    feature = FEATURES[name]
    if not feature.default:
        return False
    return certified(name, _CONTEXT.get() if context is None else context)


def _resolve(
    name: str, kwarg_value: object, whole_kwarg: object, context: str | None
) -> tuple[bool, bool]:
    """(on, through the whole): what :func:`enabled` decides before the
    reach check, and whether it was the whole feature (``part_of``) that
    asked."""
    feature = FEATURES[name]
    if kwarg_value is not None:
        return bool(kwarg_value), False
    if feature.part_of is not None and whole_kwarg is not None:
        return bool(whole_kwarg), True
    env = _env_value(feature)
    if env is not None:
        return env, False
    if feature.part_of is not None:
        env = _env_value(FEATURES[feature.part_of])
        if env is not None:
            return env, True
    return default_on(name, context), False


def enabled(
    name: str,
    kwarg_value: object = None,
    *,
    context: str | None = None,
    whole_kwarg: object = None,
) -> bool:
    """Whether `name` is on: the keyword's value when one was passed (not
    None), else its environment variable when set, else its default
    (:func:`default_on`: off, unless the feature is promoted and certified
    here). A native feature that is on outside its reach raises
    :class:`FeatureNotCertified`.

    A part (``part_of``) also reads the whole feature, after its own
    keyword and before its own variable: its own keyword, then the whole's
    keyword (`whole_kwarg`, the value the caller was given for the whole),
    then its own variable, then the whole's, then its default. Turned on
    by the whole alone, a native part outside its reach is off rather than
    refused (:func:`withheld`)."""
    on, through_whole = _resolve(name, kwarg_value, whole_kwarg, context)
    if on and through_whole and _withheld_here(name, context):
        return False
    if on:
        check(name, context)
    return on


def _withheld_here(name: str, context: str | None) -> bool:
    feature = FEATURES[name]
    context = _CONTEXT.get() if context is None else context
    return feature.kind == "native" and not certified(name, context)


def withheld(
    name: str,
    kwarg_value: object = None,
    *,
    context: str | None = None,
    whole_kwarg: object = None,
) -> bool:
    """Whether the whole feature asked for part `name` and it stayed off
    because it is native and outside its reach here (:func:`enabled`'s
    soft refusal), so a solver can say why it did not reuse."""
    on, through_whole = _resolve(name, kwarg_value, whole_kwarg, context)
    return on and through_whole and _withheld_here(name, context)


def active() -> list[str]:
    """The features the environment turns on, in registry order (a
    default-on one only when its variable says on, not by its default)."""
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
        f"{feature.reach}, {LEVELS[feature.reach - 1].id}); unset "
        f"{feature.env} to run without it"
    )


def stamp(context: str | None = None) -> str:
    """What a printout's line 2 adds for the features the environment turns
    on where their default is off: ``""`` when there are none (the line is
    then byte-identical to one without features), else ``"+a+b"``, followed
    by ``" uncertified on this platform"`` when any of them is outside its
    reach here (`context` defaulting to the running one). A default-on
    feature is never named: on, it is the stable path; turned off, the
    printout is the one before it was promoted."""
    context = _CONTEXT.get() if context is None else context
    names = [n for n in active() if not default_on(n, context)]
    if not names:
        return ""
    text = "".join(f"+{n}" for n in names)
    if not all(certified(n, context) for n in names):
        text += " uncertified on this platform"
    return text
