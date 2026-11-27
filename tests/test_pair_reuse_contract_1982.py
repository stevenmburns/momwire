"""One pair-reuse contract (momwire#1982 step 1; one name since momwire#2063
step 4).

`pair_reuse` is the one pair-reuse switch: one keyword, taken by every solver
that reuses (bspline, sinusoidal, razor2p), and one variable,
MOMWIRE_X_PAIR_REUSE. What each family runs of it, and how far that is
certified, is data on its one row (`Feature.families`). What is held here:

- the registry: one feature, a row per family, none of them a switch, and no
  default turned on;
- the order it is read in on a family: its keyword, the family's deprecated
  keyword, its variable, the family's deprecated variable, a pragma, its
  default;
- a native family row (razor2p's) asked for outside its reach is OFF (and
  `withheld` says so), by any spelling;
- line 2 names `pair_reuse` alone, and is uncertified where the family row
  that runs is past its reach;
- each solver: the switch turns its reuse on and off, and
  `pair_reuse_declined` is None or a str everywhere.

The deprecated spellings themselves, and their precedence against
`pair_reuse`, are tests/test_one_reuse_switch_2063.py's.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pytest

from momwire import _features as F
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

ALIASES = {"offset_reuse": "bspline", "razor_pair_reuse": "razor"}


@pytest.fixture
def no_features(monkeypatch):
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
        for fam in f.families:
            if fam.alias_env is not None:
                monkeypatch.delenv(fam.alias_env, raising=False)
    return monkeypatch


def _hold_row(monkeypatch, solver, reach):
    """Hold `pair_reuse`'s row for `solver` to `reach` for one test."""
    pr = F.FEATURES["pair_reuse"]
    rows = tuple(
        dataclasses.replace(r, reach=reach) if r.solver == solver else r
        for r in pr.families
    )
    monkeypatch.setitem(
        F.FEATURES, "pair_reuse", dataclasses.replace(pr, families=rows)
    )


# --- the registry ---------------------------------------------------------------


def test_pair_reuse_is_the_one_switch():
    whole = F.FEATURES["pair_reuse"]
    assert whole.kwarg == "pair_reuse" and whole.env == "MOMWIRE_X_PAIR_REUSE"
    assert not whole.default and not whole.offer
    assert {r.solver for r in whole.families} == set(whole.solvers)
    assert {r.alias: r.solver for r in whole.families if r.alias} == ALIASES
    # No row of the registry is one of its old parts.
    assert not set(ALIASES) & set(F.FEATURES)
    assert not any(hasattr(f, "part_of") for f in F.FEATURES.values())


def test_every_reusing_solver_takes_the_one_keyword():
    # antennaknobs probes this keyword on the constructor (`_takes_pair_reuse`).
    for cls in (BSplineSolver, SinusoidalSolver, RazorSolver):
        assert "pair_reuse" in inspect.signature(cls.__init__).parameters, cls
    # The deprecated spellings are still taken, for two minor releases.
    assert "offset_reuse" in inspect.signature(BSplineSolver.__init__).parameters
    assert "razor_pair_reuse" in inspect.signature(RazorSolver.__init__).parameters


# --- the order it is read in ------------------------------------------------------


def test_the_switch_on_each_family(no_features):
    for solver in ("bspline", "razor", "sinusoidal", None):
        assert F.enabled("pair_reuse", solver=solver) is False
        assert F.enabled("pair_reuse", True, solver=solver) is True
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    for solver in ("bspline", "razor", "sinusoidal", None):
        assert F.enabled("pair_reuse", solver=solver) is True
        # a keyword over a variable: a request can say no
        assert F.enabled("pair_reuse", False, solver=solver) is False
    assert F.active() == ["pair_reuse"]


def test_a_feature_without_family_rows_ignores_the_spellings(no_features):
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    assert F.enabled("transmitted_anchors", solver="bspline") is False
    assert F.active(solver="bspline") == ["pair_reuse"]


def test_a_native_family_row_is_off_outside_its_reach(no_features):
    row = F.family("pair_reuse", "razor")
    assert row is not None and row.kind == "native"
    # This branch must not change a reach another branch widens, so it holds
    # the row to an earlier reach, 4, which the macOS drop-in lies past.
    no_features.setattr(F, "_os", lambda platform=None: "macos")
    _hold_row(no_features, "razor", 4)
    for kw in ({"kwarg_value": True}, {"alias_value": True}, {}):
        if not kw:
            no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
        assert F.enabled("pair_reuse", context="dropin", solver="razor", **kw) is False
        assert F.withheld("pair_reuse", context="dropin", solver="razor", **kw) is True
        # never refused, and the other families run it
        assert F.enabled("pair_reuse", context="dropin", solver="bspline", **kw)
        assert not F.withheld("pair_reuse", context="dropin", solver="bspline", **kw)
    assert F.certified("pair_reuse", "dropin") is True
    assert F.certified("pair_reuse", "dropin", solver="razor") is False


def test_withheld_is_false_unless_asked(no_features):
    for solver in ("bspline", "razor", "sinusoidal"):
        assert F.withheld("pair_reuse", solver=solver) is False
        assert F.withheld("pair_reuse", False, solver=solver) is False
    # A Python family row runs outside its reach, so it is never withheld.
    assert F.withheld("pair_reuse", True, solver="bspline", context="dropin") is False


def test_the_stamp_names_the_switch(no_features):
    """Line 2 names `pair_reuse` only. Its certification is what RUNS: the
    switch reaches every level, and on bspline its row reaches 9, every level
    bspline runs on, so no shipped level stamps it uncertified."""
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    for context in (None, "dropin"):
        args = (context,) if context else ()
        assert F.stamp(*args) == "+pair_reuse", context
        for solver in ("bspline", "razor", "sinusoidal"):
            assert F.stamp(*args, solver=solver) == "+pair_reuse", (context, solver)
    assert F.stamp(solver="") == ""


def test_pair_reuse_reaches_every_level():
    """Reach 11, the deepest: every Python route and drop-in, and every
    executable, each of which accepts MOMWIRE_X_PAIR_REUSE and runs its own
    basis's reuse. Still Python-kind, off by default and not offered. Each
    family row keeps the reach its route is certified to."""
    whole = F.FEATURES["pair_reuse"]
    assert whole.reach == F.DEEPEST and whole.stable
    assert whole.kind == "python" and not whole.default and not whole.offer
    for platform in ("linux", "win32", "darwin"):
        assert F.certified("pair_reuse", "dropin", platform=platform)
        assert F.certified("pair_reuse", platform=platform, variant="sse2")
    for basis in (
        "razor2p",
        "bs2",
        "nec42-bs2",
        "nec42-sinusoidal",
        "nec2c-sinusoidal",
    ):
        assert F.certified("pair_reuse", "exe", basis=basis, platform="linux")
        family = F.basis_solver(basis)
        assert F.certified(
            "pair_reuse", "exe", basis=basis, platform="linux", solver=family
        ), basis
    # bspline's row reaches the bs2 executables (nec42-bs2 the deepest),
    # razor2p's the nec5 executables, the sinusoidal row every level
    assert F.family("pair_reuse", "bspline").reach == F._level("exe-nec42-bs2")
    assert F.family("pair_reuse", "razor").reach == F._level(
        "exe-nec5-bs2-razor2p-linux-macos"
    )
    assert F.family("pair_reuse", "sinusoidal").reach == F.DEEPEST


def test_a_family_row_past_its_reach_stamps_uncertified(no_features):
    """The switch certified, the family row that runs not: line 2 says
    uncertified on that family alone. No shipped level puts bspline past its
    row's reach (9), so the row is held to an earlier reach, 4, which the
    macOS drop-in (level 5) lies past, inside razor2p's 8."""
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    no_features.setattr(F, "_os", lambda platform=None: "macos")
    assert F.current_level("dropin") == F._level("macos")
    assert F.stamp("dropin", solver="bspline") == "+pair_reuse"
    _hold_row(no_features, "bspline", 4)
    tail = " uncertified on this platform"
    assert F.stamp("dropin", solver="bspline") == "+pair_reuse" + tail
    assert F.stamp("dropin", solver="razor") == "+pair_reuse"
    assert F.stamp("dropin", solver="sinusoidal") == "+pair_reuse"
    # unknown family: every row the environment turns on counts
    assert F.stamp("dropin") == "+pair_reuse" + tail
    # the switch off: nothing runs
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "0")
    assert F.stamp("dropin", solver="bspline") == ""


# --- bspline -------------------------------------------------------------------


def _bspline(**kw):
    wire = [[[0.0, -2.5, 10.0], [0.0, 2.5, 10.0]]]
    return BSplineSolver(wires=wire, n_per_edge_per_wire=[[21]], wire_radius=1e-3, **kw)


def test_bspline_pair_reuse_turns_on_the_census_and_the_toeplitz_form(no_features):
    s = _bspline()
    assert s._offset_reuse is False and s._reuse_on is False
    assert s.pair_reuse_declined is None
    s = _bspline(pair_reuse=True)  # on: the census finds the classes
    assert s._offset_reuse is True and s._reuse_on is True
    s = _bspline(pair_reuse="verify")  # the other solvers' check: on here
    assert s._offset_reuse is True and s._reuse_on is True
    s = _bspline(pair_reuse=False)
    assert s._offset_reuse is False and s._reuse_on is False
    hint = [[(0, 0), (0, 0)]]
    s = _bspline(pair_reuse=hint)  # a front end's hint reads as on, unkept
    assert s._offset_reuse is True and s._reuse_on is True
    assert not any(v is hint for v in vars(s).values())
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert _bspline()._offset_reuse is True
    assert _bspline(pair_reuse=False)._offset_reuse is False


def test_bspline_pair_reuse_runs_the_toeplitz_form_below_the_census(no_features):
    # One 21-segment wire, under the census's small-deck threshold: the plan
    # declines by name, and the switch still fills the wire's own block in
    # its Toeplitz form, which sums in another order, a few ulp from off.
    whole = _bspline(pair_reuse=True)
    z_whole = np.atleast_1d(np.asarray(whole.compute_impedance()[0], dtype=complex))
    assert whole.pair_reuse_declined.startswith("too few segments to repay the plan")
    off = np.atleast_1d(np.asarray(_bspline().compute_impedance()[0], dtype=complex))
    assert not np.array_equal(z_whole.view(np.uint64), off.view(np.uint64))
    # the Toeplitz form is what moved it: held off, the declined plan is off
    held = _bspline(pair_reuse=True)
    held._offset_reuse = False
    z_held = np.atleast_1d(np.asarray(held.compute_impedance()[0], dtype=complex))
    assert np.array_equal(z_held.view(np.uint64), off.view(np.uint64))


# --- razor2p ------------------------------------------------------------------


def _razor(**kw):
    return RazorSolver(
        wires=[np.array([[0.0, -5.0, 8.0], [0.0, 5.0, 8.0]])],
        nsegs=31,
        wire_radius=0.001,
        wavelength=20.0,
        nec5_quadrature=True,
        **kw,
    )


@pytest.mark.skipif(
    not F.certified("pair_reuse", solver="razor"),
    reason="razor2p's pair reuse is not certified on this platform",
)
def test_razor_takes_the_one_switch(no_features):
    assert _razor()._pair_reuse == 0
    assert _razor(pair_reuse=True)._pair_reuse == 1
    assert _razor(pair_reuse="verify")._pair_reuse == 2
    # a front end's bspline hint is on, as on the sinusoidal solver
    assert _razor(pair_reuse=[[(0, 0), (0, 0)]])._pair_reuse == 1
    assert _razor(pair_reuse=False)._pair_reuse == 0
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert _razor()._pair_reuse == 1
    assert _razor(pair_reuse=False)._pair_reuse == 0
    for s in (_razor(), _razor(pair_reuse=True)):
        assert s.pair_reuse_declined is None


def test_razor_outside_its_reach_is_off_and_says_so(no_features):
    _hold_row(no_features, "razor", 4)
    no_features.setattr(F, "_os", lambda platform=None: "macos")
    with F.context_scope("dropin"):
        for kw in ({"pair_reuse": True}, {"razor_pair_reuse": True}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                s = _razor(**kw)
            assert s._pair_reuse == 0, kw
            assert s.pair_reuse_declined == "not certified on this platform", kw
            assert s.features_withheld == ("pair_reuse",), kw


# --- one readout type ----------------------------------------------------------


def test_pair_reuse_declined_is_none_or_a_str_on_every_solver(no_features):
    sin = SinusoidalSolver(
        wires=[np.array([[0.0, -2.6, 8.0], [0.0, 2.6, 8.0]])],
        n_per_edge_per_wire=[[41]],
        wavelength=10.7,
        wire_radius=1e-3,
        pair_reuse=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sin.compute_port_solution()
    assert isinstance(sin.pair_reuse_declined, str)  # a small deck
    for s in (_bspline(), _bspline(pair_reuse=True), _razor()):
        assert s.pair_reuse_declined is None
