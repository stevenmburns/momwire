"""One pair-reuse contract (momwire#1982 step 1).

`pair_reuse` is the one user-facing pair-reuse switch: one keyword, taken by
every solver that reuses (bspline, sinusoidal, razor2p), and one variable,
MOMWIRE_X_PAIR_REUSE. Its parts, bspline's `offset_reuse` and razor2p's
`razor_pair_reuse`, follow it; their own keywords and variables are kept as
aliases and win. What is held here:

- the registry: two parts, both of `pair_reuse`, neither offered alone, and
  no default turned on;
- the order a part is read in: its keyword, the whole's keyword, its
  variable, the whole's variable, its default;
- a native part asked for only through the whole is OFF outside its reach
  (and `withheld` says so), where its own keyword still refuses;
- line 2 names only what the environment set, so the whole's variable
  stamps "+pair_reuse" and not its parts;
- each solver: the whole turns its reuse on, the alias wins, and
  `pair_reuse_declined` is None or a str everywhere.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from momwire import _features as F
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

PARTS = {"offset_reuse", "razor_pair_reuse"}


@pytest.fixture
def no_features(monkeypatch):
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    return monkeypatch


# --- the registry ---------------------------------------------------------------


def test_pair_reuse_is_the_one_user_facing_switch():
    parts = {n for n, f in F.FEATURES.items() if f.part_of is not None}
    assert parts == PARTS
    whole = F.FEATURES["pair_reuse"]
    assert whole.part_of is None and whole.kwarg == "pair_reuse"
    assert whole.env == "MOMWIRE_X_PAIR_REUSE"
    for name in PARTS:
        f = F.FEATURES[name]
        assert f.part_of == "pair_reuse"
        # A part is never offered alone, and nothing here is on by default.
        assert not f.offer and not f.default
    assert not whole.default
    # A part of a part would need an order nobody wrote down.
    for f in F.FEATURES.values():
        if f.part_of is not None:
            assert f.part_of in F.FEATURES
            assert F.FEATURES[f.part_of].part_of is None


def test_offset_reuse_does_not_read_as_retired():
    summary = F.FEATURES["offset_reuse"].summary.lower()
    assert "folded" not in summary and "pair reuse" in summary


def test_every_reusing_solver_takes_the_one_keyword():
    # antennaknobs probes this keyword on the constructor (`_takes_pair_reuse`).
    for cls in (BSplineSolver, SinusoidalSolver, RazorSolver):
        assert "pair_reuse" in inspect.signature(cls.__init__).parameters, cls
    # The aliases stay.
    assert "offset_reuse" in inspect.signature(BSplineSolver.__init__).parameters
    assert "razor_pair_reuse" in inspect.signature(RazorSolver.__init__).parameters


# --- the order a part is read in -----------------------------------------------


def test_the_whole_turns_a_part_on(no_features):
    assert F.enabled("offset_reuse") is False
    assert F.enabled("offset_reuse", whole_kwarg=True) is True
    assert F.enabled("offset_reuse", whole_kwarg=False) is False
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert F.enabled("offset_reuse") is True
    # The whole's variable is not the part's: only the whole is active.
    assert F.active() == ["pair_reuse"]


def test_the_part_s_own_spelling_wins(no_features):
    # its keyword over the whole's keyword
    assert F.enabled("offset_reuse", False, whole_kwarg=True) is False
    assert F.enabled("offset_reuse", True, whole_kwarg=False) is True
    # its variable over the whole's variable
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "0")
    assert F.enabled("offset_reuse") is False
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "0")
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    assert F.enabled("offset_reuse") is True
    # a keyword (the whole's) over a variable (the part's): a request can say no
    assert F.enabled("offset_reuse", whole_kwarg=False) is False


def test_a_feature_that_is_no_part_ignores_the_whole(no_features):
    assert F.enabled("transmitted_anchors", whole_kwarg=True) is False
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert F.enabled("transmitted_anchors") is False


def test_a_native_part_through_the_whole_is_off_outside_its_reach(no_features):
    f = F.FEATURES["razor_pair_reuse"]
    assert f.kind == "native"
    # The drop-in is past reach 1 everywhere; this branch must not change a
    # reach another branch widens, so it only asks where the part reaches.
    if f.reach >= F.current_level("dropin"):
        pytest.skip("razor_pair_reuse reaches the drop-in here")
    for kw in ({"whole_kwarg": True}, {}):
        if not kw:
            no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
        assert F.enabled("razor_pair_reuse", context="dropin", **kw) is False
        assert F.withheld("razor_pair_reuse", context="dropin", **kw) is True
    # Its own keyword and variable still refuse there.
    with pytest.raises(F.FeatureNotCertified):
        F.enabled("razor_pair_reuse", True, context="dropin")
    no_features.setenv("MOMWIRE_X_RAZOR_PAIR_REUSE", "1")
    with pytest.raises(F.FeatureNotCertified):
        F.enabled("razor_pair_reuse", context="dropin")
    assert F.withheld("razor_pair_reuse", context="dropin") is False


def test_withheld_is_false_unless_the_whole_asked(no_features):
    for name in PARTS:
        assert F.withheld(name) is False
        assert F.withheld(name, whole_kwarg=False) is False
    # A Python part runs outside its reach, so it is never withheld.
    assert F.withheld("offset_reuse", whole_kwarg=True, context="dropin") is False


def test_the_stamp_names_the_whole_not_its_parts(no_features):
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    plain = "" if F.certified("pair_reuse") else " uncertified on this platform"
    assert F.stamp() == "+pair_reuse" + plain
    dropin = (
        "" if F.certified("pair_reuse", "dropin") else " uncertified on this platform"
    )
    assert F.stamp("dropin") == "+pair_reuse" + dropin


def test_pair_reuse_reaches_every_level():
    """Reach 11, the deepest: every Python route and drop-in, and every
    executable, each of which accepts MOMWIRE_X_PAIR_REUSE and runs its own
    basis's reuse (`EXE_PARTS`, held to the registry in
    tests/test_features_registry.py). Still Python-kind, off by default and
    not offered. Its parts keep their own reaches: on a family the whole acts
    on through a part, what runs is certified only as far as the part."""
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
    assert F.FEATURES["offset_reuse"].reach == F._level("dropin-linux-windows")
    assert F.FEATURES["razor_pair_reuse"].reach == F._level(
        "exe-nec5-bs2-razor2p-linux-macos"
    )


def test_a_part_past_its_reach_stamps_the_whole_uncertified(no_features):
    """The whole certified, the part that runs not: line 2 says uncertified
    on that family alone (the macOS drop-in: level 5, past offset_reuse's
    4, inside razor_pair_reuse's 8; sinusoidal has no part)."""
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    no_features.setattr(F, "_os", lambda platform=None: "macos")
    assert F.current_level("dropin") == F._level("macos")
    tail = " uncertified on this platform"
    assert F.stamp("dropin", solver="bspline") == "+pair_reuse" + tail
    assert F.stamp("dropin", solver="razor") == "+pair_reuse"
    assert F.stamp("dropin", solver="sinusoidal") == "+pair_reuse"
    # unknown family: every part that runs counts
    assert F.stamp("dropin") == "+pair_reuse" + tail
    # the bspline part set off: nothing of the whole runs there
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "0")
    assert F.stamp("dropin", solver="bspline") == ""


# --- bspline -------------------------------------------------------------------


def _bspline(**kw):
    wire = [[[0.0, -2.5, 10.0], [0.0, 2.5, 10.0]]]
    return BSplineSolver(wires=wire, n_per_edge_per_wire=[[21]], wire_radius=1e-3, **kw)


def test_bspline_pair_reuse_turns_on_offset_reuse(no_features):
    s = _bspline()
    assert s._offset_reuse is False and s._pair_reuse_given is None
    assert s.pair_reuse_declined is None
    s = _bspline(pair_reuse=True)  # on, with no hint to copy
    assert s._offset_reuse is True and s._pair_reuse_given is None
    s = _bspline(pair_reuse="verify")  # the other solvers' check: on here
    assert s._offset_reuse is True and s._pair_reuse_given is None
    s = _bspline(pair_reuse=False)
    assert s._offset_reuse is False and s._pair_reuse_given is None
    hint = [[(0, 0), (0, 0)]]
    s = _bspline(pair_reuse=hint)
    assert s._offset_reuse is True and s._pair_reuse_given is hint
    # The alias wins.
    assert _bspline(pair_reuse=hint, offset_reuse=False)._offset_reuse is False
    assert _bspline(pair_reuse=False, offset_reuse=True)._offset_reuse is True
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert _bspline()._offset_reuse is True
    assert _bspline(pair_reuse=False)._offset_reuse is False


def test_bspline_with_pair_reuse_true_is_offset_reuse_to_the_bit(no_features):
    z_alias = _bspline(offset_reuse=True).compute_impedance()[0]
    z_whole = _bspline(pair_reuse=True).compute_impedance()[0]
    z_off = _bspline().compute_impedance()[0]
    a, b = (np.atleast_1d(np.asarray(z, dtype=complex)) for z in (z_alias, z_whole))
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    # and it ran: offset reuse sums in another order, a few ulp from off
    off = np.atleast_1d(np.asarray(z_off, dtype=complex))
    assert not np.array_equal(a.view(np.uint64), off.view(np.uint64))


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
    not F.certified("razor_pair_reuse"),
    reason="razor_pair_reuse is not certified on this platform",
)
def test_razor_takes_the_one_switch(no_features):
    assert _razor()._pair_reuse == 0
    assert _razor(pair_reuse=True)._pair_reuse == 1
    assert _razor(pair_reuse="verify")._pair_reuse == 2
    # a front end's bspline hint is on, as on the sinusoidal solver
    assert _razor(pair_reuse=[[(0, 0), (0, 0)]])._pair_reuse == 1
    assert _razor(pair_reuse=False)._pair_reuse == 0
    # the alias wins, both ways
    assert _razor(pair_reuse=True, razor_pair_reuse=False)._pair_reuse == 0
    assert _razor(pair_reuse=False, razor_pair_reuse="verify")._pair_reuse == 2
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert _razor()._pair_reuse == 1
    assert _razor(pair_reuse=False)._pair_reuse == 0
    no_features.setenv("MOMWIRE_X_RAZOR_PAIR_REUSE", "0")
    assert _razor()._pair_reuse == 0
    for s in (_razor(), _razor(pair_reuse=True)):
        assert s.pair_reuse_declined is None


def test_razor_through_the_whole_outside_its_reach_is_off_and_says_so(
    no_features,
):
    if F.certified("razor_pair_reuse", "dropin"):
        pytest.skip("razor_pair_reuse reaches the drop-in here")
    with F.context_scope("dropin"):
        s = _razor(pair_reuse=True)
        assert s._pair_reuse == 0
        assert s.pair_reuse_declined == "not certified on this platform"
        with pytest.raises(F.FeatureNotCertified):
            _razor(razor_pair_reuse=True)


@pytest.mark.skipif(
    not F.certified("razor_pair_reuse"),
    reason="razor_pair_reuse is not certified on this platform",
)
def test_razor_pair_reuse_true_is_the_alias_to_the_bit(no_features):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _razor(razor_pair_reuse=True).compute_port_solution().y
        b = _razor(pair_reuse=True).compute_port_solution().y
    assert np.array_equal(
        np.asarray(a, dtype=complex).view(np.uint64),
        np.asarray(b, dtype=complex).view(np.uint64),
    )


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
