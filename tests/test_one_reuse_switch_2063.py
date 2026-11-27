"""One fill-reuse switch, `pair_reuse` (momwire#2063 step 4).

`pair_reuse=True` (or MOMWIRE_X_PAIR_REUSE=1) turns on everything each
solver reuses: bs2's census and its same-edge Toeplitz form (what was
`offset_reuse`), razor2p's fill-pair reuse (what was `razor_pair_reuse`), the
sinusoidal class map. False turns all of it off. The old names are deprecated
spellings of the switch on their own family (`FamilyReach.alias`), read and
warned in 0.99.0 and removed in 0.100.0:

- the precedence: `pair_reuse` given (keyword or variable) wins; a
  deprecated keyword beats a variable, as any keyword does; a deprecated
  spelling alone means `pair_reuse` on its own family only;
- the registry carries no row that is separately switchable;
- a deprecated spelling answers to the bit what `pair_reuse` answers, and
  line 2 names `pair_reuse`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _features as F
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

OFFSET = "MOMWIRE_X_OFFSET_REUSE"
RAZOR = "MOMWIRE_X_RAZOR_PAIR_REUSE"
WHOLE = "MOMWIRE_X_PAIR_REUSE"


@pytest.fixture
def no_features(monkeypatch):
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    for var in (OFFSET, RAZOR):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _bspline(n=21, **kw):
    wire = [[[0.0, -2.5, 10.0], [0.0, 2.5, 10.0]]]
    return BSplineSolver(wires=wire, n_per_edge_per_wire=[[n]], wire_radius=1e-3, **kw)


def _razor(**kw):
    return RazorSolver(
        wires=[np.array([[0.0, -5.0, 8.0], [0.0, 5.0, 8.0]])],
        nsegs=31,
        wire_radius=0.001,
        wavelength=20.0,
        nec5_quadrature=True,
        **kw,
    )


def _sin(**kw):
    return SinusoidalSolver(
        wires=[np.array([[0.0, -2.6, 8.0], [0.0, 2.6, 8.0]])],
        n_per_edge_per_wire=[[41]],
        wavelength=10.7,
        wire_radius=1e-3,
        **kw,
    )


def _bs_on(s):
    assert s._reuse_on is s._offset_reuse, "the census and the Toeplitz form apart"
    return s._reuse_on


def _quiet(fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fn(**kw)


razor_certified = pytest.mark.skipif(
    not F.certified("pair_reuse", solver="razor"),
    reason="razor2p's pair reuse is not certified on this platform",
)


# --- the whole switch -----------------------------------------------------------


def test_the_switch_turns_bs2s_census_and_toeplitz_form_on_and_off(no_features):
    assert _bs_on(_bspline()) is False
    assert _bs_on(_bspline(pair_reuse=True)) is True
    assert _bs_on(_bspline(pair_reuse=False)) is False
    no_features.setenv(WHOLE, "1")
    assert _bs_on(_bspline()) is True
    assert _bs_on(_bspline(pair_reuse=False)) is False


@razor_certified
def test_the_switch_turns_razor2ps_reuse_on_and_off(no_features):
    assert _razor()._pair_reuse == 0
    assert _razor(pair_reuse=True)._pair_reuse == 1
    assert _razor(pair_reuse=False)._pair_reuse == 0
    no_features.setenv(WHOLE, "1")
    assert _razor()._pair_reuse == 1
    assert _razor(pair_reuse=False)._pair_reuse == 0


def test_the_switch_turns_the_sinusoidal_class_map_on_and_off(no_features):
    assert _sin().features_bound["pair_reuse"] is False
    assert _sin(pair_reuse=True).features_bound["pair_reuse"] is True
    no_features.setenv(WHOLE, "1")
    assert _sin().features_bound["pair_reuse"] is True
    assert _sin(pair_reuse=False).features_bound["pair_reuse"] is False


def test_the_cpp_routes_read_the_one_slot():
    # The executables' C++ fills read the switch's one slot: no route keeps
    # a request slot of an old part (tests/test_features_registry.py holds
    # the executables' table of deprecated variables).
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "momwire"
    bs2 = (src / "_nec5_bspline_printout_core.cpp").read_text(encoding="utf-8")
    assert "fp.offset_reuse = fp.pair_reuse;" in bs2
    razor = (src / "_nec5_razor_seam_core.cpp").read_text(encoding="utf-8")
    assert "requested_features().pair_reuse ? 1 : 0" in razor


# --- the deprecated spellings ---------------------------------------------------


def test_offset_reuse_warns_and_is_honoured(no_features):
    with pytest.warns(DeprecationWarning, match="'offset_reuse' is deprecated"):
        assert _bs_on(_bspline(offset_reuse=True)) is True
    with pytest.warns(DeprecationWarning, match="'offset_reuse'"):
        assert _bs_on(_bspline(offset_reuse=False)) is False
    no_features.setenv(OFFSET, "1")
    with pytest.warns(DeprecationWarning, match=f"{OFFSET} is deprecated"):
        assert _bs_on(_bspline()) is True


@razor_certified
def test_razor_pair_reuse_warns_and_is_honoured(no_features):
    with pytest.warns(DeprecationWarning, match="'razor_pair_reuse' is deprecated"):
        assert _razor(razor_pair_reuse=True)._pair_reuse == 1
    with pytest.warns(DeprecationWarning, match="'razor_pair_reuse'"):
        assert _razor(razor_pair_reuse="verify")._pair_reuse == 2
    no_features.setenv(RAZOR, "1")
    with pytest.warns(DeprecationWarning, match=f"{RAZOR} is deprecated"):
        assert _razor()._pair_reuse == 1


def test_no_spelling_warns_when_none_is_used(no_features):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _bspline(pair_reuse=True)
        _razor(pair_reuse=True)
        _bspline(offset_reuse=None)


# --- the precedence ------------------------------------------------------------


def test_pair_reuse_given_wins_over_a_deprecated_keyword(no_features):
    assert _bs_on(_quiet(_bspline, pair_reuse=True, offset_reuse=False)) is True
    assert _bs_on(_quiet(_bspline, pair_reuse=False, offset_reuse=True)) is False
    assert _quiet(_razor, pair_reuse=False, razor_pair_reuse=True)._pair_reuse == 0
    # the switch on, the old spelling's "verify" mode is not read
    assert _quiet(_razor, pair_reuse=True, razor_pair_reuse="verify")._pair_reuse == 1


def test_the_variable_wins_over_a_deprecated_variable(no_features):
    no_features.setenv(WHOLE, "0")
    no_features.setenv(OFFSET, "1")
    no_features.setenv(RAZOR, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # not read, not warned
        assert _bs_on(_bspline()) is False
        assert _razor()._pair_reuse == 0
    no_features.setenv(WHOLE, "1")
    no_features.setenv(OFFSET, "0")
    no_features.setenv(RAZOR, "0")
    assert _bs_on(_bspline()) is True
    assert F.enabled("pair_reuse", solver="razor") is True


def test_a_keyword_wins_over_a_variable_whichever_spelling(no_features):
    no_features.setenv(WHOLE, "1")
    assert _bs_on(_quiet(_bspline, offset_reuse=False)) is False
    no_features.setenv(WHOLE, "0")
    assert _bs_on(_quiet(_bspline, offset_reuse=True)) is True
    no_features.setenv(OFFSET, "0")
    assert _bs_on(_bspline(pair_reuse=True)) is True


def test_a_deprecated_spelling_alone_is_its_own_familys_only(no_features):
    no_features.setenv(OFFSET, "1")
    assert F.enabled("pair_reuse", solver="bspline") is True
    assert F.enabled("pair_reuse", solver="razor") is False
    assert F.enabled("pair_reuse", solver="sinusoidal") is False
    assert _razor()._pair_reuse == 0
    assert _sin().features_bound["pair_reuse"] is False
    no_features.delenv(OFFSET)
    no_features.setenv(RAZOR, "1")
    assert F.enabled("pair_reuse", solver="razor") is True
    assert _bs_on(_bspline()) is False
    assert _sin().features_bound["pair_reuse"] is False
    # a keyword another family spelt is no keyword of this one
    with pytest.raises(TypeError, match="offset_reuse"):
        _razor(offset_reuse=True)
    with pytest.raises(TypeError, match="razor_pair_reuse"):
        _bspline(razor_pair_reuse=True)


def test_a_deprecated_variable_at_0_alone_overrides_a_pragma(no_features):
    no_features.setenv(OFFSET, "0")
    with F.pragma_scope("CM momwire: pair_reuse\nCE\n"):
        assert F.enabled("pair_reuse", solver="bspline") is False
        assert F.enabled("pair_reuse", solver="razor") is True


# --- line 2 -----------------------------------------------------------------------


def test_line_2_names_the_switch_for_a_deprecated_variable(no_features):
    no_features.setenv(OFFSET, "1")
    assert F.stamp(solver="bspline") == "+pair_reuse"
    assert F.stamp(solver="razor") == ""
    assert F.stamp(solver="sinusoidal") == ""
    assert F.stamp() == "+pair_reuse"
    no_features.setenv(WHOLE, "0")
    assert F.stamp(solver="bspline") == "" and F.stamp() == ""
    no_features.setenv(WHOLE, "1")
    no_features.setenv(RAZOR, "1")
    assert F.stamp(solver="razor") == "+pair_reuse"
    assert F.stamp() == "+pair_reuse"


# --- the registry -------------------------------------------------------------------


def test_the_registry_carries_no_switchable_part():
    for old in ("offset_reuse", "razor_pair_reuse"):
        assert old not in F.FEATURES
        with pytest.raises(KeyError):
            F.enabled(old)
    pr = F.FEATURES["pair_reuse"]
    # per-family reach, kind and drift are data on the one row
    rows = {r.solver: (r.reach, r.kind, r.drift, r.alias) for r in pr.families}
    assert rows == {
        "bspline": (9, "python", 5e-10, "offset_reuse"),
        "razor": (8, "native", 2e-9, "razor_pair_reuse"),
        "sinusoidal": (11, "python", 5e-10, None),
    }
    assert pr.drift == max(r.drift for r in pr.families)
    assert pr.drift <= F.DEFAULT_DRIFT_MAX
    # a front end's fields stay the one row's
    assert pr.label and pr.summary and pr.offer is False
    # the only bound answer for the switch is the switch's
    s = _bspline(pair_reuse=True)
    assert set(s.features_bound) & {"offset_reuse", "razor_pair_reuse"} == set()
    assert s.features_bound["pair_reuse"] is True


# --- the bits -------------------------------------------------------------------------


def _bits(z):
    return np.atleast_1d(np.asarray(z, dtype=complex)).view(np.uint64)


def test_offset_reuse_answers_as_pair_reuse_to_the_bit(no_features):
    # 61 segments: over the census's small-deck threshold on one wire.
    a = _quiet(_bspline, n=61, offset_reuse=True).compute_impedance()[0]
    b = _bspline(n=61, pair_reuse=True).compute_impedance()[0]
    assert np.array_equal(_bits(a), _bits(b))


@razor_certified
def test_razor_pair_reuse_answers_as_pair_reuse_to_the_bit(no_features):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _razor(razor_pair_reuse=True).compute_port_solution().y
        b = _razor(pair_reuse=True).compute_port_solution().y
    assert np.array_equal(_bits(a), _bits(b))


def test_the_aliases_go_in_their_removal_release():
    """The deprecated spellings are removed in ALIAS_REMOVED_IN (Steve,
    2026-11-27: one minor after ALIAS_DEPRECATED_IN). A version bump to it
    with the aliases still in the registry fails here."""
    import tomllib
    from pathlib import Path

    def v(s):
        return tuple(int(p) for p in s.split(".")[:3])

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("no source tree")
    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    aliases = [
        fam.alias for f in F.FEATURES.values() for fam in f.families if fam.alias
    ]
    assert v(F.ALIAS_REMOVED_IN) == (
        v(F.ALIAS_DEPRECATED_IN)[0],
        v(F.ALIAS_DEPRECATED_IN)[1] + 1,
        0,
    )
    if v(version) >= v(F.ALIAS_REMOVED_IN):
        assert not aliases, f"remove {aliases} in {F.ALIAS_REMOVED_IN}"
