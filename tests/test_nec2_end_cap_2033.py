"""`nec2_end_cap`: NEC-2's free-end condition on the sinusoidal basis
(momwire#2033).

The reference impedances below were captured on 2026-11-18 from nec2c 1.3.1,
run as a black box on these synthetic decks, built with the SI speed of light
so that the only difference left is the end condition. `CAPPED` is that build
as NEC-2 defines it. `UNCAPPED` is the same build with its end-cap term zeroed,
which is what momwire's default basis answers. nec2c prints five figures, so
`PRINTED` is the agreement these pins can hold.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pytest
from scipy.special import j0, j1

from momwire import _features, _nec2_end_cap
from momwire.dropin.nec2c import _portal
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

ENV = "MOMWIRE_X_NEC2_END_CAP"
PRINTED = 1e-4


def _deck(cards, drive, ground=""):
    head = "CM synthetic end-cap deck\nCE\n"
    return f"{head}{cards}{ground or 'GE 0' + chr(10)}EX 0 {drive} 0 1 0\nFR 0 1 0 0 100 0\nXQ\nEN\n"


DECKS = {
    "fat_dipole": _deck("GW 1 21 0 0 -0.7045 0 0 0.7045 0.03\n", "1 11"),
    "thin_dipole": _deck("GW 1 21 0 0 -0.7045 0 0 0.7045 0.0003\n", "1 11"),
    "yagi": _deck(
        "GW 1 15 -0.6 -0.75 0 -0.6 0.75 0 0.01\n"
        "GW 2 15 0 -0.7 0 0 0.7 0 0.01\n"
        "GW 3 15 0.45 -0.65 0 0.45 0.65 0 0.01\n",
        "2 8",
    ),
    "monopole": _deck("GW 1 15 0 0 0 0 0 0.7 0.01\n", "1 1", "GE 1\nGN 1\n"),
    "tee": _deck(
        "GW 1 10 0 0 -0.7 0 0 0 0.01\n"
        "GW 2 10 0 0 0 0 0 0.7 0.01\n"
        "GW 3 8 0 0 0 0.3 0 0 0.01\n",
        "1 10",
    ),
    "one_segment": _deck(
        "GW 1 21 0 0 -0.7 0 0 0.7 0.01\nGW 2 1 0.1 0 -0.06 0.1 0 0.06 0.01\n",
        "1 11",
    ),
}

CAPPED = {
    "fat_dipole": 87.750 + 17.267j,
    "thin_dipole": 66.132 - 37.499j,
    "yagi": 24.801 + 3.0165j,
    "monopole": 36.930 + 2.7050j,
    "tee": 74.018 - 1.0429j,
    "one_segment": 73.549 + 4.0218j,
}
UNCAPPED = {
    "fat_dipole": 87.563 + 17.118j,
    "thin_dipole": 66.096 - 37.740j,
    "yagi": 25.862 + 0.032979j,
    "monopole": 36.722 + 2.1226j,
    "tee": 73.035 - 2.8533j,
    "one_segment": 72.852 + 2.1832j,
}

# A closed square loop: no wire end is free.
LOOP = _deck(
    "GW 1 7 0 0 0 0.7 0 0 0.003\n"
    "GW 2 7 0.7 0 0 0.7 0.7 0 0.003\n"
    "GW 3 7 0.7 0.7 0 0 0.7 0 0.003\n"
    "GW 4 7 0 0.7 0 0 0 0 0.003\n",
    "1 4",
)


def _impedance(deck, prog="momwire-nec2c-sinusoidal"):
    """Input impedance through the drop-in's own entry, as a front end asks."""
    out = io.StringIO()
    rc = _portal.main(
        [], stdin=io.StringIO(deck), stdout=out, stderr=io.StringIO(), prog=prog
    )
    text = out.getvalue()
    assert rc == 0, text
    lines = text.splitlines()
    at = next(i for i, s in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in s)
    row = next(s for s in lines[at:] if re.match(r"\s*\d+\s+\d+\s+[-\d.]+E", s)).split()
    return complex(float(row[6]), float(row[7]))


@pytest.fixture
def cap_calls(monkeypatch):
    """How many times the capped basis was built, with the variable unset."""
    monkeypatch.delenv(ENV, raising=False)
    calls = []
    real = _nec2_end_cap.basis_view

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_nec2_end_cap, "basis_view", counted)
    return calls


def _dipole(**kw):
    wire = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    return SinusoidalSolver(
        wires=[wire],
        n_per_edge_per_wire=[[21]],
        wavelength=3.0,
        wire_radius=0.01,
        **kw,
    )


def test_the_registry_entry_is_an_answer_changing_opt_in():
    f = _features.FEATURES["nec2_end_cap"]
    assert (f.reach, f.env, f.kwarg, f.solvers) == (
        1,
        ENV,
        "nec2_end_cap",
        ("sinusoidal",),
    )
    # Offered (Steve, 2026-11-19) but never on by default or by a seed.
    assert f.default is False and f.offer is True
    assert f.drift > _features.DEFAULT_DRIFT_MAX
    assert "changes the answer" in f.label


@pytest.mark.parametrize("name", sorted(DECKS))
def test_on_matches_capped_nec2c(name, cap_calls, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    z = _impedance(DECKS[name])
    assert abs(z - CAPPED[name]) / abs(CAPPED[name]) < PRINTED
    assert cap_calls, "the capped basis never ran"


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_launchers_default_matches_capped_nec2c(name, cap_calls):
    """Unset, the NEC-2 launcher's sinusoidal route takes the cap (Steve,
    2026-11-19)."""
    z = _impedance(DECKS[name])
    assert abs(z - CAPPED[name]) / abs(CAPPED[name]) < PRINTED
    assert cap_calls, "the capped basis never ran"


@pytest.mark.parametrize("name", sorted(DECKS))
def test_off_matches_uncapped_nec2c(name, cap_calls, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    z = _impedance(DECKS[name])
    assert abs(z - UNCAPPED[name]) / abs(UNCAPPED[name]) < PRINTED
    assert not cap_calls, "the capped basis ran with the option off"


def test_the_pins_can_tell_the_two_conditions_apart():
    for name in DECKS:
        assert abs(CAPPED[name] - UNCAPPED[name]) / abs(CAPPED[name]) > 5 * PRINTED


@pytest.mark.parametrize(
    "prog", ["momwire-nec2c-sinusoidal-galerkin", "momwire-nec2c-bspline"]
)
def test_the_other_solvers_ignore_it(prog, cap_calls, monkeypatch):
    off = _impedance(DECKS["fat_dipole"], prog)
    monkeypatch.setenv(ENV, "1")
    assert _impedance(DECKS["fat_dipole"], prog) == off
    assert not cap_calls


def test_a_deck_with_no_free_end_is_untouched(cap_calls, monkeypatch):
    off = _impedance(LOOP)
    monkeypatch.setenv(ENV, "1")
    assert _impedance(LOOP) == off
    assert not cap_calls


def test_the_keyword_beats_the_variable(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert _dipole()._nec2_end_cap is True
    assert _dipole(nec2_end_cap=False)._nec2_end_cap is False
    monkeypatch.delenv(ENV)
    assert _dipole()._nec2_end_cap is False
    assert _dipole(nec2_end_cap=True)._nec2_end_cap is True


def _self_entry(view, seg):
    lo, hi = view["starts"][seg], view["starts"][seg + 1]
    at = lo + int(np.flatnonzero(view["jbasis"][lo:hi] == seg)[0])
    return view["A"][at].real, view["B"][at].real, view["C"][at].real


def test_the_basis_meets_the_reports_end_condition():
    """NEC Part I, Eqs. 29 and 32: I = +-(X/k) dI/ds at each free end."""
    s = _dipole(nec2_end_cap=True)
    geom = s._build_geometry()
    view = s._basis_coefs(geom, s.k)
    ka = s.k * 0.01
    x = j1(ka) / j0(ka)
    u = 0.5 * s.k * geom["seg_h"][0]

    a, b, c = _self_entry(view, 0)
    at_end = a - b * np.sin(u) + c * np.cos(u)
    assert at_end == pytest.approx(x * (b * np.cos(u) + c * np.sin(u)), rel=1e-12)
    assert abs(at_end) > 1e-4  # the end current is not zero

    a, b, c = _self_entry(view, geom["n_segs"] - 1)
    at_end = a + b * np.sin(u) + c * np.cos(u)
    assert at_end == pytest.approx(-x * (b * np.cos(u) - c * np.sin(u)), rel=1e-12)


def test_only_the_end_segments_basis_functions_change():
    on = _dipole(nec2_end_cap=True)
    off = _dipole(nec2_end_cap=False)
    geom_on, geom_off = on._build_geometry(), off._build_geometry()
    view_on = on._basis_coefs(geom_on, on.k)
    view_off = off._basis_coefs(geom_off, off.k)
    n = geom_on["n_segs"]
    for basis in range(1, n - 1):
        for key in "ABC":
            mine = view_on["jbasis"] == basis
            assert view_on[key][mine] == pytest.approx(view_off[key][mine], rel=1e-10)
    for basis in (0, n - 1):
        mine = view_on["jbasis"] == basis
        assert np.abs(view_on["C"][mine] - view_off["C"][mine]).max() > 1e-4


def test_a_junction_the_nec4_rule_solves_refuses_by_name(cap_calls):
    """That route has its own cap, so an answer here could not carry NEC-2's
    while line 2 said it did."""
    arms = [
        np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]),
        np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    ]
    kw = {
        "wires": arms,
        "n_per_edge_per_wire": [[9], [9], [7]],
        "junctions": [[(0, "end"), (1, "start"), (2, "start")]],
        "wavelength": 3.0,
        "wire_radius": 0.01,
        "junction_rule": "nec42",
    }
    on = SinusoidalSolver(**kw, nec2_end_cap=True)
    with pytest.raises(ValueError, match="nec2_end_cap cannot act"):
        on._basis_coefs(on._build_geometry(), on.k)
    off = SinusoidalSolver(**kw, nec2_end_cap=False)
    off._basis_coefs(off._build_geometry(), off.k)
    assert not cap_calls


@pytest.mark.parametrize("complex_k", [True, False])
def test_a_wavenumber_nec2_never_defines_refuses(complex_k):
    s = _dipole(nec2_end_cap=True)
    geom = s._build_geometry()
    k = s.k * (1.0 - 0.1j) if complex_k else np.full(geom["n_segs"], s.k)
    with pytest.raises(ValueError, match="nec2_end_cap"):
        s._basis_coefs(geom, k)


# -- the NEC-2 launcher's default (Steve, 2026-11-19) -------------------------
#
# `momwire-nec2c-sinusoidal` takes the cap when MOMWIRE_X_NEC2_END_CAP is
# unset; `=0` is the no-cap answer and `=1` the refusing one. A deck the cap
# cannot act on runs without it by default, and says so in the printout.

# Wall-clock lines: the only ones two runs of one deck may print differently.
_CLOCK = re.compile(r"FILL: \d+ msec|Compute Time|TIME|msec")


def _printout(deck, prog="momwire-nec2c-sinusoidal"):
    """`main`'s printout, in an engine scope: `main` leaves its basis
    selected for the process, and a later test's `render_deck` would answer
    under it."""
    out = io.StringIO()
    with _portal.engine_scope():
        rc = _portal.main(
            [], stdin=io.StringIO(deck), stdout=out, stderr=io.StringIO(), prog=prog
        )
    assert rc == 0
    return [s for s in out.getvalue().splitlines() if not _CLOCK.search(s)]


def _run(monkeypatch, value, deck, prog="momwire-nec2c-sinusoidal"):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    return _printout(deck, prog)


def _notices(lines):
    return [s for s in lines if s.startswith("NEC-2 END CAP")]


def test_the_default_is_the_explicit_cap_and_not_the_off_answer(monkeypatch):
    """Adversarial: unset prints what `=1` prints, line for line (wall-clock
    lines aside), and NOT what `=0` prints, on a deck whose ends are free."""
    deck = DECKS["fat_dipole"]
    default = _run(monkeypatch, None, deck)
    on = _run(monkeypatch, "1", deck)
    off = _run(monkeypatch, "0", deck)
    assert default == on
    assert default != off
    monkeypatch.delenv(ENV)
    z_default = _impedance(deck)
    monkeypatch.setenv(ENV, "1")
    assert _impedance(deck) == z_default
    monkeypatch.setenv(ENV, "0")
    z_off = _impedance(deck)
    assert abs(z_off - z_default) / abs(z_default) > 1e-3
    assert not _notices(default) and not _notices(off)


@pytest.mark.parametrize(
    "prog",
    [
        "momwire-nec2c",
        "momwire-nec2c-sinusoidal-galerkin",
        "momwire-nec2c-bspline-d1",
        "momwire-nec2c-pulse",
    ],
)
def test_no_other_basis_takes_the_default(prog, monkeypatch, cap_calls):
    deck = DECKS["fat_dipole"]
    assert _run(monkeypatch, None, deck, prog) == _run(monkeypatch, "0", deck, prog)
    assert not cap_calls


def test_the_python_api_and_registry_defaults_stay_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _features.FEATURES["nec2_end_cap"].default is False
    assert _dipole()._nec2_end_cap is False
    with _nec2_end_cap.launcher_default():
        assert _dipole()._nec2_end_cap is True
        assert _dipole(nec2_end_cap=False)._nec2_end_cap is False
        monkeypatch.setenv(ENV, "0")
        assert _dipole()._nec2_end_cap is False
    monkeypatch.delenv(ENV)
    assert _dipole()._nec2_end_cap is False


# A wholly buried horizontal dipole under Sommerfeld soil: free ends below
# the interface, which NEC-2 has no condition for.
BURIED = (
    "CM buried free-end deck\nCE\n"
    "GW 1 15 -0.7 0 -0.5 0.7 0 -0.5 0.01\n"
    "GE -1\nGN 2 0 0 0 13 0.005\n"
    "EX 0 1 8 0 1 0\nFR 0 1 0 0 100 0\nXQ\nEN\n"
)


def _refused_decks():
    """(name, deck, junction rule) for the decks the cap refuses."""
    return [
        ("buried", BURIED, None, _nec2_end_cap.BELOW_INTERFACE_REFUSAL),
        ("nec4_tee", DECKS["tee"], "nec42", _nec2_end_cap.NEC4_JUNCTION_REFUSAL),
    ]


@pytest.mark.parametrize(
    "name,deck,rule,refusal", _refused_decks(), ids=lambda v: str(v)[:8]
)
def test_a_deck_the_cap_refuses_falls_back_by_default(
    name, deck, rule, refusal, monkeypatch, cap_calls
):
    """Unset: the `=0` printout plus one notice line (and its blank) above
    MATRIX TIMING. `=1`: today's refusal, as a NEC ERROR."""
    if rule is None:
        monkeypatch.delenv("MOMWIRE_JUNCTION_RULE", raising=False)
    else:
        monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", rule)
    default = _run(monkeypatch, None, deck)
    off = _run(monkeypatch, "0", deck)
    assert not cap_calls, "the capped basis was built on a deck it cannot act on"
    notice = _nec2_end_cap.withdrawn_notice(refusal)
    assert _notices(default) == [notice]
    at = default.index(notice)
    assert default[at + 1] == ""
    assert default[:at] + default[at + 2 :] == off
    assert not _notices(off)
    on = "\n".join(_run(monkeypatch, "1", deck))
    assert f"ERROR: {refusal}" in on
    assert "NEC-2 END CAP" not in on
