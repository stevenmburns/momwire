"""`nec2_far_approx`: NEC-2's interaction approximation on the point-matched
sinusoidal fill (`momwire._nec2_far_approx`, `_sinusoidal_far_core.cpp`).

The reference impedances below were captured on 2026-11-20 from nec2c 1.3.1,
run as a black box on these synthetic decks. momwire is asked the same deck
the way the NEC-2 launcher asks it (NEC-2's end cap, `nec2_end_cap`) and at
the same wavelength (nec2c's speed of light is 299.8, so the FR card is
scaled by 299.792458 / 299.8), so the only difference left is the
approximation. nec2c prints five figures; `PRINTED` is what the pins hold.

What the probes established about NEC-2 (each one pinned below):

- the approximation is ON with no KH card, at 1 wavelength (nec2c prints
  "APPROXIMATE INTEGRATION EMPLOYED FOR SEGMENTS THAT ARE MORE THAN 1.000
  WAVELENGTHS APART", and the deck answers as `KH 0 0 0 0 1`);
- KH's range is the card's fifth field;
- the far pair's field is the lumped dipole's (constant, sine and cosine
  moments 2H, 0, 2 sin(kH)/k): at KH 0.05, where it reaches pairs a few
  segments apart and moves Zin by 38 %, it agrees with nec2c to its printed
  figures, and a lumped constant term alone (sine and cosine exact) moves the
  other way (-29.5j against +4.6j on `THREE`, measured, not pinned);
- image pairs take it by the image distance, over PEC, Fresnel and
  Sommerfeld grounds (the Sommerfeld remainder exact), and the extended
  kernel takes it too.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from momwire import _features, _nec2_far_approx
from momwire._accel import acc
from momwire.deck import DeckError, build_solver, parse_nec2
from momwire.deck._nec4 import parse_nec4
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 1, native).
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "sinusoidal_field_tensor_far"),
        reason="the accelerator is not built",
    ),
]

ENV = "MOMWIRE_X_NEC2_FAR_APPROX"
PRINTED = 1e-4
KH_REFUSAL = "KH (interaction approximation limit) is not supported by this engine"

THREE = """CM three dipoles
CE
GW 1 21 0 0 -0.25 0 0 0.25 0.001
GW 2 21 0.6 0 -0.24 0.6 0 0.24 0.001
GW 3 21 2.5 0 -0.26 2.5 0 0.26 0.001
GE 0
{cards}FR 0 1 0 0 299.8 0
EX 0 1 11 0 1 0
XQ
EN
"""
# A half-wave horizontal dipole 0.08 wavelength over the ground.
LOW = """CM low dipole
CE
GW 1 21 -0.24 0 0.08 0.24 0 0.08 0.001
GE 1
{cards}FR 0 1 0 0 299.8 0
EX 0 1 11 0 1 0
XQ
EN
"""
# k * delta = 1.04e-3: the fill takes the folded cos(k xi) - 1 shapes.
FOLDED = """CM folded shapes
CE
GW 1 201 0 0 -5 0 0 5 0.001
GW 2 101 4 0 -2.5 4 0 2.5 0.001
GE 0
{cards}FR 0 1 0 0 1.0 0
EX 0 1 101 0 1 0
XQ
EN
"""

FRESNEL = "GN 0 0 0 0 13 0.005\n"
SOMMERFELD = "GN 2 0 0 0 13 0.005\n"

# nec2c 1.3.1's printed input impedance, 2026-11-20.
NEC2C = {
    ("three", ""): 76.603 + 38.37j,
    ("three", "KH 0 0 0 0 1"): 76.603 + 38.37j,
    ("three", "KH 0 0 0 0 0.1"): 76.553 + 35.208j,
    ("three", "KH 0 0 0 0 0.05"): 77.272 + 8.8271j,
    ("three_ek", "KH 0 0 0 0 0.1"): 76.542 + 35.158j,
    ("low_pec", "KH 0 0 0 0 0.1"): 14.73 + 21.928j,
    ("low_fresnel", "KH 0 0 0 0 0.1"): 45.937 + 18.276j,
    ("folded", "KH 0 0 0 0 0.005"): 0.20539 - 8504.4j,
}
# The same decks at KH 1000 (no pair that far): nec2c's answer without the
# approximation, the base each change is measured from.
NEC2C_OFF = {
    "three": 76.603 + 38.37j,
    "three_ek": 76.59 + 38.291j,
    "low_pec": 14.796 + 24.794j,
    "low_fresnel": 46.14 + 21.115j,
    "low_sommerfeld": 58.245 + 15.026j,
    "folded": 0.2054 - 8503.4j,
}
# Sommerfeld: momwire's interpolated remainder and nec2c's differ by 4e-4 on
# this deck with the approximation off, so the CHANGE is pinned instead.
NEC2C_SOMMERFELD_KH01 = 58.011 + 12.208j

DECKS = {
    "three": THREE.format(cards=""),
    "three_ek": THREE.format(cards="EK\n"),
    "low_pec": LOW.format(cards="GN 1\n"),
    "low_fresnel": LOW.format(cards=FRESNEL),
    "low_sommerfeld": LOW.format(cards=SOMMERFELD),
    "folded": FOLDED.format(cards=""),
}


def _with(deck, card):
    return deck.replace("\nFR ", f"\n{card}\nFR ", 1) if card else deck


def _si(deck):
    """The deck at nec2c's wavelength in momwire's SI speed of light."""

    def scale(m):
        f = m.group(0).split()
        f[5] = repr(float(f[5]) * 299.792458 / 299.8)
        return " ".join(f)

    return re.sub(r"^FR\s.*$", scale, deck, flags=re.M)


@pytest.fixture
def no_far(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


@pytest.fixture
def far_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")


def _zin(deck, monkeypatch, *, far, cap=True, basis="sinusoidal"):
    monkeypatch.setenv(ENV, "1" if far else "0")
    monkeypatch.setenv("MOMWIRE_X_NEC2_END_CAP", "1" if cap else "0")
    built = build_solver(parse_nec2(_si(deck)), basis=basis)
    return complex(built.solver.compute_impedance()[0])


# -- the registry ------------------------------------------------------------


def test_the_entry_is_an_approximate_native_reach_one_opt_in():
    f = _features.FEATURES["nec2_far_approx"]
    assert (f.reach, f.kind, f.env, f.kwarg) == (1, "native", ENV, "nec2_far_approx")
    assert f.solvers == ("sinusoidal",)
    assert not f.default
    # Approximate: past the default-on bar, so no front end may seed it on.
    assert f.drift > _features.DEFAULT_DRIFT_MAX
    assert f.offer and f.label and f.summary


def test_the_dropin_route_refuses_it(far_on):
    # Native, reach 1: the drop-in is level 4.
    with _features.context_scope("dropin"):
        with pytest.raises(_features.FeatureNotCertified, match="nec2_far_approx"):
            _features.enabled("nec2_far_approx")


# -- KH: refused off, read on ------------------------------------------------


@pytest.mark.parametrize("env", [None, "0"])
def test_off_kh_is_refused_by_the_same_sentence(monkeypatch, env):
    if env is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env)
    deck = _with(DECKS["three"], "KH 0 0 0 0 0.5")
    with pytest.raises(DeckError) as e:
        parse_nec2(deck)
    assert str(e.value) == KH_REFUSAL
    with pytest.raises(DeckError) as e:
        parse_nec2(deck, nec2_far_approx=False)
    assert str(e.value) == KH_REFUSAL


def test_on_kh_reads_its_fifth_field(far_on):
    assert parse_nec2(_with(DECKS["three"], "KH 0 0 0 0 0.5")).interaction_range == 0.5
    assert parse_nec2(DECKS["three"]).interaction_range is None


def test_the_keyword_reads_kh_without_the_variable(no_far):
    model = parse_nec2(_with(DECKS["three"], "KH 0 0 0 0 0.25"), nec2_far_approx=True)
    assert model.interaction_range == 0.25


def test_kh_after_an_execute_card_or_out_of_range_refuses(far_on):
    late = DECKS["three"].replace("XQ\nEN\n", "XQ\nKH 0 0 0 0 0.5\nXQ\nEN\n")
    with pytest.raises(DeckError, match="KH after an execute card"):
        parse_nec2(late)
    with pytest.raises(DeckError, match="finite, non-negative"):
        parse_nec2(_with(DECKS["three"], "KH 0 0 0 0 -1"))


def test_the_nec4_dialect_refuses_kh_whatever_the_feature_says(far_on):
    with pytest.raises(DeckError, match="KH"):
        parse_nec4(_with(DECKS["three"], "KH 0 0 0 0 0.5"))


def test_kh_reaches_the_sinusoidal_solver_and_no_other_basis(far_on):
    model = parse_nec2(_with(DECKS["three"], "KH 0 0 0 0 0.3"))
    s = build_solver(model, basis="sinusoidal").solver
    assert s.features_bound["nec2_far_approx"] is True
    assert s.nec2_far_approx_range == 0.3
    with pytest.raises(ValueError, match="KH sets NEC-2's interaction approximation"):
        build_solver(model, basis="bspline")


def test_on_without_kh_takes_nec2s_one_wavelength(far_on):
    s = build_solver(parse_nec2(DECKS["three"]), basis="sinusoidal").solver
    assert s.nec2_far_approx_range == _nec2_far_approx.DEFAULT_RANGE == 1.0


# -- off is the shipped fill --------------------------------------------------


def _dipoles(**kw):
    wires = [np.array([[x, 0.0, -0.25], [x, 0.0, 0.25]]) for x in (0.0, 0.6, 1.7, 2.5)]
    return SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[21]] * 4,
        feed_wire_index=0,
        wavelength=1.0,
        wire_radius=0.001,
        **kw,
    )


def test_off_is_the_fill_bit_for_bit(no_far):
    z0 = _dipoles().compute_impedance()[0]
    assert _dipoles(nec2_far_approx=False).compute_impedance()[0] == z0
    # The range is unread with the feature off.
    assert _dipoles(nec2_far_approx_range=0.01).compute_impedance()[0] == z0


# -- the kernel ---------------------------------------------------------------


def _tensor_args(n=40, seed=3):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-3, 3, (n, 3))
    t = rng.normal(size=(n, 3))
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    h = rng.uniform(0.02, 0.06, n)
    gx, gw = np.polynomial.legendre.leggauss(8)
    k = 2 * np.pi
    return c, t, h, 0.001, k, 376.73, gx, gw


def _dipole_field(c, t, h, a, k, eta):
    """The lumped dipole, straight from its formula (numpy)."""
    rv = c[:, None, :] - c[None, :, :]
    z = np.sum(rv * t[None, :, :], -1)
    rho_v = rv - z[..., None] * t[None, :, :]
    rho = np.sqrt(np.sum(rho_v**2, -1) + a * a)
    R = np.sqrt(z * z + rho * rho)
    G = np.exp(-1j * k * R) / (4 * np.pi * R)
    kr = k * R
    A = 1 - 1j / kr - 1 / kr**2
    B = -(1 - 3j / kr - 3 / kr**2)
    cc = -1j * k * eta * G
    Ez = cc * (A + B * (z / R) ** 2)
    Er = cc * B * (z / R) * (rho / R)
    td = t @ t.T
    rpf = np.sum(rho_v * t[:, None, :], -1) / rho
    phi = td * Ez + rpf * Er
    H = h / 2
    return phi * (2 * H), 0 * phi, phi * (2 * np.sin(k * H) / k)


def test_the_kernel_is_the_lumped_dipole_beyond_the_range_and_exact_inside():
    c, t, h, a, k, eta, gx, gw = _tensor_args()
    exact = acc.sinusoidal_field_tensor(c, t, c, t, h, a, k, eta, gx, gw)
    rv = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(rv**2, -1))
    far_r = 2.5
    got = acc.sinusoidal_field_tensor_far(c, t, c, t, h, a, k, eta, gx, gw, far_r)
    lumped = _dipole_field(c, t, h, a, k, eta)
    far = d > far_r
    assert far.any() and (~far).any()
    for g, e, lu in zip(got, exact, lumped):
        scale = np.max(np.abs(e))
        # Inside: the shipped kernel's arithmetic (a vector-lane ulp at most).
        assert np.max(np.abs(g[~far] - e[~far])) <= 1e-14 * scale
        # Beyond: the formula's.
        assert np.max(np.abs(g[far] - lu[far])) <= 1e-12 * scale
    # And the lumped dipole is the exact kernel's large-R limit, O((kH)^2).
    rel = np.abs(got[0][far] - exact[0][far]) / np.abs(exact[0][far])
    assert rel.max() < (k * h.max() / 2) ** 2


def test_the_overlay_writes_the_same_far_pairs_over_any_base():
    c, t, h, a, k, eta, gx, gw = _tensor_args()
    exact = acc.sinusoidal_field_tensor(c, t, c, t, h, a, k, eta, gx, gw)
    far = acc.sinusoidal_field_tensor_far(c, t, c, t, h, a, k, eta, gx, gw, 2.5)
    empty = np.zeros((0, 0))
    over = acc.sinusoidal_far_overlay(
        *exact, c, t, c, t, h, a, k, eta, False, False, *(empty,) * 5, 0j, 2.5
    )
    for o, f, e in zip(over, far, exact):
        assert np.max(np.abs(o - f)) <= 1e-14 * np.max(np.abs(e))
    # The folded third shape lumps to cos-moment minus constant-moment.
    folded = (exact[0], exact[1], exact[2] - exact[0])
    over = acc.sinusoidal_far_overlay(
        *folded, c, t, c, t, h, a, k, eta, False, True, *(empty,) * 5, 0j, 2.5
    )
    want = far[2] - far[0]
    assert np.max(np.abs(over[2] - want)) <= 1e-13 * np.max(np.abs(want))


def test_a_range_past_every_pair_is_the_shipped_kernel():
    c, t, h, a, k, eta, gx, gw = _tensor_args()
    exact = acc.sinusoidal_field_tensor(c, t, c, t, h, a, k, eta, gx, gw)
    got = acc.sinusoidal_field_tensor_far(c, t, c, t, h, a, k, eta, gx, gw, 1e3)
    for g, e in zip(got, exact):
        assert np.max(np.abs(g - e)) <= 1e-14 * np.max(np.abs(e))


# -- against nec2c -------------------------------------------------------------


@pytest.mark.parametrize(("name", "card"), sorted(NEC2C))
def test_on_agrees_with_nec2c_where_it_approximates(monkeypatch, name, card):
    deck = _with(DECKS[name], card)
    want = NEC2C[(name, card)]
    on = _zin(deck, monkeypatch, far=True)
    assert abs(on - want) <= PRINTED * abs(want)
    if want != NEC2C_OFF[name]:
        # The approximation moved nec2c past its printed figures: momwire
        # without it is further away (by 3.8e-1 to 1.2e-4 here).
        off = _zin(DECKS[name], monkeypatch, far=False)
        assert abs(off - want) > 10 * abs(on - want)


def test_the_sommerfeld_change_agrees_with_nec2cs(monkeypatch):
    deck = DECKS["low_sommerfeld"]
    on = _zin(_with(deck, "KH 0 0 0 0 0.1"), monkeypatch, far=True)
    off = _zin(deck, monkeypatch, far=False)
    want = NEC2C_SOMMERFELD_KH01 - NEC2C_OFF["low_sommerfeld"]
    # Each nec2c figure carries half a unit in its last place.
    assert abs((on - off) - want) <= 2e-3


def test_image_pairs_take_it_by_their_own_distance(monkeypatch):
    # A short dipole 0.05 wavelength up: its own pairs are all within 0.08,
    # its image pairs all beyond 0.1. A range of 0.09 lumps only the images,
    # and moves R as nec2c's does (0.10214 -> 0.10216, printed).
    deck = """CM short dipole
CE
GW 1 11 -0.04 0 0.05 0.04 0 0.05 0.0005
GE 1
GN 1
FR 0 1 0 0 299.8 0
EX 0 1 6 0 1 0
XQ
EN
"""
    off = _zin(deck, monkeypatch, far=False)
    on = _zin(_with(deck, "KH 0 0 0 0 0.09"), monkeypatch, far=True)
    assert round(off.real, 5) == 0.10214
    assert round(on.real, 5) == 0.10216


# -- refusals -------------------------------------------------------------------


def test_the_nec4_junction_rule_refuses(no_far):
    with pytest.raises(ValueError) as e:
        _dipoles(nec2_far_approx=True, junction_rule="nec42")
    assert str(e.value) == _nec2_far_approx.NEC4_RULE_REFUSAL


def test_a_negative_range_refuses(no_far):
    with pytest.raises(ValueError, match="finite, non-negative"):
        _dipoles(nec2_far_approx=True, nec2_far_approx_range=-1.0)


# -- pair reuse -------------------------------------------------------------------


def _reuse_pair(monkeypatch, deck, guard=True):
    """Y with the approximation on, reuse off and on."""
    out = []
    for reuse in ("0", "1"):
        monkeypatch.setenv(ENV, "1")
        monkeypatch.setenv("MOMWIRE_X_PAIR_REUSE", reuse)
        s = build_solver(parse_nec2(deck), basis="sinusoidal").solver
        if not guard:
            monkeypatch.setattr(s, "_far_range_ties", lambda *a: False)
        out.append((s.compute_y_matrix(), s))
    return out


def _radials(n_rad=8, seg=40, ground=""):
    lam = 1.0
    # Heights and lengths off any round number, so that no pair, direct or
    # image, sits at the 1-wavelength range to rounding (`ties`).
    hub, top = 0.3137, 0.9411
    cards = [f"GW 1 {round(0.625 * seg)} 0 0 {hub} 0 0 {top} 0.004"]
    for i in range(n_rad):
        ph = 2 * np.pi * i / n_rad + 0.1
        x, y = 1.5513 * np.cos(ph), 1.5513 * np.sin(ph)
        cards.append(
            f"GW {i + 2} {round(1.6 * seg)} 0 0 {hub} {x:.6f} {y:.6f} 0.1171 0.004"
        )
    ge = "GE 1\n" + ground if ground else "GE 0\n"
    return (
        "CM raised whip\nCE\n"
        + "\n".join(cards)
        + f"\n{ge}FR 0 1 0 0 {299.792458 / lam} 0\nEX 0 1 1 0 1 0\nXQ\nEN\n"
    )


@pytest.mark.parametrize("ground", ["", "GN 1\n", FRESNEL, SOMMERFELD])
def test_reuse_with_the_approximation_is_the_band_path(monkeypatch, ground):
    # Past the reuse's small-deck bounds (400 segments in free space and over
    # PEC, 200 over a finite ground), so each fill engages it.
    finite = ground not in ("", "GN 1\n")
    deck = _radials(6, 30, ground) if finite else _radials(ground=ground)
    (y0, _), (y1, s1) = _reuse_pair(monkeypatch, deck)
    assert s1.pair_reuse_declined is None and s1.pair_reuse_census
    assert np.max(np.abs(y1 - y0)) <= 1e-12 * np.max(np.abs(y0))


def _straddling_deck(spacing, offsets):
    cards, tag = [], 1
    for x in offsets:
        for dx in (0.0, spacing):
            cards.append(f"GW {tag} 40 {x + dx!r} 0 -0.24 {x + dx!r} 0 0.24 0.001")
            tag += 1
    return (
        "CM straddling pairs\nCE\n"
        + "\n".join(cards)
        + f"\nGE 0\nKH 0 0 0 0 {spacing}\nFR 0 1 0 0 299.792458 0\n"
        + "EX 0 1 20 0 1 0\nXQ\nEN\n"
    )


def test_a_class_straddling_the_range_declines_the_reuse(monkeypatch):
    # Pairs of vertical dipoles `spacing` apart, at offsets whose separation
    # rounds to just above, exactly at, and just below the range: congruent
    # pairs, one class, on both sides of it.
    spacing = 0.25
    offsets = [0.05, 0.95, 0.45, 1.95, 3.95]
    seps = [(x + spacing) - x for x in offsets]
    deck = _straddling_deck(spacing, offsets)
    monkeypatch.setenv(ENV, "1")
    s = build_solver(parse_nec2(deck), basis="sinusoidal").solver
    far_r = _nec2_far_approx.range_metres(0.25, 2 * np.pi / s.wavelength)
    assert any(d > far_r for d in seps) and any(d <= far_r for d in seps)
    (y0, _), (y1, s1) = _reuse_pair(monkeypatch, deck)
    assert s1.pair_reuse_declined == _nec2_far_approx.TIES_DECLINE
    assert np.array_equal(y1, y0)
    # Without the guard a class serves its representative's side to every
    # member, and the answer leaves the reuse's own tolerance (1e-12).
    (y0, _), (y2, s2) = _reuse_pair(monkeypatch, deck, guard=False)
    assert s2.pair_reuse_declined is None
    assert np.max(np.abs(y2 - y0)) > 1e-10 * np.max(np.abs(y0))


def test_the_ties_count_is_the_squared_distance_window():
    c = np.array([[0.0, 0.0, 0.0]])
    s = np.array([[1.0, 0, 0], [1.0 + 1e-12, 0, 0], [1.0 + 1e-6, 0, 0], [0.5, 0, 0]])
    assert acc.far_range_ties(c, s, 1.0, _nec2_far_approx.TIE_REL) == 2


def test_the_ground_is_scanned_for_ties_at_the_image_distance(monkeypatch):
    # A horizontal wire 0.125 up: its image pairs straight below are 0.25
    # apart, and no direct pair is.
    monkeypatch.setenv(ENV, "1")
    deck = """CM image ties
CE
GW 1 30 -0.4 0 0.125 0.4 0 0.125 0.001
GE 1
GN 1
KH 0 0 0 0 0.25
FR 0 1 0 0 299.792458 0
EX 0 1 15 0 1 0
XQ
EN
"""
    s = build_solver(parse_nec2(deck), basis="sinusoidal").solver
    geom = s._build_geometry()
    k = 2 * np.pi / s.wavelength
    seg_c = geom["seg_centers"]
    far_r = _nec2_far_approx.range_metres(0.25, k)
    assert _nec2_far_approx.ties(acc, seg_c, seg_c, far_r) == 0
    assert s._far_range_ties(geom, k, object()) is True
    assert os.environ[ENV] == "1"
