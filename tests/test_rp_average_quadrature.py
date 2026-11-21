"""``RP 0``'s average power gain is the trapezoid rule in phi on the NEC-5
engine (the follow-up #1967's review filed against momwire#1962).

momwire averaged the first ``NPH - 1`` phis at full weight, which is the
trapezoid rule only over a closed turn, where the last phi is the first one
again. Short of a turn it is not: on two phis 90 degrees apart x13 printed
0.9924 where momwire printed 0.7181, over the same solid angle. And over a
zero step on an axis with more than one angle x13 prints ``NaN`` (and a
solid angle of 0.0000) where momwire printed numbers.

The rule was read black-box off licensed x13 printouts of synthetic decks,
``fixtures/rp_average_quadrature/`` (its README says how). The tests hold
three things: the evidence (each capture's average follows from its own
printed |E| rows under the trapezoid rule, with one constant per deck
geometry), momwire's trailer against each capture (the solid angle and the
NaN lines exactly, the average to the two solvers' difference), and the
rule itself on a constructed pattern. The C++ printout core prints the
Python bytes (test_razor_core_standalone's
`test_the_average_quadrature_decks_print_as_python`). The NEC-4.2 slot keeps
its old trailer: no NEC-4.2 capture measures it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from momwire._far_readout import average_gain
from momwire.dropin.nec5 import _nec4 as _nec4_slot
from momwire.dropin.nec5 import _shell

FIXTURES = Path(__file__).parent / "fixtures" / "rp_average_quadrature"
STEMS = sorted(p.stem for p in FIXTURES.glob("*.nec"))
ZERO_STEP = [s for s in STEMS if s.startswith("z_")]
AVERAGED = [s for s in STEMS if not s.startswith("z_")]
_TRAILER = re.compile(
    r"AVERAGE POWER GAIN=\s*(\S+)\s+SOLID ANGLE USED IN AVERAGING=\(\s*(\S+)\)"
)
_RP = re.compile(r"^RP\s", re.M)


def _deck(stem: str) -> str:
    return (FIXTURES / f"{stem}.nec").read_bytes().decode("latin-1")


def _engine(stem: str) -> str:
    return (FIXTURES / f"{stem}.nec5.out").read_bytes().decode("latin-1")


def _card(stem: str) -> tuple[int, int, float, float, float, float]:
    line = next(x for x in _deck(stem).splitlines() if x.startswith("RP"))
    f = line.split()
    return int(f[2]), int(f[3]), float(f[5]), float(f[6]), float(f[7]), float(f[8])


def _trailer(text: str) -> tuple[str, str]:
    m = _TRAILER.search(text)
    assert m, text[-3000:]
    return m.group(1), m.group(2)


def _trailer_lines(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    k = next(i for i, x in enumerate(lines) if "AVERAGE POWER GAIN=" in x)
    return lines[k : k + 3]


def _e_squared(text: str, n_theta: int, n_phi: int) -> np.ndarray:
    """The pattern rows' |E(theta)|^2 + |E(phi)|^2, as ``[theta, phi]``."""
    rows = []
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(i for i, x in enumerate(lines) if "DEGREES  DEGREES       DB" in x)
    for line in lines[start + 1 :]:
        t = line.split()
        if len(t) < 11 or not re.match(r"-?\d", t[0]):
            break
        rows.append(float(t[-4]) ** 2 + float(t[-2]) ** 2)
    assert len(rows) == n_theta * n_phi
    return np.array(rows).reshape(n_phi, n_theta).T


def test_the_fixtures_are_there():
    assert len(STEMS) == 25
    for stem in STEMS:
        assert (FIXTURES / f"{stem}.nec5.out").exists(), stem
    assert (FIXTURES / "README.txt").exists()


def test_the_engine_integrates_phi_by_the_trapezoid_rule():
    """The evidence: |E|^2 is the gain up to one constant per geometry, so
    the right quadrature of each capture's own rows implies one constant
    across every capture. The trapezoid rule does (to the six printed
    figures); momwire's old rule spreads it by tens of percent."""
    implied: dict[str, list[float]] = {"free": [], "gn1": []}
    old: list[float] = []
    for stem in AVERAGED:
        if stem.endswith("_a2"):
            continue  # A = 2 prints no rows to integrate
        nth, nph, _t0, _p0, dth, dph = _card(stem)
        text = _engine(stem)
        e2 = _e_squared(text, nth, nph)
        thetas = _t0 + dth * np.arange(nth)
        new_avg, solid = average_gain(e2, thetas, dth, dph, nph, trapezoid_phi=True)
        printed, solid_pi = _trailer(text)
        assert f"{abs(solid) / math.pi:.4f}" == solid_pi, stem
        implied["gn1" if stem.startswith("gn1") else "free"].append(
            float(printed) / new_avg
        )
        if not stem.startswith("gn1") and dth > 0:
            old_avg, _ = average_gain(e2, thetas, dth, dph, nph)
            old.append(float(printed) / old_avg)
    for name, cs in implied.items():
        assert len(cs) >= 2, name
        assert max(cs) / min(cs) - 1 < 2e-5, (name, cs)
    assert max(old) / min(old) - 1 > 0.3, old


@pytest.mark.parametrize("stem", ZERO_STEP)
def test_the_engine_prints_nan_over_a_zero_step(stem):
    lines = _trailer_lines(_engine(stem))
    assert lines[0] == (
        "   AVERAGE POWER GAIN=         NaN       SOLID ANGLE USED IN "
        "AVERAGING=( 0.0000)*PI STERADIANS."
    )
    assert lines[2] == (
        "   POWER RADIATED ASSUMING RADIATION INTO 4*PI STERADIANS =         NaN WATTS"
    )


@pytest.mark.parametrize("basis", ["razor2p", "bs2"])
@pytest.mark.parametrize("stem", STEMS)
def test_momwire_prints_the_engines_trailer(stem, basis):
    """The solid angle exactly, a zero step's NaN lines exactly, and the
    average to within the two solvers' own difference (their full-sphere
    averages differ by about 0.5 %; the old rule missed p2_d90 by 28 %)."""
    mine = _shell.render(_deck(stem), basis=basis)
    engine = _engine(stem)
    if stem in ZERO_STEP:
        assert _trailer_lines(mine) == _trailer_lines(engine)
        return
    (avg_m, solid_m), (avg_e, solid_e) = _trailer(mine), _trailer(engine)
    assert solid_m == solid_e
    assert float(avg_m) == pytest.approx(float(avg_e), rel=0.012), (avg_m, avg_e)


def test_the_rule_on_a_constructed_pattern():
    """Over a closed turn the trapezoid rule is the old one (the end columns
    are one direction); short of a turn the end columns count half; a
    descending axis reads as its mirror; a zero step is NaN."""
    rng = np.random.default_rng(1967)
    thetas = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
    g = rng.uniform(0.5, 2.0, size=(5, 37))
    g[:, -1] = g[:, 0]
    old, s_old = average_gain(g, thetas, 30.0, 10.0, 37)
    new, s_new = average_gain(g, thetas, 30.0, 10.0, 37, trapezoid_phi=True)
    assert new == pytest.approx(old, rel=1e-14)
    assert s_new == pytest.approx(s_old, rel=1e-14)
    two = g[:, :2]
    by_hand = np.sum((two[:, 0] + two[:, 1]) * 0.5 * _bands(thetas, 30.0)) / np.sum(
        _bands(thetas, 30.0)
    )
    new, s_up = average_gain(two, thetas, 30.0, 90.0, 2, trapezoid_phi=True)
    assert new == pytest.approx(by_hand, rel=1e-14)
    down, s_down = average_gain(
        two[::-1], thetas[::-1], -30.0, 90.0, 2, trapezoid_phi=True
    )
    assert down == pytest.approx(new, rel=1e-14)
    assert abs(s_down) == pytest.approx(abs(s_up), rel=1e-14)
    avg, solid = average_gain(two, thetas, 30.0, 0.0, 2, trapezoid_phi=True)
    assert math.isnan(avg) and solid == 0.0
    avg, solid = average_gain(two, np.full(5, 30.0), 0.0, 90.0, 2, trapezoid_phi=True)
    assert math.isnan(avg) and solid == 0.0


def _bands(thetas: np.ndarray, d: float) -> np.ndarray:
    lo = np.radians(np.maximum(thetas - 0.5 * d, thetas[0]))
    hi = np.radians(np.minimum(thetas + 0.5 * d, thetas[-1]))
    return np.cos(lo) - np.cos(hi)


def test_the_nec42_slot_keeps_its_trailer():
    """Not measured on NEC-4.2: a zero phi step still prints numbers there."""
    out = _nec4_slot.render(_deck("z_dph0_p2"), basis="sinusoidal")
    avg, _solid = _trailer(out)
    assert avg != "NaN"


@pytest.mark.parametrize("basis", ["razor2p", "bs2"])
def test_momwire_differs_from_the_engine_by_one_factor(basis):
    """momwire's averages sit a fixed fraction under x13's on every deck of
    one geometry (its input power and currents are another solver's), so
    the quadrature is the engine's: the ratio is one number to 1e-3 across
    phi spans of 90 to 480 degrees, descending axes, two to 37 phis. Under
    the old rule p2_d90 alone was 28 % off."""
    ratios = {}
    for stem in AVERAGED:
        if stem.startswith("gn1"):
            continue
        avg_m, _ = _trailer(_shell.render(_deck(stem), basis=basis))
        avg_e, _ = _trailer(_engine(stem))
        ratios[stem] = float(avg_m) / float(avg_e)
    assert len(ratios) == 17
    assert max(ratios.values()) - min(ratios.values()) < 1e-3, ratios
