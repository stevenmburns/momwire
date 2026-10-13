"""Sin-Galerkin on the rise-spelled fan in free space -- momwire#1344.

`fan_rise_deck` is N radials that each run at depth and rise to the node, a
monopole above: the N rise segments coincide on one line, INSIDE N different
polylines. When the issue was filed (a69cd0aa) SinusoidalGalerkinSolver read
-5.61 - 3326.49j ohm on it in free space, where bspline reads 32.94 - 337.42j
and the point-matched sinusoidal family 32.75 - 342.05j.

Two findings, both pinned here.

**The default no longer reproduces it.** Bisected: the first commit that
reads 32.80 - 341.13j is 80ff53fa, "the mixed-potential fill is
SinusoidalGalerkinSolver's default" (momwire#1354). That fill solves the
deck: with its own family to 0.3 %, with bspline to 1.1 %, at 2, 3, 4 and 6
radials, and as an answer rather than rounding (moving every vertex by
1e-12 m moves Z by under 1e-5 ohm; the AVX2 and SSE2 builds agree to 2e-11).

**The defect is still in the direct-field fill**, which `fill="direct"`
asks for and which a reflection-coefficient ground, a wire end in the
ground plane and junction ports still take. There the answer for two
segments on one path is decided by rounding: -5.6 - 3326j as written,
3977 - 26802j with the whole deck moved 0.3 m, 32.8 - 341.1j (right) moved
1 m, and -0.5 - 1567j as written on the SSE2 build. So that fill now
refuses the spelling by name, on `_KEPT_COINCIDENT_REFUSAL`'s pattern.

The split-rise spelling (each rise its own WIRE, joined to its own radial)
is a different deck and is still refused on every fill, momwire#1333.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402
from momwire.sinusoidal_galerkin import (  # noqa: E402
    SinusoidalGalerkinSolver,
    _coincident_segments,
)
from test_crossing_serve_524 import SOIL_A, fan_rise_deck  # noqa: E402

# What the issue reported, measured again at a69cd0aa on Skylake.
NONSENSE_1344 = -5.608360597626039 - 3326.4944151821837j
# Skylake, main at ae412830.
SG_4 = 32.79917557605253 - 341.1313453694269j
BSPLINE_4 = 32.940338670789885 - 337.41988732042205j


def _free(n_radials=4, lift=0.0, **more):
    deck = fan_rise_deck(n_radials=n_radials)
    for key in ("ground_z", "ground_eps", "ground_model"):
        deck.pop(key)
    deck["wires"] = [w + np.array([0.0, 0.0, lift]) for w in deck["wires"]]
    deck.update(more)
    return deck


def _z(cls, deck):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(cls(**deck).compute_impedance()[0])


def test_the_fan_reads_its_impedance_not_the_issues():
    z = _z(SinusoidalGalerkinSolver, _free())
    assert z == pytest.approx(SG_4, abs=1e-3)
    assert abs(z - NONSENSE_1344) > 2000.0


@pytest.mark.parametrize("n_radials", [2, 3, 4, 6])
def test_it_agrees_with_the_other_families(n_radials):
    deck = _free(n_radials)
    sg = _z(SinusoidalGalerkinSolver, deck)
    bs = _z(BSplineSolver, deck)
    pm = _z(SinusoidalSolver, deck)
    assert abs(sg - bs) < 0.02 * abs(bs)
    assert abs(sg - pm) < 0.03 * abs(pm)
    if n_radials == 4:
        assert bs == pytest.approx(BSPLINE_4, abs=1e-3)


def test_the_answer_is_not_rounding():
    """Coincident conductors are where an answer can be decided by rounding
    (momwire#1333's split fan was). Breaking the coincidence at 1e-12 m is
    four decades above rounding and must not move the answer."""
    rng = np.random.default_rng(1344)
    z0 = _z(SinusoidalGalerkinSolver, _free())
    for _ in range(3):
        deck = _free()
        deck["wires"] = [
            w + 1e-12 * rng.standard_normal(w.shape) for w in deck["wires"]
        ]
        assert abs(_z(SinusoidalGalerkinSolver, deck) - z0) < 1e-5


# --------------------------------------------------------------------------
# the direct-field fill refuses the spelling
# --------------------------------------------------------------------------
REFL = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="refl-coef")


@pytest.mark.parametrize(
    "deck",
    [
        _free(fill="direct"),
        _free(lift=0.3, fill="direct"),  # read 3977 - 26802j
        _free(lift=1.0, **REFL),  # takes the direct fill unasked
        _free(n_radials=2, fill="direct"),
    ],
    ids=["as-written", "moved-0.3m", "refl-coef-ground", "two-radials"],
)
def test_the_direct_fill_refuses_two_segments_on_one_path(deck):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as caught:
            SinusoidalGalerkinSolver(**deck).compute_impedance()
    said = str(caught.value)
    assert "coincide, end for end" in said and "momwire#1344" in said
    assert "depends on rounding" in said
    # it says what does solve the deck
    assert "mixed-potential fill solves them" in said and "BSplineSolver" in said


def test_the_fills_that_solve_it_still_do():
    """Free space, PEC and Sommerfeld take the mixed-potential fill."""
    z = _z(SinusoidalGalerkinSolver, _free(lift=1.0, ground_z=0.0))
    assert z == pytest.approx(61.5699 - 287.1141j, abs=1e-3)


def test_the_direct_fill_still_solves_conductors_that_are_only_close():
    """Close is not coincident: the same fan with every vertex moved by a
    micron is two conductors a micron apart, and the direct fill serves it."""
    rng = np.random.default_rng(1344)
    deck = _free(fill="direct")
    deck["wires"] = [w + 1e-6 * rng.standard_normal(w.shape) for w in deck["wires"]]
    z = _z(SinusoidalGalerkinSolver, deck)
    assert abs(z - SG_4) < 0.5


def test_the_detector_reads_endpoints_not_wires():
    a = np.array([(0, 0, 0), (0, 0, 1), (5, 0, 0), (0, 0, 1)], dtype=float)
    b = np.array([(0, 0, 1), (0, 0, 2), (0, 0, 0), (0, 0, 3)], dtype=float)
    assert _coincident_segments(a, b) is None
    b[3] = (0, 0, 2)  # segment 3 is segment 1 reversed
    assert _coincident_segments(a, b) == (1, 3)
    b[3] = (0, 0, 2 + 1e-6)  # a micron off: another conductor
    assert _coincident_segments(a, b) is None
