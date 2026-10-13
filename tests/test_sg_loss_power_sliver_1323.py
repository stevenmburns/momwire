"""The loss-power readout on a sliver segment -- momwire#1323.

`SinusoidalSolver.wire_loss_power` and `_buried_wire_loss_power` integrate
|I|² of the three-term shape P + Q sin kξ + R cos kξ. They used the literal
closed form, which on a short segment adds terms of size (P, R)²h to return
one of size h: momwire#1283's cancellation, in the readout #1283 left alone.
Dan's buried-radial vertical written as one GW through z = 0 (thr_n204) has
a 9.46 um segment at the crossing node (kh = 2.8e-6), and with LD 5 its
Sin-Galerkin wire loss read 4.97e-4 W of 7.01e-3 W in (efficiency 92.9 %)
where bspline reads 4.76e-5 W (99.32 %).

Both readouts now take the cancellation-free spelling `_apply_loading` has
used since #1283, conjugated (`sinusoidal._abs_shape_integrals`).
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire.sinusoidal import _abs_shape_integrals  # noqa: E402
from momwire.sinusoidal_galerkin import _loading_integrals  # noqa: E402
from test_sg_loading_sliver_1283 import K14, SLIVER, _wing  # noqa: E402

LENGTHS = (SLIVER, 2.54e-5, 7.6e-5, 1e-3, 0.0254, 0.3, 0.8, 3.0)
# Real (air), soil A at 14 MHz, and two lossier media.
WAVENUMBERS = (K14, 1.1 - 0.4j, 3.0 - 2.0j, 0.01 - 5.0j)


def _quadrature(k, h):
    """Oracle: Gauss-Legendre on the pointwise integrands, cos − 1 spelled
    −2 sin²(kξ/2) so no sample cancels."""
    x, w = np.polynomial.legendre.leggauss(60)
    xi = 0.5 * h * x
    g = -2.0 * np.sin(0.5 * k * xi) ** 2
    half = 0.5 * h
    return (
        half * np.sum(w * g),
        half * np.sum(w * np.abs(np.sin(k * xi)) ** 2),
        half * np.sum(w * np.abs(g) ** 2),
        half * np.sum(w * np.abs(np.cos(k * xi)) ** 2),
    )


@pytest.mark.parametrize("k", WAVENUMBERS)
@pytest.mark.parametrize("h", LENGTHS)
def test_the_integrals_match_quadrature(k, h):
    for got, want in zip(_abs_shape_integrals(k, h), _quadrature(k, h)):
        assert abs(got - want) <= 1e-12 * abs(want)


@pytest.mark.parametrize("h", LENGTHS)
def test_at_real_k_they_are_the_loading_overlaps_integrals(h):
    """The conjugated family and #1283's bilinear one are the same three
    numbers when k is real."""
    got = _abs_shape_integrals(K14, h)
    for a, b in zip(got[:3], _loading_integrals(K14, h)):
        assert abs(a - b) <= 1e-13 * abs(b)


def _power_stable(k, h, A, B, C, AC):
    i_g, i_ss, i_gg, _i_cc = _abs_shape_integrals(k, h)
    return (
        abs(AC) ** 2 * h
        + 2.0 * np.real(AC * np.conj(C) * np.conj(i_g))
        + abs(B) ** 2 * i_ss
        + abs(C) ** 2 * i_gg
    )


def _power_literal(k, h, A, B, C, AC):
    """The pre-#1323 spelling of `wire_loss_power`'s ∫|I|²."""
    return (
        abs(A) ** 2 * h
        + 2.0 * np.real(A * np.conj(C)) * (2.0 / k) * np.sin(0.5 * k * h)
        + abs(B) ** 2 * (0.5 * h - np.sin(k * h) / (2.0 * k))
        + abs(C) ** 2 * (0.5 * h + np.sin(k * h) / (2.0 * k))
    )


def _power_quadrature(k, h, A, B, C, AC):
    x, w = np.polynomial.legendre.leggauss(24)
    xi = 0.5 * h * x
    f = AC + B * np.sin(k * xi) - 2.0 * C * np.sin(0.5 * k * xi) ** 2
    return 0.5 * h * np.sum(w * np.abs(f) ** 2)


@pytest.mark.parametrize("h", [SLIVER, 2.54e-5, 7.6e-5, 1e-3, 0.0254, 0.5])
def test_the_wing_power_matches_quadrature_at_every_length(h):
    """A crossing wing's own shape, with a complex amplitude on it."""
    amp = 0.7 - 0.4j
    args = (K14, h, *(amp * np.asarray(_wing(K14 * h))))
    exact = _power_quadrature(*args)
    assert abs(_power_stable(*args) - exact) <= 1e-12 * exact


def test_the_literal_form_is_what_lost_the_digits():
    """On the 9.46 um sliver the old spelling is off by decades of the
    answer; at an ordinary segment length the two agree. If the first
    assertion ever fails, the oracle has stopped seeing the sliver."""
    args = (K14, SLIVER, *_wing(K14 * SLIVER))
    exact = _power_quadrature(*args)
    assert abs(_power_literal(*args) - exact) > 1e3 * exact
    args = (K14, 0.5, *_wing(K14 * 0.5))
    exact = _power_quadrature(*args)
    assert abs(_power_literal(*args) - exact) <= 1e-9 * exact


def _budget(name, basis):
    from test_eznec_split_crossing_1281 import fixture

    from momwire.deck._nec5 import parse_nec5
    from momwire.eznec import _serve

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _serve.serve(parse_nec5(fixture(name)), basis=basis).power


# bspline's wire loss on thr_n204 with LD 5 (W, 1 V drive), measured on
# Skylake; the slow test below re-measures it.
BSPLINE_THR_N204_LOSS = 4.759522e-05


def test_dans_through_deck_reads_its_copper_loss():
    """thr_n204 on Sin-Galerkin: 4.761e-5 W and 99.32 %. Before the fix the
    same solve read 4.971e-4 W and 92.91 %."""
    b = _budget("thr_n204", "sinusoidal-galerkin")
    assert b.wire_loss == pytest.approx(BSPLINE_THR_N204_LOSS, rel=0.01)
    assert b.efficiency_percent == pytest.approx(99.32, abs=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["thr_n21", "hs_n204"])
def test_decks_without_a_sliver_read_what_they_did(name):
    """The spelling changes no deck whose segments are ordinary: these two
    read the same loss before and after, to the readout's own rounding."""
    want = {"thr_n21": 4.732150e-05, "hs_n204": 4.761289e-05}[name]
    assert _budget(name, "sinusoidal-galerkin").wire_loss == pytest.approx(
        want, rel=2e-6
    )


@pytest.mark.slow
def test_dans_through_deck_loses_what_bspline_loses():
    """The issue's gate: the same deck on both bases."""
    sg = _budget("thr_n204", "sinusoidal-galerkin")
    bs = _budget("thr_n204", "bspline")
    assert sg.wire_loss == pytest.approx(bs.wire_loss, rel=0.01)
    assert sg.efficiency_percent == pytest.approx(bs.efficiency_percent, abs=0.01)
