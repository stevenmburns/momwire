"""Sin-Galerkin's loading overlap on a sliver segment — momwire#1283.

Dan's buried-radial vertical written as ONE GW through z = 0 (thr_n204) is cut
at the plane 9.46 um above node 0, so the crossing node's wing lives on a
9.46 um segment (kh = 2.8e-6). The wing's shape there has A = -C = 2.6e11 and
A + C = 0.25, and the literal closed form of the overlap integral added terms
of size (A, C)^2 h to return one of size h: it read -256 against an exact
1.89e-6, and the deck answered 460 ohm with LD 5 against 71 on bspline.

Measured on the way to the fix (scratch, Haswell): the length of the segment
against its RADIUS is not the axis. A free-space lossy dipole with one
segment of 0.001 a spliced in agrees with bspline to 3 mOhm, and so does the
hand-split deck in free space; the break needs the crossing wing, and on Dan's
deck it sets in between s = 7.6e-5 m (agrees) and 2.5e-5 m (5 ohm off).
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire.sinusoidal_galerkin import _loading_integrals  # noqa: E402

K14 = 2.0 * np.pi * 14e6 / 299792458.0
SLIVER = 9.46e-6


def _wing(kd):
    """The crossing wing's (A, B, C, A + C) on a segment of electrical
    length kd (`_crossing_wing_view`)."""
    q = np.cos(0.5 * kd) / np.sin(0.5 * kd)
    s4 = np.sin(0.25 * kd)
    ac = q * s4 * s4 / (np.sin(0.5 * kd) * np.cos(0.5 * kd))
    return (
        q / np.sin(kd),
        q / (2.0 * np.cos(0.5 * kd)),
        -q / (2.0 * np.sin(0.5 * kd)),
        ac,
    )


def _overlap_stable(k, h, A, B, C, AC):
    i_g, i_ss, i_gg = _loading_integrals(k, h)
    return AC * AC * h + 2.0 * AC * C * i_g + B * B * i_ss + C * C * i_gg


def _overlap_quadrature(k, h, A, B, C, AC):
    """Oracle: Gauss-Legendre on the pointwise shape, spelled
    AC + B sin + C (cos - 1) so no sample cancels."""
    x, w = np.polynomial.legendre.leggauss(24)
    xi = 0.5 * h * x
    f = AC + B * np.sin(k * xi) - 2.0 * C * np.sin(0.5 * k * xi) ** 2
    return 0.5 * h * np.sum(w * f * f)


@pytest.mark.parametrize("h", [SLIVER, 2.54e-5, 7.6e-5, 1e-3, 0.0254, 0.5])
def test_the_wing_overlap_matches_quadrature_at_every_length(h):
    args = (K14, h, *_wing(K14 * h))
    exact = _overlap_quadrature(*args)
    assert abs(_overlap_stable(*args) - exact) <= 1e-12 * abs(exact)


def test_the_literal_form_is_what_broke():
    """The pre-#1283 spelling on the same wing: off by eight decades of the
    answer's own size, with the WRONG SIGN. If this ever passes, the oracle
    has stopped seeing the sliver."""
    A, B, C, _ac = _wing(K14 * SLIVER)
    h, k = SLIVER, K14
    literal = (
        A * A * h
        + 2.0 * A * C * (2.0 / k) * np.sin(0.5 * k * h)
        + B * B * (0.5 * h - np.sin(k * h) / (2.0 * k))
        + C * C * (0.5 * h + np.sin(k * h) / (2.0 * k))
    )
    exact = _overlap_quadrature(K14, SLIVER, *_wing(K14 * SLIVER))
    assert abs(literal - exact) > 1e3 * abs(exact)


def test_the_integrals_hold_for_a_buried_k():
    """Complex k_m (soil A at 14 MHz is ~1.1 - 0.4j rad/m): the series and the
    S-form both carry it."""
    k = 1.1 - 0.4j
    for h in (SLIVER, 0.3, 0.8, 3.0):
        x, w = np.polynomial.legendre.leggauss(40)
        xi = 0.5 * h * x
        g = -2.0 * np.sin(0.5 * k * xi) ** 2
        want = [
            0.5 * h * np.sum(w * g),
            0.5 * h * np.sum(w * np.sin(k * xi) ** 2),
            0.5 * h * np.sum(w * g * g),
        ]
        for got, ref in zip(_loading_integrals(k, h), want):
            assert abs(got - ref) <= 1e-12 * abs(ref)


@pytest.mark.slow
def test_dans_through_deck_loads_like_bspline():
    """The issue's gate: thr_n204 on Sin-Galerkin with and without LD 5, next
    to bspline. The copper's effect (Z_L - Z_0) agrees across the two bases;
    before the fix Sin-Galerkin's read 390 ohm."""
    from test_eznec_split_crossing_1281 import LD5_ALL, fixture, serve_z

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = fixture("thr_n204")
        bare = loaded.replace(LD5_ALL, "")
        assert bare != loaded
        d_sg = serve_z(loaded, "sinusoidal-galerkin") - serve_z(
            bare, "sinusoidal-galerkin"
        )
        d_bs = serve_z(loaded, "bspline") - serve_z(bare, "bspline")
    assert abs(d_sg - d_bs) < 0.1 * abs(d_bs), (d_sg, d_bs)
