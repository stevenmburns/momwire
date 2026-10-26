"""A lossy TL's characteristic impedance is complex: z0·(1 − jα/β) — momwire#1706.

EZNEC's lossy line (and antennaknobs' ``.ez`` importer) uses that Zc; the
stamp used the real z0, which put a native model 0.2 Ω from its EZNEC twin on
a 66 ft lossy coax. Lossless lines are unchanged bit for bit.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from momwire.networks import tl_abcd, tl_admittance_2x2
from momwire.networks._reduce import _tl_gamma_l

WL = 299.792458 / 14.2  # a wavelength at 14.2 MHz
RG58 = dict(vf=0.66, k1=0.33, k2=0.0016)  # a lossy coax, dB/100 ft


def _zc(z0, length, **kw):
    gl = _tl_gamma_l(
        length, WL, kw.get("vf", 1.0), kw.get("k1", 0.0), kw.get("k2", 0.0)
    )
    return z0 * (1 - 1j * gl.real / gl.imag), gl


def test_a_lossless_line_is_bit_for_bit_the_real_z0_stamp():
    gl = _tl_gamma_l(7.3, WL, 0.66, 0.0, 0.0)
    sh, ch = np.sinh(gl), np.cosh(gl)
    assert tl_abcd(50.0, 7.3, WL, vf=0.66) == (ch, 50.0 * sh, sh / 50.0, ch)
    y = tl_admittance_2x2(50.0, 7.3, WL, vf=0.66)
    assert np.array_equal(y, (1 / (50.0 * sh)) * np.array([[ch, -1], [-1, ch]]))


@pytest.mark.parametrize("length", [1.0, 7.3, 20.1168, 100.0])
def test_a_lossy_line_terminated_in_its_zc_reflects_nothing(length):
    zc, _ = _zc(50.0, length, **RG58)
    a, b, c, d = tl_abcd(50.0, length, WL, **RG58)
    zin = (a * zc + b) / (c * zc + d)
    assert abs(zin - zc) < 1e-12 * abs(zc)


def test_the_chain_and_admittance_forms_describe_one_line():
    a, b, c, d = tl_abcd(50.0, 20.1168, WL, **RG58)
    y = tl_admittance_2x2(50.0, 20.1168, WL, **RG58)
    # Y from ABCD (i_b out of port B): y11 = d/b, y12 = -1/b
    assert y[0, 0] == pytest.approx(d / b, rel=1e-12)
    assert y[0, 1] == pytest.approx(-1 / b, rel=1e-12)


def test_zc_is_the_first_order_form_eznec_uses():
    """α/β from the cable model, at 14.2 MHz: Zc = 50·(1 − jα/β), its
    imaginary part −50·α/β: −0.530 Ω for this cable at 14.2 MHz."""
    zc, gl = _zc(50.0, 20.1168, **RG58)
    alpha_np_m = (0.33 * math.sqrt(14.2) + 0.0016 * 14.2) / (20 / math.log(10)) / 30.48
    beta = 2 * math.pi / (0.66 * WL)
    assert zc == pytest.approx(complex(50.0, -50.0 * alpha_np_m / beta), rel=1e-9)
    assert zc.imag == pytest.approx(-0.5303449, abs=1e-6)
    # and it is the series-loss Zc to first order: Z0·√(1 + R/(jωL)) with
    # R/(ωL) = 2α/β, i.e. Z0·√(1 − 2jα/β) ≈ Z0·(1 − jα/β)
    exact = 50.0 * cmath.sqrt(1 - 2j * gl.real / gl.imag)
    # the two part at second order: Z0·(α/β)²/2, 0.0028 Ω here
    ab = gl.real / gl.imag
    assert abs(zc - exact) == pytest.approx(50.0 * ab**2 / 2, rel=0.01)
