"""A `TL` with ``skin=True`` is the telegrapher line its k1/k2 describe (momwire#1855).

k1 is skin-effect loss, and skin effect carries an internal reactance equal to
its resistance. The default line keeps the resistance only (EZNEC's lossy
line, momwire#1706); ``skin=True`` adds the reactance, which slows the line by
about α₁/β. The oracle here is built from per-metre R, L, G and C directly,
not from the module's normalised form.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from momwire.networks import (
    TL,
    C_LIGHT,
    Driven,
    Network,
    NetworkReducer,
    PortOnWire,
    PortVirtual,
    tl_abcd,
    tl_admittance_2x2,
)
from momwire.networks._reduce import _tl_gamma_l, _tl_zc

# RG-58 as SimNEC's AC6LA table gives it: dB/100 ft = k0 + k1·√f + k2·f
# (k0 is not a TL term). vf 0.66, 50 Ω.
RG58 = dict(vf=0.66, k1=0.436326, k2=0.009218)
DB_PER_NP = 20.0 / math.log(10.0)


def _wl(f_mhz):
    return C_LIGHT / (f_mhz * 1e6)


def _oracle_abcd(z0, length, f_mhz, vf, k1, k2):
    """The line from its per-metre constants: R from k1 (and an equal ωLᵢ), G
    from k2, L and C from z0 and vf."""
    w = 2 * math.pi * f_mhz * 1e6
    v = vf * C_LIGHT
    a1 = k1 * math.sqrt(f_mhz) / DB_PER_NP / 30.48  # Np/m
    a2 = k2 * f_mhz / DB_PER_NP / 30.48
    r1 = 2 * z0 * a1
    g = 2 * a2 / z0
    zs = r1 + 1j * (w * z0 / v + r1)
    ys = g + 1j * w / (z0 * v)
    gamma, zc = cmath.sqrt(zs * ys), cmath.sqrt(zs / ys)
    ch, sh = cmath.cosh(gamma * length), cmath.sinh(gamma * length)
    return (ch, zc * sh, sh / zc, ch), gamma, zc


@pytest.mark.parametrize("f_mhz", [1.8, 3.55, 14.2, 50.0])
@pytest.mark.parametrize("length", [4.0, 12.0, 30.48])
def test_the_skin_line_is_the_telegrapher_line(f_mhz, length):
    got = tl_abcd(50.0, length, _wl(f_mhz), skin=True, **RG58)
    want, _g, _zc = _oracle_abcd(50.0, length, f_mhz, **RG58)
    for x, y in zip(got, want):
        assert abs(x - y) <= 1e-12 * max(1.0, abs(y))


def test_twelve_metres_of_rg58_at_3_55_mhz_is_2_75_percent_longer():
    """The figure that made the SimNEC station match (antennaknobs#2200)."""
    wl = _wl(3.55)
    _a, b, _c, _d = tl_abcd(50.0, 12.0, wl, skin=True, **RG58)
    _o, gamma, _zc = _oracle_abcd(50.0, 12.0, 3.55, **RG58)
    beta0 = 2 * math.pi / (RG58["vf"] * wl)
    assert gamma.imag / beta0 == pytest.approx(1.02750, abs=5e-5)
    # and the default line has the vf phase exactly
    assert _tl_gamma_l(12.0, wl, RG58["vf"], RG58["k1"], RG58["k2"]).imag == (
        pytest.approx(beta0 * 12.0, rel=1e-15)
    )


def test_the_matched_loss_sits_below_the_table_by_the_internal_inductance():
    """R comes from the table (R = 2·z0·α), and the internal reactance raises
    |Zc|, so the matched loss sits below the table: to second order the ratio
    is 1 − r(r − g)/(2(r + g)), r = R₁/(ωL) and g = G/(ωC) (about r/2 when the
    dielectric term is small: 3.5 % at 1.8 MHz on RG-58). Calibrating R so the
    loss equals the table instead fits the SimNEC station worse (1.8 % against
    1.5 % median, antennaknobs#2200), so the table R stays."""
    for f in (1.8, 7.1, 28.0):
        _o, gamma, _zc = _oracle_abcd(50.0, 1.0, f, **RG58)
        a1 = RG58["k1"] * math.sqrt(f) / DB_PER_NP / 30.48
        a2 = RG58["k2"] * f / DB_PER_NP / 30.48
        beta = 2 * math.pi / (RG58["vf"] * _wl(f))
        r, g = 2 * a1 / beta, 2 * a2 / beta
        ratio = 1 - r * (r - g) / (2 * (r + g))
        assert gamma.real / (a1 + a2) == pytest.approx(ratio, abs=r * r)
        assert tl_abcd(50.0, 1.0, _wl(f), skin=True, **RG58)[0] == pytest.approx(
            cmath.cosh(gamma), rel=1e-12
        )


def test_the_two_forms_describe_one_skin_line():
    wl = _wl(3.7)
    a, b, c, d = tl_abcd(50.0, 4.0, wl, skin=True, **RG58)
    y = tl_admittance_2x2(50.0, 4.0, wl, skin=True, **RG58)
    assert y[0, 0] == pytest.approx(d / b, rel=1e-12)
    assert y[0, 1] == pytest.approx(-1 / b, rel=1e-12)


def test_the_reducer_stamps_the_skin_line():
    """A virtual driver through the line onto a load: the reducer's driving-point
    Z is the oracle's chain-matrix transform, and differs from the default
    line's by far more than the tolerance (so the flag really reached it)."""
    f = 3.55
    wl = _wl(f)
    z_load = 22.0 - 37.0j
    y = np.array([[1.0 / z_load]])

    def z_in(skin):
        net = Network(
            ports={"f": PortOnWire("f"), "in": PortVirtual("in")},
            branches=[TL(a="in", b="f", z0=50.0, length=12.0, skin=skin, **RG58)],
            sources=[Driven(port="in")],
        )
        red = NetworkReducer(net, {"f": 0, "in": 1}, 2)
        return complex(red.driven_impedance(y, wl)[0])

    (a, b, c, d), _g, _zc = _oracle_abcd(50.0, 12.0, f, **RG58)
    want = (a * z_load + b) / (c * z_load + d)
    assert z_in(True) == pytest.approx(want, rel=1e-9)
    assert abs(z_in(False) - want) > 1e-2 * abs(want)


def test_skin_off_is_the_line_as_before():
    wl = _wl(14.2)
    gl = _tl_gamma_l(20.0, wl, RG58["vf"], RG58["k1"], RG58["k2"])
    zc = _tl_zc(50.0, gl)
    sh, ch = np.sinh(gl), np.cosh(gl)
    assert tl_abcd(50.0, 20.0, wl, **RG58) == (ch, zc * sh, sh / zc, ch)
    assert tl_abcd(50.0, 20.0, wl, skin=False, **RG58) == (ch, zc * sh, sh / zc, ch)


def test_skin_on_a_lossless_line_changes_nothing():
    wl = _wl(14.2)
    assert tl_abcd(50.0, 7.3, wl, vf=0.66, skin=True) == tl_abcd(50.0, 7.3, wl, vf=0.66)


def test_skin_must_be_a_bool():
    with pytest.raises(TypeError, match="skin must be True or False"):
        TL("a", "b", z0=50.0, length=1.0, skin=0.4)
