"""A `TL` with ``skin=True`` is AC6LA's Johnson line (momwire#1855).

TLDetails, Zplots, AutoEZ and SimNEC solve a K0K1K2 cable from per-length
elements: R_dc from k0, R_hf from k1, G from k2, L and C from z0 and vf, and
the conductor's internal impedance Z_int = √(R_dc² + [R_hf·(1 + j)]²), which
carries the skin effect's reactance. The default line keeps the resistance
only (EZNEC's lossy line, momwire#1706). The oracle here is that model built
in SI from its per-length elements, not from the module's normalised form.
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

# RG-58A/U (Belden 8259) as the AC6LA table gives it: dB/100 ft =
# k0 + k1·√f + k2·f, vf 0.66, 50 Ω.
RG58 = dict(vf=0.66, k0=0.12942, k1=0.436326, k2=0.009218)
DB_PER_NP = 20.0 / math.log(10.0)


def _wl(f_mhz):
    return C_LIGHT / (f_mhz * 1e6)


def _np_per_m(db_per_100ft):
    return db_per_100ft / DB_PER_NP / 30.48


def _oracle(z0, length, f_mhz, vf, k0, k1, k2):
    """``(ABCD, γ, Zc)`` of the Johnson line from its per-metre elements."""
    w = 2 * math.pi * f_mhz * 1e6
    v = vf * C_LIGHT
    r_dc = 2 * z0 * _np_per_m(k0)
    r_hf = 2 * z0 * _np_per_m(k1 * math.sqrt(f_mhz))
    g = 2 * _np_per_m(k2 * f_mhz) / z0
    z_int = cmath.sqrt(r_dc**2 + (r_hf * (1 + 1j)) ** 2)
    zs = z_int + 1j * w * z0 / v
    ys = g + 1j * w / (z0 * v)
    gamma, zc = cmath.sqrt(zs * ys), cmath.sqrt(zs / ys)
    ch, sh = cmath.cosh(gamma * length), cmath.sinh(gamma * length)
    return (ch, zc * sh, sh / zc, ch), gamma, zc


@pytest.mark.parametrize("f_mhz", [0.137, 1.8, 3.55, 14.2, 50.0])
@pytest.mark.parametrize("length", [4.0, 12.0, 30.48])
def test_the_skin_line_is_the_johnson_line(f_mhz, length):
    got = tl_abcd(50.0, length, _wl(f_mhz), skin=True, **RG58)
    want, _g, _zc = _oracle(50.0, length, f_mhz, **RG58)
    for x, y in zip(got, want):
        assert abs(x - y) <= 1e-12 * max(1.0, abs(y))


def test_ac6las_published_worked_example():
    """AC6LA's "T-Line Model.xls" (ac6la.com/tlmath.html), the model Zplots and
    TLDetails use, works Belden 8240 RG-58 (51.5 Ω, vf 0.66, k0 0.118904,
    k1 0.321239, k2 0.004695) over 100 ft at 14 MHz into 25 + j10 Ω. Its
    printed Zo, γ·length, input impedance and S-parameters (50 Ω reference)
    are the numbers below."""
    kw = dict(vf=0.66, k0=0.118904, k1=0.321239, k2=0.004695)
    a, b, c, d = tl_abcd(51.5, 100 * 0.3048, _wl(14.0), skin=True, **kw)
    _o, gamma, zc = _oracle(51.5, 100 * 0.3048, 14.0, **kw)
    assert zc == pytest.approx(52.0248832435808 - 0.492846813222545j, rel=1e-13)
    assert gamma * 100 * 0.3048 == pytest.approx(
        0.144966144371374 + 13.6885522543927j, rel=1e-13
    )
    z_load = 25 + 10j
    assert (a * z_load + b) / (c * z_load + d) == pytest.approx(
        85.2118304205344 + 19.4658680905111j, rel=1e-12
    )
    den = a + b / 50 + c * 50 + d
    assert (a + b / 50 - c * 50 - d) / den == pytest.approx(
        3.19054814135953e-2 + 4.6713369379071e-3j, rel=1e-12
    )
    assert 2 / den == pytest.approx(0.374987441974728 - 0.77892045789011j, rel=1e-12)


def test_twelve_metres_of_rg58_at_3_55_mhz_is_2_73_percent_longer():
    """The line that carried a SimNEC station (antennaknobs#2200): its phase is
    2.73 % more than vf alone gives. SimNEC's saved readout for that line,
    79.62693048724897 degrees, is this phase; its readout is not always
    current, so the number is pinned to this one line only."""
    wl = _wl(3.55)
    _o, gamma, zc = _oracle(50.0, 12.0, 3.55, **RG58)
    assert math.degrees(gamma.imag * 12.0) == pytest.approx(
        79.62693048724897, rel=1e-12
    )
    _a, b, _c, _d = tl_abcd(50.0, 12.0, wl, skin=True, **RG58)
    assert b == pytest.approx(zc * cmath.sinh(gamma * 12.0), rel=1e-12)
    beta0 = 2 * math.pi / (RG58["vf"] * wl)
    assert gamma.imag / beta0 == pytest.approx(1.02734, abs=1e-5)


def test_k0_alone_is_a_resistance_with_no_internal_inductance():
    """With k1 = 0 the internal impedance is R_dc, real: k0 adds loss, and
    phase only at second order (d²/8), where k1 adds it at first order."""
    f = 3.55
    only_k0 = dict(vf=0.66, k0=0.5, k1=0.0, k2=0.0)
    _o, gamma, _zc = _oracle(50.0, 1.0, f, **only_k0)
    beta0 = 2 * math.pi / (0.66 * _wl(f))
    d = 2 * _np_per_m(0.5) / beta0
    # fourth order is ~d⁴, 1e-6 here
    assert gamma.imag / beta0 == pytest.approx(1 + d * d / 8, abs=d**4)
    assert gamma.real == pytest.approx(_np_per_m(0.5), rel=1e-3)
    got = tl_abcd(50.0, 1.0, _wl(f), skin=True, **only_k0)
    assert got[0] == pytest.approx(cmath.cosh(gamma), rel=1e-12)


def test_the_matched_loss_sits_below_the_table_by_the_internal_inductance():
    """The α's are the table's, and the internal reactance raises |Zc|, so the
    matched loss sits below the table: with k0 = 0, to second order the ratio
    is 1 − r(r − g)/(2(r + g)), r = R_hf/(ωL) and g = G/(ωC), about r/2 when
    the dielectric term is small (3.5 % at 1.8 MHz on RG-58)."""
    no_k0 = dict(RG58, k0=0.0)
    for f in (1.8, 7.1, 28.0):
        _o, gamma, _zc = _oracle(50.0, 1.0, f, **no_k0)
        a1 = _np_per_m(no_k0["k1"] * math.sqrt(f))
        a2 = _np_per_m(no_k0["k2"] * f)
        beta = 2 * math.pi / (no_k0["vf"] * _wl(f))
        r, g = 2 * a1 / beta, 2 * a2 / beta
        ratio = 1 - r * (r - g) / (2 * (r + g))
        assert gamma.real / (a1 + a2) == pytest.approx(ratio, abs=r * r)


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

    (a, b, c, d), _g, _zc = _oracle(50.0, 12.0, f, **RG58)
    want = (a * z_load + b) / (c * z_load + d)
    assert z_in(True) == pytest.approx(want, rel=1e-9)
    assert abs(z_in(False) - want) > 1e-2 * abs(want)


def test_skin_off_is_the_line_as_before():
    wl = _wl(14.2)
    kw = dict(vf=0.66, k1=0.436326, k2=0.009218)
    gl = _tl_gamma_l(20.0, wl, 0.66, 0.436326, 0.009218)
    zc = _tl_zc(50.0, gl)
    sh, ch = np.sinh(gl), np.cosh(gl)
    old = (ch, zc * sh, sh / zc, ch)
    assert tl_abcd(50.0, 20.0, wl, **kw) == old
    assert tl_abcd(50.0, 20.0, wl, skin=False, k0=0.0, **kw) == old


def test_k0_without_skin_is_matched_loss_on_the_resistive_line():
    gl = _tl_gamma_l(1.0, _wl(3.55), 0.66, 0.0, 0.0, None, 0.5)
    assert gl.real == pytest.approx(_np_per_m(0.5), rel=1e-12)


def test_skin_on_a_lossless_line_changes_nothing():
    wl = _wl(14.2)
    on = tl_abcd(50.0, 7.3, wl, vf=0.66, skin=True)
    off = tl_abcd(50.0, 7.3, wl, vf=0.66)
    for x, y in zip(on, off):
        assert x == pytest.approx(y, rel=1e-15, abs=1e-15)


def test_skin_must_be_a_bool():
    with pytest.raises(TypeError, match="skin must be True or False"):
        TL("a", "b", z0=50.0, length=1.0, skin=0.4)
