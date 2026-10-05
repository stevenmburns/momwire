"""The reducer reads frequency through the design's c (antennaknobs#1685).

A deck-derived design solves at NEC's wavelength, 299.8/f, while NEC evaluates
its lumped loads and line loss at the deck's own f.  Before
``c_light_mhz_m`` the reducer converted the wavelength back with the SI c, so
every frequency-dependent branch of such a design sat 25 ppm below the file's
frequency: 14.17464 MHz for a 14.175 MHz deck.

Three branch kinds read a frequency, and each is pinned here at the file's
own f when the design's c is passed: a lumped part's omega, a line's matched
loss, and a Touchstone block's sample frequency.  The default (no keyword) is
pinned bit-identical to the SI conversion every caller relied on before.
"""

import numpy as np

from momwire.networks import (
    TL,
    C_LIGHT,
    Driven,
    Network,
    NetworkReducer,
    PortOnWire,
    PortVirtual,
    TouchstoneLoad,
    TwoPort,
    tl_abcd,
)
from momwire.networks._reduce import frequency_mhz

NEC_C = 299.8  # NEC's metre-megahertz product
F_MHZ = 14.175  # AC6LA's LC1 file frequency, the one AK#1685 measured on
WL = NEC_C / F_MHZ  # the wavelength a deck-derived design solves at

Y_ANT = np.array([[0.012 + 0.004j]])


def _series_c_reducer(c_farads, **kw):
    """A driven feed reaching the antenna through a series capacitor: the
    driven impedance is Z_ant + 1/(j omega C), so omega is read straight off
    the answer."""
    net = Network(
        ports={"feed": PortVirtual("feed"), "ant": PortOnWire("ant")},
        branches=[TwoPort("feed", "ant", c=c_farads)],
        sources=[Driven(port="feed")],
    )
    return NetworkReducer(net, {"ant": 0, "feed": 1}, 2, **kw)


def _omega_read_back(reducer, c_farads):
    z = complex(reducer.driven_impedance(Y_ANT, WL, z_ref=0.0)[0])
    x_cap = (z - 1.0 / Y_ANT[0, 0]).imag
    return -1.0 / (x_cap * c_farads)


def test_a_lumped_part_sits_at_the_file_frequency_under_the_design_c():
    c = 100e-12
    omega = _omega_read_back(_series_c_reducer(c, c_light_mhz_m=NEC_C), c)
    assert abs(omega / (2 * np.pi * F_MHZ * 1e6) - 1.0) < 1e-12


def test_without_the_keyword_the_part_sits_25_ppm_low():
    """The defect, kept visible: the SI c reads this wavelength at
    14.175 x 299.792458/299.8 MHz."""
    c = 100e-12
    omega = _omega_read_back(_series_c_reducer(c), c)
    shifted = F_MHZ * (C_LIGHT / 1e6) / NEC_C
    assert abs(omega / (2 * np.pi * shifted * 1e6) - 1.0) < 1e-12
    assert abs(shifted / F_MHZ - 1.0 - (-25.2e-6)) < 0.1e-6


def test_the_default_is_bit_identical_to_the_si_conversion():
    """None must be the pre-keyword expression, not a numerically equal one."""
    assert frequency_mhz(WL) == C_LIGHT / WL / 1e6
    net_kw = _series_c_reducer(100e-12, c_light_mhz_m=None)
    net = _series_c_reducer(100e-12)
    a = net.driven_impedance(Y_ANT, WL)
    b = net_kw.driven_impedance(Y_ANT, WL)
    assert np.array_equal(a, b)
    line = tl_abcd(50.0, 3.0, WL, vf=0.66, k1=0.5, k2=0.01)
    assert line == tl_abcd(50.0, 3.0, WL, vf=0.66, k1=0.5, k2=0.01, c_light_mhz_m=None)


def test_a_line_keeps_the_wavelength_for_its_phase_and_the_file_f_for_its_loss():
    """Phase from the wavelength as handed over (NEC's own), loss from the
    design's frequency: a lossless line does not move with the keyword at all,
    and a lossy one's attenuation is the cable model's at 14.175 MHz."""
    lossless = tl_abcd(50.0, 7.0, WL, vf=0.66)
    assert lossless == tl_abcd(50.0, 7.0, WL, vf=0.66, c_light_mhz_m=NEC_C)

    k1, k2, length = 0.4, 0.002, 7.0
    a, _b, _c, _d = tl_abcd(50.0, length, WL, k1=k1, k2=k2, c_light_mhz_m=NEC_C)
    gl = np.arccosh(a)
    db_per_100ft = k1 * np.sqrt(F_MHZ) + k2 * F_MHZ
    alpha = db_per_100ft * np.log(10) / 20 / (100 * 0.3048)
    assert abs(gl.real / (alpha * length) - 1.0) < 1e-9
    assert abs(abs(gl.imag) - (2 * np.pi * length / WL) % (2 * np.pi)) < 1e-9


class _RecordingY:
    """The whole `AdmittanceData` contract: ``nports`` and ``y_at(f_hz)``,
    recording the frequency it was asked at."""

    nports = 1

    def __init__(self):
        self.asked = []

    def y_at(self, f_hz):
        self.asked.append(f_hz)
        return np.array([[0.02 + 0.0j]])


def test_a_touchstone_block_is_sampled_at_the_file_frequency():
    data = _RecordingY()
    net = Network(
        ports={"feed": PortOnWire("feed"), "tip": PortVirtual("tip")},
        branches=[TL("feed", "tip", z0=50.0, length=1.0), TouchstoneLoad("tip", data)],
        sources=[Driven(port="feed")],
    )
    NetworkReducer(net, {"feed": 0, "tip": 1}, 2, c_light_mhz_m=NEC_C).driven_impedance(
        Y_ANT, WL
    )
    assert data.asked and all(abs(f / (F_MHZ * 1e6) - 1.0) < 1e-12 for f in data.asked)
