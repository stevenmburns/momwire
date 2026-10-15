"""The pattern readout over a whole grid (momwire#1448) is the per-point one.

``eznec._serve._pattern`` used to build its rows one direction at a time
through the scalar readouts (``_polarisation``, ``_gain_db``, ``_phase_deg``);
on SimNEC's small decks that loop cost more than the solve. It is now the same
arithmetic over the grid, and every printed digit has to survive the move. The
captures and the byte-identity census over every fixture deck are the main
gate; this file pins the rules one at a time, each against the scalar form it
replaced, because each exists for a printed-digit reason:

* an EXACTLY zero component prints phase ``0.00``, never ``-0.00``;
* a DUST component is zeroed, and the row's TILT still reads +90 or -90 off
  the sign of ``cos δ`` (a signed zero inside ``atan2``);
* below the horizon over ground the row is all zeros, gains at the floor;
* RFLD > 0 adds the range phase in degrees WITHOUT re-wrapping (-332.80).

"Equal" here means equal bits, sign of zero included: a printed ``-0.00``
against ``0.00`` is exactly a sign-of-zero difference.
"""

import math

import numpy as np
import pytest

from momwire._far_readout import (
    _DEG_TO_RAD,
    _RAD_TO_DEG,
    _gain_db,
    _gain_db_grid,
    _phase_deg,
    _phase_deg_grid,
    _polarisation,
    _polarisation_grid,
)
from momwire.eznec import _serve
from momwire.eznec._printout import (
    ChargeRow,
    PatternBlock,
    PatternRow,
    RunData,
    WireCurrentRow,
    _charges,
    _currents,
    _pattern,
)


def _same_bits(got, want) -> bool:
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    return got.shape == want.shape and np.array_equal(
        got.view(np.int64), want.view(np.int64)
    )


def _adversarial_components(rng) -> np.ndarray:
    """Field components that reach every branch: signed exact zeros, dust on
    either side of the printout's floor, the NEC blank floor, phases on the
    ±180 wrap, and ordinary values over many decades."""
    special = [
        0j,
        complex(-0.0, 0.0),
        complex(0.0, -0.0),
        complex(-0.0, -0.0),
        1e-8 + 0j,
        complex(-1e-8, -0.0),
        complex(0.0, 3e-8),
        1e-11 + 1e-11j,
        1.0 + 0j,
        -1.0 + 0j,
        complex(-1.0, -0.0),
        complex(-1.0, 1e-300),
        1j,
        -1j,
        3.5 - 2.25j,
    ]
    mags = 10.0 ** rng.uniform(-12, 3, 4000)
    phases = rng.uniform(-math.pi, math.pi, 4000)
    random = mags * np.exp(1j * phases)
    return np.concatenate([np.array(special, dtype=complex), random])


def test_degrees_and_radians_are_one_multiplication():
    """The grid forms multiply by these constants where the scalar ones call
    ``math.degrees`` / ``math.radians``; CPython defines those functions as
    exactly that product, and this pins it on the interpreter in hand."""
    rng = np.random.default_rng(1448)
    for x in np.concatenate([rng.uniform(-10, 10, 2000), [0.0, -0.0, math.pi]]):
        x = float(x)
        assert math.degrees(x).hex() == (x * _RAD_TO_DEG).hex()
        assert math.radians(x).hex() == (x * _DEG_TO_RAD).hex()


def test_the_phase_grid_is_the_scalar_phase_sign_of_zero_included():
    values = _adversarial_components(np.random.default_rng(1))
    want = [_phase_deg(complex(v)) for v in values]
    assert _same_bits(_phase_deg_grid(values), want)


def test_the_gain_grid_is_the_scalar_gain_floor_included():
    rng = np.random.default_rng(2)
    gains = np.concatenate(
        [
            [0.0, 1e-21, 1e-20, np.nextafter(1e-20, 0.0), 1.0],
            10.0 ** rng.uniform(-25, 2, 3000),
        ]
    )
    assert _same_bits(_gain_db_grid(gains), [_gain_db(float(g)) for g in gains])


@pytest.mark.parametrize("floor_scale", [1.0, 1.0 / 7.3, 1000.0 / 41.07])
@pytest.mark.parametrize("linear_below", [5e-5, 5e-6])
def test_the_polarisation_grid_is_the_scalar_ellipse(floor_scale, linear_below):
    rng = np.random.default_rng(3)
    et = _adversarial_components(rng)
    ep = np.concatenate(
        [et[::-1][: len(et) // 2], rng.permutation(et)[: len(et) - len(et) // 2]]
    )
    a = np.array([abs(complex(v)) for v in et])
    b = np.array([abs(complex(v)) for v in ep])
    axial, tilt, sense = _polarisation_grid(
        a, b, _phase_deg_grid(et), _phase_deg_grid(ep), floor_scale, linear_below
    )
    want = [
        _polarisation(complex(t), complex(p), floor_scale, linear_below)
        for t, p in zip(et, ep, strict=True)
    ]
    assert _same_bits(axial, [w[0] for w in want])
    assert _same_bits(tilt, [w[1] for w in want])
    assert sense.tolist() == [w[2] for w in want]


def _scalar_rows(
    thetas, phis, e_theta, e_phi, g_v, g_h, below, floor_scale, range_m, shift
):
    """The per-point loop exactly as ``_serve._pattern`` carried it before
    momwire#1448, kept as the oracle the grid form is held to."""
    rows = []
    for j in range(len(phis)):
        for i in range(len(thetas)):
            if below is not None and below[i]:
                rows.append(
                    PatternRow(
                        float(thetas[i]), float(phis[j]),
                        _gain_db(0.0), _gain_db(0.0), _gain_db(0.0),
                        0.0, 0.0, "", 0.0, 0.0, 0.0, 0.0,
                    )
                )  # fmt: skip
                continue
            et, ep = complex(e_theta[i, j]), complex(e_phi[i, j])
            if et == 0:
                et = 0j
            if ep == 0:
                ep = 0j
            if et.real * et.real + et.imag * et.imag <= _serve._PRINTED_DUST_FLOOR2:
                et = 0j
            if ep.real * ep.real + ep.imag * ep.imag <= _serve._PRINTED_DUST_FLOOR2:
                ep = 0j
            axial, tilt, sense = _polarisation(
                et, ep, floor_scale, linear_below=_serve._AXIAL_PRINTS_ZERO
            )
            et_mag, ep_mag = abs(et), abs(ep)
            et_deg = math.degrees(math.atan2(et.imag, et.real))
            ep_deg = math.degrees(math.atan2(ep.imag, ep.real))
            if shift is not None:
                et_mag, ep_mag = et_mag / range_m, ep_mag / range_m
                et_deg += shift
                ep_deg += shift
            rows.append(
                PatternRow(
                    float(thetas[i]), float(phis[j]),
                    _gain_db(float(g_v[i, j])), _gain_db(float(g_h[i, j])),
                    _gain_db(float(g_v[i, j] + g_h[i, j])),
                    axial, tilt, sense, et_mag, et_deg, ep_mag, ep_deg,
                )
            )  # fmt: skip
    return tuple(rows)


def _grid(n_theta=19, n_phi=7, seed=0):
    rng = np.random.default_rng(seed)
    thetas = 0.0 + 5.0 * np.arange(n_theta)
    phis = 0.0 + 60.0 * np.arange(n_phi)
    shape = (n_theta, n_phi)
    mags = 10.0 ** rng.uniform(-10, 2, shape)
    e_theta = mags * np.exp(1j * rng.uniform(-math.pi, math.pi, shape))
    e_phi = 10.0 ** rng.uniform(-10, 2, shape) * np.exp(
        1j * rng.uniform(-math.pi, math.pi, shape)
    )
    # The cells each rule is about.
    e_theta[0, 0] = complex(-0.0, -0.0)  # exact zero carrying its signs
    e_phi[0, 0] = complex(0.0, -0.0)
    e_theta[1, 0] = 2e-8 + 1e-8j  # dust (|E|^2 under 1e-14) ...
    e_phi[1, 0] = 2.0 * np.exp(1j * math.radians(120.0))  # ... cos(delta) < 0
    e_theta[1, 1] = complex(-3e-8, 0.0)
    e_phi[1, 1] = 2.0 * np.exp(1j * math.radians(60.0))  # cos(delta) > 0
    e_theta[2, 0] = np.exp(1j * math.radians(-170.0))  # for the unwrapped shift
    g_v = np.abs(e_theta) ** 2 / 100.0
    g_h = np.abs(e_phi) ** 2 / 100.0
    g_v[3, 0] = 0.0
    return thetas, phis, e_theta, e_phi, g_v, g_h


def _rows_equal(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        for name in PatternRow._fields:
            gv, wv = getattr(g, name), getattr(w, name)
            if isinstance(wv, str):
                assert gv == wv, name
            else:
                assert type(gv) is float, name
                assert gv.hex() == wv.hex(), (name, g, w)


@pytest.mark.parametrize("range_m", [0.0, 1000.0])
@pytest.mark.parametrize("over_ground", [False, True])
def test_the_grid_rows_are_the_scalar_loops_rows(range_m, over_ground):
    thetas, phis, e_theta, e_phi, g_v, g_h = _grid(n_theta=25)
    below = (thetas > 90.0 + 1e-9) if over_ground else None
    if below is not None:
        # the caller zeroes the gains of rows it does not compute
        g_v = np.where(below[:, None], 0.0, g_v)
        g_h = np.where(below[:, None], 0.0, g_h)
    floor_scale = 1.0 / 41.07
    shift = -164.02 if range_m else None
    got = _serve._pattern_rows(
        thetas, phis, e_theta, e_phi, g_v, g_h, below, floor_scale, range_m, shift
    )
    want = _scalar_rows(
        thetas, phis, e_theta, e_phi, g_v, g_h, below, floor_scale, range_m, shift
    )
    _rows_equal(got, want)


def _printed(rows):
    lines = _pattern(PatternBlock(rows=rows))
    return {
        (r.theta_deg, r.phi_deg): line for r, line in zip(rows, lines[-len(rows) :])
    }


def test_an_exact_zero_component_prints_phase_plus_zero():
    thetas, phis, e_theta, e_phi, g_v, g_h = _grid()
    rows = _serve._pattern_rows(
        thetas, phis, e_theta, e_phi, g_v, g_h, None, 1.0, 0.0, None
    )
    row = rows[0]  # theta 0, phi 0: both components are signed exact zeros
    assert (row.theta_deg, row.phi_deg) == (0.0, 0.0)
    assert math.copysign(1.0, row.e_theta_phase_deg) == 1.0
    assert math.copysign(1.0, row.e_phi_phase_deg) == 1.0
    line = _printed(rows)[(0.0, 0.0)]
    assert "-0.00" not in line
    assert line.rstrip().endswith("0.00000E+00     0.00")


def test_a_dust_component_tilts_by_the_sign_of_cos_delta():
    """E(THETA) under the dust floor is zeroed, and the row's tilt is then
    ``½·atan2(±0, -b²)``: -90 when ``cos δ < 0`` and +90 when it is > 0, the
    signed zero coming through the grid's products intact."""
    thetas, phis, e_theta, e_phi, g_v, g_h = _grid()
    rows = _serve._pattern_rows(
        thetas, phis, e_theta, e_phi, g_v, g_h, None, 1.0, 0.0, None
    )
    by_angle = {(r.theta_deg, r.phi_deg): r for r in rows}
    negative, positive = by_angle[(5.0, 0.0)], by_angle[(5.0, 60.0)]
    assert negative.e_theta_magnitude == 0.0 and positive.e_theta_magnitude == 0.0
    assert negative.tilt_deg == -90.0
    assert positive.tilt_deg == 90.0
    assert f"{negative.tilt_deg:9.2f}" == "   -90.00"


def test_below_the_horizon_over_ground_a_row_is_all_zeros():
    thetas, phis, e_theta, e_phi, g_v, g_h = _grid(n_theta=25)
    below = thetas > 90.0 + 1e-9
    assert below.any()
    g_v = np.where(below[:, None], 0.0, g_v)
    g_h = np.where(below[:, None], 0.0, g_h)
    rows = _serve._pattern_rows(
        thetas, phis, e_theta, e_phi, g_v, g_h, below, 1.0, 1000.0, -164.02
    )
    dropped = [r for r in rows if r.theta_deg > 90.0]
    assert len(dropped) == below.sum() * len(phis)
    for r in dropped:
        assert (r.vert_db, r.hor_db, r.total_db) == (-999.99, -999.99, -999.99)
        assert r.sense == ""
        values = (r.axial_ratio, r.tilt_deg, r.e_theta_magnitude, r.e_theta_phase_deg)
        values += (r.e_phi_magnitude, r.e_phi_phase_deg)
        # zeros, and no range phase added to them
        assert all(v.hex() == (0.0).hex() for v in values)


def test_the_range_phase_is_added_without_rewrapping():
    """RFLD > 0: -170 + -164.02 prints -334.02, not its wrap at +25.98."""
    thetas, phis, e_theta, e_phi, g_v, g_h = _grid()
    rows = _serve._pattern_rows(
        thetas, phis, e_theta, e_phi, g_v, g_h, None, 1.0 / 1000.0, 1000.0, -164.02
    )
    row = {(r.theta_deg, r.phi_deg): r for r in rows}[(10.0, 0.0)]
    assert f"{row.e_theta_phase_deg:9.2f}" == "  -334.02"
    assert row.e_theta_magnitude == pytest.approx(1.0 / 1000.0, rel=1e-15)
    # a zeroed component prints the bare shift
    zero = rows[0]
    assert f"{zero.e_theta_phase_deg:9.2f}" == "  -164.02"


def _old_row_line(row) -> str:
    """The renderer's per-row f-string as it stood before momwire#1448."""
    return (
        f"{row.theta_deg:8.2f}{row.phi_deg:9.2f}"
        f"{row.vert_db:11.2f}{row.hor_db:8.2f}{row.total_db:8.2f}"
        f"{row.axial_ratio:11.5f}{row.tilt_deg:9.2f}"
        + "  "
        + row.sense.ljust(6)
        + f"{row.e_theta_magnitude:15.5E}"
        + f"{row.e_theta_phase_deg:9.2f}"
        + f"{row.e_phi_magnitude:15.5E}"
        + f"{row.e_phi_phase_deg:9.2f}"
    )


def test_the_row_template_is_the_old_fstring_byte_for_byte():
    """The renderer formats a row through one ``%`` template now; every cell
    must be the f-string cell it replaced, the awkward values included."""
    rng = np.random.default_rng(5)
    awkward = [0.0, -0.0, -0.004, 0.005, -999.99, 1e-300, 9.999995, 12345.678]
    awkward += [-1e5, math.inf, -math.inf, math.nan]
    rows = [
        PatternRow(*([value] * 7), sense, *([value] * 4))
        for sense in ("", "LINEAR", "LEFT", "RIGHT")
        for value in awkward
    ]
    for _ in range(500):
        v = rng.normal(size=11) * 10.0 ** rng.integers(-12, 4, size=11)
        rows.append(PatternRow(*v[:7].tolist(), "LEFT", *v[7:].tolist()))
    lines = _pattern(PatternBlock(rows=tuple(rows)))
    assert lines[-len(rows) :] == [_old_row_line(r) for r in rows]


def test_the_current_and_charge_templates_are_the_old_fstrings():
    """Same move for the Wire Currents / Wire Charge Densities rows."""

    def old_current(row):
        cells = (*row.center, row.length, row.real, row.imag, row.magnitude)
        return (
            f"{row.element:7d}{row.tag:5d}"
            + "".join(f"{value:13.5E}" for value in cells)
            + f"{row.phase_deg:10.4f}"
        )

    def old_charge(row):
        cells = (*row.center, row.length, row.magnitude)
        return (
            f"{row.element:7d}{row.tag:5d}"
            + "".join(f"{value:13.5E}" for value in cells)
            + f"{row.phase_deg:10.4f}"
        )

    rng = np.random.default_rng(6)
    awkward = [0.0, -0.0, 1e-300, -5e-5, 99999.99995, math.inf, math.nan]
    currents, charges = [], []
    for k in range(300):
        v = rng.normal(size=8) * 10.0 ** rng.integers(-12, 4, size=8)
        if k < len(awkward):
            v[:] = awkward[k]
        v = v.tolist()
        center = (v[0], v[1], v[2])
        currents.append(WireCurrentRow(k + 1, k % 7 + 1, center, *v[3:]))
        charges.append(ChargeRow(k + 1, k % 7 + 1, center, v[3], v[4], v[5]))
    data = RunData(currents=tuple(currents), charges=tuple(charges))
    assert _currents(data)[-len(currents) :] == [old_current(r) for r in currents]
    assert _charges(data)[-len(charges) :] == [old_charge(r) for r in charges]


def test_the_gains_take_abs_from_libm_hypot():
    """`pattern_fields`' gains are ``norm·|M|²`` with ``|M|`` from libm's
    hypot (CPython's ``abs(complex)``), not numpy's complex absolute, whose
    AVX-512 and AVX2 loops disagree in the last bit: a gain printed on one
    machine is the gain printed on another, and a C++ replay of the readout
    (momwire#1479) has one function to call."""
    from momwire._far_readout import ETA0, Ground, _far_moments, pattern_fields

    rng = np.random.default_rng(1479)
    mid = rng.standard_normal((40, 3))
    moment = rng.standard_normal((40, 3)) + 1j * rng.standard_normal((40, 3))
    thetas = np.linspace(0.0, 180.0, 37)
    phis = np.linspace(0.0, 360.0, 25)
    k, p_in = 2.0 * math.pi / 10.0, 0.37
    _e_t, _e_p, g_v, g_h = pattern_fields(
        mid, moment, k, thetas, phis, Ground("free"), 0.0, 30e6, p_in
    )
    m_theta, m_phi = _far_moments(
        mid, moment, k, np.radians(thetas), np.radians(phis), Ground("free"), 0.0, 30e6
    )
    norm = ETA0 * k * k / (8.0 * math.pi * p_in)
    want_v = (
        norm
        * np.array([abs(v) for v in m_theta.ravel().tolist()]).reshape(m_theta.shape)
        ** 2
    )
    want_h = (
        norm
        * np.array([abs(v) for v in m_phi.ravel().tolist()]).reshape(m_phi.shape) ** 2
    )
    assert _same_bits(g_v, want_v)
    assert _same_bits(g_h, want_h)
