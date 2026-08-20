"""The type-free network core: TL/chain math and the MNA system (momwire#456).

Direct math and MNA-level only — no antennas, no `Network` spec, no reducer.
Every system here is hand-built and every expectation is a closed form, so a
failure names the stamp that broke rather than a design that moved.

The moved code arrived from antennaknobs, whose own MNA tests all drive
through `NetworkReducer`; none were portable as-is, so this battery is written
against the primitives directly.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from momwire.networks import (
    RCOND_SINGULAR,
    C_LIGHT,
    MNASystem,
    SingularNetworkError,
    balanced_admittance_4x4,
    magnetizing_impedance,
    poison_singular_sample,
    tl_abcd,
    tl_admittance_2x2,
)
from momwire.networks._reduce import _Group2Element, _stamp_abcd

FREQ_MHZ = 28.0
WL = C_LIGHT / (FREQ_MHZ * 1e6)
OMEGA = 2.0 * np.pi * FREQ_MHZ * 1e6

# The matched-loss model, recomputed independently of the module under test so
# the loss assertions are not tautologies: k1·√f_MHz + k2·f_MHz dB per 100 ft.
_NEPER_PER_DB = math.log(10.0) / 20.0
_FEET_PER_M = 1.0 / 0.3048


def _alpha(k1, k2, wavelength=WL):
    """Attenuation in nepers/m for the cable-table model."""
    f_mhz = C_LIGHT / wavelength / 1e6
    return (k1 * math.sqrt(f_mhz) + k2 * f_mhz) * _NEPER_PER_DB * _FEET_PER_M / 100.0


def _abcd_to_y(abcd):
    """Standard chain → short-circuit admittance conversion."""
    a, b, c, d = abcd
    return np.array([[d, -(a * d - b * c)], [-1.0 + 0j, a]], dtype=np.complex128) / b


# ---------------------------------------------------------------------------
# tl_abcd — the chain matrix the reducer stamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "z0,length,vf,k1,k2",
    [
        (50.0, 0.37 * WL, 1.0, 0.0, 0.0),
        (75.0, 1.83 * WL, 0.66, 0.0, 0.0),
        (450.0, 0.11 * WL, 0.91, 0.2, 0.008),
        (50.0, 2.4 * WL, 0.82, 1.1, 0.05),
    ],
)
def test_tl_abcd_is_reciprocal(z0, length, vf, k1, k2):
    """AD − BC = 1 exactly: cosh² − sinh² for any γl, lossy or not.

    Reciprocity is the one invariant that survives every parameter, so it is
    the cheapest guard against an entry being dropped or misplaced.
    """
    a, b, c, d = tl_abcd(z0, length, WL, vf=vf, k1=k1, k2=k2)
    assert a * d - b * c == pytest.approx(1.0 + 0j, abs=1e-9)


def test_lossless_tl_abcd_has_textbook_cos_jsin_structure():
    """γl = jθ ⇒ [[cos θ, j·z0·sin θ], [j·sin θ/z0, cos θ]]."""
    z0, vf = 50.0, 0.66
    length = 0.3 * WL
    theta = 2.0 * np.pi * length / (vf * WL)
    a, b, c, d = tl_abcd(z0, length, WL, vf=vf)

    assert a == pytest.approx(np.cos(theta))
    assert d == pytest.approx(np.cos(theta))
    assert b == pytest.approx(1j * z0 * np.sin(theta))
    assert c == pytest.approx(1j * np.sin(theta) / z0)
    # The diagonal is real and the off-diagonal purely imaginary — a lossless
    # line neither dissipates nor phase-shifts its own reciprocity.
    assert a.imag == pytest.approx(0.0, abs=1e-12)
    assert b.real == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("halves,sign", [(1, -1.0), (2, 1.0), (3, -1.0), (4, 1.0)])
def test_lossless_half_wave_line_is_signed_identity(halves, sign):
    """k·λ/2 ⇒ [[±1, 0], [0, ±1]] — the through-line that has no admittance
    matrix at all, which is the whole reason the reducer stamps chains."""
    z0, vf = 50.0, 0.8
    length = halves * vf * WL / 2.0
    a, b, c, d = tl_abcd(z0, length, WL, vf=vf)

    assert a == pytest.approx(sign, abs=1e-9)
    assert d == pytest.approx(sign, abs=1e-9)
    assert abs(b) == pytest.approx(0.0, abs=1e-6)
    assert abs(c) == pytest.approx(0.0, abs=1e-6)
    # ...and the admittance form genuinely cannot spell it.
    with pytest.raises(SingularNetworkError, match="no admittance matrix"):
        tl_admittance_2x2(z0, length, WL, vf=vf)


@pytest.mark.parametrize("z_load", [12.5, 200.0, 50.0, 30.0 - 45.0j])
def test_lossless_quarter_wave_inverts_impedance(z_load):
    """z_in = z0²/z_load through the ABCD terms — the classic λ/4 transformer.

    z_in = (A·z_L + B)/(C·z_L + D), and at θ = π/2 the diagonal vanishes.
    """
    z0, vf = 50.0, 0.66
    length = vf * WL / 4.0
    a, b, c, d = tl_abcd(z0, length, WL, vf=vf)
    z_in = (a * z_load + b) / (c * z_load + d)

    assert z_in == pytest.approx(z0**2 / z_load, rel=1e-9)


def test_matched_line_transfer_is_exactly_exp_gamma_l():
    """A + B/z0 = e^{γl}, so |A + B/z0| = e^{αl}: attenuation, exactly.

    Feeding the line its own z0 makes the chain matrix collapse to the raw
    propagation factor (v_a = (A + B/z0)·v_b), which pins the loss model to a
    closed form computed here from the cable-table coefficients rather than
    from the module under test.
    """
    z0, vf, k1, k2 = 50.0, 0.82, 0.9, 0.02
    for length in (0.2 * WL, 1.0 * WL, 3.7 * WL):
        a, b, _, _ = tl_abcd(z0, length, WL, vf=vf, k1=k1, k2=k2)
        assert abs(a + b / z0) == pytest.approx(
            math.exp(_alpha(k1, k2) * length), rel=1e-9
        )


def test_loss_grows_with_length_and_attenuation_and_is_positive():
    """Attenuation is positive, monotone in length, and monotone in k1/k2.

    A lossless line moves the same power out that it took in; every lossy one
    must move strictly less, by strictly more as either knob rises.
    """
    z0, vf = 50.0, 1.0

    def gain(length, k1, k2):
        a, b, _, _ = tl_abcd(z0, length, WL, vf=vf, k1=k1, k2=k2)
        return abs(a + b / z0)

    # Lossless is exactly transparent.
    assert gain(1.3 * WL, 0.0, 0.0) == pytest.approx(1.0, rel=1e-12)

    # Positive attenuation, monotone in length.
    lengths = [0.5 * WL, 1.0 * WL, 2.0 * WL, 4.0 * WL]
    gains = [gain(x, 0.5, 0.01) for x in lengths]
    assert all(g > 1.0 for g in gains)
    assert gains == sorted(gains)
    assert all(hi > lo * 1.001 for lo, hi in zip(gains, gains[1:]))

    # Monotone in each loss coefficient independently.
    assert gain(2.0 * WL, 1.0, 0.0) > gain(2.0 * WL, 0.5, 0.0) > 1.0
    assert gain(2.0 * WL, 0.0, 0.02) > gain(2.0 * WL, 0.0, 0.01) > 1.0

    # A lossy open stub presents a resistive part; a lossless one is reactive.
    a_l, _, c_l, _ = tl_abcd(z0, 0.3 * WL, WL, k1=0.0, k2=0.0)
    a_y, _, c_y, _ = tl_abcd(z0, 0.3 * WL, WL, k1=0.8, k2=0.02)
    assert (a_l / c_l).real == pytest.approx(0.0, abs=1e-9)
    assert (a_y / c_y).real > 0.0


# ---------------------------------------------------------------------------
# tl_admittance_2x2 — the closed form the oracles are written against
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "z0,length,vf,k1,k2",
    [
        (50.0, 0.37 * WL, 1.0, 0.0, 0.0),
        (75.0, 0.83 * WL, 0.66, 0.0, 0.0),
        (450.0, 1.11 * WL, 0.91, 0.3, 0.01),
    ],
)
def test_tl_admittance_matches_the_chain_matrix(z0, length, vf, k1, k2):
    """The same line, two descriptions: converting one must give the other."""
    abcd = tl_abcd(z0, length, WL, vf=vf, k1=k1, k2=k2)
    y_direct = tl_admittance_2x2(z0, length, WL, vf=vf, k1=k1, k2=k2)

    np.testing.assert_allclose(y_direct, _abcd_to_y(abcd), rtol=1e-9, atol=1e-12)
    # Reciprocal and symmetric, as any passive 2-port must be.
    np.testing.assert_allclose(y_direct, y_direct.T, rtol=1e-12, atol=1e-14)


def test_transposed_flips_the_off_diagonal_only():
    """A half-twist is a polarity inversion on port B: the transfer terms
    change sign, the self terms emphatically do not (that would be a
    negative z0, which is a different and wrong element)."""
    args = (50.0, 0.31 * WL, WL)
    y = tl_admittance_2x2(*args, vf=0.66)
    y_t = tl_admittance_2x2(*args, transposed=True, vf=0.66)

    assert y_t[0, 0] == y[0, 0]
    assert y_t[1, 1] == y[1, 1]
    assert y_t[0, 1] == -y[0, 1]
    assert y_t[1, 0] == -y[1, 0]
    # Guard the specific confusion the docstring names: negating z0 would
    # negate the diagonal too, so the two must not coincide.
    y_negz0 = tl_admittance_2x2(-50.0, 0.31 * WL, WL, vf=0.66)
    assert not np.allclose(y_t, y_negz0)


# ---------------------------------------------------------------------------
# balanced_admittance_4x4 — the even/odd decomposition
# ---------------------------------------------------------------------------

_DIFF_VECTORS = (
    np.array([1.0, -1.0, 0.0, 0.0], dtype=np.complex128),
    np.array([0.0, 0.0, 1.0, -1.0], dtype=np.complex128),
    np.array([0.7, -0.7, -1.9j, 1.9j], dtype=np.complex128),
)


def test_balanced_without_common_mode_is_rank_two():
    """zcomm=None: the common mode is structurally open, so I(a1) = −I(a2) is
    forced at each end by wiring and the 4×4 carries only two modes."""
    y4 = balanced_admittance_4x4(100.0, 0.29 * WL, WL, vf=0.82)

    assert y4.shape == (4, 4)
    assert np.linalg.matrix_rank(y4, tol=1e-9) == 2
    # Every column current sums to zero: nothing returns through the common.
    np.testing.assert_allclose(y4.sum(axis=0), 0.0, atol=1e-9)


def test_common_mode_raises_the_rank_to_four():
    y4 = balanced_admittance_4x4(100.0, 0.29 * WL, WL, vf=0.82)
    y4c = balanced_admittance_4x4(100.0, 0.29 * WL, WL, vf=0.82, zcomm=300.0)

    assert np.linalg.matrix_rank(y4, tol=1e-9) == 2
    assert np.linalg.matrix_rank(y4c, tol=1e-9) == 4


def test_common_mode_leaves_the_differential_block_untouched():
    """The ¼-ones incidence annihilates differential vectors, so adding a
    common-mode path cannot perturb differential behaviour — the exact
    even/odd decomposition, and the reason the two blocks superpose."""
    args = (100.0, 0.29 * WL, WL)
    y4 = balanced_admittance_4x4(*args, vf=0.82)
    y4c = balanced_admittance_4x4(*args, vf=0.82, zcomm=300.0)

    for v in _DIFF_VECTORS:
        np.testing.assert_allclose(y4c @ v, y4 @ v, rtol=1e-9, atol=1e-12)

    # The perturbation is entirely orthogonal to the differential subspace.
    delta = y4c - y4
    for v in _DIFF_VECTORS:
        np.testing.assert_allclose(delta @ v, 0.0, atol=1e-12)


def test_grounding_conductor_two_collapses_to_the_2x2():
    """Dropping rows/cols a2, b2 from the zcomm=None stamp returns the
    ordinary coax 2×2 exactly — the docstring's own consistency claim."""
    args = (100.0, 0.29 * WL, WL)
    y4 = balanced_admittance_4x4(*args, vf=0.82)
    y2 = tl_admittance_2x2(100.0, 0.29 * WL, WL, vf=0.82)

    np.testing.assert_allclose(y4[np.ix_([0, 2], [0, 2])], y2, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# MNASystem — hand-built systems with closed-form answers
# ---------------------------------------------------------------------------

E_SRC = 2.0 + 0j
R_LOAD = 73.0
Z_SRC = 50.0


def _driven_resistor(r_load=R_LOAD, z_src=Z_SRC, emf=E_SRC, z_ref_at=None):
    """One node: an EMF behind z_src driving a conductance 1/r_load.

    The whole system is a voltage divider, which is the smallest thing that
    exercises a Group-2 source row, a Group-1 admittance row and the
    termination bookkeeping at once.
    """
    g = np.array([[1.0 / r_load]], dtype=np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=z_src, e=emf)
    return MNASystem(
        g,
        [src],
        {0: (0, emf, "v", 0j)},
        probes=[("src", "termination", 0)],
        z_ref_at=z_ref_at,
    )


def test_driven_source_and_series_element_obey_ohms_law():
    v, j = _driven_resistor().solve()

    assert j[0] == pytest.approx(E_SRC / (Z_SRC + R_LOAD))
    assert v[0] == pytest.approx(E_SRC * R_LOAD / (Z_SRC + R_LOAD))
    # KCL closes: the branch current is exactly what the load draws.
    assert j[0] == pytest.approx(v[0] / R_LOAD)


def test_driven_impedance_readout_subtracts_the_reference():
    """Kind "v": Z = emf/j − z_ref_at[k] (issue #746). Stamping the EMF
    behind a reference rather than pinning the node is what lets a short
    across the port be an answer instead of an unsolvable system."""
    sys_ = _driven_resistor(z_ref_at={0: Z_SRC})
    _, j = sys_.solve()

    z_seen = E_SRC / j[0] - sys_.z_ref_at[0]
    assert z_seen == pytest.approx(R_LOAD + 0j)


def test_hard_short_across_the_port_is_solvable_and_reads_zero():
    """The case an IDEAL generator cannot answer (issue #746). A Group-2 short
    pins the port node to the datum; because the EMF sits behind a reference
    impedance rather than pinning the node itself, the system stays finite and
    Z = emf/j − z_ref comes out as exactly 0 instead of raising.
    """
    g = np.zeros((1, 1), dtype=np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    short = _Group2Element(0, None, c_v=1.0 + 0j, c_j=0j, e=0j)
    v, j = MNASystem(
        g, [src, short], {0: (0, E_SRC, "v", 0j)}, z_ref_at={0: Z_SRC}
    ).solve()

    assert v[0] == pytest.approx(0.0, abs=1e-12)
    assert j[0] == pytest.approx(E_SRC / Z_SRC)
    assert j[1] == pytest.approx(j[0])  # the short carries the whole current
    assert E_SRC / j[0] - Z_SRC == pytest.approx(0.0 + 0j, abs=1e-12)


def test_solve_is_cached():
    sys_ = _driven_resistor()
    assert sys_.solve() is sys_.solve()


def test_group2_ideal_short_identifies_two_nodes():
    """z = 0 in the impedance form: (v_b − v_a) + 0·j = 0 pins the two nodes
    to the same potential exactly, with no value ever inverted."""
    g = np.diag([1.0 / 50.0, 1.0 / 200.0]).astype(np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    short = _Group2Element(0, 1, c_v=1.0 + 0j, c_j=0j, e=0j)
    v, j = MNASystem(g, [src, short], {0: (0, E_SRC, "v", 0j)}).solve()

    assert v[0] == pytest.approx(v[1])
    # And the identified pair draws the parallel combination.
    z_par = 1.0 / (1.0 / 50.0 + 1.0 / 200.0)
    assert j[0] == pytest.approx(E_SRC / (Z_SRC + z_par))


def test_open_branch_contributes_nothing():
    """y = 0 in the admittance form: 0·(v_b − v_a) + 1·j = 0 forces j = 0, so
    the branch's KCL stamps vanish and the rest of the system is untouched."""
    bare = _driven_resistor()
    g = np.array([[1.0 / R_LOAD]], dtype=np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    dangling = _Group2Element(0, None, c_v=0j, c_j=1.0 + 0j, e=0j)
    v_open, j_open = MNASystem(g, [src, dangling], {0: (0, E_SRC, "v", 0j)}).solve()
    v_bare, j_bare = bare.solve()

    assert j_open[1] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(v_open, v_bare, rtol=1e-12, atol=1e-14)
    assert j_open[0] == pytest.approx(j_bare[0])


def test_termination_probe_reports_the_source_chain_dissipation():
    """½·Re(z)·|j|² in the source's own series impedance — the drop across
    the Load part of the branch, with the modelling reference removed."""
    sys_ = _driven_resistor()
    j = sys_.solve()[1][0]

    assert sys_.branch_power(("src", "termination", 0)) == pytest.approx(
        0.5 * Z_SRC * abs(j) ** 2
    )


def test_group2_branch_power_is_the_drop_times_the_current():
    g = np.diag([1.0 / 50.0, 1.0 / 200.0]).astype(np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    series = _Group2Element(0, 1, c_v=1.0 + 0j, c_j=25.0 + 0j, e=0j)
    sys_ = MNASystem(g, [src, series], {0: (0, E_SRC, "v", 0j)})
    v, j = sys_.solve()

    assert sys_.branch_power(("r", "group2", 1)) == pytest.approx(
        0.5 * float(np.real((v[0] - v[1]) * np.conj(j[1])))
    )
    # An explicit resistance dissipates exactly ½R|j|², and positively.
    assert sys_.branch_power(("r", "group2", 1)) == pytest.approx(
        0.5 * 25.0 * abs(j[1]) ** 2
    )
    assert sys_.branch_power(("r", "group2", 1)) > 0.0


def test_lossless_chain_stamped_line_conserves_power():
    """The chain-matrix stamp carries power in at A and out at B; a lossless
    line reports ~0 net, which is the invariant `_stamp_abcd`'s two coupled
    constitutive rows exist to preserve."""
    elements, couplings = [], []
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    elements.append(src)
    leg = _stamp_abcd(
        elements,
        couplings,
        [(0, 1.0 + 0j)],
        [(1, 1.0 + 0j)],
        tl_abcd(50.0, 0.31 * WL, WL, vf=0.66),
    )
    g = np.diag([0.0 + 0j, 1.0 / R_LOAD]).astype(np.complex128)
    sys_ = MNASystem(g, elements, {0: (0, E_SRC, "v", 0j)}, couplings=couplings)

    p_line = sys_.branch_power(("line", "group2abcd", (leg,)))
    assert p_line == pytest.approx(0.0, abs=1e-9)

    # ...and the line actually transformed something: a λ/4-ish section of
    # 50 Ω feeding 73 Ω cannot leave the input impedance at 73 Ω.
    _, j = sys_.solve()
    z_in = E_SRC / j[0] - Z_SRC
    assert abs(z_in - R_LOAD) > 1.0

    # A lossy line reports strictly positive dissipation through the same leg.
    elements_l, couplings_l = [], []
    elements_l.append(_Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC))
    leg_l = _stamp_abcd(
        elements_l,
        couplings_l,
        [(0, 1.0 + 0j)],
        [(1, 1.0 + 0j)],
        tl_abcd(50.0, 0.31 * WL, WL, vf=0.66, k1=1.5, k2=0.05),
    )
    sys_l = MNASystem(g, elements_l, {0: (0, E_SRC, "v", 0j)}, couplings=couplings_l)
    assert sys_l.branch_power(("line", "group2abcd", (leg_l,))) > 1e-6


def test_crossed_line_is_a_negated_port_weight():
    """A half-twist is not a flag on the chain matrix: it is port B's weights
    negated where the pair is stamped, which inverts the far-end voltage."""
    out = {}
    for name, weight in (("plain", 1.0 + 0j), ("crossed", -1.0 + 0j)):
        elements, couplings = [], []
        elements.append(_Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC))
        _stamp_abcd(
            elements,
            couplings,
            [(0, 1.0 + 0j)],
            [(1, weight)],
            tl_abcd(50.0, 0.17 * WL, WL, vf=0.66),
        )
        g = np.diag([0.0 + 0j, 1.0 / R_LOAD]).astype(np.complex128)
        out[name] = MNASystem(
            g, elements, {0: (0, E_SRC, "v", 0j)}, couplings=couplings
        ).solve()

    # Same input current magnitude, opposite far-end polarity.
    assert out["crossed"][1][0] == pytest.approx(out["plain"][1][0])
    assert out["crossed"][0][1] == pytest.approx(-out["plain"][0][1])


# ---------------------------------------------------------------------------
# Singularity policy
# ---------------------------------------------------------------------------


def test_floating_node_raises_with_rcond_diagnostics():
    """Genuine rank deficiency: node 1 is reachable only through an open, so
    nothing pins its potential. The message must carry the equilibrated
    reciprocal condition — that number is the whole reason `zgesvx` is used
    instead of a bare solve, which would answer with confident garbage.
    """
    g = np.diag([1.0 / R_LOAD, 0.0]).astype(np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    open_br = _Group2Element(0, 1, c_v=0j, c_j=1.0 + 0j, e=0j)
    sys_ = MNASystem(g, [src, open_br], {0: (0, E_SRC, "v", 0j)})

    with pytest.raises(SingularNetworkError) as exc:
        sys_.solve()

    msg = str(exc.value)
    assert "no finite solution" in msg
    assert "reciprocal condition" in msg
    assert "after equilibration" in msg
    # The diagnostic number is present and actually below the policy floor.
    rcond = float(msg.split("reciprocal condition")[1].split("after")[0].strip(" ()"))
    assert rcond < RCOND_SINGULAR
    # A ValueError subclass, so guards predating the type keep their contract.
    assert isinstance(exc.value, ValueError)


def test_diagnose_closure_is_appended_to_the_message():
    """The walk that names the offending branch runs only on the failing
    path — a closure, not stored state."""
    calls = []

    def diagnose():
        calls.append(1)
        return "branch 'feedline' dangles at node 1"

    g = np.diag([1.0 / R_LOAD, 0.0]).astype(np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    open_br = _Group2Element(0, 1, c_v=0j, c_j=1.0 + 0j, e=0j)

    with pytest.raises(SingularNetworkError, match="dangles at node 1"):
        MNASystem(
            g, [src, open_br], {0: (0, E_SRC, "v", 0j)}, diagnose=diagnose
        ).solve()
    assert calls == [1]

    # The happy path never pays for it.
    calls.clear()
    _driven_resistor().solve()
    assert calls == []


def test_healthy_system_does_not_warn(caplog):
    """A well-scaled MNA matrix stays quiet; the near-singular warning is
    reserved for rcond between the two thresholds."""
    with caplog.at_level(logging.WARNING, logger="momwire.networks._reduce"):
        _driven_resistor().solve()
    assert caplog.records == []


def test_poison_singular_sample_returns_none_and_logs(caplog):
    """One frequency landing on an unsolvable topology is no reason to lose
    the other forty samples: the bad one becomes None (→ NaN upstream) and
    the reason is logged once, with the attribution a point solve would raise.
    """
    g = np.diag([1.0 / R_LOAD, 0.0]).astype(np.complex128)
    src = _Group2Element(None, 0, c_v=1.0 + 0j, c_j=Z_SRC, e=E_SRC)
    open_br = _Group2Element(0, 1, c_v=0j, c_j=1.0 + 0j, e=0j)
    bad = MNASystem(g, [src, open_br], {0: (0, E_SRC, "v", 0j)})

    with caplog.at_level(logging.WARNING, logger="momwire.networks._reduce"):
        out = poison_singular_sample(bad.solve, where=" at 28.000 MHz")

    assert out is None
    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert "at 28.000 MHz" in text
    assert "this sample is NaN" in text
    assert "reciprocal condition" in text


def test_poison_singular_sample_passes_through_healthy_results():
    """Args and kwargs reach the callee untouched, and a good sample is
    returned as-is rather than being wrapped."""
    sentinel = object()

    def fn(a, b, c=None):
        assert (a, b, c) == (1, 2, 3)
        return sentinel

    assert poison_singular_sample(fn, 1, 2, c=3, where=" somewhere") is sentinel


def test_poison_singular_sample_does_not_swallow_other_errors():
    def boom():
        raise ZeroDivisionError("not a singular network")

    with pytest.raises(ZeroDivisionError):
        poison_singular_sample(boom)


# ---------------------------------------------------------------------------
# magnetizing_impedance — duck-typed, no spec layer needed
# ---------------------------------------------------------------------------


class _Br:
    def __init__(self, lmag=None, qlmag=None, core=None):
        self.lmag, self.qlmag, self.core = lmag, qlmag, core


class _Core:
    """Minimal stand-in for a `ferrite.FerriteCore`: impedance(f_MHz)."""

    def __init__(self):
        self.seen = []

    def impedance(self, f_mhz):
        self.seen.append(f_mhz)
        return 100.0 + 900.0j


def test_magnetizing_impedance_none_when_no_branch_declared():
    assert magnetizing_impedance(_Br(), OMEGA) is None


def test_magnetizing_impedance_scalar_model():
    """Ideal: jωL. Finite Q adds the series loss ωL/Q."""
    lmag = 30e-6
    assert magnetizing_impedance(_Br(lmag=lmag), OMEGA) == pytest.approx(
        1j * OMEGA * lmag
    )
    z_q = magnetizing_impedance(_Br(lmag=lmag, qlmag=40.0), OMEGA)
    assert z_q == pytest.approx(1j * OMEGA * lmag + OMEGA * lmag / 40.0)
    assert z_q.real > 0.0


def test_core_supersedes_the_scalar_model_and_is_asked_in_mhz():
    """`core` IS the core: it replaces lmag/qlmag wholesale rather than
    combining with them, and the material is asked at f in MHz."""
    core = _Core()
    z = magnetizing_impedance(_Br(lmag=30e-6, qlmag=40.0, core=core), OMEGA)

    assert z == 100.0 + 900.0j
    assert core.seen == [pytest.approx(FREQ_MHZ)]
