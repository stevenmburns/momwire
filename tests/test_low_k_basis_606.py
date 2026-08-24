"""The low-kΔ regime of the point-matched sinusoidal basis (momwire#606).

`SinusoidalSolver` is momwire's NEC-2 twin, and on feed impedance it agrees
with nec2c to four decimals across three decades of kΔ. Below kΔ ≈ 1e-3 it
used to change sign and run away while nec2c converged cleanly on the same
deck: 6.79 % off at kΔ = 2.1e-4, and an N-ladder that walked 797 kΩ through
two sign changes where nec2c's walked 17.8 kΩ monotonically onto the
electrostatic referee.

The cause was not a missing small-argument expansion — this family has had
careful ones since #205. It was the basis SCALING. The fill spelled

    Φ_c @ σA + Φ_s @ B + Φ_co @ σC

in which A ≈ −C and Φ_co → Φ_c as k → 0, so an O((kΔ)²) answer came out of
O(1) terms: a relative error of ~8ε/(kΔ)², measured at exactly that scaling
over four decades. #203 had already published the remedy for the coefficient
half — a pre-summed `AC` — but only the Galerkin sibling ever consumed it,
and the sum itself was formed as `A + C`, which is correctly rounded to the
sum without being the rounded value of the sum.

These gates pin all three legs of the fix: the coefficient (`AC` from a
closed form), the fill (the well-scaled shape set below the threshold), and
the readout (`_feed_segment_current`, which is what the port algebra reads).
And they pin the regime that already worked, which a fix must not disturb.
"""

from fractions import Fraction

import numpy as np
import pytest

from momwire.deck import build_solver, parse
from momwire.sinusoidal import _WELL_SCALED_KD, _recip_sin_gap

# ----------------------------------------------------------------------
# The deck: W7EL's coupled-loop model, a 300 m vertical standing on the rim
# of a closed 80x80 m loop, driven at 404 kV. #606's reproduction case.
# ----------------------------------------------------------------------
_DECK = """CM momwire#606
CE
GW 1,{n1},20.,-40.,300.,20.,-40.,0.,.005
GW 2,4,40.,-40.,0.,40.,40.,0.,.005
GW 3,4,40.,40.,0.,-40.,40.,0.,.005
GW 4,4,-40.,40.,0.,-40.,-40.,0.,.005
GW 5,3,-40.,-40.,0.,20.,-40.,0.,.005
GW 6,1,20.,-40.,0.,40.,-40.,0.,.005
GE 0
FR 0,1,0,0,{fmhz!r}
GN -1
EX 0,1,{fseg},0,0.,-404675.9
XQ
EN
"""

# nec2c's own answer on that deck, one row per decade of kΔ. Captured from
# nec2c 1.3 (`nec2c -i deck -o out`, ANTENNA INPUT PARAMETERS) and identical
# to the table in the issue. These are an EXTERNAL engine's printed output,
# not recorded momwire values — the thing this family is a twin OF, and the
# only reference here that is not machine-dependent.
_NEC2C = {
    0.0005: (6.1962e-05, -3.9064e05),
    0.005: (6.1961e-03, -3.9056e04),
    0.05: (6.2637e-01, -3.8199e03),
    0.5: (9.5113e02, +1.9582e03),
    5.0: (1.4707e03, +1.1742e03),
}


def _zin(fmhz, n1=15, basis="sinusoidal"):
    """Feed impedance of the #606 deck at `fmhz`, meshed `n1` on the vertical."""
    fseg = max(1, int(round(n1 * 14 / 15)))
    text = _DECK.format(n1=n1, fmhz=fmhz, fseg=fseg)
    built = build_solver(parse(text), basis=basis)
    return 1.0 / built.solver.compute_port_solution().y[0, 0]


# ----------------------------------------------------------------------
# Leg 1: the coefficient
# ----------------------------------------------------------------------
def _exact_recip_gap(kd, nterms=16):
    """1/sin(kΔ) − 1/(2 sin(kΔ/2)) in exact rational arithmetic.

    An independent reference, not a rearrangement of the thing under test:
    the sine series summed over `Fraction`, so there is no float anywhere
    until the comparison.
    """

    def sin_series(z):
        z = Fraction(z)
        total, term = Fraction(0), z
        for i in range(nterms):
            total += term if i % 2 == 0 else -term
            term = term * z * z / Fraction((2 * i + 2) * (2 * i + 3))
        return total

    x = Fraction(kd)
    return Fraction(1) / sin_series(x) - Fraction(1) / (2 * sin_series(x / 2))


@pytest.mark.parametrize("kd", [2.0, 1e-1, 1e-2, 2.1e-4, 1e-5, 1e-6])
def test_recip_sin_gap_is_exact_where_the_literal_difference_is_noise(kd):
    """`_recip_sin_gap` IS `A + C` on a neighbour entry, up to its a·Q factor.

    The literal difference of the two reciprocals loses 8 to 15 digits at the
    kΔ this solver has to serve; the half-angle identity keeps all of them.
    """
    truth = float(_exact_recip_gap(kd))
    literal = 1.0 / np.sin(kd) - 1.0 / (2.0 * np.sin(kd / 2))
    closed = float(_recip_sin_gap(kd))

    assert abs(closed - truth) / abs(truth) < 1e-14
    # ...and it is a real improvement, not a different spelling of the same
    # error: below kΔ = 1e-2 the literal difference is measurably worse.
    if kd <= 1e-2:
        assert abs(literal - truth) > 100 * abs(closed - truth)


def test_published_ac_is_not_the_float_sum_of_a_and_c_at_low_kd():
    """The whole of #606's coefficient leg, as a property.

    At a sane kΔ the closed form and the float sum agree to a rounding — the
    fix is not a change of value. At the deck's kΔ they differ by percent,
    and it is the SUM that is wrong: `A` and `C` are O(1) carrying an
    absolute ε against an O((kΔ)²) answer.
    """
    for fmhz, bound in ((5.0, 1e-13), (0.0005, None)):
        s = build_solver(
            parse(_DECK.format(n1=15, fmhz=fmhz, fseg=14)), basis="sinusoidal"
        ).solver
        geom = s._build_geometry()
        view = s._basis_coefs(geom, s.k)
        rel = np.abs(view["AC"] - (view["A"] + view["C"])) / np.abs(view["AC"])
        if bound is not None:
            assert rel.max() < bound, rel.max()
        else:
            assert rel.max() > 1e-3, rel.max()


# ----------------------------------------------------------------------
# Leg 2 + 3: the fill and the readout, end to end against nec2c
# ----------------------------------------------------------------------
def test_feed_impedance_tracks_nec2c_at_500hz():
    """The headline. kΔ = 2.1e-4, where the twin used to be 6.79 % off.

    The bar is 1e-3 relative on both parts: three orders tighter than the
    failure, and loose enough that it pins the FORMULATION rather than this
    machine's last bits (nec2c's own printout carries only five digits).
    """
    z = _zin(0.0005)
    r_ref, x_ref = _NEC2C[0.0005]
    assert abs(z.real - r_ref) / abs(r_ref) < 1e-3, z
    assert abs(z.imag - x_ref) / abs(x_ref) < 1e-3, z


@pytest.mark.parametrize("fmhz", [0.005, 0.05, 0.5])
def test_the_regime_that_already_worked_is_undisturbed(fmhz):
    """The three rows the issue reports at 0.00 %, pinned so a fix to the
    low-kΔ end cannot pay for itself out of the end that was already right.

    5 MHz is deliberately NOT here: 20 m segments at λ = 60 m is λ/3, coarse
    enough to strain both engines, and the issue excludes it for that reason.
    """
    z = _zin(fmhz)
    r_ref, x_ref = _NEC2C[fmhz]
    assert abs(z.real - r_ref) / abs(r_ref) < 1e-3, z
    assert abs(z.imag - x_ref) / abs(x_ref) < 1e-3, z


def test_n_ladder_converges_instead_of_changing_sign():
    """Refining SHRINKS kΔ, so it pushes further into the regime that broke.

    The pre-#606 ladder read −417, −410, −362, −426, −750 kΩ — two sign
    changes against nec2c's monotone 17.8 kΩ walk. What this gate pins is the
    qualitative failure, not a converged value: one sign throughout, and a
    spread bounded well inside the 797 kΩ excursion it used to take.
    """
    xs = np.array([_zin(0.0005, n1=n).imag for n in (15, 21, 31, 45, 61)])

    assert np.all(xs < 0), xs
    assert np.ptp(xs) < 25e3, xs
    # every rung within 5 % of the mean — the ladder is flat, not walking off
    assert np.abs(xs - xs.mean()).max() / abs(xs.mean()) < 0.05, xs


# ----------------------------------------------------------------------
# The threshold: what it must NOT do
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nsegs", [81, 801])
def test_ordinary_decks_keep_the_literal_path_and_its_kernel(nsegs):
    """The well-scaled path costs the C++ accelerator, so it must engage only
    where the literal one is actually losing digits.

    A half-wave dipole is the canonical shape this family serves; at N=801 it
    sits at kΔ = 3.8e-3 carrying the 1.2e-10 fill error #203 measured and
    accepted. Raising `_WELL_SCALED_KD` far enough to swallow it would trade
    70 % of a single-k solve for digits nothing reads — this gate is what
    makes that a test failure rather than a silent slowdown.
    """
    from momwire.sinusoidal import SinusoidalSolver

    lam = 22.0
    ys = np.linspace(-0.962 * lam / 4, 0.962 * lam / 4, 2)
    wire = np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 4.0)])
    s = SinusoidalSolver(wires=[wire], nsegs=nsegs, wavelength=lam, wire_radius=0.0005)
    geom = s._build_geometry()
    kd_min = float(np.min(s.k * np.asarray(geom["seg_h"], dtype=float)))
    assert kd_min >= _WELL_SCALED_KD, (nsegs, kd_min, _WELL_SCALED_KD)


def test_the_two_shape_sets_are_the_same_operator():
    """The reformulation is an identity, not a model change.

    `Φ_c @ σAC + Φ_s @ B + Φ_d @ σC` and `Φ_c @ σA + Φ_s @ B + Φ_co @ σC` are
    equal in exact arithmetic (AC = A + C, Φ_d = Φ_co − Φ_c). Evaluated at a
    kΔ where BOTH spellings are accurate they must therefore agree to a fill
    tolerance — if they did not, the well-scaled path would be answering a
    different question below the threshold rather than the same one better.
    """
    s = build_solver(
        parse(_DECK.format(n1=15, fmhz=5.0, fseg=14)), basis="sinusoidal"
    ).solver
    geom = s._build_geometry()
    k = s.k
    view = s._basis_coefs(geom, k)
    N = geom["n_segs"]
    starts = view["starts"]
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(starts))
    cols = view["jbasis"]
    sig = view["sigma"]

    def scatter(vals):
        M = np.zeros((N, N), dtype=np.complex128)
        M[rows, cols] = vals
        return M

    M_A, M_B = scatter(sig * view["A"]), scatter(view["B"])
    M_C, M_AC = scatter(sig * view["C"]), scatter(sig * view["AC"])

    Pc, Ps, Pco = s._field_tensor(geom, k, cos_shape="cos")
    _Pc, _Ps, Pd = s._field_tensor(geom, k, cos_shape="cos-1")

    literal = Pc @ M_A + Ps @ M_B + Pco @ M_C
    scaled = _Pc @ M_AC + _Ps @ M_B + Pd @ M_C
    rel = np.linalg.norm(scaled - literal) / np.linalg.norm(literal)
    assert rel < 1e-12, rel


# ----------------------------------------------------------------------
# The currents, not just the port scalar
# ----------------------------------------------------------------------
# nec2c 1.3's CURRENTS AND LOCATION table for the same 500 Hz deck, driven
# wire only (tag 1, segments 1-15). The imaginary parts are ~1e-10 against
# real parts of order 1, so magnitude is the whole of the comparison.
_NEC2C_DRIVEN_WIRE = np.array(
    [
        4.1938e-02,
        1.1951e-01,
        1.9420e-01,
        2.6792e-01,
        3.4128e-01,
        4.1463e-01,
        4.8827e-01,
        5.6249e-01,
        6.3762e-01,
        7.1408e-01,
        7.9249e-01,
        8.7389e-01,
        9.6075e-01,
        1.0359e00,
        9.8754e-01,
    ]
)
_DRIVE = -404675.9j  # the deck's EX 0 voltage


def _segment_centre_currents(n1=15, fmhz=0.0005):
    """Current at every segment CENTRE — the quantity nec2c tabulates."""
    fseg = max(1, int(round(n1 * 14 / 15)))
    s = build_solver(
        parse(_DECK.format(n1=n1, fmhz=fmhz, fseg=fseg)), basis="sinusoidal"
    ).solver
    sol = s.compute_port_solution()
    alpha = sol.coeffs[:, 0] * _DRIVE
    n = sol.basis.geom["n_segs"]
    return s._evaluate_basis_at_points(
        sol.basis.seg_view, np.arange(n, dtype=np.int64), np.zeros(n), alpha
    )


def test_driven_wire_currents_match_nec2c():
    """The port scalar is one number; the current distribution is the solve.

    `_feed_segment_current` was not the only readout summing `A + C` —
    `_currents_at` and the node-value evaluation did too, so a fix that only
    moved the impedance would leave the printed currents wrong. Pre-#606 this
    wire sat a flat 6.2 % below nec2c; the bar here is 1e-3, which that could
    not have passed.
    """
    got = np.abs(_segment_centre_currents()[:15])
    rel = np.abs(got - _NEC2C_DRIVEN_WIRE) / _NEC2C_DRIVEN_WIRE
    assert rel.max() < 1e-3, (rel.max(), got)


def test_the_loop_pathology_is_still_reproduced():
    """The twin's job is to reproduce NEC-2, defects included.

    W7EL's model puts ~220 A in the loop off a ~1 A source — quadrature error
    in the line integral of grad(phi), NEC-2's own defect and the thing the
    model was built to show. A "fix" that quietly cured it would mean this
    family had stopped being the NEC-2 twin, which is a worse outcome than
    the impedance bug. The loop current is a discretization artifact, so the
    bar is loose: same order, same sign of the effect, within 5 %.

    Sign is deliberately not compared. momwire traverses the closed loop in
    the opposite direction from nec2c's per-wire GW order, so every loop
    segment reads exactly 180 degrees out — a convention, uniform across all
    sixteen, not a disagreement about the physics.
    """
    loop = np.abs(_segment_centre_currents()[15:])
    source = abs(_segment_centre_currents()[13])

    assert loop.min() > 100.0 * source, (loop.min(), source)
    assert np.abs(loop - 2.2e2).max() / 2.2e2 < 0.05, loop


# ----------------------------------------------------------------------
# The one regime the fix does not reach, and how it says so
# ----------------------------------------------------------------------
def test_extended_kernel_at_low_kd_warns_instead_of_going_quiet():
    """`extended_kernel=True` has no well-scaled path (momwire#246 left the
    folded EKSCX forms unwired), so it keeps the literal fill at every kΔ.

    That is a known limit, not a regression — the EK solve gets exactly what
    it got before #606. What it must NOT do is go quiet: handing back a
    percent-wrong impedance with no signal is the same failure #246's own
    guard exists to prevent, one level up. This gate is what stops a future
    reader from "simplifying" the warning away.
    """
    import warnings as _w

    fseg = 14
    text = _DECK.format(n1=15, fmhz=0.0005, fseg=fseg)
    built = build_solver(parse(text), basis="sinusoidal")
    built.solver.extended_kernel = True

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        built.solver.compute_port_solution()

    msgs = [str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)]
    assert any("606" in m and "extended_kernel" in m for m in msgs), msgs


def test_no_warning_where_the_literal_fill_is_the_right_answer():
    """The warning has to be specific to the bad regime or it is noise.

    Above the threshold the literal fill IS the accurate one, EK or not.
    """
    import warnings as _w

    built = build_solver(
        parse(_DECK.format(n1=15, fmhz=5.0, fseg=14)), basis="sinusoidal"
    )
    built.solver.extended_kernel = True

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        built.solver.compute_port_solution()

    assert not [str(x.message) for x in caught if "606" in str(x.message)], [
        str(x.message) for x in caught
    ]
