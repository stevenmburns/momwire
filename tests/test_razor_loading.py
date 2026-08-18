"""RazorSolver's wire loading: distributed loss + lumped loads (momwire#427).

The loading term for razor-blade testing is derived in
`RazorSolver._loading_stencil`: a loaded wire's surface condition is
E_tan = Z_s(l)·I(l), razor tests the condition on a PATH, so the extra
matrix is L[m, n] = ∫_{P_m} Z_s(l) Λ_n(l) dl and `Z = Z_free + L`. In the
wing idiom it is `3h/8` when the row's path half and the column's tent ramp
rise at the same end of a shared segment, `h/8` when they rise at opposite
ends, times σ_row·σ_col. A lumped load is the delta case of the same
integral and collapses to one diagonal entry.

The gates below are in oracle order, strongest first — the order momwire#427
asked for:

1. **loading off is bit-frozen.** The term is structurally absent when
   nothing is loaded (`prepared["loading"] is None`), and a Z = 0 load is
   bit-identical to no load at all. The vs-branch-point half of that claim
   was measured out of tree against a shadow copy of the pre-#427 module
   (free space, PEC clear, PEC contact, refl-coef and Sommerfeld, both
   lanes, matrix and swept solve: 10/10 `array_equal`).
2. **the drive-point identity, exact.** A lumped Z_L at the FED knot gives
   `Z_driven = Z_unloaded + Z_L` to LU roundoff, on a dipole and on a
   base-fed monopole over PEC — i.e. on the grounded tent. This is not
   arranged: it is Sherman-Morrison on a rank-1 diagonal stamp, and it is
   what fixes the term's SIGN (the siblings assemble G with the opposite
   global sign and therefore subtract; this formulation adds).
3. **the Thevenin identity, exact.** A lumped Z_L at a NON-driven knot
   equals the two-port reduction of razor's own unloaded Y matrix
   terminated in Z_L, to LU roundoff, on an asymmetric deck — including
   the razor-blade non-reciprocity, since the reduction uses Y_pq and Y_qp
   separately.
4. **the grounded tent, exactly halved.** A loaded monopole over PEC is
   half its loaded mirror dipole — with distributed loss, with a mirrored
   pair of lumped loads, and (the base-gap convention) with a base load Z_L
   answering to a centre load of 2·Z_L on the dipole.
5. **the NEC-5 twin lane.** `tests/golden_razor_loading_nec5.py` carries the
   binary's printed impedances for an unloaded ladder, an `LD 4` at the fed
   knot, the same `LD 4` mid-element and `LD 5` copper. What is gated is the
   loading INCREMENT, since razor and NEC-5 never agree pointwise.
6. **cross-formulation** difference-of-differences against `BSplineSolver`
   and `SinusoidalGalerkinSolver` at N = 192 (the units-4/5 protocol),
   plus the `wire_loss_power` readout against the same two.
7. **the schedule.** The stencil is geometry and rides the k-independent
   prepare half; Z_s(ω) is not, and is rebuilt per solved wavenumber — which
   the swept gates prove by sweeping a skin-effect loss that moves with ω.
"""

import numpy as np
import pytest
import scipy.linalg

from momwire import RazorSolver, _wire_loading
from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver
from golden_razor_loading_nec5 import LOAD_Z, LOADED_LADDERS, SIGMA_CU

FREQ_MHZ = 14.0
NEC5_WL = 299792458.0 / (FREQ_MHZ * 1e6)
NEC5_RAD = 1.0262e-3
NEC5_LEN = 10.18946

WL = 22.0
RAD = 1.0e-3
SIGMA = 5.8e7
LANES = ({}, {"nec5_quadrature": True})
LANE_IDS = ("gl", "n5q")

# A trap-shaped lumped load: big enough to move the answer by tens of ohms,
# so the term is under test rather than the rounding.
ZL = 40.0 - 120.0j

DIP_HALF = 0.239 * WL
DIPOLE = [np.array([[0.0, -DIP_HALF, 0.0], [0.0, DIP_HALF, 0.0]])]
MONO_LEN = 5.35
MONOPOLE = [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])]
MIRROR = [np.array([[0.0, 0.0, -MONO_LEN], [0.0, 0.0, MONO_LEN]])]

# An inverted-V: bent, three anchors, and fed OFF the apex so the deck is
# asymmetric and the two-port reduction has to carry a non-symmetric Y.
_IV_C = 0.19 * WL / np.sqrt(2.0)
_IV_ARM = float(np.hypot(_IV_C, _IV_C))
INVVEE = [
    np.array(
        [
            [-_IV_C, 0.0, 0.3 * WL - _IV_C],
            [0.0, 0.0, 0.3 * WL],
            [_IV_C, 0.0, 0.3 * WL - _IV_C],
        ]
    )
]
IV_FEED_ARC = _IV_ARM * 0.75  # three quarters along the first arm
IV_LOAD_ARC = _IV_ARM * 1.5  # half way along the SECOND arm


def _sim(wires, npe, **kw):
    return RazorSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wire_radius=kw.pop("wire_radius", RAD),
        wavelength=kw.pop("wavelength", WL),
        **kw,
    )


def _z(wires, npe, **kw):
    z, coeffs = _sim(wires, npe, **kw).compute_impedance()
    return complex(z), coeffs


def _terminate(Y, z_load):
    """Port 0's impedance with port 1 terminated in `z_load`.

    Textbook two-port reduction, written out because razor's Y is NOT
    symmetric: a series impedance in port 1's gap makes that gap's applied
    voltage V_1 = −Z_L·I_1, and eliminating it gives

        Z_in = 1 / (Y_00 − Y_01·Y_10·Z_L / (1 + Y_11·Z_L)).
    """
    return complex(
        1.0 / (Y[0, 0] - Y[0, 1] * Y[1, 0] * z_load / (1.0 + Y[1, 1] * z_load))
    )


# --------------------------------------------------------------------------
# 1. loading off is bit-frozen
# --------------------------------------------------------------------------
def test_loading_is_structurally_absent_when_nothing_is_loaded():
    """No loading configured ⇒ no stencil, not a stencil of zeros.

    The `extended_kernel=False` standard the architecture doc's §6 gate (b)
    holds every capability to: an unloaded fill must not execute one float
    operation the pre-capability fill did not. `prepared["loading"] is None`
    is the structural form of that, and it is what makes the vs-branch-point
    bit-identity a consequence rather than a coincidence.
    """
    sim = _sim(DIPOLE, [[16]])
    prepared = sim._assemble_Z_prepare(sim._build_geometry())
    assert prepared["loading"] is None
    loaded = _sim(DIPOLE, [[16]], wire_conductivity=SIGMA)
    assert loaded._assemble_Z_prepare(loaded._build_geometry())["loading"] is not None


@pytest.mark.parametrize(
    "kw",
    [
        {"lumped_loads": []},
        {"lumped_loads": [(0, 0.0, 0.0)]},
        {"lumped_loads": [(0, 0.0, 0.0), (0, DIP_HALF, 0.0j)]},
    ],
    ids=("empty", "one-zero", "two-zero"),
)
def test_zero_impedance_loading_is_bit_identical_to_no_loading(kw):
    """A Z_L = 0 load adds nothing, to the bit.

    The weaker half of gate (b) — the stencil IS built here, and the term is
    executed — so what this pins is that executing it with a zero impedance
    is an exact no-op rather than a rounding-level one.
    """
    ref = _sim(DIPOLE, [[16]])
    got = _sim(DIPOLE, [[16]], **kw)
    Z_ref = ref._assemble_Z(ref._build_geometry(), ref.k)
    Z_got = got._assemble_Z(got._build_geometry(), got.k)
    assert np.array_equal(Z_ref, Z_got)


# --------------------------------------------------------------------------
# 2. the drive-point identity, exact — and the sign it fixes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
@pytest.mark.parametrize("deck", ("dipole", "monopole"))
def test_a_lumped_load_at_the_fed_knot_is_exactly_in_series(deck, lane):
    """`Z_driven = Z_unloaded + Z_L`, to LU roundoff.

    Sherman-Morrison on the rank-1 stamp: with G' = G + Z_L·e_p e_pᵀ and a
    drive e_p, (G'⁻¹)_pp = a/(1 + Z_L a) with a = (G⁻¹)_pp, so
    1/(G'⁻¹)_pp = 1/a + Z_L exactly. It holds only because a lumped load at
    knot p lands on ONE diagonal entry and on the SAME knot the feed
    resolves to (`_snap_to_knot` is shared), so this is the gate that fixes
    both the sign of the loading term and the load's site convention.

    The monopole row is the grounded tent: the load sits in the BASE gap, in
    series with the base gap's source, and enters at full value — the same
    convention that makes the feed voltage the base gap's rather than the
    equivalent dipole's. Measured relatives: 1.6e-14 / 3.8e-15 (dipole,
    GL / NEC-5 lane) and 9.4e-17 / 1.9e-16 (monopole).
    """
    if deck == "dipole":
        wires, npe, arc, ground = DIPOLE, [[24]], DIP_HALF, {}
    else:
        wires, npe, arc, ground = MONOPOLE, [[24]], 0.0, {"ground_z": 0.0}
    common = dict(feed_arclength=arc, **ground, **lane)
    z0, _ = _z(wires, npe, **common)
    z1, _ = _z(wires, npe, lumped_loads=[(0, arc, ZL)], **common)
    rel = abs(z1 - (z0 + ZL)) / abs(z0 + ZL)
    assert rel < 1e-11, f"{z1} != {z0} + {ZL} (rel {rel:.3e})"
    # ...and the load is not being quietly dropped.
    assert abs(z1 - z0) > 10.0


# --------------------------------------------------------------------------
# 3. the Thevenin identity, exact
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_a_lumped_load_elsewhere_is_the_y_matrix_termination(lane):
    """A load at a NON-driven knot == the unloaded Y matrix terminated in it.

    Two independent numeric paths to one number: the loading term stamps
    Z_L into the matrix and factors it once, while the reference builds the
    unloaded two-port Y (one factorization, two back-substitutions) and
    eliminates port 1 algebraically. They agree to LU roundoff (measured
    9.6e-16 GL, 1.2e-15 NEC-5 lane), which says the stamp IS a series
    impedance in that knot's current path and nothing else.

    The deck is deliberately asymmetric — an inverted-V fed three quarters
    along one arm, loaded half way along the OTHER — so Y_01 != Y_10 (razor
    is non-reciprocal by ~1.5e-5 here) and the reduction has to use both.
    """
    npe = [[16, 16]]
    z_stamped, _ = _z(
        INVVEE,
        npe,
        feed_arclength=IV_FEED_ARC,
        lumped_loads=[(0, IV_LOAD_ARC, ZL)],
        **lane,
    )
    Y = _sim(
        INVVEE, npe, feeds=[(0, IV_FEED_ARC, 1.0), (0, IV_LOAD_ARC, 0.0)], **lane
    ).compute_y_matrix()
    z_thevenin = _terminate(Y, ZL)
    rel = abs(z_stamped - z_thevenin) / abs(z_thevenin)
    assert rel < 1e-11, f"{z_stamped} vs {z_thevenin} (rel {rel:.3e})"
    # The reference really is the non-reciprocal one — if Y were symmetric
    # this deck would not be exercising the two off-diagonals separately.
    assert abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1]) > 1e-7


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_loading_composes_with_multi_feed_and_the_y_matrix(lane):
    """Loading reaches `compute_y_matrix` and the multi-feed solve.

    Both go through `_assemble_Z`, so this is a plumbing gate rather than a
    physics one — but the plumbing is what a consumer touches. The claim is
    the same reduction as above read the other way round: terminate the
    LOADED two-port in a further Z and it equals the solve with both loads
    stamped.
    """
    npe = [[16, 16]]
    z2 = 15.0 + 60.0j
    sim = _sim(
        INVVEE,
        npe,
        feeds=[(0, IV_FEED_ARC, 1.0), (0, _IV_ARM * 0.5, 0.0)],
        lumped_loads=[(0, IV_LOAD_ARC, ZL)],
        **lane,
    )
    z_th = _terminate(sim.compute_y_matrix(), z2)
    z_both, _ = _z(
        INVVEE,
        npe,
        feed_arclength=IV_FEED_ARC,
        lumped_loads=[(0, IV_LOAD_ARC, ZL), (0, _IV_ARM * 0.5, z2)],
        **lane,
    )
    assert abs(z_both - z_th) / abs(z_th) < 1e-11, f"{z_both} vs {z_th}"


def test_two_loads_at_one_knot_are_in_series():
    """Two `lumped_loads` naming the same knot add, they do not fight.

    `np.add.at` on the diagonal is what makes this true; the deck front end
    refuses the same case rather than merging it, so the behaviour is worth
    pinning where it is defined.
    """
    a, b = 30.0 + 10.0j, -5.0 + 200.0j
    z_two, _ = _z(DIPOLE, [[24]], lumped_loads=[(0, DIP_HALF, a), (0, DIP_HALF, b)])
    z_one, _ = _z(DIPOLE, [[24]], lumped_loads=[(0, DIP_HALF, a + b)])
    assert abs(z_two - z_one) < 1e-9 * abs(z_one)


# --------------------------------------------------------------------------
# 4. the grounded tent: a loaded monopole is half its loaded mirror dipole
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
@pytest.mark.parametrize("kind", ("distributed", "mirrored-pair", "base-gap"))
def test_loading_on_the_grounded_tent_halves_the_mirror_model(kind, lane):
    """Loading a grounded radiator over PEC == loading its mirror model.

    The unloaded statement is `test_razor_ground_contact.py`'s first gate:
    the grounded matrix is the mirror model's symmetric reduction, so
    Z_mono = Z_dipole/2 exactly. Loading has to survive that reduction, and
    the three rows here are the three ways it can be asked to:

    * **distributed** — copper on the monopole is copper on the whole mirror
      dipole. The grounded tent's own loading is its REAL wing only (σ = 0
      empties the image wing in `_loading_stencil`), which is exactly the
      half of the dipole's centre tent that lies above the plane;
    * **mirrored-pair** — a lumped Z_L at a knot h·(n/2) up the monopole is
      the SAME Z_L at both of the dipole's mirrored knots. The image of a
      loaded conductor is a loaded conductor;
    * **base-gap** — a lumped Z_L at the CONTACT knot answers to 2·Z_L in
      the dipole's centre gap, which is the load-side statement of the
      halved row: the base gap is half the dipole's gap, so an impedance in
      it counts double when the model is unfolded.

    Measured relatives 3e-15 … 1e-13 across the three rows, both lanes,
    n ∈ {8, 24} — LU roundoff, not agreement.
    """
    n = 24
    h = MONO_LEN / n
    arc = h * (n // 2)
    if kind == "distributed":
        mono_kw = dict(wire_conductivity=SIGMA)
        dip_kw = dict(wire_conductivity=SIGMA)
    elif kind == "mirrored-pair":
        mono_kw = dict(lumped_loads=[(0, arc, ZL)])
        dip_kw = dict(lumped_loads=[(0, MONO_LEN + arc, ZL), (0, MONO_LEN - arc, ZL)])
    else:
        mono_kw = dict(lumped_loads=[(0, 0.0, ZL)])
        dip_kw = dict(lumped_loads=[(0, MONO_LEN, 2.0 * ZL)])

    common = dict(wire_radius=NEC5_RAD, wavelength=NEC5_WL, **lane)
    z_mono, _ = _z(
        MONOPOLE, [[n]], feed_arclength=0.0, ground_z=0.0, **mono_kw, **common
    )
    z_dip, _ = _z(MIRROR, [[2 * n]], feed_arclength=MONO_LEN, **dip_kw, **common)
    rel = abs(z_mono - z_dip / 2.0) / abs(z_mono)
    assert rel < 1e-11, f"{z_mono} vs {z_dip / 2.0} (rel {rel:.3e})"


# --------------------------------------------------------------------------
# 5. the NEC-5 twin lane, on loading
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", ("lumped-at-feed", "lumped-mid", "copper"))
def test_loading_increment_matches_nec5(case):
    """Loading adds no new twin gap: the INCREMENT tracks the binary's.

    Razor and NEC-5 never agree pointwise at finite N — that is the twin's
    whole story (`tests/test_razor_nec5_twin.py`) and it shows up here as a
    steady 0.003 + 0.037j Ω offset in the unloaded column. So what is gated
    is what loading itself contributes:

        | (Z_loaded − Z_unloaded)_razor − (Z_loaded − Z_unloaded)_NEC5 |

    in the `nec5_quadrature` lane (momwire#316's identified path rule).
    Measured worst over the N = 24…96 ladder: 0.003 Ω (`LD 4` at the fed
    knot), 0.021 Ω (`LD 4` mid-element), 0.001 Ω (`LD 5` copper) — the last
    two at or below the binary's own 1e-3 print resolution on the copper
    row. The bar is 0.05 Ω per rung.

    The conventions were verified BEFORE this was gated, from printed output
    alone; `scripts/capture_razor_loading_nec5_lane.py` records how. In
    short: an `LD 4` on the fed segment moves the printed impedance by
    exactly R + jX, so it is a series impedance in the port's current path;
    and since `EX 0 tag j 2` feeds the knot at segment j's far end, an `LD`
    on segment j lands on that same knot — which is one basis coefficient
    here, i.e. `lumped_loads`. `LD 5` is the per-unit-length conductor
    impedance `wire_conductivity` drives.
    """
    base = {n: (zn, zq) for n, zn, zq, _zg in LOADED_LADDERS["unloaded"]}
    worst = 0.0
    for n, zn, zq, _zg in LOADED_LADDERS[case]:
        bn, bq = base[n]
        incr = (zq - bq) - (zn - bn)
        worst = max(worst, abs(incr))
        assert abs(incr) < 0.05, f"N={n}: increment gap {incr} ohm"
    # ...and the loading really did move the printed impedance, so the
    # increment claim is not a claim about two zeros.
    assert max(abs(zn - base[n][0]) for n, zn, _q, _g in LOADED_LADDERS[case]) > 0.5
    assert worst > 0.0


def test_the_golden_lanes_still_reproduce():
    """The recorded razor columns are what this module produces today.

    The golden file carries NEC-5's printouts *and* razor's two lanes; the
    binary is not run in CI, so this is what keeps the razor half honest —
    if the fill or the loading term moves, the twin test above would still
    pass against stale razor numbers unless something re-derives them.
    Two rungs of each case, which is the cheapest sample that touches all
    three loading spellings.
    """
    wires = [np.array([[0.0, 0.0, 0.0], [0.0, NEC5_LEN, 0.0]])]
    extra = {
        "unloaded": {},
        "lumped-at-feed": {"lumped_loads": [(0, NEC5_LEN / 2.0, LOAD_Z)]},
        "lumped-mid": {"lumped_loads": [(0, NEC5_LEN / 4.0, LOAD_Z)]},
        "copper": {"wire_conductivity": SIGMA_CU},
    }
    for case, rows in LOADED_LADDERS.items():
        for n, _zn, zq, zg in rows[:2]:
            for want, lane in ((zq, {"nec5_quadrature": True}), (zg, {})):
                got, _ = _z(
                    wires,
                    [[n]],
                    wire_radius=NEC5_RAD,
                    wavelength=NEC5_WL,
                    **extra[case],
                    **lane,
                )
                assert abs(got - want) < 5e-6, f"{case} N={n}: {got} vs {want}"


# --------------------------------------------------------------------------
# 6. perturbation sanity
# --------------------------------------------------------------------------
def test_distributed_loss_scales_as_the_skin_effect_says():
    """ΔR_in tracks Re[Z'] linearly, and Re[Z'] goes as 1/√σ.

    Two magnitudes two decades apart in conductivity: in the strong-skin
    limit Z'_int = (1+j)/(2πaσδ) with δ = √(2/ωμσ), so Z' ∝ σ^{-1/2} and a
    hundredfold drop in σ is a tenfold rise in ΔR. Measured ratio 10.08 —
    the 0.8 % excess is the exact I₀/I₁ solid-cylinder form pulling away
    from the asymptote at the LOWER conductivity, where a/δ is ten times
    smaller, which is the right direction and the right size.

    The linearity half is read off the same pair: ΔR/Re[Z'] is one number
    at both magnitudes, because the current distribution barely moves.
    """
    z0, _ = _z(DIPOLE, [[24]])
    zs = {}
    for sigma in (5.8e9, 5.8e7):
        z, _ = _z(DIPOLE, [[24]], wire_conductivity=sigma)
        zs[sigma] = z - z0
    ratio = zs[5.8e7].real / zs[5.8e9].real
    assert 9.8 < ratio < 10.4, f"ΔR ratio {ratio} is not the √σ law"

    # ΔR / Re[Z'] is the same effective length at both magnitudes: the term
    # is LINEAR in Z', which a quadratic bug in the stencil would break.
    sim = _sim(DIPOLE, [[24]], wire_conductivity=5.8e7)
    eff = [
        zs[s].real
        / float(
            np.real(
                _wire_loading.loading_for(
                    _sim(DIPOLE, [[24]], wire_conductivity=s), sim.omega
                ).z_wire[0]
            )
        )
        for s in (5.8e9, 5.8e7)
    ]
    assert abs(eff[0] - eff[1]) / eff[0] < 0.02, eff


def _dense_stencil(sim, geom):
    """`_loading_stencil` as a dense (n_basis, n_basis) geometric matrix."""
    st = sim._loading_stencil(geom)
    L = np.zeros((geom["n_basis_total"],) * 2)
    np.add.at(L, (st["rows"], st["cols"]), st["vals"])
    return L


def _quadrature_stencil(sim, geom):
    """∫_{P_m} Λ_n dl by QUADRATURE, over the solver's own testing paths.

    The independent path to `_loading_stencil`'s closed form: walk the
    `_testing_paths` nodes row by row, find every tent whose support covers
    the node's segment, evaluate that tent's linear shape at the node from
    the node's own arc coordinate, and dot the path tangent with the tent's
    current direction there. The default Gauss-Legendre path rule integrates
    a linear integrand exactly, so the two must agree to roundoff.
    """
    pts, tans, wts = sim._testing_paths(geom)
    n_b = geom["n_basis_total"]
    n_qp = pts.shape[1] // 2
    p0, tvec, h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
    ws, wr, wg = geom["wing_seg"], geom["wing_rise"], geom["wing_sigma"]
    L = np.zeros((n_b, n_b))
    for m in range(n_b):
        for q in range(pts.shape[1]):
            i = 0 if q < n_qp else 1
            if wg[m, i] == 0.0:  # the grounded tent's image half-path
                continue
            s = ws[m, i]
            tau = float((pts[m, q] - p0[s]) @ tvec[s])
            for n in range(n_b):
                for j in (0, 1):
                    if ws[n, j] != s or wg[n, j] == 0.0:
                        continue
                    shape = tau / h[s] if wr[n, j] else 1.0 - tau / h[s]
                    direction = float(tans[m, q] @ (wg[n, j] * tvec[s]))
                    L[m, n] += wts[m, q] * direction * shape
    return L


@pytest.mark.parametrize(
    "wires,npe,kw",
    [
        (DIPOLE, [[12]], {}),
        (INVVEE, [[8, 8]], {}),
        (MONOPOLE, [[12]], {"ground_z": 0.0}),
    ],
    ids=("dipole", "invvee", "monopole"),
)
def test_the_stencil_is_the_path_integral_it_claims_to_be(wires, npe, kw):
    """The closed form equals a direct quadrature of ∫_{P_m} Λ_n dl.

    `_loading_stencil` reduces the integral to two constants — 3h/8 when the
    row's path half and the column's tent ramp rise at the same end of a
    shared segment, h/8 when they rise at opposite ends — times σ_row·σ_col.
    This is that claim checked against the integral itself, on the solver's
    own testing paths, over an interior tent (dipole), a bent wire and a
    junction (inverted-V) and a grounded tent (monopole). It catches a
    swapped 3/8-vs-1/8, a missed neighbour and a mis-signed wing at once,
    without going near a solve.

    Two structural readings ride along. L is SYMMETRIC — swapping row and
    column swaps the two rise flags, which the "same end?" test does not see
    — even though razor's field matrix is not; a surface impedance is a
    local reciprocal object and the testing rule does not spoil that. And
    the row sums are the path lengths WHERE THE BASIS CAN CARRY A CONSTANT
    CURRENT: h in the interior of a wire, but 7h/8 on the tent next to a
    free end, because the tent basis pins the current to zero there and no
    column exists to carry the missing eighth.
    """
    sim = _sim(wires, npe, wire_conductivity=SIGMA, **kw)
    geom = sim._build_geometry()
    L = _dense_stencil(sim, geom)
    assert np.allclose(L, _quadrature_stencil(sim, geom), rtol=1e-12, atol=1e-15)
    assert np.allclose(L, L.T, rtol=1e-13)


# --------------------------------------------------------------------------
# 7. the schedule: the stencil is geometry, Z_s(ω) is not
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_swept_loading_matches_the_per_k_solves(lane):
    """A swept loaded solve equals rebuilding the solver at every point.

    The gate on the prepare/replay split. The stencil is pure geometry and
    is built ONCE in `_assemble_Z_prepare`; Z'(ω) is not — the skin-effect
    internal impedance goes as √ω and the insulation reactance as ω — so it
    is rebuilt per solved wavenumber. Caching it in the prepare half would
    freeze the loading at the first k and show up here as a growing error
    across the sweep, since these three wavelengths span ±20 %.
    """
    wls = np.array([WL * 0.8, WL, WL * 1.2])
    ks = 2.0 * np.pi / wls
    kw = dict(
        wire_conductivity=SIGMA,
        insulation_radius=3.0 * RAD,
        insulation_eps_r=2.3,
        lumped_loads=[(0, 0.1 * DIP_HALF, ZL)],
        **lane,
    )
    swept = _sim(DIPOLE, [[24]], **kw).compute_impedance_swept(ks)
    for i, wl in enumerate(wls):
        want, _ = _z(DIPOLE, [[24]], wavelength=float(wl), **kw)
        assert abs(swept[i] - want) < 1e-9 * abs(want), f"{swept[i]} vs {want}"
    # The sweep must actually be moving the loading, or it proves nothing:
    # a k-independent stencil times a k-independent Z' would be flat.
    plain = _sim(DIPOLE, [[24]], **{k: v for k, v in kw.items() if k in lane})
    unloaded = plain.compute_impedance_swept(ks)
    incr = swept - unloaded
    assert abs(incr[-1] - incr[0]) > 0.05, incr


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
def test_swept_y_matrix_with_loading_matches_the_per_k_solves(lane):
    """The same schedule gate on the two-port path."""
    wls = np.array([WL * 0.9, WL * 1.1])
    ks = 2.0 * np.pi / wls
    kw = dict(
        wire_conductivity=SIGMA,
        lumped_loads=[(0, 0.4 * DIP_HALF, ZL)],
        feeds=[(0, DIP_HALF, 1.0), (0, 1.6 * DIP_HALF, 0.0)],
        **lane,
    )
    swept = _sim(DIPOLE, [[24]], **kw).compute_y_matrix_swept(ks)
    for i, wl in enumerate(wls):
        want = _sim(DIPOLE, [[24]], wavelength=float(wl), **kw).compute_y_matrix()
        assert np.allclose(swept[i], want, rtol=1e-11, atol=0.0)


@pytest.mark.parametrize(
    "ground",
    [
        {},
        {"ground_z": 0.0},
        {"ground_z": 0.0, "ground_eps": 13.0 - 0.5j},
        {"ground_z": 0.0, "ground_eps": 13.0 - 0.5j, "ground_model": "sommerfeld"},
    ],
    ids=("free", "pec", "refl-coef", "sommerfeld"),
)
def test_loading_composes_with_every_served_ground(ground):
    """Loading is a property of the conductor, not of the half-space.

    `Z = (Z_free − Z_image) + L`: the term takes no image and no Fresnel
    weight, so it needs no branch per ground and the four rows here differ
    only in what the fill under it was. What each row asserts is that the
    loaded answer is the unloaded one moved by roughly the same increment —
    the ground changes the current distribution, so not identically, but a
    term that was being dropped or double-counted over one ground would not
    land within a factor of two of the others.
    """
    wires = [np.array([[0.0, -DIP_HALF, 0.3 * WL], [0.0, DIP_HALF, 0.3 * WL]])]
    npe = [[16]]
    z0, _ = _z(wires, npe, **ground)
    z1, c1 = _z(wires, npe, wire_conductivity=SIGMA, **ground)
    incr = (z1 - z0).real
    assert 0.4 < incr < 2.0, f"{ground}: ΔR = {incr}"
    p, per_wire = _sim(wires, npe, wire_conductivity=SIGMA, **ground).wire_loss_power(
        c1
    )
    assert p > 0.0 and np.isclose(p, per_wire.sum())


# --------------------------------------------------------------------------
# refusals and validation
# --------------------------------------------------------------------------
def test_capabilities_declares_wire_loading_served():
    assert RazorSolver.capabilities.wire_loading is True
    assert RazorSolver.capabilities.refusal("wire_loading") is None


def test_a_lumped_load_at_a_k3_junction_is_refused():
    """Same ambiguity as a source there: which branch pair is it between?"""
    a = 0.1 * WL
    wires = [
        np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-a, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, a, 0.0]]),
    ]
    sim = _sim(wires, [[6]] * 3, lumped_loads=[(0, 0.0, ZL)])
    with pytest.raises(NotImplementedError, match="lumped_loads"):
        sim.compute_impedance()


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"lumped_loads": [(0, 0.0)]}, "wire_index, arclength, impedance"),
        ({"lumped_loads": [(3, 0.0, 1.0)]}, "out of range"),
        ({"lumped_loads": [(0, 0.0, complex("nan"))]}, "must be finite"),
        ({"wire_conductivity": -1.0}, "must be > 0"),
        ({"insulation_radius": 2.0 * RAD}, "must be given together"),
        ({"insulation_radius": 0.5 * RAD, "insulation_eps_r": 2.0}, "must exceed"),
    ],
)
def test_loading_kwargs_are_validated(kw, match):
    with pytest.raises(ValueError, match=match):
        _sim(DIPOLE, [[8]], **kw)


# --------------------------------------------------------------------------
# 8. cross-formulation (slow)
# --------------------------------------------------------------------------
LADDER = (24, 48, 96, 192)
REFERENCES = {
    "bspline": (BSplineSolver, {"degree": 2}),
    "sin-galerkin": (SinusoidalGalerkinSolver, {}),
}
# A trap of MODERATE size for the gated column; see the docstring of
# `test_loading_adds_no_cross_formulation_gap` for the 30+400j column and
# why it is recorded rather than gated.
IV_TRAP = 20.0 + 80.0j


def _ref_z(cls, extra, wires, npe, **kw):
    z, _ = cls(
        wires=wires,
        n_per_edge_per_wire=npe,
        wire_radius=RAD,
        wavelength=WL,
        **extra,
        **kw,
    ).compute_impedance()
    return complex(z)


def _ref_terminated(cls, extra, wires, npe, arc_load, z_load):
    """A reference formulation's loaded impedance, by the house idiom.

    The siblings have no lumped-load kwarg — they serve a lumped load as
    deck-level port algebra over a `node_gaps` port (`momwire.deck._solver`)
    — so the honest cross-formulation reference is that same algebra: a
    zero-voltage gap port at the load site, terminated in Z_L. Razor's own
    stamp is proved equal to this construction on razor
    (`test_a_lumped_load_elsewhere_is_the_y_matrix_termination`), so the
    comparison is of one physical object across three formulations.
    """
    Y = cls(
        wires=wires,
        n_per_edge_per_wire=npe,
        wire_radius=RAD,
        wavelength=WL,
        feeds=[(0, None, 1.0), (0, arc_load, 0.0)],
        **extra,
    ).compute_y_matrix()
    return _terminate(Y, z_load)


@pytest.fixture(scope="module")
def cross_ladders():
    """Both decks × loaded/unloaded × two razor lanes × two references ×
    the N ladder. Measured ~4 min, which is why every reader is `slow`."""
    out = {}
    for n in LADDER:
        npe_d, npe_v = [[n]], [[n // 2, n // 2]]
        for lane, mode in (("razor-GL", {}), ("razor-NEC5", LANES[1])):
            out[("dipole", lane, "off", n)] = _ref_z(RazorSolver, mode, DIPOLE, npe_d)
            out[("dipole", lane, "on", n)] = _ref_z(
                RazorSolver, mode, DIPOLE, npe_d, wire_conductivity=SIGMA
            )
            out[("invvee", lane, "off", n)] = _ref_z(RazorSolver, mode, INVVEE, npe_v)
            out[("invvee", lane, "on", n)] = _ref_z(
                RazorSolver,
                mode,
                INVVEE,
                npe_v,
                lumped_loads=[(0, _IV_ARM * 0.5, IV_TRAP)],
            )
        for rname, (cls, extra) in REFERENCES.items():
            out[("dipole", rname, "off", n)] = _ref_z(cls, extra, DIPOLE, npe_d)
            out[("dipole", rname, "on", n)] = _ref_z(
                cls, extra, DIPOLE, npe_d, wire_conductivity=SIGMA
            )
            out[("invvee", rname, "off", n)] = _ref_z(cls, extra, INVVEE, npe_v)
            out[("invvee", rname, "on", n)] = _ref_terminated(
                cls, extra, INVVEE, npe_v, _IV_ARM * 0.5, IV_TRAP
            )
    return out


@pytest.mark.slow
@pytest.mark.parametrize("lane", ("razor-GL", "razor-NEC5"))
@pytest.mark.parametrize("ref", tuple(REFERENCES))
@pytest.mark.parametrize("deck", ("dipole", "invvee"))
def test_loading_adds_no_cross_formulation_gap(deck, ref, lane, cross_ladders):
    """Razor's disagreement with a Galerkin solver is the SAME loaded as
    unloaded — the units-4/5 protocol, applied to loading.

    Razor walks its own O(1/N) walk, so |Z_razor − Z_reference| is a real
    number at any finite N (0.49 Ω against BSpline on the inverted-V at
    N = 192, and that is free space, not loading). The sharp claim this
    formulation can make about the loading TERM is therefore the
    difference-of-differences: loading must not widen that gap.

    Measured at N = 192, gap_loaded − gap_unloaded, both decks:

        deck      reference       razor-GL   razor-NEC5
        dipole    BSpline          +0.0025      +0.0025   (copper)
        dipole    SinGalerkin      +0.0023      +0.0023
        invvee    BSpline          −0.1824      −0.2100   (20+80j trap)
        invvee    SinGalerkin      −0.0403      −0.3298

    Every entry is well under the 0.25 Ω bar, and the loaded gaps on the
    inverted-V are SMALLER than the unloaded ones — the trap suppresses the
    current in the outer arm, which is where razor's testing error lives.
    The dipole's copper column agrees far more sharply than the bar, because
    there the increment itself can be read: 0.8686 + 0.7602j (razor, both
    lanes to six figures) against 0.8691 + 0.7627j (BSpline) and 0.8693 +
    0.7624j (SG) — 0.0025 Ω and 0.0023 Ω apart on a 1.15 Ω increment.

    **Recorded, not gated: a very large trap.** At Z_L = 30 + 400j the
    BSpline column stays inside the bar (−0.32 / −0.33) but
    `SinusoidalGalerkinSolver`'s goes to +3.12 / +2.68, and its loaded
    increment is not converged at N = 192 — it swings 252, 206, 211, 223 Ω
    down the ladder while BSpline's (218.5, 218.9, 219.2, 219.4) and
    razor's (217.1, 218.1, 218.7, 219.1) walk smoothly to the same place.
    A 400 Ω series impedance nearly opens the wire, which is the near-open
    high-Q regime the sinusoidal family is known to converge slowly in
    (antennaknobs#478); it is a property of the reference, not of the term
    under test, so the gate uses a trap of ordinary size and this paragraph
    carries the finding.
    """
    gaps = {}
    for state in ("off", "on"):
        gaps[state] = abs(
            cross_ladders[(deck, lane, state, 192)]
            - cross_ladders[(deck, ref, state, 192)]
        )
    assert gaps["on"] - gaps["off"] < 0.25, gaps


@pytest.mark.slow
def test_wire_loss_power_agrees_across_formulations():
    """The dissipation readout, against the two Galerkin siblings.

    `wire_loss_power` is a PHYSICAL integral — ½ Σ_w Re[Z'_w]·∫|I|² — so
    unlike the loading matrix it uses the tent basis's Galerkin overlap
    (h/3, h/6 per segment), not the testing-path stencil. That is the same
    choice `SinusoidalGalerkinSolver` makes in inheriting
    `SinusoidalSolver.wire_loss_power` unchanged, and the evidence it is
    right is that three formulations land on one number. Measured at
    N = 192, V = 1 V on the copper dipole: 7.7180e-5 W (razor GL), 7.7170e-5
    (razor NEC-5 lane), 7.7550e-5 (BSpline), 7.7544e-5 (SG) — 0.48 % apart,
    against a 0.15 % spread in |I_in| between the formulations, which is
    where most of it comes from (the readout is quadratic in the current).
    """
    got = {}
    for label, cls, extra in (
        ("razor", RazorSolver, {}),
        ("bspline", BSplineSolver, {"degree": 2}),
        ("sin-galerkin", SinusoidalGalerkinSolver, {}),
    ):
        sim = cls(
            wires=DIPOLE,
            n_per_edge_per_wire=[[192]],
            wire_radius=RAD,
            wavelength=WL,
            wire_conductivity=SIGMA,
            **extra,
        )
        _z_in, coeffs = sim.compute_impedance()
        got[label] = sim.wire_loss_power(coeffs)[0]
    for ref in ("bspline", "sin-galerkin"):
        rel = abs(got["razor"] - got[ref]) / got[ref]
        assert rel < 0.01, f"{got}"


@pytest.mark.slow
def test_the_loaded_impedance_converges_toward_the_references():
    """The gap that loading does not widen still CLOSES with N.

    Belt and braces on the test above: a difference-of-differences gate
    passes trivially if both columns are garbage in the same way. Here the
    loaded absolute gap against BSpline must fall down the ladder, which it
    does at razor's own O(1/N) rate — measured on the inverted-V, GL lane:
    2.4069 → 1.3376 → 0.7829 → 0.4907 Ω unloaded and 1.5280 → 0.8398 →
    0.4897 → 0.3083 Ω with the trap.
    """
    sim = _sim(DIPOLE, [[24]], wire_conductivity=SIGMA)
    assert sim._loading_active
    seq = []
    for n in LADDER:
        z_r, _ = _z(
            INVVEE,
            [[n // 2, n // 2]],
            lumped_loads=[(0, _IV_ARM * 0.5, IV_TRAP)],
        )
        z_b = _ref_terminated(
            BSplineSolver,
            {"degree": 2},
            INVVEE,
            [[n // 2, n // 2]],
            _IV_ARM * 0.5,
            IV_TRAP,
        )
        seq.append(abs(z_r - z_b))
    for a, b in zip(seq, seq[1:]):
        assert b < a, f"loaded cross-formulation gap is not closing: {seq}"
    assert seq[-1] < 0.5, seq


# --------------------------------------------------------------------------
# the term, read straight off the matrix
# --------------------------------------------------------------------------
def test_the_lumped_stamp_is_one_diagonal_entry():
    """A lumped load touches G[p, p] and nothing else.

    The delta case of L[m, n] = ∫_{P_m} Z_s Λ_n dl, read directly: only P_p
    contains knot p, and Λ_n(l_p) = δ_np, so the whole term is one number
    on the diagonal. Anything else — a neighbour smear, a row scale — would
    show up as a second nonzero here, and would break gate 2 next.
    """
    sim = _sim(DIPOLE, [[16]])
    geom = sim._build_geometry()
    Z0 = sim._assemble_Z(geom, sim.k)
    loaded = _sim(DIPOLE, [[16]], lumped_loads=[(0, DIP_HALF, ZL)])
    g2 = loaded._build_geometry()
    Z1 = loaded._assemble_Z(g2, loaded.k)
    diff = Z1 - Z0
    p = loaded._feed_basis_indices(g2)[0]
    assert diff[p, p] == ZL
    diff[p, p] = 0.0
    assert np.array_equal(diff, np.zeros_like(diff))


def test_the_grounded_tent_loads_its_real_wing_only():
    """The contact tent's image wing carries no conductor, so no loading.

    `_loading_stencil` drops σ = 0 entries, which is the grounded tent's
    side-A (image) wing. The consequence is checkable without a solve, in
    the row sums of a uniformly meshed monopole:

        grounded row   h/2   — its testing path is the real half only
        interior rows  h     — a whole path, both halves
        topmost row    7h/8  — a whole path, but the free end above it
                               carries no basis to hold the missing h/8

    and the grounded COLUMN is the mirror statement: only the contact
    segment appears in it, because the image wing is not metal.
    """
    sim = _sim(MONOPOLE, [[12]], ground_z=0.0, wire_conductivity=SIGMA)
    geom = sim._build_geometry()
    L = _dense_stencil(sim, geom)
    n_b = geom["n_basis_total"]
    h = MONO_LEN / 12
    grounded = geom["grounded_bases"]
    assert grounded.size == 1
    g = int(grounded[0])
    rows = L.sum(axis=1)
    assert np.isclose(rows[g], 0.5 * h, rtol=1e-13)
    # The grounded tent is basis n_interior..; the free-end neighbour is the
    # LAST interior tent, whose upper wing sits on the terminal segment.
    top = geom["n_basis_interior"] - 1
    assert np.isclose(rows[top], 0.875 * h, rtol=1e-13)
    mid = [i for i in range(n_b) if i not in (g, top)]
    assert np.allclose(rows[mid], h, rtol=1e-13)
    # The grounded column reaches only the contact segment's two tents.
    assert np.count_nonzero(L[:, g]) == 2


def test_the_swept_and_single_fills_agree_bit_for_bit_with_loading():
    """`_assemble_Z` and the prepare/replay pair build the same loaded
    matrix, to the bit — the loading term takes the same path through both,
    which is what lets the swept loop reuse one stencil."""
    sim = _sim(DIPOLE, [[16]], wire_conductivity=SIGMA, lumped_loads=[(0, 1.0, ZL)])
    geom = sim._build_geometry()
    one = sim._assemble_Z(geom, sim.k)
    prepared = sim._assemble_Z_prepare(geom)
    two = sim._assemble_Z_from_prepared(geom, prepared, sim.k, sim.omega)
    assert np.array_equal(one, two)
    # and the replayed matrix at a second k is the one a fresh solver builds
    k2 = sim.k * 1.15
    three = sim._assemble_Z_from_prepared(geom, prepared, k2, sim.c * k2)
    fresh = _sim(
        DIPOLE,
        [[16]],
        wavelength=2.0 * np.pi / k2,
        wire_conductivity=SIGMA,
        lumped_loads=[(0, 1.0, ZL)],
    )
    g4 = fresh._build_geometry()
    assert np.allclose(three, fresh._assemble_Z(g4, fresh.k), rtol=1e-13, atol=0.0)
    assert scipy.linalg.norm(three - one) > 0.0
