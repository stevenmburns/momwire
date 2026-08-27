"""momwire#680 U2 — the C++ twin of the designed near-interface walk.

`_near_interface_accel.near_interface_six_batch` is a WALK port of
`six_point` (shared walk + limit, the house twin rule): the same head
detour, the same real-axis mid, the same rotated-ray tail with the same
panel-start rule, quiet rule and far-pair kill cap, on the very same
Gauss nodes — so parity is rounding class, and the gates here are
RELATIVE at 1e-12, never bit (house rule: no cross-build bit equality;
the transcendental libraries and Gauss dot products differ in the last
bits).

The ledger spans every structural branch of the walk (the restart's
pinned set):

  * the corner (ρ = a, z = z′ = 0) — reached by design, not clamp;
  * the ε̃ = 1 identity points (test_g524_3's) — the accel must ALSO
    satisfy the free-space collapse identity at ≤ 1e-12, not merely
    match the numpy walk;
  * a high-σ far pair — the exact-underflow quiet path (the whole tail
    underflows to 0.0; that is the tail being zero, not a stall);
  * a ρ = 0 pair — the single up-ray J₀ = 1 path;
  * a kill-cap pair — s > 60/λ_top, the capped-extents branch.

The relative scale is the VECTOR scale max|ref| over the six components:
W ≡ 0 at ε̃ = 1 and a per-component scale would divide by an exact zero.

Parity gates run BEFORE integrated gates (the #568 lesson: an integrated
gate cannot see a conditioning defect); g524_4 and the crossgate
collapses re-run through the accel path separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _ground_refl, _near_interface as ni

C0 = 299792458.0
F7 = 7e6
WL7 = C0 / F7
K7 = 2.0 * np.pi / WL7
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
A_WIRE = 0.001

SOIL_A = _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0)
HIGH_SIGMA = _ground_refl.eps_tilde((13.0, 5.0), OM7, EPS0)

# (label, eps_t, (rho, z, zp)) — one row per structural branch.
LEDGER = [
    ("corner", SOIL_A, (A_WIRE, 0.0, 0.0)),
    ("eps1-corner", 1.0, (A_WIRE, 0.0, 0.0)),
    ("eps1-generic", 1.0, (0.3, 0.2, -0.4)),
    ("eps1-rho0", 1.0, (0.0, 1.0, -0.5)),
    ("high-sigma-far", HIGH_SIGMA, (0.5, 3.0, -3.0)),
    ("rho0", SOIL_A, (0.0, 0.05, -0.03)),
    ("kill-cap", SOIL_A, (1.0, 8.0, -6.0)),
    ("generic", SOIL_A, (0.3, 0.5, -0.2)),
    ("tiny-s", SOIL_A, (1e-5, 1e-6, -1e-6)),
]

pytestmark = pytest.mark.skipif(
    not ni._HAVE_NEAR_INTERFACE_ACCEL,
    reason="near-interface accel not built (pure-Python install)",
)


def _accel_six(eps_t, points, rtol=1e-10):
    k_p = float(K7)
    k_m = ni.k_medium(complex(eps_t), k_p)
    tri = np.asarray(points, dtype=float).reshape(-1, 3)
    return ni._nia.near_interface_six_batch(
        k_p,
        k_m,
        np.ascontiguousarray(tri[:, 0]),
        np.ascontiguousarray(tri[:, 1]),
        np.ascontiguousarray(tri[:, 2]),
        rtol,
        ni._LAM_MULT,
        ni._ADAPT_DEPTH,
        ni._DETOUR,
        ni._GX,
        ni._GW,
    )


@pytest.mark.parametrize("label,eps_t,pt", LEDGER, ids=[r[0] for r in LEDGER])
def test_g680_1_pointwise_parity(label, eps_t, pt):
    """C++ vs numpy on the pinned ledger, 1e-12 relative on the vector
    scale. Measured at first light: worst 3.9e-16 — rounding class, four
    decades inside the gate."""
    ref = ni.six_point(eps_t, K7, *pt)
    got = _accel_six(eps_t, [pt])[0]
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    rel = float(np.max(np.abs(got - ref))) / scale
    assert rel <= 1e-12, f"{label}: accel departs the reference walk at {rel:.3e}"


def test_g680_2_eps1_identity_through_the_accel():
    """test_g524_3's collapse identity, ON the accel values: at ε̃ = 1,
    U_T = k²V_T = e^{−jkR}/R exactly and W ≡ ∂zW ≡ 0. The twin must own
    the identity itself, not merely track the numpy walk."""
    pts = [(A_WIRE, 0.0, 0.0), (0.3, 0.2, -0.4), (0.0, 1.0, -0.5)]
    vals = _accel_six(1.0, pts, rtol=1e-12)
    for (rho, z, zp), six in zip(pts, vals):
        R = np.hypot(rho, z - zp)
        g = np.exp(-1j * K7 * R) / R
        assert abs(six[0] - g) <= 1e-12 * abs(g)
        assert abs(K7 * K7 * six[1] - g) <= 1e-12 * abs(g)
        assert abs(six[2]) <= 1e-12 * abs(g)
        assert abs(six[3]) <= 1e-12 * abs(g) / max(R, A_WIRE)


def test_g680_3_designed_tables_routes_and_matches():
    """`designed_tables` through the accel vs forced-numpy, on a mesh with
    IEEE-exact duplicate triples: same dedup, same scatter, 1e-12 relative
    — and the duplicated columns are still the SAME floats (the U1 memo
    lives in Python on both paths)."""
    rho = np.array([[0.3, 0.5, 0.3], [0.3, 0.5, 0.3]])
    z = np.array([[0.2], [0.4]]) * np.ones((1, 3))
    zp = -0.15
    got = ni.designed_tables(SOIL_A, K7, rho, z, zp, rtol=1e-10)
    try:
        ni._FORCE_NUMPY = True
        ref = ni.designed_tables(SOIL_A, K7, rho, z, zp, rtol=1e-10)
    finally:
        ni._FORCE_NUMPY = False
    for key in ni.KEYS:
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        rel = float(np.max(np.abs(got[key] - ref[key]))) / scale
        assert rel <= 1e-12, (key, rel)
        assert np.array_equal(got[key][:, 0], got[key][:, 2])


def test_g680_4_domain_raises_survive_the_accel():
    """The walk's contract raises hold on the batch path: z < 0, zp > 0
    and R = 0 refuse rather than evaluate."""
    with pytest.raises(ValueError):
        _accel_six(SOIL_A, [(0.1, -0.2, -0.3)])
    with pytest.raises(ValueError):
        _accel_six(SOIL_A, [(0.1, 0.2, 0.3)])
    with pytest.raises(ValueError):
        _accel_six(SOIL_A, [(0.0, 0.0, 0.0)])


def test_g680_5_high_sigma_far_pair_is_quiet_zero_class():
    """The underflow-quiet branch: at σ = 5 S/m the far pair's tail
    underflows to exact 0.0 on both machines — the answer is head+mid
    dominated and the twins still agree at rounding class (parity row
    high-sigma-far); this row additionally pins that the values are
    finite and small rather than NaN (xsf's underflow contract: Amos
    underflow returns 0.0, never NaN)."""
    got = _accel_six(HIGH_SIGMA, [(0.5, 3.0, -3.0)])[0]
    assert np.all(np.isfinite(got))
    assert float(np.max(np.abs(got))) < 1e-6
