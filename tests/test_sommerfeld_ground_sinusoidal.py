"""SinusoidalSolver ground_model="sommerfeld" tests
(docs/sommerfeld-everywhere-plan.md Phase 2).

Golden gates use the nec2c-captured gn 2 values ("finite" key), same
oracle discipline as test_sommerfeld_ground.py. Because this solver
shares NEC's three-term basis, its residual floor against nec2c is
~0.1 ohm — 20x sharper than bspline's ~1.5-2.4 ohm cross-solver floor,
which is exactly why this suite is the strongest validation of the
Sommerfeld engine itself. Measured residuals at capture (2026-07-07),
full 39-case matrix, |Z_somm - Z_gn2|: dipole max 0.10 ohm at heights
0.02-0.35 wl and ~0.91 at 0.5 wl (image-ray distances there run past
NEC's own 1-wavelength grid edge, where nec2c switches to asymptotic
approximations — the drift is consistent with the oracle coarsening,
not with the heights where our machinery is stressed); inverted_l max
0.14; yagi max 0.21. Gates are set ~1.5x above those measurements.

The remainder-tensor SIGN is pinned empirically by
`test_matches_bspline_sommerfeld_cross_solver` on a 0.05-wl dipole,
where the remainder moves Z by ~20 ohm — a sign error is a ~2x miss,
far outside the ~1.5 ohm cross-solver floor asserted there.
"""

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _ground_refl
from momwire.sinusoidal import SinusoidalSolver

from fixtures_refl_coef_geoms import GEOMS
from golden_refl_coef_ground import GOLDEN


def _solver(name, frac, cls=SinusoidalSolver, **overrides):
    kw = dict(GEOMS[(name, frac)])
    kw["ground_z"] = 0.0
    kw.update(overrides)
    return cls(**kw)


def _solve(name, frac, **overrides):
    z, _ = _solver(name, frac, **overrides).compute_impedance()
    return z


SOMM = {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"}


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


def test_ground_model_validation():
    kw = dict(GEOMS[("dipole", 0.2)])
    with pytest.raises(ValueError, match="ground_model"):
        SinusoidalSolver(
            **kw, ground_z=0.0, ground_eps=(10, 0.002), ground_model="nope"
        )
    with pytest.raises(ValueError, match="requires ground_eps"):
        SinusoidalSolver(**kw, ground_z=0.0, ground_model="sommerfeld")
    with pytest.raises(ValueError, match="requires ground_z"):
        SinusoidalSolver(**kw, ground_eps=(10, 0.002), ground_model="sommerfeld")


def test_wires_below_ground_rejected():
    kw = dict(GEOMS[("dipole", 0.2)])
    z_top = max(p[2] for wire in kw["wires"] for p in wire)
    s = SinusoidalSolver(**kw, ground_z=z_top + 1.0, **SOMM)
    with pytest.raises(ValueError, match="strictly above"):
        s.compute_impedance()


def test_default_model_is_refl_coef():
    """ground_model defaults to refl-coef: explicit and default solves are
    bit-identical, so pre-existing behavior is unchanged."""
    z_default = _solve("dipole", 0.2, ground_eps=(10.0, 0.002))
    z_explicit = _solve(
        "dipole", 0.2, ground_eps=(10.0, 0.002), ground_model="refl-coef"
    )
    assert z_default == z_explicit


# ---------------------------------------------------------------------------
# Exact limits
# ---------------------------------------------------------------------------


def test_free_space_limit():
    """eps -> 1: C2 = 0 and the remainder integrands vanish identically,
    so the grounded solve reproduces the no-ground solve."""
    kw = dict(GEOMS[("dipole", 0.2)])
    z_free, _ = SinusoidalSolver(**kw).compute_impedance()
    z_g = _solve("dipole", 0.2, ground_eps=1.0 + 0.0j, ground_model="sommerfeld")
    assert abs(z_g - z_free) / abs(z_free) < 1e-9


def test_pec_limit_collapses_to_image():
    """eps -> 1e16: C2 -> 1 and the remainder scales away (~1/sqrt(eps)),
    reproducing the PEC-image solve."""
    z_pec = _solve("dipole", 0.1)  # ground_z set, no ground_eps -> PEC image
    z_g = _solve("dipole", 0.1, ground_eps=1e16 + 0.0j, ground_model="sommerfeld")
    assert abs(z_g - z_pec) / abs(z_pec) < 1e-5


def test_tuple_and_complex_eps_equivalent():
    s = _solver("dipole", 0.2, **SOMM)
    eps_c = _ground_refl.eps_tilde((10.0, 0.002), s.omega, s.eps)
    z_t = _solve("dipole", 0.2, **SOMM)
    z_c = _solve("dipole", 0.2, ground_eps=eps_c, ground_model="sommerfeld")
    assert abs(z_t - z_c) / abs(z_t) < 1e-12


# ---------------------------------------------------------------------------
# Numerics contracts
# ---------------------------------------------------------------------------


def test_swept_matches_single_k():
    s = _solver("dipole", 0.2, **SOMM)
    k0 = s.k
    ks = np.array([0.97 * k0, k0, 1.03 * k0])
    z_swept = s.compute_impedance_swept(ks)
    z_single = _solve("dipole", 0.2, **SOMM)
    assert abs(z_swept[1] - z_single) / abs(z_single) < 1e-10
    # the flanking entries used per-k eps(omega) and grids: they must
    # differ from the center (guards a frozen-omega bug)
    assert abs(z_swept[0] - z_single) > 1e-3


def test_y_matrix_consistent_with_impedance():
    s = _solver("dipole", 0.2, **SOMM)
    y = s.compute_y_matrix()
    z = _solve("dipole", 0.2, **SOMM)
    assert abs(1.0 / y[0, 0] - z) / abs(z) < 1e-9


def test_quadrature_order_converged():
    """The remainder kernel is smooth (image point below the plane):
    n_qp_sommerfeld=3 vs 5 must agree far inside the physics residual."""
    z3 = _solve("dipole", 0.05, **SOMM, n_qp_sommerfeld=3)
    z5 = _solve("dipole", 0.05, **SOMM, n_qp_sommerfeld=5)
    assert abs(z3 - z5) / abs(z3) < 1e-3


def test_matches_bspline_sommerfeld_cross_solver():
    """SIGN PIN for the remainder tensor (see module docstring): at
    0.05 wl the remainder moves Z by ~20 ohm; agreement with the
    independently-validated bspline sommerfeld solve inside the ~1.5 ohm
    cross-solver floor rules out a sign or normalization error."""
    z_sin = _solve("dipole", 0.05, **SOMM)
    z_bsp, _ = _solver("dipole", 0.05, cls=BSplineSolver, **SOMM).compute_impedance()
    assert abs(z_sin - z_bsp) < 2.5
    # ... and the gap being bridged is real: refl-coef sits ~20 ohm away
    z_refl = _solve("dipole", 0.05, ground_eps=(10.0, 0.002))
    assert abs(z_refl - z_sin) > 15.0


# ---------------------------------------------------------------------------
# Golden gn 2 gates (nec2c oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frac,ground",
    [
        (0.02, (10.0, 0.002)),
        (0.02, (3.0, 0.001)),
        (0.05, (10.0, 0.002)),
        (0.2, (13.0, 0.005)),
    ],
)
def test_dipole_tracks_gn2(frac, ground):
    """Measured max residual 0.10 ohm across the 15 dipole cases at
    0.02-0.35 wl — gate at 0.2 (the same-basis floor)."""
    gn2 = GOLDEN[("dipole", frac, *ground)]["finite"]
    z = _solve("dipole", frac, ground_eps=ground, ground_model="sommerfeld")
    assert abs(z - gn2) < 0.2


def test_dipole_tracks_gn2_half_wave_height():
    """0.5 wl: measured ~0.91 ohm — the image ray runs past nec2c's own
    1-wavelength grid edge (see module docstring). Gate at 1.2."""
    gn2 = GOLDEN[("dipole", 0.5, 10.0, 0.002)]["finite"]
    z = _solve("dipole", 0.5, **SOMM)
    assert abs(z - gn2) < 1.2


@pytest.mark.parametrize("frac", [0.02, 0.2])
def test_inverted_l_tracks_gn2(frac):
    """Junction + vertical-current geometry (exercises the I_z^V/I_rho^V
    surfaces); measured max 0.14 ohm — gate at 0.3."""
    gn2 = GOLDEN[("inverted_l", frac, 10.0, 0.002)]["finite"]
    z = _solve(
        "inverted_l", frac, ground_eps=(10.0, 0.002), ground_model="sommerfeld"
    )
    assert abs(z - gn2) < 0.3


def test_yagi_tracks_gn2_large_r1():
    """>1.2-wavelength boom: image-ray distances past NEC's 1-wavelength
    grid edge exercise the geometry-sized grid; measured max 0.21 ohm —
    gate at 0.4."""
    gn2 = GOLDEN[("yagi", 0.2, 10.0, 0.002)]["finite"]
    z = _solve("yagi", 0.2, ground_eps=(10.0, 0.002), ground_model="sommerfeld")
    assert abs(z - gn2) < 0.4


def test_beats_refl_coef_below_010():
    """The point of the exercise: at 0.02 wl the refl-coef model is
    >20 ohm from gn 2; sommerfeld must recover >80% of that gap."""
    gn2 = GOLDEN[("dipole", 0.02, 10.0, 0.002)]["finite"]
    z_somm = _solve("dipole", 0.02, **SOMM)
    z_refl = _solve("dipole", 0.02, ground_eps=(10.0, 0.002))
    assert abs(z_somm - gn2) < 0.2 * abs(z_refl - gn2)
    assert abs(z_refl - gn2) > 20.0  # the gap being closed is real
