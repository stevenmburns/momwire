"""momwire#980 step A: the sinusoidal fills at an in-medium (complex) wavenumber.

A buried segment lives at k_m = k₀·√ε̃ with Im k_m ≤ 0. `SinusoidalGalerkinSolver`
has no buried fill yet (that is #980's build); what this file pins is the
PRECONDITION the design spike measured: with the real-only casts gone, the
closed-form sinusoidal fields, the NEC three-term coefficient recipe and the
Galerkin assembly are complex-clean, and the solve lands on the same answer as
`BSplineSolver`'s infinite-medium block — the k_m direct fill assembled with
ε̃_m, the oracle `test_gu5_8_a` uses for the deep-burial limit.

The medium is entered the way the spike did it, by overriding `k` and `eta`
on a constructed solver. That is a test seam, not an API: the buried serve
(#980 step D1) will set both per pair class from the ground spec. What must
hold whichever way they arrive is what is gated here.

Gates (soil A, 7 MHz, radius 1 mm; the spike's numbers on 2026-09-08):

    vertical 1 m, N=41     in-medium gap 1.25e-7   free-space gap 7.0e-9
    T junction, 41+21      in-medium gap 1.65e-4   free-space gap 1.63e-4

The in-medium gap on a junction deck EQUALS the free-space cross-basis gap:
what is left is the two bases disagreeing, not a complex-k defect. So the
junction gate is relative to the free-space gap on the same mesh, and the
straight-wire gate is absolute.
"""

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver, _ground_refl
from momwire import _sommerfeld_below

SOIL_A = (13.0, 0.005)
C0 = 299792458.0
WL7 = C0 / 7e6

VERTICAL = (
    [np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])],
    [[41]],
    [(0, 0.5, 1 + 0j)],
)
T_JUNCTION = (
    [
        np.array([(-1.0, 0.0, -0.5), (1.0, 0.0, -0.5)]),
        np.array([(0.0, 0.0, -0.5), (0.0, 0.0, -1.5)]),
    ],
    [[41], [21]],
    [(0, 0.5, 1 + 0j)],
)


def _medium(solver):
    eps_t = _ground_refl.eps_tilde(SOIL_A, solver.omega, solver.eps)
    return eps_t, _sommerfeld_below.k_medium(eps_t, solver.k)


def bspline_infinite_medium(wires, n_per_edge, feeds):
    """`test_gu5_8_a`'s oracle: the k_m direct block at ε̃_m, no interface."""
    ref = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feeds=feeds,
        wavelength=WL7,
        wire_radius=0.001,
    )
    eps_t, k_m = _medium(ref)
    geom = ref._build_geometry()
    supp_seg, polys, kcl_A, wk, wbg = ref._build_basis_polynomials(geom)
    z_op = ref._assemble_Z(
        ref._build_J_blocks(geom, k_m), supp_seg, polys, geom, eps=ref.eps * eps_t
    )
    v, port_vectors, _x, volts, kcl_con = ref._feed_drive_and_readout(
        geom, wk, wbg, supp_seg.shape[0], kcl_A
    )
    c = ref._solve_with_kcl(ref._apply_loading(z_op), v, kcl_con)
    z_inf = volts[0] / (port_vectors[0] @ c[: port_vectors[0].shape[0]])
    return z_inf, ref.compute_impedance()[0]


def sg_in_medium(wires, n_per_edge, feeds):
    """SG at k_m and η_m, with every complex→real cast promoted to an error."""
    s = SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feeds=feeds,
        wavelength=WL7,
        wire_radius=0.001,
    )
    z_free = s.compute_impedance()[0]
    eps_t, k_m = _medium(s)
    s.k = k_m
    s.eta = np.sqrt(s.mu / (s.eps * eps_t))
    with warnings.catch_warnings():
        # A `dtype=float` cast on a complex quantity is the silent-truncation
        # class this file exists to keep out; make it fail, not warn.
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        z_med = s.compute_impedance()[0]
    return z_med, z_free


@pytest.mark.parametrize(
    "deck, abs_gate, rel_to_free_gate",
    [
        pytest.param(VERTICAL, 1e-6, None, id="vertical-41"),
        pytest.param(T_JUNCTION, None, 3.0, id="t-junction-41+21"),
    ],
)
def test_sg_at_k_m_lands_on_the_infinite_medium_dipole(
    deck, abs_gate, rel_to_free_gate, record_property
):
    wires, n_per_edge, feeds = deck
    z_inf, z_free_bs = bspline_infinite_medium(wires, n_per_edge, feeds)
    z_med, z_free_sg = sg_in_medium(wires, n_per_edge, feeds)

    gap_med = abs(z_med - z_inf) / abs(z_inf)
    gap_free = abs(z_free_sg - z_free_bs) / abs(z_free_bs)
    record_property("z_infinite_medium_bspline", f"{z_inf:.5f}")
    record_property("z_infinite_medium_sg", f"{z_med:.5f}")
    record_property("gap_in_medium", float(gap_med))
    record_property("gap_free_space", float(gap_free))

    # The medium is a real load: the free-space dipole is electrically tiny
    # (R ~ 0.1 ohm) and the lossy medium's conduction current is what the
    # feed sees. Guards against a solve that quietly ran at the real k.
    assert z_med.real > 100.0, f"in-medium R {z_med.real:.3f} reads like free space"
    if abs_gate is not None:
        assert gap_med < abs_gate, f"in-medium gap {gap_med:.3e}"
    if rel_to_free_gate is not None:
        assert gap_med < rel_to_free_gate * gap_free + 1e-6, (
            f"in-medium gap {gap_med:.3e} vs free-space cross-basis gap {gap_free:.3e}"
        )


def test_the_well_scaled_identities_return_float64_for_float_input():
    """The real path's bytes: dropping the `dtype=float` cast must not promote
    a float input. float in, float64 out, same values."""
    from momwire.sinusoidal import _asinh_minus_arg, _recip_sin_gap, _sin_minus_arg

    u = np.array([1e-4, 0.05, 0.3, 2.0])
    for f in (_sin_minus_arg, _recip_sin_gap, _asinh_minus_arg):
        out = f(u)
        assert out.dtype == np.float64, (f.__name__, out.dtype)
        assert f(1e-3).dtype == np.float64, f.__name__
    # and complex in, complex out, agreeing with the real branch on the real axis
    uc = u.astype(np.complex128)
    for f in (_sin_minus_arg, _recip_sin_gap, _asinh_minus_arg):
        outc = f(uc)
        assert np.iscomplexobj(outc), f.__name__
        np.testing.assert_allclose(outc.real, f(u), rtol=1e-14, atol=0.0)
        assert np.all(outc.imag == 0.0), f.__name__
