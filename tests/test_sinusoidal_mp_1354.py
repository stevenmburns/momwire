"""momwire#1354: the sinusoidal Galerkin fill in mixed-potential form, on the
B-spline pair machinery (`_sinusoidal_mp`, `SinusoidalGalerkinSolver(fill=
"mixed-potential")`).

What is gated:

* the C++ tiered moment kernel and assembler against their numpy references
  (reassociation level);
* the parallel-pair one-dimensional reduction against a brute-force product
  rule on a pair far enough apart for the product rule to converge, and
  against itself under refinement on the self pair;
* the mixed-potential G against the direct-field G on small free-space, PEC
  and Sommerfeld decks: the two are one integration by parts apart, so they
  agree to quadrature error, and the impedance to better than the basis's
  own discretisation step;
* reciprocity of the mixed-potential G, which the direct form only reaches
  through its near-pair correction.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire._sinusoidal_mp as mp
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver, _solve_constrained

C0 = 299792458.0
WL = C0 / 7e6
K0 = 2 * np.pi / WL
A = 0.001


def _dipole(n=21, z=10.0, **kw):
    return dict(
        wires=[np.array([(-10.0, 0.0, z), (10.0, 0.0, z)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
        **kw,
    )


DECKS = {
    "dipole": _dipole(),
    "bent-L": dict(
        wires=[np.array([(0, 0, 0.0), (10.0, 0, 0.0), (10.0, 8.0, 0.0)])],
        n_per_edge_per_wire=[[11, 9]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
    ),
    "tee": dict(
        wires=[
            np.array([(-10.0, 0, 0.0), (0, 0, 0.0)]),
            np.array([(0, 0, 0.0), (10.0, 0, 0.0)]),
            np.array([(0, 0, 0.0), (0, 0, 7.0)]),
        ],
        n_per_edge_per_wire=[[11], [11], [7]],
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
    ),
    "dipole-pec": _dipole(z=2.0, ground_z=0.0),
    "dipole-pec-low": _dipole(z=0.3, ground_z=0.0),
    "dipole-somm": _dipole(
        z=2.0, ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ),
    # A junction basis wider than the remainder kernel's three wings, over
    # a Sommerfeld ground: the wing tables split it and fold its rows.
    "tee-somm": dict(
        wires=[
            np.array([(-10.0, 0, 3.0), (0, 0, 3.0)]),
            np.array([(0, 0, 3.0), (10.0, 0, 3.0)]),
            np.array([(0, 0, 3.0), (0, 0, 10.0)]),
            np.array([(0, 0, 3.0), (0, 7.0, 3.0)]),
        ],
        n_per_edge_per_wire=[[11], [11], [7], [7]],
        junctions=[[(0, "end"), (1, "start"), (2, "start"), (3, "start")]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=A,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    ),
}


def _G_and_Z(deck, fill):
    s = SinusoidalGalerkinSolver(**deck, fill=fill)
    geom = s._build_geometry()
    with s._operating_medium(geom) as medium:
        k = s.k
        G, sv = s._assemble_Z(geom, k, s._medium_eta(medium))
        U = s._drive_columns(geom, sv, k)
        V = s._port_voltages()
        alpha = _solve_constrained(
            G.copy(order="F"), U @ V, s._crossing_continuity_rows(geom, sv, k)
        )
        z = (V / s._port_currents(alpha, geom, sv, U))[0]
    return G, z


def _random_segments(rng, n, spread=5.0):
    c = rng.uniform(-spread, spread, size=(n, 3))
    t = rng.normal(size=(n, 3))
    t /= np.linalg.norm(t, axis=1)[:, None]
    h = rng.uniform(0.2, 0.6, size=n)
    return c, t, h


# ----------------------------------------------------------------------
# kernels against their numpy references
# ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [K0, K0 * (3.6 - 0.4j)])
def test_tiered_moment_kernel_matches_numpy(k):
    if not mp._HAVE_SIN_TIERED:
        pytest.skip("accelerator not built")
    rng = np.random.default_rng(1354)
    c_i, t_i, h_i = _random_segments(rng, 12)
    c_j, t_j, h_j = _random_segments(rng, 15)
    sl_i, sr_i = mp.segment_ends(c_i, t_i, h_i)
    sl_j, sr_j = mp.segment_ends(c_j, t_j, h_j)
    ladder = ((2.0, 8), (16.0, 4))
    J_cpp = mp.pair_moments_tiered(sl_i, sr_i, sl_j, sr_j, A * A, k, 32, ladder)
    J_np = mp._pair_moments_tiered_numpy(sl_i, sr_i, sl_j, sr_j, A * A, k, 32, ladder)
    assert J_cpp.shape == (3, 3, 12, 15)
    scale = np.abs(J_np).max()
    assert np.abs(J_cpp - J_np).max() / scale < 1e-13


def test_assembler_matches_numpy():
    if not mp._HAVE_SIN_ASSEMBLE:
        pytest.skip("accelerator not built")
    s = SinusoidalGalerkinSolver(**DECKS["tee"])
    geom = s._build_geometry()
    k = s.k
    sv = s._basis_coefs(geom, k)
    starts, jbasis, coef, dcoef = mp.basis_csr(sv, k)
    N = geom["n_segs"]
    rng = np.random.default_rng(7)
    J = rng.normal(size=(3, 3, N, N)) + 1j * rng.normal(size=(3, 3, N, N))
    t = geom["seg_tangents"]
    w_a = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    w_phi = rng.normal(size=(N, N)) + 0j
    rows = np.arange(N)
    out = {}
    for label, fn in (
        ("cpp", mp.assemble_window),
        ("np", mp._assemble_window_numpy),
    ):
        Z = np.zeros((N, N), dtype=np.complex128, order="F")
        fn(
            Z,
            J,
            rows,
            rows,
            starts,
            jbasis,
            coef,
            dcoef,
            t,
            t,
            1.5 - 0.2j,
            0.3 + 1j,
            w_a=w_a,
            w_phi=w_phi,
            scale=-0.7,
        )
        out[label] = Z
    assert np.abs(out["cpp"] - out["np"]).max() / np.abs(out["np"]).max() < 1e-13


# ----------------------------------------------------------------------
# the parallel-pair reduction
# ----------------------------------------------------------------------


def test_parallel_reduction_matches_product_rule_on_a_separated_pair():
    """Two collinear segments one segment length apart: the product rule
    converges there, so the reduction must meet it."""
    k = K0 * (3.6 - 0.4j)
    h = 0.5
    c_i = np.array([[0.0, 0.0, 0.0]])
    c_j = np.array([[2.0 * h, 0.0, 0.0]])
    t = np.array([[1.0, 0.0, 0.0]])
    J = mp.parallel_pair_moments(c_i, t, [h], c_j, -t, [h], A * A, k)
    sl_i, sr_i = mp.segment_ends(c_i, t, [h])
    sl_j, sr_j = mp.segment_ends(c_j, -t, [h])
    ref = mp.pair_moments_product(sl_i, sr_i, sl_j, sr_j, A * A, k, *mp._gl01(48))
    assert np.abs(J[:, :, 0] - ref[:, :, 0, 0]).max() / np.abs(ref).max() < 1e-11


def test_parallel_reduction_is_converged_on_the_self_pair():
    k = K0
    h = 0.5
    c = np.array([[0.0, 0.0, 0.0]])
    t = np.array([[0.0, 0.0, 1.0]])
    J = mp.parallel_pair_moments(c, t, [h], c, t, [h], A * A, k)
    J_fine = mp.parallel_pair_moments(c, t, [h], c, t, [h], A * A, k, n_t=64, n_xi=16)
    assert np.abs(J - J_fine).max() / np.abs(J_fine).max() < 1e-12
    # Symmetric in (p, q) on the self pair, and the const moment is the
    # static self term's size: log(h/a)-ish times h / (2 pi).
    assert np.abs(J[:, :, 0] - J[:, :, 0].T).max() / np.abs(J).max() < 1e-13
    assert 0.05 < abs(J[0, 0, 0]) / (h / (2 * np.pi)) < 20


# ----------------------------------------------------------------------
# the fill against the direct form
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DECKS))
def test_mixed_potential_G_agrees_with_the_direct_form(name):
    G_dir, z_dir = _G_and_Z(DECKS[name], "direct")
    G_mp, z_mp = _G_and_Z(DECKS[name], "mixed-potential")
    rel = np.abs(G_mp - G_dir).max() / np.abs(G_dir).max()
    assert rel < 2e-6, f"{name}: G differs by {rel:.2e}"
    assert abs(z_mp - z_dir) / abs(z_dir) < 1e-5, (name, z_mp, z_dir)


@pytest.mark.parametrize("name", sorted(DECKS))
def test_mixed_potential_G_is_reciprocal(name):
    G_mp, _ = _G_and_Z(DECKS[name], "mixed-potential")
    assert np.abs(G_mp - G_mp.T).max() / np.abs(G_mp).max() < 1e-9


def test_fill_kwarg_is_validated():
    with pytest.raises(ValueError, match="fill must be"):
        SinusoidalGalerkinSolver(**DECKS["dipole"], fill="potential")


def test_extended_kernel_and_contacts_keep_the_direct_fill():
    s = SinusoidalGalerkinSolver(
        **DECKS["dipole"], fill="mixed-potential", extended_kernel=True
    )
    assert not s._mp_serves(s._build_geometry())
    contact = _dipole(
        z=0.0, ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    )
    contact["wires"] = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])]
    contact["feeds"] = [(0, 0.5, 1 + 0j)]
    s = SinusoidalGalerkinSolver(**contact, fill="mixed-potential")
    assert not s._mp_serves(s._build_geometry())


# ----------------------------------------------------------------------
# crossing decks: the within-medium classes in mixed form plus completions
# ----------------------------------------------------------------------


def _crossing_cases():
    from test_crossing_serve_524 import crossing_deck, hub_deck

    return {"crossing1": crossing_deck(1), "hub4": hub_deck(n_radials=4)}


@pytest.mark.parametrize("name", ["crossing1", "hub4"])
def test_crossing_deck_mixed_form_agrees_with_the_direct_form(name):
    deck = _crossing_cases()[name]
    G_dir, z_dir = _G_and_Z(deck, "direct")
    G_mp, z_mp = _G_and_Z(deck, "mixed-potential")
    rel = np.abs(G_mp - G_dir).max() / np.abs(G_dir).max()
    assert rel < 5e-5, f"{name}: G differs by {rel:.2e}"
    assert abs(z_mp - z_dir) / abs(z_dir) < 1e-3, (name, z_mp, z_dir)
    assert np.abs(G_mp - G_mp.T).max() / np.abs(G_mp).max() < 1e-9


def test_crossing_completions_are_load_bearing(monkeypatch):
    """With bspline's self completions left off, the node's value-1 bases
    lose their by-parts content and the fill is O(1) wrong — which is the
    evidence that this route takes them (the direct form must not)."""
    deck = _crossing_cases()["crossing1"]
    G_dir, _ = _G_and_Z(deck, "direct")
    monkeypatch.setattr(mp, "COMPLETION_SIGN", 0.0)
    G_off, _ = _G_and_Z(deck, "mixed-potential")
    rel = np.abs(G_off - G_dir).max() / np.abs(G_dir).max()
    assert rel > 1e-2, rel
