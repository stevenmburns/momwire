"""momwire#1362: the sinusoidal Galerkin fill's mixed-potential form under
the extended kernel.

The extended kernel is NEC Eq 89's coaxial factor on G (`_bspline_kernels.
_ek_factor`, a function of R alone) for a pair whose `_ek_axis_groups` labels
match. Along a test segment the pair's label is fixed, so the integration by
parts that turns the direct form into the mixed-potential one stays exact for
every deck whose bases each lie on one coaxial group (`_mp_ek_exact`); a basis
that turns a corner keeps the direct fill.

The oracle is the direct fill with the extended kernel on the same deck. Its
EK delta is a quadrature (`_folded_ek_delta_fields`) whose default panel
count is under-resolved by up to ~4e-5 ohm on a 21-segment dipole, so the
deck gate compares the EK EFFECT (Z_EK − Z_reduced) of the two fills with the
direct fill's delta refined (`_N_PANEL_EK_DELTA_NEAR` x4): measured agreement
1.6e-9 to 2.3e-7 ohm on effects of 4e-3 to 0.4 ohm, while the mixed-potential
effect moves by at most 1.4e-10 ohm under its own refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire._sinusoidal_mp as mp
import momwire.sinusoidal as sn
import momwire.sinusoidal_galerkin as sgm
from momwire._bspline_kernels import _ek_factor
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

C0 = 299792458.0
WL = C0 / 7e6
K0 = 2 * np.pi / WL
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")


def _dipole(n=21, z=10.0, L=20.0, a=0.001, vertical=False, **kw):
    if vertical:
        w = np.array([(0, 0, z - L / 2), (0, 0, z + L / 2)])
    else:
        w = np.array([(-L / 2, 0.0, z), (L / 2, 0.0, z)])
    return dict(
        wires=[w],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, L / 2, 1 + 0j)],
        wavelength=WL,
        wire_radius=a,
        **kw,
    )


DECKS = {
    "dipole": _dipole(),
    "dipole-fat": _dipole(n=41, a=0.05),
    # A vertical dipole's image is coaxial with it: the image block extends.
    "vdipole-pec": _dipole(vertical=True, z=12.0, ground_z=0.0),
    "vdipole-somm": _dipole(vertical=True, z=12.0, **SOMM),
    "hdipole-somm": _dipole(z=5.0, **SOMM),
    # A collinear junction: one group across the node.
    "collinear": dict(
        wires=[
            np.array([(-10.0, 0, 10.0), (0.0, 0, 10.0)]),
            np.array([(0.0, 0, 10.0), (10.0, 0, 10.0)]),
        ],
        n_per_edge_per_wire=[[11], [10]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=0.001,
    ),
}


def _z(deck, fill, ek):
    s = SinusoidalGalerkinSolver(**deck, fill=fill, extended_kernel=ek)
    return complex(np.atleast_1d(s.compute_impedance()[0])[0])


def _random_segments(rng, n, spread=5.0):
    c = rng.uniform(-spread, spread, size=(n, 3))
    t = rng.normal(size=(n, 3))
    t /= np.linalg.norm(t, axis=1)[:, None]
    h = rng.uniform(0.2, 0.6, size=n)
    return c, t, h


# ----------------------------------------------------------------------
# the kernels against their numpy references
# ----------------------------------------------------------------------


@pytest.mark.skipif(not mp._HAVE_SIN_TIERED_EK, reason="needs the accelerator")
def test_tiered_ek_kernel_matches_numpy_and_moves_only_the_matched_pairs():
    rng = np.random.default_rng(1362)
    c, t, h = _random_segments(rng, 9, spread=1.5)
    sl, sr = mp.segment_ends(c, t, h)
    gi = rng.integers(0, 3, size=9)
    gj = rng.integers(0, 3, size=9)
    a = 0.02
    ladder = ((2.0, 4), (6.0, 2))
    ek = (gi, gj, a)
    got = mp.pair_moments_tiered(sl, sr, sl, sr, a * a, K0, 8, ladder, ek=ek)
    ref = mp._pair_moments_tiered_numpy(sl, sr, sl, sr, a * a, K0, 8, ladder, ek=ek)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()
    red = mp.pair_moments_tiered(sl, sr, sl, sr, a * a, K0, 8, ladder)
    elig = gi[:, None] == gj[None, :]
    moved = np.abs(got - red).max(axis=(0, 1)) > 0
    assert np.array_equal(moved, elig)
    # An unmatched pair is the reduced kernel to the bit.
    assert np.array_equal(got[:, :, ~elig], red[:, :, ~elig])


@pytest.mark.skipif(not mp._HAVE_SIN_PARALLEL, reason="needs the accelerator")
def test_parallel_ek_reduction_matches_numpy():
    rng = np.random.default_rng(7)
    n = 12
    t = np.tile([0.0, 0.0, 1.0], (n, 1))
    c_i = np.zeros((n, 3))
    c_i[:, 2] = rng.uniform(-1.0, 1.0, n)
    c_j = np.zeros((n, 3))
    c_j[:, 2] = rng.uniform(-1.0, 1.0, n)
    h_i = rng.uniform(0.3, 0.7, n)
    h_j = rng.uniform(0.3, 0.7, n)
    a2 = np.full(n, 0.01**2)
    a_ek = np.where(np.arange(n) % 3 == 0, 0.0, 0.01)
    got = mp.parallel_pair_moments(c_i, t, h_i, c_j, t, h_j, a2, K0, a_ek=a_ek)
    ref = mp._parallel_pair_moments_numpy(c_i, t, h_i, c_j, t, h_j, a2, K0, a_ek=a_ek)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()
    red = mp.parallel_pair_moments(c_i, t, h_i, c_j, t, h_j, a2, K0)
    off = a_ek == 0.0
    assert np.array_equal(got[:, :, off], red[:, :, off])
    assert (np.abs(got[:, :, ~off] - red[:, :, ~off]).max(axis=(0, 1)) > 0).all()


def test_the_factor_is_one_at_zero_radius():
    R = np.linspace(0.01, 3.0, 7)
    assert np.array_equal(_ek_factor(R, 0.0, K0), np.ones_like(R, dtype=complex))


# ----------------------------------------------------------------------
# the route, and the decks
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_mixed_potential_fill_serves_the_extended_kernel(monkeypatch, name):
    """Served, and the extended moments actually ran (a spy on the tiered
    kernel sees `ek`)."""
    seen = []
    real = mp.pair_moments_tiered

    def spy(*a, **kw):
        seen.append(kw.get("ek") is not None)
        return real(*a, **kw)

    monkeypatch.setattr(mp, "pair_moments_tiered", spy)
    s = SinusoidalGalerkinSolver(**DECKS[name], extended_kernel=True)
    assert s.fill == "mixed-potential"
    assert s._mp_serves(s._build_geometry())
    s.compute_impedance()
    assert any(seen), seen


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_ek_effect_agrees_with_the_direct_form(monkeypatch, name):
    """Z_EK − Z_reduced, mixed-potential against direct with the direct EK
    delta refined: the extended kernel is the same object on both sides."""
    deck = DECKS[name]
    eff_mp = _z(deck, "mixed-potential", True) - _z(deck, "mixed-potential", False)
    monkeypatch.setattr(sn, "_N_PANEL_EK_DELTA_NEAR", 4 * sn._N_PANEL_EK_DELTA_NEAR)
    monkeypatch.setattr(sgm, "_N_PANEL_EK_DELTA_NEAR", 4 * sgm._N_PANEL_EK_DELTA_NEAR)
    eff_dir = _z(deck, "direct", True) - _z(deck, "direct", False)
    assert abs(eff_dir) > 1e-3, eff_dir  # the extended kernel moved Z
    assert abs(eff_mp - eff_dir) <= 1e-6 + 1e-6 * abs(eff_dir), (eff_mp, eff_dir)


@pytest.mark.parametrize("name", ["dipole", "vdipole-pec", "collinear"])
def test_the_ek_matrix_is_reciprocal(name):
    s = SinusoidalGalerkinSolver(**DECKS[name], extended_kernel=True)
    geom = s._build_geometry()
    with s._operating_medium(geom) as medium:
        G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    assert np.abs(G - G.T).max() <= 1e-9 * np.abs(G).max()
