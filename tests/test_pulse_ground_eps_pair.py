"""The pulse family takes the ground as the (eps_r, sigma) pair, like every
other family.

Found by driving antennaknobs' new Pulse tab (antennaknobs#1148) in the real
app: with the app's default finite ground the solve died with
``TypeError: complex() argument must be a string or a number, not tuple``.
`PulseSolver.__init__` coerced `ground_eps` with `complex()`, while
`BSplineSolver` and the sinusoidal family store the spec as given and let
`_ground_refl.eps_tilde` fold a pair at solve time. The capability row
declared refl-coef and sommerfeld served, so a consumer had no reason to
expect the spelling to matter — and the only spelling the engine passes is
the pair.

The gate pins the two spellings to the same answer: the pair (eps_r, sigma)
and the complex eps-tilde it folds to at this frequency must give the same Z
to the digit, on both pulse-family classes and under both finite grounds.
"""

import numpy as np
import pytest

from momwire import HarringtonSolver, PulseSolver
from momwire._ground_refl import eps_tilde

WAVELENGTH = 10.0
EPS0 = 8.8541878128e-12
C0 = 299_792_458.0


def _dipole(cls, ground_eps, ground_model="refl-coef"):
    """A horizontal dipole a quarter wave above the plane."""
    h = 0.25 * WAVELENGTH
    half = 0.24 * WAVELENGTH
    wires = [np.array([[-half, 0.0, h], [half, 0.0, h]])]
    kw = {} if ground_model == "refl-coef" else {"ground_model": ground_model}
    z, _c = cls(
        wires=wires,
        n_per_edge_per_wire=[[21]],
        wire_radius=0.001,
        wavelength=WAVELENGTH,
        feeds=[(0, 0.5, 1.0)],
        ground_z=0.0,
        ground_eps=ground_eps,
        **kw,
    ).compute_impedance()
    return complex(np.asarray(z).ravel()[0])


@pytest.mark.parametrize("cls", [PulseSolver, HarringtonSolver])
@pytest.mark.parametrize("ground_model", ["refl-coef", "sommerfeld"])
def test_the_pair_spelling_constructs_and_matches_the_complex_spelling(
    cls, ground_model
):
    pair = (13.0, 0.005)
    omega = 2.0 * np.pi * C0 / WAVELENGTH
    folded = eps_tilde(pair, omega, EPS0)
    assert folded.imag < 0  # the pair really folds to a lossy eps-tilde
    z_pair = _dipole(cls, pair, ground_model)  # raised TypeError before the fix
    z_cplx = _dipole(cls, folded, ground_model)
    # rel 1e-9, not bit equality: the solver folds the pair with its own
    # c0 / eps0 constants, and this test's fold uses CODATA values, so the
    # two eps-tildes differ in the 12th digit. The claim is that the pair
    # is accepted and folded, which a TypeError (the defect) or a wrong
    # fold (orders of magnitude) both fail.
    assert z_pair == pytest.approx(z_cplx, rel=1e-9), (
        cls.__name__,
        ground_model,
        z_pair,
        z_cplx,
    )
    # And the ground is doing something: a PEC image answers differently.
    z_pec = _dipole(cls, None, "refl-coef")
    assert abs(z_pair - z_pec) > 0.1


def test_the_spec_is_stored_as_given():
    """Like BSplineSolver: no coercion at construction, the fold happens at
    solve time through the shared helper, so a pair stays a pair."""
    h = 0.25 * WAVELENGTH
    sim = PulseSolver(
        wires=[np.array([[-1.0, 0.0, h], [1.0, 0.0, h]])],
        n_per_edge_per_wire=[[5]],
        wire_radius=0.001,
        wavelength=WAVELENGTH,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
    )
    assert sim.ground_eps == (13.0, 0.005)
