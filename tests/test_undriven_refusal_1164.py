"""An undriven deck refuses by name on EVERY family — momwire#1164, #962.

Z = V/I per port is 0/0 when no port is driven: the RHS is zero, the solution
is identically zero, every port current reads exactly 0. #1162 refused that on
SinusoidalGalerkinSolver; every other family returned NaN (or `inf` beside a 0
on a two-port, #962's `z = [inf, 0]` from four wild-corpus decks whose `EX`
cards drove 0 V), counted as solved, and poisoned every median computed over
the census it sat in.

What this module pins, family by family:

  * an all-0-V deck refuses with `NO_DRIVEN_PORT_REFUSAL` on the single-k
    entry and on the sweep, an EMPTY sweep included (the check runs before
    any fill);
  * a 0 V port beside a driven one still answers, reading Z = 0/I = 0;
  * the backstop, `port_impedances`: a zero current on a driven port refuses
    on both the single-k and the swept shapes instead of returning inf.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire._port_solution import NO_DRIVEN_PORT_REFUSAL, port_impedances
from momwire.bspline import BSplineSolver
from momwire.harrington import HarringtonSolver
from momwire.hmatrix import HMatrixSolver
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WIRE = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
FAMILIES = [
    ("pulse", PulseSolver, {}),
    ("harrington", HarringtonSolver, {}),
    ("sinusoidal", SinusoidalSolver, {}),
    ("sin-galerkin", SinusoidalGalerkinSolver, {}),
    ("razor", RazorSolver, {}),
    ("bspline-d1", BSplineSolver, {"degree": 1}),
    ("bspline-d2", BSplineSolver, {"degree": 2}),
    ("hmatrix", HMatrixSolver, {}),
]
IDS = [f[0] for f in FAMILIES]


def _solver(cls, feeds, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(
            wires=[WIRE],
            n_per_edge_per_wire=[[21]],
            wavelength=20.0,
            feeds=feeds,
            **kw,
        )


@pytest.mark.parametrize("_name, cls, kw", FAMILIES, ids=IDS)
def test_a_zero_volt_deck_refuses_on_every_entry(_name, cls, kw):
    s = _solver(cls, [(0, 5.0, 0j)], **kw)
    calls = [s.compute_impedance]
    # The pulse family has no swept impedance entry to guard.
    if hasattr(s, "compute_impedance_swept"):
        calls += [
            lambda: s.compute_impedance_swept(np.array([s.k])),
            lambda: s.compute_impedance_swept(np.array([])),
        ]
    for call in calls:
        with pytest.raises(ValueError) as exc:
            call()
        assert str(exc.value) == NO_DRIVEN_PORT_REFUSAL


@pytest.mark.parametrize("_name, cls, kw", FAMILIES, ids=IDS)
def test_a_zero_volt_port_beside_a_driven_one_answers(_name, cls, kw):
    z, _ = _solver(cls, [(0, 2.0, 1 + 0j), (0, 8.0, 0j)], **kw).compute_impedance()
    assert np.all(np.isfinite(z))
    assert z[1] == 0
    assert abs(z[0]) > 1.0


def test_the_sentence_names_the_route_that_answers():
    assert "compute_port_solution().y" in NO_DRIVEN_PORT_REFUSAL
    y = _solver(SinusoidalSolver, [(0, 5.0, 0j)]).compute_y_matrix()
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize(
    "voltages, currents",
    [
        (np.array([1.0 + 0j, 0j]), np.array([0.02 + 0j, 0j])),
        (np.array([[1.0 + 0j, 0j]]), np.array([[0.02, 0.01], [0.0, 0.01]])),
    ],
    ids=["single-k", "swept"],
)
def test_the_backstop_refuses_a_zero_current_on_a_driven_port(voltages, currents):
    """Single-k: port 1 is 0/0. Swept: port 0 reads 1/0 at the second k."""
    with pytest.raises(FloatingPointError, match=r"port\(s\) \[\d\]"):
        port_impedances(voltages, currents)


def test_the_backstop_passes_a_real_answer_through_unchanged():
    v = np.array([1.0 + 0j, 0j])
    i = np.array([0.02 - 0.01j, 0.003 + 0j])
    assert np.array_equal(port_impedances(v, i), v / i)
