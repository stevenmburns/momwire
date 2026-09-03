"""A bundle refuses BY NAME instead of dying in LAPACK — momwire#846.

N geometrically COINCIDENT segments give the tent basis N identical columns,
so razor's matrix is singular by construction — at any mesh, in free space and
in soil alike, whatever the quadrature. What came back was
`LinAlgError: singular matrix`, which says nothing about the deck and nothing
about what to do instead.

Where the check sits is the interesting decision. NOT at construction and NOT
in the assembly: the FILL on such a deck is well defined, and momwire#813's
collapse gates measure it against bspline's to 7.3e-11. Those gates compare
matrices and never solve, which is exactly why the singularity was invisible
to them until #813 step 4 ran one. It is the SOLVE that has no answer, so the
sentence is raised from the solve entry points and the matrix gates keep
working — which this file pins, because moving the check "somewhere earlier"
is the obvious tidy-up and it would silently cost #813 its adjudicators.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import (  # noqa: E402
    crossing_deck,
    fan_rise_deck,
    hub_deck,
)

DECLARED = RazorSolver.capabilities.refusals["bundle"]

# Every deck here is FREE SPACE unless a test says otherwise. N identical
# columns are N identical columns with no ground under them, so soil buys this
# file nothing but 40 s a solve — and one soil gate below pins that the answer
# really is the same there.


@pytest.fixture(autouse=True)
def _serve(monkeypatch):
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)


def _free_space(build):
    return {
        k: v
        for k, v in build.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------


def test_the_bundle_refuses_with_the_sentence_the_row_declares():
    with pytest.raises(ValueError) as exc:
        RazorSolver(
            **_free_space(fan_rise_deck()), nec5_quadrature=True
        ).compute_impedance()
    raised = str(exc.value)
    assert raised.endswith(DECLARED)
    assert "run between the same two points" in raised


def test_the_sentence_names_the_spelling_that_IS_served():
    """A refusal that does not say what to do instead is half a refusal. The
    hub respelling it names is a real deck and razor solves it — asserted
    here, in the same file, so the sentence cannot go stale on its own."""
    assert "buried HUB" in DECLARED
    assert "BSplineSolver" in DECLARED
    z, _ = RazorSolver(
        **_free_space(hub_deck()), nec5_quadrature=True
    ).compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_it_is_the_coincidence_and_not_the_fan():
    """One radial is the same deck shape with nothing coincident in it, and it
    solves — so the refusal is keyed on the geometry it names."""
    z, _ = RazorSolver(
        **_free_space(fan_rise_deck(n_radials=1)), nec5_quadrature=True
    ).compute_impedance()
    assert np.isfinite(z.real)


@pytest.mark.slow
def test_soil_refuses_with_the_same_sentence():
    """The one soil rung, in the push lane because a buried fill costs 40 s:
    the singularity is the BASIS's and not the interface's, so the answer over
    soil A must be the same sentence the free-space rungs above get."""
    with pytest.raises(ValueError) as exc:
        RazorSolver(**fan_rise_deck(), nec5_quadrature=True).compute_impedance()
    assert str(exc.value).endswith(DECLARED)


@pytest.mark.parametrize("call", ["compute_impedance", "compute_y_matrix", "swept"])
def test_every_solve_entry_point_refuses(call):
    """Four ways into a solve and one sentence. A guard on one of them is a
    guard a caller walks around."""
    s = RazorSolver(**_free_space(fan_rise_deck()), nec5_quadrature=True)
    with pytest.raises(ValueError) as exc:
        if call == "swept":
            s.compute_impedance_swept(np.array([s.k]))
        else:
            getattr(s, call)()
    assert str(exc.value).endswith(DECLARED)


# ---------------------------------------------------------------------------
# what the check must NOT break
# ---------------------------------------------------------------------------


def test_the_matrix_still_builds_on_the_bundle():
    """The reason the check is not at construction. momwire#813's eps~ = 1
    collapse adjudicators fill this deck and compare the MATRIX; they never
    solve it. Refusing earlier would take those gates away and the arc would
    lose its free-space truth on the fan."""
    s = RazorSolver(**_free_space(fan_rise_deck()), nec5_quadrature=True)
    geom = s._build_geometry()
    assert int(geom["n_basis_total"]) > 0
    prepared = s._assemble_Z_prepare(geom)
    Z = s._assemble_Z_from_prepared(geom, prepared, s.k, s.c * s.k)
    assert Z.shape[0] == Z.shape[1] == int(geom["n_basis_total"])
    assert np.isfinite(Z).all()


def test_bspline_serves_the_bundle_because_it_has_a_bundle_rule():
    """momwire#524 phase 2's fan widening. The refusal is razor's own gap and
    must not read as momwire's."""
    z, _ = BSplineSolver(**_free_space(fan_rise_deck())).compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)


@pytest.mark.parametrize(
    "name, build", [("hub", hub_deck()), ("crossing", crossing_deck(1))]
)
def test_the_served_decks_are_untouched(name, build):
    z, _ = RazorSolver(**_free_space(build), nec5_quadrature=True).compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
