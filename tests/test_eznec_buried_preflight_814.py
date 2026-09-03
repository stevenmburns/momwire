"""The eznec drop-in refuses a buried deck BY NAME — momwire#814 prep.

The seam pre-flights what a basis cannot do with a deck, reading the prose off
the solver's own `Capabilities` row so that a consumer stops keeping its own
copy of what each family serves. It checked node gaps, knot feeds and
ground contact; it did NOT check `buried`, so a buried deck on a basis without
a buried fill fell through to the solver's constructor and came back as a bare
`ValueError` from inside momwire rather than a named refusal in the printout.

momwire#814 will flip razor's `buried` cell. These gates are what make that
flip land here without a second edit: the seam asks the ROW, so the same code
refuses today and serves after the flip. Both arms are exercised — the flipped
one by patching the row itself, because the row is what the seam reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.eznec._serve import _Mesh, _Piece, ServeRefusal, _check_basis_can_host
from momwire.razor import RazorSolver

SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}
PERFECT = {"ground_z": 0.0}
FREE: dict = {}


def _piece(tag, pts):
    a = np.asarray(pts, dtype=float)
    return _Piece(tag=tag, first_node=0, last_node=len(a) - 1, points=a, radius=1e-3)


def _buried_mesh():
    """One wholly-below radial: the `buried` cell, no crossing anywhere.

    Two elements, not one: a lone SINGLE-element polyline is refused earlier
    by the inert-piece rule (momwire#608) and would never reach the buried
    check at all."""
    return _Mesh(
        pieces=[_piece(1, [(0.0, 0.0, -0.15), (2.5, 0.0, -0.15), (5.0, 0.0, -0.15)])]
    )


def _crossing_mesh():
    """A below wire ending in the plane, an above wire starting there, and the
    junction between them declared: the `buried+crossing_junction` cell."""
    return _Mesh(
        pieces=[
            _piece(1, [(0.0, 0.0, -2.0), (0.0, 0.0, 0.0)]),
            _piece(2, [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]),
        ],
        junctions=[[(0, "end"), (1, "start")]],
    )


def _above_mesh():
    return _Mesh(
        pieces=[_piece(1, [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 10.0)])]
    )


def _flipped(caps):
    """razor's row as momwire#814 will leave it: the cell True and the two
    "not served YET" entries gone. Patched on the ROW because the row is what
    the seam reads — monkeypatching `_SERVE_BURIED` cannot reach a class
    attribute built at import."""
    return caps._replace(
        buried=True,
        refusals={
            k: v
            for k, v in caps.refusals.items()
            if k not in ("buried", "buried+crossing_junction")
        },
    )


# ---------------------------------------------------------------------------
# the refusal, and that it is the ROW's sentence and not a copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh, cells",
    [
        (_buried_mesh(), ("buried",)),
        (_crossing_mesh(), ("buried", "crossing_junction")),
    ],
)
def test_razor_refuses_a_buried_deck_with_the_sentence_its_row_declares(mesh, cells):
    declared = RazorSolver.capabilities.refusal(*cells)
    assert declared is not None, "this gate is about the pre-flip row"
    with pytest.raises(ServeRefusal) as exc:
        _check_basis_can_host(mesh, SOMMERFELD, "razor-2p", RazorSolver)
    assert str(exc.value).endswith(declared)
    assert "runs below a FINITE ground plane" in str(exc.value)


def test_the_two_buried_decks_get_DIFFERENT_sentences():
    """The point of momwire#850's separate cell: a declared crossing junction
    and a lone buried wire are two refusals under one geometry word, and the
    seam has to pick the one the deck earns."""

    def _reason(mesh):
        with pytest.raises(ServeRefusal) as exc:
            _check_basis_can_host(mesh, SOMMERFELD, "razor-2p", RazorSolver)
        return str(exc.value)

    assert "cross the interface at a junction" in _reason(_crossing_mesh())
    assert "has no buried fill" in _reason(_buried_mesh())


# ---------------------------------------------------------------------------
# the flipped arm: the same code serves, with no second edit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh", [_buried_mesh(), _crossing_mesh()])
def test_the_flip_makes_the_seam_serve_without_touching_the_seam(mesh, monkeypatch):
    monkeypatch.setattr(RazorSolver, "capabilities", _flipped(RazorSolver.capabilities))
    _check_basis_can_host(mesh, SOMMERFELD, "razor-2p", RazorSolver)


@pytest.mark.parametrize("mesh", [_buried_mesh(), _crossing_mesh()])
def test_bspline_already_serves_both(mesh):
    """The family that has served buried decks since momwire#553 must not be
    refused by a check written for the one that does not."""
    _check_basis_can_host(mesh, SOMMERFELD, "bspline", BSplineSolver)


# ---------------------------------------------------------------------------
# what the check must NOT fire on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ground", [PERFECT, FREE])
def test_no_lower_medium_means_no_buried_question(ground):
    """`ground_eps` is the key, exactly as the contact check above uses it: a
    `GD` or a perfect image has no half-space for a wire to be buried IN, so a
    wire below THAT plane is a different geometry with its own refusal, raised
    where that geometry is diagnosed and not here."""
    _check_basis_can_host(_buried_mesh(), ground, "razor-2p", RazorSolver)


def test_a_deck_with_nothing_below_is_untouched():
    _check_basis_can_host(_above_mesh(), SOMMERFELD, "razor-2p", RazorSolver)


def test_an_in_plane_junction_that_cannot_cross_is_not_the_crossing_cell():
    """momwire#848's rule, reached through the seam: a grounded junction whose
    members are all at-or-below the plane cannot span it, so it earns no
    crossing exemption and the deck is the plain `buried` cell. The seam gets
    this right because it ASKS `_medium_spec` rather than keeping a second
    copy of the test."""
    mesh = _Mesh(
        pieces=[
            _piece(1, [(0.0, 0.0, -2.0), (0.0, 0.0, 0.0)]),
            _piece(2, [(0.0, 0.0, 0.0), (5.0, 0.0, -0.15)]),
        ],
        junctions=[[(0, "end"), (1, "start")]],
    )
    with pytest.raises(ServeRefusal) as exc:
        _check_basis_can_host(mesh, SOMMERFELD, "razor-2p", RazorSolver)
    assert "has no buried fill" in str(exc.value)
    assert "cross the interface at a junction" not in str(exc.value)
