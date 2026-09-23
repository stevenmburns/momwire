"""The eznec drop-in refuses a buried deck BY NAME — momwire#814 prep.

The seam pre-flights what a basis cannot do with a deck, reading the prose off
the solver's own `Capabilities` row so that a consumer stops keeping its own
copy of what each family serves. It checked node gaps, knot feeds and
ground contact; it did NOT check `buried`, so a buried deck on a basis without
a buried fill fell through to the solver's constructor and came back as a bare
`ValueError` from inside momwire rather than a named refusal in the printout.

momwire#1149 flipped razor's row in two stages. U0/U1 turned the `buried`
cell True and left `buried+crossing_junction` declared; U2 retired that one
too. These gates are what made each stage land here without a second edit:
the seam asks the ROW, so the same code that refused the crossing deck before
U2 serves it now. The refusal arm is kept honest by patching the pre-U2
cell back ONTO the row, because the row is what the seam reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import razor as _razor
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
    """razor's row as momwire#1149 U4 will leave it: the cell True and both
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


def _with_the_pre_u2_cell(caps):
    """razor's row as it stood between U0 and U2: the crossing cell declared
    with the sentence `_SERVE_CROSSING` still owns."""
    return caps._replace(
        refusals={
            **caps.refusals,
            "buried+crossing_junction": _razor._CROSSING_NOT_SERVED_REFUSAL,
        }
    )


def test_razor_serves_a_crossing_deck_since_u2():
    assert RazorSolver.capabilities.refusal("buried", "crossing_junction") is None
    _check_basis_can_host(_crossing_mesh(), SOMMERFELD, "razor-2p", RazorSolver)


def test_a_declared_crossing_cell_is_refused_with_the_rows_sentence(monkeypatch):
    monkeypatch.setattr(
        RazorSolver, "capabilities", _with_the_pre_u2_cell(RazorSolver.capabilities)
    )
    declared = RazorSolver.capabilities.refusal("buried", "crossing_junction")
    with pytest.raises(ServeRefusal) as exc:
        _check_basis_can_host(_crossing_mesh(), SOMMERFELD, "razor-2p", RazorSolver)
    assert str(exc.value).endswith(declared)
    assert "runs below a FINITE ground plane" in str(exc.value)


def test_the_two_buried_decks_get_DIFFERENT_answers(monkeypatch):
    """The point of momwire#850's separate cell: a declared crossing junction
    and a lone buried wire are two cells under one geometry word, and the
    seam has to pick the one the deck earns. On the pre-U2 row they get
    different ANSWERS — the plain cell served, the crossing cell refused —
    which is a sharper test of the pick than two sentences."""
    monkeypatch.setattr(
        RazorSolver, "capabilities", _with_the_pre_u2_cell(RazorSolver.capabilities)
    )
    assert RazorSolver.capabilities.refusal("buried") is None
    _check_basis_can_host(_buried_mesh(), SOMMERFELD, "razor-2p", RazorSolver)
    with pytest.raises(ServeRefusal) as exc:
        _check_basis_can_host(_crossing_mesh(), SOMMERFELD, "razor-2p", RazorSolver)
    assert "cross the interface at a junction" in str(exc.value)


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


def test_an_in_plane_junction_that_cannot_cross_is_not_the_crossing_cell(monkeypatch):
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
    # On the pre-U2 row the plain cell is served and the crossing cell is
    # not, so passing there is the seam asking the RIGHT cell.
    monkeypatch.setattr(
        RazorSolver, "capabilities", _with_the_pre_u2_cell(RazorSolver.capabilities)
    )
    _check_basis_can_host(mesh, SOMMERFELD, "razor-2p", RazorSolver)
