"""An asymmetric feed tie warns by default; a symmetric one stays silent —
momwire#1262, #623's step 2.

#623 found 530 of 3341 snaps in the quick suite to be exact ties, every one
harmless because its deck is symmetric about the site. #1224's bench then
found the harmful one: `invl_deck` fed 13/3 m below the monopole top, a
segment END at every even rung, over ground with a hub below and a top wire
above. SinusoidalSolver silently took the centre above. What separates the
two is whether the deck has a reflection swapping the two candidates, which
`_feed_snap.mirror_images` decides. Everything here is geometry-only: the
snap runs before any fill.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _feed_snap  # noqa: E402
from momwire.pulse import PulseSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_crossing_serve_524 import invl_deck  # noqa: E402

WIRE = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # 20 segments: h = 0.5


def _snap(solver):
    """Run the family's feed snap and nothing after it: sinusoidal snaps in
    `_build_geometry`, pulse and razor when the drive is placed on it."""
    geom = solver._build_geometry()
    if isinstance(solver, PulseSolver):
        solver._feed_basis_indices(geom)
    elif isinstance(solver, RazorSolver):
        solver._port_columns(geom)


def _ties(solver):
    """The #1262 warnings one snap raises."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _snap(solver)
    return [
        w
        for w in caught
        if issubclass(w.category, _feed_snap.AmbiguousSite)
        and "momwire#1262" in str(w.message)
    ]


def _quiet(cls, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**kw)


def _straight(cls, arc, **kw):
    return _quiet(
        cls,
        wires=[WIRE],
        n_per_edge_per_wire=[[20]],
        wavelength=20.0,
        feeds=[(0, arc, 1 + 0j)],
        **kw,
    )


# (id, class, kwargs, an off-centre arclength this family ties on)
GRID_LOCKED = [
    ("sinusoidal", SinusoidalSolver, {}, 3.0),
    ("galerkin-segment", SinusoidalGalerkinSolver, {"feed_model": "segment"}, 3.0),
    ("pulse", PulseSolver, {}, 3.0),
    ("razor", RazorSolver, {}, 2.75),
]


def test_the_bench_deck_warns_and_names_the_site():
    """The gate the issue names: invl at x2 (1/3 m segments, 13/3 m a knot)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("MOMWIRE_623_TALLY", raising=False)
        found = _ties(_quiet(SinusoidalSolver, **invl_deck(x=2)))
    assert len(found) == 1
    msg = str(found[0].message)
    assert "arclength 4.3333333333 on wire 17" in msg
    assert "4.16666666667" in msg and "4.5" in msg  # both candidates
    assert "took 4.16666666667" in msg


@pytest.mark.parametrize(
    "_id, cls, kw, _arc", GRID_LOCKED, ids=[r[0] for r in GRID_LOCKED]
)
def test_an_off_centre_tie_warns(_id, cls, kw, _arc):
    assert len(_ties(_straight(cls, _arc, **kw))) == 1


@pytest.mark.parametrize(
    "_id, cls, kw, _arc", GRID_LOCKED, ids=[r[0] for r in GRID_LOCKED]
)
def test_the_symmetric_tie_stays_silent(_id, cls, kw, _arc):
    """The wire's midpoint: a knot for the centre-snappers on an even mesh, a
    centre for razor on an odd one — the tie #623 counted 530 of."""
    n = 21 if cls is RazorSolver else 20
    s = _quiet(
        cls,
        wires=[WIRE],
        n_per_edge_per_wire=[[n]],
        wavelength=20.0,
        feeds=[(0, None, 1 + 0j)],
        **kw,
    )
    assert _ties(s) == []


def test_a_point_gap_never_warns():
    """Position-capable: the gap sits where it was asked whichever centre
    won (#648), so its tie moves nothing on any deck."""
    assert _ties(_straight(SinusoidalGalerkinSolver, 3.0)) == []


def test_ground_breaks_the_mirror():
    """A vertical wire fed at its middle knot is symmetric in free space and
    not over a ground plane: the mirror plane is horizontal."""
    kw = dict(
        wires=[np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]])],
        n_per_edge_per_wire=[[20]],
        wavelength=20.0,
        feeds=[(0, None, 1 + 0j)],
    )
    assert _ties(_quiet(SinusoidalSolver, **kw)) == []
    assert len(_ties(_quiet(SinusoidalSolver, ground_z=0.0, **kw))) == 1


def test_a_bent_symmetric_deck_is_a_mirror_image():
    """An inverted V fed at its apex: the two candidates sit on different
    arms, and the vertical plane through the apex swaps them."""
    s = _quiet(
        SinusoidalSolver,
        wires=[np.array([[-4.0, 0.0, 7.0], [0.0, 0.0, 10.0], [4.0, 0.0, 7.0]])],
        n_per_edge_per_wire=[[10, 10]],
        wavelength=20.0,
        feeds=[(0, None, 1 + 0j)],
        ground_z=0.0,
    )
    assert _ties(s) == []
    assert _feed_snap.mirror_images(s, 0, 4.75, 5.25)
    assert not _feed_snap.mirror_images(s, 0, 4.25, 5.25)


def test_a_second_feed_breaks_the_mirror():
    """Geometry alone is symmetric; an off-centre second feed is not."""
    s = _quiet(
        SinusoidalSolver,
        wires=[WIRE],
        n_per_edge_per_wire=[[20]],
        wavelength=20.0,
        feeds=[(0, None, 1 + 0j), (0, 1.25, 1 + 0j)],
    )
    assert len(_ties(s)) == 1
