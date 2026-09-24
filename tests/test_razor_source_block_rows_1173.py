"""Razor's source block is finished one row window at a time — momwire#1173.

`_assemble_Z_source_block` used to form the charge term T2 whole (the
(n_basis, n_seg) potential difference and its two (n_basis, n_basis)
gathers), a separate remainder matrix Q on the composing ground, and three
whole-matrix temporaries for `c_A·T1 − T2/c_Φ (+ Q)`. Each T1 row window now
builds its own T2 rows and writes the finished block rows over T1 in place.
Every operation involved is elementwise on the row axis, so the claim is
exact, not approximate: the matrix may not depend on where the row windows
fall, bit for bit.

What this module gates:

  * **the schedule does not reach the answer**, on every branch the row
    window has to carry: free space (the accelerated T1 kernel), grounded
    rows (contact over PEC and over Sommerfeld), the weighted fold
    (`refl-coef`), the composing ground's Q rows (Sommerfeld, elevated), and
    the chopped rows of a crossing deck. A spy asserts that each of those
    branches really ran and that the tiny budget really split the rows, so
    a green result cannot come from a fill that never windowed;
  * **the bar can see what the change must not do.** A one-ulp change in the
    charge term's prefactor — the size of difference a reassociated
    combination produces — moves the matrix under `array_equal`.

The cross-revision half of the momwire#762 protocol (the same decks, plus the
#1173 bench arrays at N = 352/704, against the pre-change revision
c8b3103) was run by hand for this change and was bit-identical on every deck
and both lanes; what is gated here is the standing claim a later edit could
break.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import _potential_ground
from momwire import razor as _razor
from momwire.razor import RazorSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck  # noqa: E402

C0 = 299792458.0
LAM = C0 / 7.0e6

LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}}

# Small enough that every deck below splits into several row windows on
# both budgets (the spy checks it).
TINY = {"_CHUNK_ELEMS": 3_000, "_WEIGHTED_CHUNK_ELEMS": 500}


def _vertical(n, ground=None, *, contact=False):
    z0 = 0.0 if contact else 1.0
    deck = dict(
        wires=[np.array([[0.0, 0.0, z0], [0.0, 0.0, z0 + LAM / 4]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=0.005,
        wavelength=LAM,
        feeds=[(0, 0.0 if contact else LAM / 8, 1 + 0j)],
    )
    if ground is not None:
        deck["ground_z"] = 0.0
        if ground != "pec":
            deck["ground_eps"] = (13.0, 0.005)
            deck["ground_model"] = ground
    return deck


# deck, and the branch of the row window it exists to exercise
DECKS = {
    "free space": (lambda: _vertical(40), None),
    "contact / pec": (lambda: _vertical(40, "pec", contact=True), "grounded"),
    "elevated / refl-coef": (lambda: _vertical(40, "refl-coef"), "weighted"),
    "elevated / sommerfeld": (lambda: _vertical(40, "sommerfeld"), "compose"),
    "contact / sommerfeld": (
        lambda: _vertical(40, "sommerfeld", contact=True),
        "grounded",
    ),
    "crossing": (lambda: crossing_deck(1), "chop"),
}


class _Spy:
    """Counts what each source block actually carried."""

    def __init__(self, mp):
        self.seen = {"split": 0, "grounded": 0, "chop": 0, "compose": 0}
        self.seen["weighted"] = 0
        inner = RazorSolver._assemble_Z_source_block

        def spy(rs, geom, prepared, sources, k, omega, *args, ground=None, **kw):
            if len(sources["t1_row_chunks"]) > 1:
                self.seen["split"] += 1
            if prepared["grounded"].size:
                self.seen["grounded"] += 1
            if prepared["t2_chop"] is not None:
                self.seen["chop"] += 1
            if ground is not None and ground.eps_tilde is not None:
                self.seen["weighted"] += 1
                if ground.remainder() is not None:
                    self.seen["compose"] += 1
            return inner(
                rs, geom, prepared, sources, k, omega, *args, ground=ground, **kw
            )

        mp.setattr(RazorSolver, "_assemble_Z_source_block", spy)


def _Z(deck, lane, **attrs):
    rs = RazorSolver(**deck, **LANES[lane])
    for name, value in attrs.items():
        setattr(rs, name, value)
    return rs._assemble_Z(rs._build_geometry(), rs.k)


# The crossing deck on the Gauss-Legendre lane takes ~11 s: the push lane
# carries it, and the nec5 lane covers the chopped rows on every PR.
CASES = [
    pytest.param(
        name,
        lane,
        marks=[pytest.mark.slow]
        if (name, lane) == ("crossing", "gauss-legendre")
        else [],
        id=f"{lane}-{name}",
    )
    for name in sorted(DECKS)
    for lane in sorted(LANES)
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("name", "lane"), CASES)
def test_the_block_does_not_depend_on_the_row_windows(name, lane):
    make, branch = DECKS[name]
    ref = _Z(make(), lane)
    with pytest.MonkeyPatch.context() as mp:
        for attr, value in TINY.items():
            mp.setattr(_razor, attr, value)
        spy = _Spy(mp)
        got = _Z(make(), lane)
    assert spy.seen["split"], "the tiny budget never split the rows"
    if branch is not None:
        assert spy.seen[branch], f"the {branch} branch never ran"
    assert np.array_equal(got, ref), "the row-window schedule moved the matrix"


@pytest.mark.filterwarnings("ignore")
def test_a_one_ulp_prefactor_change_is_visible():
    """The negative control: `array_equal` sees a last-bit change in c_Φ."""
    make, _ = DECKS["elevated / sommerfeld"]
    rs = RazorSolver(**make(), **LANES["nec5"])
    ref = _Z(make(), "nec5")
    got = _Z(make(), "nec5", eps=np.nextafter(rs.eps, np.inf))
    assert not np.array_equal(got, ref)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "name", ["contact / pec", "elevated / refl-coef", "elevated / sommerfeld"]
)
def test_out_is_the_same_fold_as_subtracting_the_returned_block(name):
    """`out=` folds the image block into Z window by window; the matrix is
    the one `Z -= block` gives on the whole returned block, bit for bit."""
    make, _ = DECKS[name]
    rs = RazorSolver(**make(), **LANES["nec5"])
    geom = rs._build_geometry()
    prep = rs._assemble_Z_prepare(geom)
    k, omega = rs.k, rs.omega
    ground = _potential_ground.potential_ground_for(rs, geom, k, omega)
    free = rs._assemble_Z_source_block(geom, prep, prep, k, omega)
    block = rs._assemble_Z_source_block(
        geom, prep, prep["image"], k, omega, ground=ground
    )
    want = free - block
    got = free.copy()
    ret = rs._assemble_Z_source_block(
        geom, prep, prep["image"], k, omega, ground=ground, out=got
    )
    assert ret is got
    assert np.array_equal(got, want)
