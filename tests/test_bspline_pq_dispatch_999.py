"""The same-edge moment dispatch is indexed, not flattened — momwire#999 step 2.

`J_static_dispatch` and `D_ek_dispatch` flattened (p, q) as `p * 3 + q` and
switched on nine cases. That is a correct bijection for p, q in {0, 1, 2}, and
it COLLIDES the moment the degree axis moves: (0, 3) and (1, 0) both flatten to
3, (1, 3) and (2, 0) both to 6. So the patch the old in-tree comment recommended
-- "extends automatically when the header file is regenerated ... (and the
case-list in J_static_dispatch below is extended)" -- compiles, runs, and
returns the WRONG moment. Measured at a near pair before the fix:

    J(0, 3) = 1.302e-06   but the collision returns J(1, 0) = 2.191e-03
    J(1, 3) = 3.716e-08   but the collision returns J(2, 0) = 8.400e-05

Three orders of magnitude, silently. Half the degree-3 pairs collided and half
fell through to the throw, which is the worst available mix: a smoke test
asserting "degree 3 no longer raises" goes green on the wrong numbers.

The comment was false on its own terms too. #883 regenerated the header for
MAX_D = 3 a release before this, and nothing extended automatically.

What this module pins:

  * **every (p, q) in the square, in both families, against the numpy twins.**
    Seven of the sixteen only became reachable in step 2.

  * **the pairs that used to collide are distinct AND correct.** Distinctness
    alone is not the claim -- two wrong values are also distinct. Each is
    checked against its own numpy twin, and the colliding partner is checked
    to be far away, so a future re-flattening cannot pass by accident.

  * **the flattening does not come back.** A source tripwire, comments
    stripped, because the header explains the defect by quoting it.

  * **the domain is closed.** An index with no bound check is step 1's defect
    one dimension up, so out-of-range throws rather than reading a table.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from momwire._bspline_ek_moments import D_ek_moment
from momwire._bspline_static_moments import MAX_D, J_static_moment

acc = pytest.importorskip("momwire._accelerators")

pytestmark = pytest.mark.skipif(
    not hasattr(acc, "bspline_j_static_moment"),
    reason="near-dispatch leaf bindings not built",
)

# A NEAR pair — the closed forms are the branch under test here, not the
# multipole series that step 1 gated. far_ratio is 1.0 at this geometry
# against a switch at 0.5.
ALPHA, BETA = 0.0, 0.05
A_LEFT, B_RIGHT = 0.05, 0.10
A_WIRE = 5e-4

# (p, q) -> the pair it used to be confused with under `p * 3 + q`.
COLLISIONS = {(0, 3): (1, 0), (1, 3): (2, 0), (2, 3): (3, 0)}


@pytest.mark.parametrize("q", range(MAX_D + 1))
@pytest.mark.parametrize("p", range(MAX_D + 1))
def test_the_near_dispatch_matches_the_numpy_twin(p, q):
    cxx = acc.bspline_j_static_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
    npy = float(J_static_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE))
    assert np.isfinite(cxx)
    assert abs(cxx - npy) <= 1e-12 * abs(npy), f"(p, q) = ({p}, {q})"


@pytest.mark.parametrize("q", range(MAX_D + 1))
@pytest.mark.parametrize("p", range(MAX_D + 1))
def test_the_ek_dispatch_matches_the_numpy_twin(p, q):
    cxx = acc.bspline_d_ek_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
    npy = float(D_ek_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE))
    assert np.isfinite(cxx)
    assert abs(cxx - npy) <= 1e-12 * abs(npy), f"(p, q) = ({p}, {q})"


@pytest.mark.parametrize("pq,partner", sorted(COLLISIONS.items()))
def test_the_pairs_that_used_to_collide_are_distinct_and_correct(pq, partner):
    """Distinctness alone would not be the claim — two wrong values are also
    distinct. Each side is checked against its OWN numpy twin, and the two are
    checked to be far apart, so a re-flattening cannot slip through by
    returning a plausible neighbour."""
    p, q = pq
    for pp, qq in (pq, partner):
        cxx = acc.bspline_j_static_moment(pp, qq, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
        npy = float(J_static_moment(pp, qq, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE))
        assert abs(cxx - npy) <= 1e-12 * abs(npy), f"({pp}, {qq})"
    a = acc.bspline_j_static_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
    b = acc.bspline_j_static_moment(*partner, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
    # Measured ratios are 1683x and 2260x; an order of magnitude is a floor
    # that leaves room for geometry changes without being satisfiable by noise.
    assert max(abs(a), abs(b)) / min(abs(a), abs(b)) > 10.0


def test_out_of_range_throws_rather_than_indexing():
    """Step 1's defect was an index with no bound check. The fix for the
    stride must not reintroduce it one dimension up."""
    for p, q in ((MAX_D + 1, 0), (0, MAX_D + 1), (-1, 0), (0, -1)):
        with pytest.raises(Exception):
            acc.bspline_j_static_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
        with pytest.raises(Exception):
            acc.bspline_d_ek_moment(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)


def test_the_exported_degree_matches_the_generated_family():
    assert acc.BSPLINE_MOMENT_MAX_D == MAX_D
    assert acc.BSPLINE_MOMENT_MAX_D == acc.BSPLINE_FAR_MAX_P


def test_the_flattening_does_not_come_back():
    """COMMENTS ARE STRIPPED. The file explains the defect by quoting the
    expression that caused it, so a search over raw text matches the prose and
    reds on the fix — the same shape that bit step 1's tripwire, the #936 AST
    gate and the #988 importorskip guard."""
    import momwire

    src = pathlib.Path(momwire.__file__).parent / "_accel_bspline.cpp"
    code = re.sub(r"//[^\n]*", "", src.read_text())
    assert not re.search(r"\bp\s*\*\s*\d+\s*\+\s*q\b", code), (
        "a flattened (p, q) dispatch is back; index the two independently"
    )
    # And the comment that recommended the broken patch is gone for good.
    assert "extends automatically when the header file" not in src.read_text()
