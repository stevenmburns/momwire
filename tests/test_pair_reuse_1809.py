"""bs2's `pair_reuse` hint: fill one block per repeated edge-pair pose and copy
the rest (momwire#1809, antennaknobs#2165).

The hint is only a claim. Each member is checked against the solver's own
geometry (segment counts, radii, extended-kernel eligibility, and the six
end-point distances of the edge pair in the hint's order), and one that fails
is filled normally and named. So:

- no hint, or a hint with nothing that checks out, is the fill bit for bit;
- a true hint moves Z by no more than copying a block across an exact
  isometry can, a few ulp of the block's entries;
- a false claim never reaches the matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire.bspline import BSplineSolver, PairReuseDropped

LAM = 21.1


def _yagi(D, h=None):
    """Reflector, driver (fed) and D identical directors, 0.2 lambda apart."""
    Ld = 0.47 * LAM
    elems = [(1.05 * Ld, 22), (Ld, 21)] + [(0.95 * Ld, 20)] * D
    wires, n = [], []
    z = 0.0 if h is None else h
    for i, (L, ns) in enumerate(elems):
        x = 0.2 * LAM * i
        wires.append(np.array([[x, -L / 2, z], [x, L / 2, z]]))
        n.append([ns])
    return wires, n


def _director_hint(D):
    """Every director pair at one spacing is one pose (wire 2 is the first
    director), and every director's own block is one pose: the classes a
    front end would compute for an evenly spaced Yagi."""
    hint = [[((w, 0), (w, 0)) for w in range(2, 2 + D)]]
    for s in range(1, D):
        hint.append([((w, 0), (w + s, 0)) for w in range(2, 2 + D - s)])
    return hint


def _z(D, hint, extended_kernel=False, ground=None):
    wires, n = _yagi(D, h=None if ground is None else 15.0)
    kw = {}
    if ground is not None:
        kw = {
            "ground_z": 0.0,
            "ground_eps": complex(13.0, -60 * LAM * 0.005),
            "ground_model": "refl-coef",
        }
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n,
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=0.006,
        extended_kernel=extended_kernel,
        pair_reuse=hint,
        **kw,
    )
    return s, np.atleast_1d(np.asarray(s.compute_impedance()[0], dtype=complex))


@pytest.mark.parametrize("ek", [False, True])
def test_no_hint_is_the_fill_bit_for_bit(ek):
    _, z0 = _z(6, None, ek)
    _, z1 = _z(6, [], ek)
    assert np.array_equal(z0.view(np.uint64), z1.view(np.uint64))


@pytest.mark.parametrize("ek", [False, True])
def test_a_true_hint_moves_z_by_a_few_ulp(ek):
    _, z0 = _z(8, None, ek)
    s, z1 = _z(8, _director_hint(8), ek)
    assert s.pair_reuse_dropped == ()
    rel = np.max(np.abs(z1 - z0) / np.abs(z0))
    assert rel <= 1e-13, rel


def test_the_hint_skips_the_copied_blocks(monkeypatch):
    """Counts the segment pairs the off-edge kernel is asked for: with the hint,
    the director-director blocks after the first at each spacing are not filled."""
    import momwire.bspline as bs

    asked = []
    real = bs._seg_seg_full_moments_offedge

    def spy(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw):
        asked.append(len(seg_l_i) * len(seg_l_j))
        return real(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw)

    monkeypatch.setattr(bs, "_seg_seg_full_moments_offedge", spy)
    _z(8, None)
    full = sum(asked)
    asked.clear()
    _z(8, _director_hint(8))
    # 8 directors of 20 segments: 28 ordered director pairs at 7 spacings,
    # so 2 * (28 - 7) blocks of 400 pairs are copied, not filled.
    assert full - sum(asked) == 2 * (28 - 7) * 400


def test_a_false_claim_is_filled_normally_and_named():
    """The reflector is not a copy of a director: the claim is dropped, with a
    warning, and Z is the no-hint fill bit for bit."""
    _, z0 = _z(4, None)
    bad = [[((2, 0), (2, 0)), ((0, 0), (0, 0))]]
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        s, z1 = _z(4, bad)
    assert any(issubclass(w.category, PairReuseDropped) for w in got)
    assert s.pair_reuse_dropped == (((0, 0), (0, 0)),)
    assert np.array_equal(z0.view(np.uint64), z1.view(np.uint64))


def test_a_near_miss_is_refused():
    """A member whose end points miss the representative's distances by more
    than the structure's size times 1e-12 is not copied."""
    wires, n = _yagi(3)
    # The last director slid 1 um along itself: a distance changes at first
    # order. (Lifting it in z would not do: that turns the parallel pair
    # rigidly about the element axis, a true repeat.)
    wires[4] = wires[4] + np.array([0.0, 1e-6, 0.0])
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n,
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=0.006,
        pair_reuse=[[((2, 0), (3, 0)), ((3, 0), (4, 0))]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PairReuseDropped)
        plan = s._pair_reuse_plan(s._build_geometry())
    assert plan == {}
    assert s.pair_reuse_dropped == (((3, 0), (4, 0)),)


def test_a_hint_over_ground_leaves_the_image_term_alone():
    """The copies are free-space moments; the ground's image term is filled as
    before, so Z over refl-coef ground still moves by a few ulp only."""
    _, z0 = _z(6, None, ground=True)
    _, z1 = _z(6, _director_hint(6), ground=True)
    assert np.max(np.abs(z1 - z0) / np.abs(z0)) <= 1e-13
