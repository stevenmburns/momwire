"""Razor's observer restriction on the BURIED fills (momwire#1337 phase 2,
PR 6): `RazorSolver._compute_Z_operator(rows=, compact=)` on the all-below,
detached and crossing decks, which PR 5 refused by name.

The decks are `test_crossing_serve_524.py`'s: `hub_deck(4)` (four buried
radials junction-joined at a buried hub to a rise, the rise joined to an
above monopole at the interface -- the crossing tent), the same radials with
no rise or mast (wholly below), and `crossing_deck(1)` pulled apart into a
buried wire and a mast with no junction between them (detached).

What is gated:

* **the fill to the bit** -- the restricted rows against the DENSE fill's
  rows as uint64, square (every other row exactly zero) and compact, on the
  NEC-5 path lane, the Gauss-Legendre lane and the numpy (no-accelerator)
  lane, with and without loading (distributed, lumped on a requested and an
  unrequested wire) and with a buried jacket's charge term; on the crossing
  deck both with and without the crossing tent's row in R;
* **the restricted path really ran** -- spies on the T1 window prepares and
  on the cross blocks' supports show each medium built only its requested
  rows and each cross block was asked only for them, and the streamed
  folds (the below block's and the cross blocks') went through the row
  map;
* **window independence at sub-window sizes 1 and 2** (proposal §5): every
  T1 window forced to 1 or 2 rows, still the dense default fill's rows;
* **the prepared fill binds its restriction** -- a replay under another R,
  under none, or an unrestricted prepare replayed restricted, raises, on
  every buried route;
* **negative controls** -- another radial's rows are not these rows, and a
  restricted fill that loses one below window, or one cross block, moves
  Z_in while the intact one leaves it bit-identical.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill, _schedule
from momwire import razor as _razor
from momwire.razor import RazorSolver
from test_crossing_serve_524 import crossing_deck, hub_deck

LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}, "numpy": {}}
JACKET = dict(insulation_radius=2.0e-3, insulation_eps_r=3.0)


def below_deck(n=4):
    """`hub_deck`'s radials alone: every wire below the interface."""
    d = hub_deck(n)
    d["wires"] = d["wires"][:n]
    d["n_per_edge_per_wire"] = d["n_per_edge_per_wire"][:n]
    d["junctions"] = [[(i, "end") for i in range(n)]]
    d["feeds"] = [(0, 2.0, 1 + 0j)]
    return d


def detached_deck():
    """`crossing_deck(1)` with the buried wire and the mast pulled apart:
    no junction in the plane, so no crossing tent."""
    d = crossing_deck(1)
    below, above = d["wires"]
    d["wires"] = [below + np.array([0.5, 0.0, -0.3]), above + np.array([0, 0, 0.3])]
    d.pop("junctions")
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return d


# deck name -> (builder, route attribute, the requested wires)
#
# "crossing" requests radial 0, the rise and the mast: R holds the hub
# junction tent joining radial 0 to the rise AND the crossing tent (both of
# its wires are requested). "crossing-r1" requests radial 1 alone: R holds
# its interior tents only -- the hub tent pairing it with radial 0 has a
# wing outside, so it is not held (`_ObserverRows`), and nothing of the
# crossing node is in R. "below" requests radials 0 and 1, so one hub
# junction tent is in R.
DECKS = {
    "below": (below_deck, "_below_plane", (0, 1)),
    "crossing": (hub_deck, "_crossing", (0, 4, 5)),
    "crossing-r1": (hub_deck, "_crossing", (1,)),
    "detached-above": (detached_deck, "_detached", (1,)),
    "detached-below": (detached_deck, "_detached", (0,)),
}


def solver(deck, *, lane="nec5", **kw):
    d = DECKS[deck][0]()
    d.pop("junctions", None)
    d.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = RazorSolver(**d, **LANES[lane])
    assert getattr(s, DECKS[deck][1]), deck
    return s


@pytest.fixture
def lane_switch(monkeypatch):
    def apply(lane):
        monkeypatch.setattr(_razor, "_FORCE_NUMPY", lane == "numpy")

    return apply


def wire_rows(s, wires):
    off = s._build_geometry()["seg_offsets"]
    return np.concatenate([np.arange(off[w], off[w + 1]) for w in wires])


def requested(s, deck):
    return wire_rows(s, DECKS[deck][2])


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def fills(s, rows):
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)
    Z_sq = quiet(s._compute_Z_operator, geom, rows=rows)
    Z_c, R = quiet(s._compute_Z_operator, geom, rows=rows, compact=True)
    return Z, Z_sq, Z_c, R


def assert_rows_are_dense(Z, Z_sq, Z_c, R):
    off = np.ones(Z.shape[0], dtype=bool)
    off[R] = False
    assert R.size and off.any()
    assert np.array_equal(bits(Z_sq[R]), bits(Z[R]))
    assert not np.any(Z_sq[off])
    assert Z_sq.flags.f_contiguous
    assert Z_c.shape == (R.size, Z.shape[1])
    assert np.array_equal(bits(Z_c), bits(Z[R]))


def crossing_tents_held(s, R):
    return sorted(set(m for m, _ in s._crossing_tents(s._build_geometry())) & set(R))


# ----------------------------------------------------------------------
# The fill, to the bit
# ----------------------------------------------------------------------


# The Gauss-Legendre and numpy fills of the crossing and detached decks are
# 4-7 s each: the push lane carries them. Every PR runs all five decks on the
# NEC-5 lane, the all-below deck on all three, and the crossing deck's radial
# 1 (no crossing tent in R) on all three.
FILL_CASES = [
    pytest.param(
        deck,
        lane,
        id=f"{deck}-{lane}",
        marks=[pytest.mark.slow]
        if lane != "nec5" and deck not in ("below", "crossing-r1")
        else [],
    )
    for deck in DECKS
    for lane in LANES
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("deck", "lane"), FILL_CASES)
def test_the_rows_fill_is_the_dense_fill_row_for_row(deck, lane, lane_switch):
    lane_switch(lane)
    assert _razor._use_razor_fill_accel() is (lane != "numpy")
    s = solver(deck, lane=lane)
    Z, Z_sq, Z_c, R = fills(s, requested(s, deck))
    assert_rows_are_dense(Z, Z_sq, Z_c, R)
    if deck == "crossing":
        # The crossing tent's row is one of the rows gated.
        assert len(crossing_tents_held(s, R)) == 1
    if deck == "crossing-r1":
        assert crossing_tents_held(s, R) == []


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deck", ["below", "crossing", "detached-below"])
def test_loading_and_a_buried_jacket_go_through_the_loading_map(deck):
    """Distributed loading on every wire, a lumped load on a requested AND
    an unrequested wire, and a jacket on every wire -- whose charge-side
    term (momwire#1154) lands on the buried ones: the stencils' unrequested
    entries are dropped, the requested ones land in the target's rows."""
    probe = solver(deck)
    n_w = len(probe.wires_polylines)
    want = DECKS[deck][2]
    other = next(w for w in range(n_w) if w not in want)
    kw = dict(
        wire_conductivity=3.5e7,
        lumped_loads=[(want[0], 0.4, 7.0 + 3.0j), (other, 0.4, 5.0 - 2.0j)],
        insulation_radius=[JACKET["insulation_radius"]] * n_w,
        insulation_eps_r=[JACKET["insulation_eps_r"]] * n_w,
    )
    s = solver(deck, **kw)
    prep = s._assemble_Z_prepare(s._build_geometry())
    assert prep["loading"] is not None and prep["charge"] is not None
    Z, Z_sq, Z_c, R = fills(s, requested(s, deck))
    assert_rows_are_dense(Z, Z_sq, Z_c, R)
    # The loading really is in those rows: the bare fill differs there.
    bare = solver(deck)
    Z_bare = quiet(bare._compute_Z_operator, bare._build_geometry())
    assert not np.array_equal(Z[R], Z_bare[R])


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deck", ["below", "crossing"])
def test_the_full_restriction_is_the_dense_fill(deck):
    s = solver(deck)
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)
    Z_c, R = quiet(
        s._compute_Z_operator, geom, rows=np.arange(geom["n_segs_total"]), compact=True
    )
    assert np.array_equal(R, np.arange(Z.shape[0]))
    assert np.array_equal(bits(Z_c), bits(Z))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deck", ["crossing", "detached-above"])
def test_the_whole_below_block_reference_route_agrees(deck, monkeypatch):
    """`_BELOW_FOLD_INTO = False` (the in-process reference: the whole below
    block, then one fold) under the restriction gives the same rows."""
    s = solver(deck)
    Z, *_ = fills(s, requested(s, deck))
    monkeypatch.setattr(_razor, "_BELOW_FOLD_INTO", False)
    _Z, Z_sq, Z_c, R = fills(s, requested(s, deck))
    assert_rows_are_dense(Z, Z_sq, Z_c, R)


# ----------------------------------------------------------------------
# The restricted path really ran
# ----------------------------------------------------------------------


class _Spies:
    """The T1 rows each `_schedule.prepare` built, the rows each cross
    block was asked for, and the streamed folds that went through a row
    map."""

    def __init__(self, mp):
        self.t1_rows, self.cross_rows, self.mapped_folds = [], [], 0
        prepare = _schedule.prepare

        def spy_prepare(n_rows, step, build, **kw):
            out = prepare(n_rows, step, build, **kw)
            self.t1_rows.append(sum(w[1] - w[0] for w in out))
            return out

        mp.setattr(_schedule, "prepare", spy_prepare)
        for name in ("cross_complete_block", "cross_complete_block_reversed"):
            inner = getattr(_crossing_fill, name)

            def spy_block(*a, _inner=inner, **kw):
                self.cross_rows.append(len(kw["support"][0]))
                return _inner(*a, **kw)

            mp.setattr(_crossing_fill, name, spy_block)
        fold = _crossing_fill._FusedEnds._fold

        def spy_fold(ends, *a, **kw):
            if ends.into is not None and ends.into_row_of is not None:
                self.mapped_folds += 1
            return fold(ends, *a, **kw)

        mp.setattr(_crossing_fill._FusedEnds, "_fold", spy_fold)

    def reset(self):
        self.t1_rows, self.cross_rows, self.mapped_folds = [], [], 0


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deck", ["crossing", "crossing-r1", "detached-below"])
def test_the_restricted_crossing_fill_builds_only_requested_rows(deck, monkeypatch):
    s = solver(deck)
    geom = s._build_geometry()
    rows = requested(s, deck)
    spies = _Spies(monkeypatch)

    _schedule.reset_stats()
    _razor._BELOW_FOLD_ROUTES["windows"] = 0
    s._compute_Z_operator(geom)
    dense = dict(
        t1=spies.t1_rows[:],
        cross=spies.cross_rows[:],
        below=_razor._BELOW_FOLD_ROUTES["windows"],
        stats=dict(_schedule.STATS),
    )
    assert spies.mapped_folds == 0
    spies.reset()
    _schedule.reset_stats()
    _razor._BELOW_FOLD_ROUTES["windows"] = 0
    _Z, R = s._compute_Z_operator(geom, rows=rows, compact=True)
    rest_stats = dict(_schedule.STATS)

    n = geom["n_basis_total"]
    # Prepares: the whole deck's (loading only, for this route), then each
    # medium's direct and image source sets -- the same count either way.
    assert len(spies.t1_rows) == len(dense["t1"]) == rest_stats["prepares"]
    assert dense["stats"]["basis_windows"] == 0 < rest_stats["basis_windows"]
    # No prepare built more rows than the dense one, the whole deck's built
    # R's, and a medium holding an unrequested row built fewer. (A medium
    # whose rows are all requested -- the mast side of "crossing" -- builds
    # them all.)
    assert spies.t1_rows[0] == R.size < dense["t1"][0] == n
    assert all(r <= d for r, d in zip(spies.t1_rows, dense["t1"]))
    assert sum(spies.t1_rows[1:]) < sum(dense["t1"][1:])
    # The cross blocks were asked only for requested test rows (a block
    # none of R reads is not formed at all), and the streamed ones folded
    # through the row map.
    assert sum(spies.cross_rows) < sum(dense["cross"])
    assert all(c <= R.size for c in spies.cross_rows)
    assert spies.mapped_folds > 0
    # The below block's windowed fold ran (fewer windows' rows, not fewer
    # windows necessarily: a window holding any requested row stays).
    assert _razor._BELOW_FOLD_ROUTES["windows"] > 0


@pytest.mark.filterwarnings("ignore")
def test_the_all_below_fill_builds_only_requested_rows(monkeypatch):
    s = solver("below")
    geom = s._build_geometry()
    spies = _Spies(monkeypatch)
    s._compute_Z_operator(geom)
    dense = spies.t1_rows[:]
    spies.reset()
    _schedule.reset_stats()
    _Z, R = s._compute_Z_operator(geom, rows=requested(s, "below"), compact=True)
    # Direct and image source sets, each R's rows only.
    assert dense == [geom["n_basis_total"]] * 2
    assert spies.t1_rows == [R.size] * 2
    assert _schedule.STATS["basis_windows"] > 0
    assert spies.cross_rows == []


@pytest.mark.filterwarnings("ignore")
def test_rows_none_never_builds_a_restriction(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("rows=None reached the restricted fill")

    monkeypatch.setattr(_razor, "_ObserverRows", boom)
    monkeypatch.setattr(_razor, "_medium_restriction", boom)
    for deck in ("below", "crossing", "detached-above"):
        s = solver(deck)
        quiet(s._compute_Z_operator, s._build_geometry())


# ----------------------------------------------------------------------
# Window independence at sub-window sizes 1 and 2 (proposal §5)
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize(
    ("deck", "lane"),
    [
        ("below", "nec5"),
        ("below", "gauss-legendre"),
        ("crossing", "nec5"),
        pytest.param("crossing", "numpy", marks=pytest.mark.slow),
        pytest.param("detached-above", "gauss-legendre", marks=pytest.mark.slow),
    ],
)
def test_one_and_two_row_windows_are_the_dense_rows(
    step, deck, lane, lane_switch, monkeypatch
):
    lane_switch(lane)
    s = solver(deck, lane=lane)
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)  # the default windows
    prepare = _schedule.prepare
    sizes = set()

    def forced(n_rows, _step, build, **kw):
        out = prepare(n_rows, step, build, **kw)
        sizes.update(w[1] - w[0] for w in out)
        return out

    monkeypatch.setattr(_schedule, "prepare", forced)
    rows = requested(s, deck)
    Z_sq = quiet(s._compute_Z_operator, geom, rows=rows)
    Z_c, R = quiet(s._compute_Z_operator, geom, rows=rows, compact=True)
    assert_rows_are_dense(Z, Z_sq, Z_c, R)
    # Every T1 window reached the kernels at 1 (or 2) rows: at step 2 a run
    # of odd length leaves a 1-row tail, so 2 occurs and 1 may.
    assert sizes == {1} if step == 1 else (2 in sizes and sizes <= {1, 2})


# ----------------------------------------------------------------------
# The prepared fill binds its restriction
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deck", ["below", "crossing", "detached-above"])
def test_a_replay_under_another_restriction_raises(deck):
    s = solver(deck)
    geom = s._build_geometry()
    omega = s.c * s.k
    r0 = _razor._ObserverRows(requested(s, deck), geom, compact=True)
    others = [w for w in range(len(s.wires_polylines)) if w not in DECKS[deck][2]]
    r1 = _razor._ObserverRows(wire_rows(s, others[:1]), geom, compact=True)
    assert not np.array_equal(r0.basis_rows, r1.basis_rows)
    prep0 = s._assemble_Z_prepare(geom, restrict=r0)
    with pytest.raises(ValueError, match="binds the rows= it was prepared for"):
        quiet(s._assemble_Z_from_prepared, geom, prep0, s.k, omega, restrict=r1)
    with pytest.raises(ValueError, match="replayed under no restriction"):
        quiet(s._assemble_Z_from_prepared, geom, prep0, s.k, omega)
    dense = s._assemble_Z_prepare(geom)
    with pytest.raises(ValueError, match="built for every row"):
        quiet(s._assemble_Z_from_prepared, geom, dense, s.k, omega, restrict=r0)
    # The same R on the other target is the same windows: served.
    sq = _razor._ObserverRows(requested(s, deck), geom, compact=False)
    Z_sq = quiet(s._assemble_Z_from_prepared, geom, prep0, s.k, omega, restrict=sq)
    Z_c = quiet(s._assemble_Z_from_prepared, geom, prep0, s.k, omega, restrict=r0)
    assert np.array_equal(bits(Z_sq[r0.basis_rows]), bits(Z_c))


@pytest.mark.filterwarnings("ignore")
def test_the_all_below_restriction_composes_with_the_sweep():
    """Restrict once at prepare, replay per k: each k's rows are that k's
    dense rows (the all-below fill replays its prepare; the crossing fill
    prepares its media per call)."""
    s = solver("below")
    geom = s._build_geometry()
    restrict = _razor._ObserverRows(requested(s, "below"), geom, compact=True)
    prep = s._assemble_Z_prepare(geom, restrict=restrict)
    dense = s._assemble_Z_prepare(geom)
    for kk in np.array([0.97, 1.03]) * s.k:
        Z = quiet(s._assemble_Z_from_prepared, geom, dense, kk, s.c * kk)
        Z_c = quiet(
            s._assemble_Z_from_prepared, geom, prep, kk, s.c * kk, restrict=restrict
        )
        assert np.array_equal(bits(Z_c), bits(Z[restrict.basis_rows]))


def test_a_square_restriction_cannot_fold_into_a_medium():
    s = solver("below")
    geom = s._build_geometry()
    sq = _razor._ObserverRows(requested(s, "below"), geom, compact=False)
    prep = s._assemble_Z_prepare(geom, restrict=sq)
    n = geom["n_basis_total"]
    Z = np.zeros((n, n), dtype=np.complex128)
    with pytest.raises(ValueError, match="folds a compact target"):
        quiet(
            s._assemble_Z_below_plane,
            geom,
            prep,
            s.k,
            s.c * s.k,
            into=(Z, np.arange(n), np.arange(n)),
            restrict=sq,
        )


def test_a_split_wire_is_refused_by_name_on_the_crossing_deck():
    s = solver("crossing")
    with pytest.raises(ValueError, match="restricts by whole wires"):
        s._compute_Z_operator(s._build_geometry(), rows=np.arange(3))


# ----------------------------------------------------------------------
# Negative controls
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deck", ["below", "crossing"])
def test_negative_another_radials_rows_are_not_these_rows(deck):
    """The bitwise gate can fail: radial 2's tents are a rotation of radial
    1's, the same shape and the same physics, and not the same entries."""
    s = solver(deck)
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)
    Z1, R1 = quiet(s._compute_Z_operator, geom, rows=wire_rows(s, (1,)), compact=True)
    Z2, R2 = quiet(s._compute_Z_operator, geom, rows=wire_rows(s, (2,)), compact=True)
    assert Z1.shape == Z2.shape
    assert np.array_equal(bits(Z1), bits(Z[R1]))
    assert np.array_equal(bits(Z2), bits(Z[R2]))
    assert not np.array_equal(bits(Z1), bits(Z2))


def _z_in(s, Z):
    cols = s._port_columns(s._build_geometry())
    c = np.linalg.solve(Z, cols @ s._port_voltages())
    return complex(s._port_voltages()[0] / (cols.T @ c)[0])


def _hybrid(s, rows):
    """Z_in from the dense Z with R's rows replaced by the restricted fill,
    and the dense Z_in."""
    geom = s._build_geometry()
    Z = quiet(s._compute_Z_operator, geom)
    Z_c, R = quiet(s._compute_Z_operator, geom, rows=rows, compact=True)
    Z_h = Z.copy()
    Z_h[R] = Z_c
    return _z_in(s, Z_h), _z_in(s, Z)


@pytest.mark.filterwarnings("ignore")
def test_negative_a_fill_that_loses_a_below_window_moves_z_in(monkeypatch):
    """The restricted rows feed an answer, and a corruption of the restricted
    path reaches it: the hybrid Z_in is the dense Z_in EXACTLY. Dropping the
    first window of the below medium's restricted DIRECT prepare alone is
    caught by the lockstep fold (the image would walk other windows); dropping
    it from both of the below medium's source sets (radial 0's tents lose
    their below block) moves Z_in."""
    s = solver("crossing")
    rows = requested(s, "crossing")
    z_h, z_d = _hybrid(s, rows)
    assert z_h == z_d

    prepare = _schedule.prepare
    seen = []
    drop = set()

    def lossy(n_rows, step, build, *, checkpoint=None, restrict=None):
        out = prepare(n_rows, step, build, checkpoint=checkpoint, restrict=restrict)
        if restrict is not None:
            seen.append(n_rows)
            # The whole deck's two source sets, then the above medium's two,
            # then the below medium's direct and image sets.
            if len(seen) in drop:
                out = out[1:]
        return out

    monkeypatch.setattr(_schedule, "prepare", lossy)
    drop.add(5)
    with pytest.raises(AssertionError, match="the image walks other row windows"):
        _hybrid(s, rows)
    seen.clear()
    drop.add(6)
    z_bad, _ = _hybrid(s, rows)
    assert len(seen) == 6
    assert abs(z_bad - z_d) / abs(z_d) > 1e-2, (z_bad, z_d)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "name", ["cross_complete_block", "cross_complete_block_reversed"]
)
def test_negative_a_fill_that_loses_a_cross_block_moves_z_in(name, monkeypatch):
    """Each restricted cross block reaches Z_in: answering it as zeros (and
    folding nothing) under the restriction moves the hybrid Z_in, where the
    intact restricted fill leaves it the dense Z_in exactly."""
    s = solver("crossing")
    rows = requested(s, "crossing")
    z_h, z_d = _hybrid(s, rows)
    assert z_h == z_d
    inner = getattr(_crossing_fill, name)
    dropped = []

    def lose(*a, **kw):
        if "into_row_of" in kw:
            dropped.append(len(kw["support"][0]))
            return np.zeros((len(kw["support"][0]), len(kw["support"][1])), complex)
        return inner(*a, **kw)

    monkeypatch.setattr(_crossing_fill, name, lose)
    z_bad, _ = _hybrid(s, rows)
    assert dropped and all(dropped)
    assert abs(z_bad - z_d) / abs(z_d) > 1e-3, (z_bad, z_d)
