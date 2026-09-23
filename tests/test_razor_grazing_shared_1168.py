"""momwire#1168 U3: razor's below-plane grazing floor, on the shared extents.

`RazorSolver._below_plane_grazing_refusal` used to build all-pairs (n, n)
rho / hh / angle arrays; it now asks bspline's chunked, C++-backed
`_pair_extents_below` (the helper bspline and SG already share), and
`_below_remainder_th_min`'s hand-chunked (obs x src) loop became
`bspline._pair_extents_below_rect`. Razor's currents are unchanged bit for
bit (measured on hub_deck(16) x2 / x4 and crossing_deck(1) against 69f94e7,
not re-run here: those are minutes, not a gate's seconds).

The trap the move had to keep: the shared helper DROPS rho = 0 pairs, and
razor's all-pairs form read them through atan2 — a point kept exactly IN the
plane met itself at atan2(0, 0) = 0 deg, and one above it at -90 deg. That
self-pair is the whole refusal of a plane-touching point that is not a
declared crossing node, so razor restores it explicitly. The decks below
reach it; `test_a_plane_touching_end_that_is_not_a_crossing_member_still_
refuses` (tests/test_razor_crossing_fill_813.py) does not — its deck is refused
at construction as a mid-span crossing.

Mutation record (2026-09-23): with the explicit zero-depth term removed,
the zero-depth deck is SERVED (`buried_serve_refusal()` returns None and
the fill completes) and the poking deck's pre-flight returns None while its
fill raises a different sentence, so both tests below fail.
"""

import math

import numpy as np
import pytest

from momwire import _medium_spec, _sommerfeld_below
from momwire import bspline as _bs
from momwire import razor as _razor
from momwire.razor import RazorSolver

from test_crossing_serve_524 import A_WIRE, SOIL_A, WL7, crossing_deck


def _floor():
    return _sommerfeld_below._SOMM_BELOW_TH_MIN_DEG


def _sentence(th_deg, depth):
    return _bs._BURIED_GRAZING_REFUSAL.format(th=th_deg, floor=_floor(), depth=depth)


def _junction_fan(end_of_second):
    """Two below wires and an above riser in one DECLARED crossing junction.

    The second below wire's plane end sits `end_of_second` (x, z) off the
    node — inside the declared-junction coincidence check (1e-3 of the
    shortest terminal segment), outside razor's 1e-9 `plan_skip` radius. So
    that point survives the skip: it is exactly the plane point that is not
    a crossing node, and only its self-pair in the all-pairs form sees it.
    """
    dx, dz = end_of_second
    return dict(
        wires=[
            np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 0.0)]),
            np.array([(2.0, 0.0, -1.0), (dx, 0.0, dz)]),
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]),
        ],
        n_per_edge_per_wire=[[8], [8], [15]],
        junctions=[[(0, "end"), (1, "end"), (2, "start")]],
        feeds=[(2, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def _grazing_deck():
    """A wholly-below wire running 0.1 mm under the plane for 3 m: its two
    shallow endpoints pair at atan2(2e-4, 3) = 0.0038 deg, a real grazing
    pair (rho > 0) the shared helper itself finds."""
    return dict(
        wires=[np.array([(0.0, 0.0, -2.0), (0.0, 0.0, -1e-4), (3.0, 0.0, -1e-4)])],
        n_per_edge_per_wire=[[8, 10]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def _spy_helpers(monkeypatch):
    """Record what each shared helper returned to razor."""
    seen = {"square": [], "rect": []}
    real_sq, real_rect = _bs._pair_extents_below, _bs._pair_extents_below_rect

    def square(x, y, d_b, **kw):
        out = real_sq(x, y, d_b, **kw)
        seen["square"].append((x.shape[0], out))
        return out

    def rect(obs, src, d_obs, d_src, **kw):
        out = real_rect(obs, src, d_obs, d_src, **kw)
        seen["rect"].append((obs.shape[0], src.shape[0], out))
        return out

    monkeypatch.setattr(_bs, "_pair_extents_below", square)
    monkeypatch.setattr(_bs, "_pair_extents_below_rect", rect)
    return seen


# ----------------------------------------------------------------------
# the zero-depth self-pair refusal, kept explicitly
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "th_deg", "depth"),
    [
        # IN the plane: the old self-pair atan2(0, 0) = 0 deg, depth sum 0.
        ((1e-7, 0.0), 0.0, 0.0),
        # 1 nm ABOVE it (inside the touch tolerance): atan2(-2e-9, 0).
        ((1e-7, 1e-9), -90.0, -2e-9),
    ],
    ids=["in-plane", "poking"],
)
def test_a_kept_plane_point_refuses_on_its_self_pair(
    monkeypatch, offset, th_deg, depth
):
    seen = _spy_helpers(monkeypatch)
    s = RazorSolver(**_junction_fan(offset), n_qp_path=8)
    assert s._crossing, "the deck must take the crossing route"
    want = _sentence(th_deg, depth)
    # Byte-equal to what 69f94e7's all-pairs spelling printed for this deck.
    assert s.buried_serve_refusal() == want
    # It is the ZERO-DEPTH term that refused, not a real pair: both shared
    # helpers, which never see a rho = 0 pair's angle, read clear of the floor.
    floor = math.radians(_floor())
    (_n, (_r1, th_sq)), *_ = seen["square"]
    (_no, _ns, (th_rect, _hh)), *_ = seen["rect"]
    assert th_sq > floor and th_rect > floor, (th_sq, th_rect)
    # And the fill refuses with the same sentence, before any grid.
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    assert str(exc.value) == want


def test_a_rho_zero_pair_at_depth_still_reads_pi_over_2():
    """The other half of the rho = 0 rule: a straight-down vertical has
    rho = 0 between EVERY pair of plan points, all at depth, so the
    all-pairs form read pi/2 everywhere and the helper (dropping them all)
    reads atan(inf) = pi/2 — no zero-depth term fires, nothing refuses."""
    deck = dict(
        wires=[np.array([(0.0, 0.0, -3.0), (0.0, 0.0, -0.5)])],
        n_per_edge_per_wire=[[6]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    s = RazorSolver(**deck, n_qp_path=8)
    geom = s._build_geometry()
    assert s._below_plane_grazing_refusal(geom) is None


# ----------------------------------------------------------------------
# a real grazing pair: the helper's own minimum, same sentence
# ----------------------------------------------------------------------


def test_a_grazing_deck_refuses_with_the_same_sentence(monkeypatch):
    seen = _spy_helpers(monkeypatch)
    s = RazorSolver(**_grazing_deck(), n_qp_path=8)
    assert s._below_plane
    got = s.buried_serve_refusal()
    # Byte-equal to 69f94e7's text. The plan pair wins here (the Gauss nodes
    # of the remainder sit inside the segments, a little deeper-in-angle).
    assert got == _sentence(math.degrees(math.atan2(2e-4, 3.0)), 2e-4)
    assert "theta = 0.00382 deg" in got
    (_n, (_r1, th_sq)), *_ = seen["square"]
    (_no, _ns, (th_rect, _hh)), *_ = seen["rect"]
    assert th_sq <= th_rect
    with pytest.raises(ValueError) as exc:
        s.compute_impedance()
    assert str(exc.value) == got


# ----------------------------------------------------------------------
# the shared helpers are the ones reached
# ----------------------------------------------------------------------


def test_razor_reads_its_plan_extents_through_bspline(monkeypatch):
    """`test_g980c_4_the_plan_reads_pair_extents_through_bspline`'s razor
    twin: the served crossing FILL (not only the pre-flight) reaches both
    shared helpers. The fill is stopped at the rectangular one — the grazing
    check runs before any grid, so nothing past it is needed."""

    class _Reached(Exception):
        pass

    seen = {}
    real_sq = _bs._pair_extents_below

    def square(x, y, d_b, **kw):
        seen["square"] = x.shape[0]
        return real_sq(x, y, d_b, **kw)

    def rect(obs, src, d_obs, d_src, **kw):
        seen["rect"] = (obs.shape[0], src.shape[0])
        raise _Reached

    monkeypatch.setattr(_bs, "_pair_extents_below", square)
    monkeypatch.setattr(_bs, "_pair_extents_below_rect", rect)
    with pytest.raises(_Reached):
        RazorSolver(**crossing_deck(1), n_qp_path=8).compute_impedance()
    assert seen.get("square", 0) > 0, "razor's plan did not call bspline's extents"
    assert min(seen["rect"]) > 0


# ----------------------------------------------------------------------
# the rectangular twin is razor's old hand-chunked loop, bit for bit
# ----------------------------------------------------------------------


def _hand_chunked(obs, src, gz):
    """`RazorSolver._below_remainder_th_min`'s loop as of 69f94e7, verbatim."""
    d_s = gz - src[:, 2]
    best, best_hh = np.inf, 0.0
    step = max(1, 4_000_000 // max(1, src.shape[0]))
    for i0 in range(0, obs.shape[0], step):
        o = obs[i0 : i0 + step]
        rho = np.hypot(
            o[:, 0][:, None] - src[:, 0][None, :],
            o[:, 1][:, None] - src[:, 1][None, :],
        )
        hh = (gz - o[:, 2])[:, None] + d_s[None, :]
        th = np.arctan2(hh, rho)
        k = int(np.argmin(th))
        if th.flat[k] < best:
            best, best_hh = float(th.flat[k]), float(hh.flat[k])
    return best, best_hh


def _remainder_clouds(s, geom):
    obs = s._testing_paths(geom)[0].reshape(-1, 3)
    xg, _wg = np.polynomial.legendre.leggauss(s.n_qp_sommerfeld)
    tq = 0.5 * (xg + 1.0)
    src = (
        geom["seg_p0"][:, None, :]
        + (tq[None, :, None] * geom["seg_h"][:, None, None]) * geom["seg_t"][:, None, :]
    ).reshape(-1, 3)
    return obs, src


@pytest.mark.parametrize("name", ["crossing", "grazing"])
@pytest.mark.parametrize("pairs", [1 << 20, 997, 1])
def test_the_rectangular_twin_is_the_hand_chunked_loop(name, pairs):
    """Any chunking gives the FIRST row-major minimum, so the pair and its
    depth sum are the old loop's bit for bit — asserted across a chunk size
    that splits rows mid-deck and one that walks a row at a time."""
    if name == "crossing":
        s = RazorSolver(**crossing_deck(1), n_qp_path=8)
        geom, _rows, _chop = s._medium_geometry(s._build_geometry(), _medium_spec.BELOW)
    else:
        s = RazorSolver(**_grazing_deck(), n_qp_path=8)
        geom = s._build_geometry()
    gz = float(s.ground_z)
    obs, src = _remainder_clouds(s, geom)
    want = _hand_chunked(obs, src, gz)
    got = _bs._pair_extents_below_rect(
        obs, src, gz - obs[:, 2], gz - src[:, 2], pairs=pairs
    )
    assert got == want
    # ...and it is what razor's own method now returns.
    assert s._below_remainder_th_min(geom) == want


def test_the_module_still_owns_one_copy_of_the_sentence():
    """Razor formats bspline's template; it keeps no copy of its own."""
    assert not hasattr(_razor, "_BURIED_GRAZING_REFUSAL")
