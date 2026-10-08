"""Bit-identity gate for the FUSED mixed fill — momwire#1224.

`SinusoidalSolver._assemble_Z_mixed` used to hold three whole Φ tables
(3N², 1.65 GB at invl x32): the class blocks wrote their own quadrants, a
non-crossing deck's transmitted pair classes were added into the other two,
and the whole triple was then reduced into Z. It now walks observer bands:
each band's three Φ rows are built from both classes' contributions in a
band-sized buffer and reduced into Z's rows at once, so Z is the fill's
only (N, N) array.

The argument (in `_assemble_Z_mixed`'s band-loop comment): every Φ element
had exactly one writer (the four quadrants partition the matrix), the Z
bands are sub-bands of each class band so no class value is computed over
a different row set, and the sparse reduction's bits do not follow the
band's row count. So the gates here are:

  * the fused fill equals the whole-Φ fill (the old algorithm, spelled out
    below from `_class_block` / `_transmitted_tensor` and whole products —
    none of the fused path's own helpers) BYTE FOR BYTE, on a crossing deck
    (invl x4, N = 734) and a non-crossing buried deck (a detached monopole
    over the same 16 buried radials, N = 700, which is the transmitted-term
    path), at Z band heights that cut a class band finely (1, 7, 100) and
    ones that do not, and with the class bands themselves split (a 1 MB
    budget);
  * the fused path ran: a spy on `_reduce_phi_band` records every band's
    row count, and the whole-table reduction is never reached;
  * Z is the only (N, N) the fill holds: the traced peak of the Φ phase is
    under 2 Z (the old triple put it at ≥ 3 Z).

Mutations checked by hand (momwire#1224; not committed): reordering
`_reduce_phi_band` to `A + C + B` and dropping the `− s` from the
transmitted block's row offset each fail the byte gates.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from test_crossing_serve_524 import hub_deck, invl_deck

from momwire import _schedule
from momwire import sinusoidal as sn
from momwire.sinusoidal import SinusoidalSolver


def detached_hub_deck(x=4):
    """A NON-crossing mixed deck: `hub_deck`'s 16 buried radials joined at
    the buried hub, no rise, under a detached 10 m monopole starting 0.5 m
    up — so both transmitted directions are filled. N = 175·x."""
    d = hub_deck(n_radials=16)
    wires = list(d["wires"][:16])
    npe = [[n * x for n in e] for e in d["n_per_edge_per_wire"][:16]]
    wires.append(np.array([(0.0, 0.0, 0.5), (0.0, 0.0, 10.5)]))
    npe.append([15 * x])
    d.update(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(16)]],
        feeds=[(16, 4.3333333333, 1 + 0j)],
    )
    return d


DECKS = {"invl": lambda: invl_deck(x=4), "detached": detached_hub_deck}


def _bytes_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def whole_phi_tables(s, geom):
    """The pre-fuse `_assemble_Z_mixed`'s three whole Φ tables and its Ms,
    written against the solver's building blocks only; also returns what
    the rest of that fill reads."""
    eta = s._fill_eta(s.k, None)
    below = s._below_segments(geom)
    medium = s._fill_medium(geom)
    N = int(geom["n_segs"])
    crossing = s._is_crossing(geom)
    view = s._mixed_view(geom, below, medium)
    plan = s._mixed_serve_plan(geom, below, medium, crossing)
    h = np.asarray(geom["seg_h"], dtype=np.float64)
    k_abs = np.where(below, abs(medium.k_m), medium.k_p)
    cos_shape = "cos-1" if float(np.min(k_abs * h)) < sn._WELL_SCALED_KD else "cos"
    starts = view["starts"]
    n_idx = np.repeat(np.arange(N, dtype=np.int64), starts[1:] - starts[:-1])
    sigma = view["sigma"]
    A_eff = sigma * (view["A"] if cos_shape == "cos" else view["AC"])
    coefs = (A_eff, view["B"], sigma * view["C"])
    if N < sn._DENSE_ASSEMBLY_THRESHOLD:
        Ms = []
        for c in coefs:
            M = np.zeros((N, N), dtype=np.complex128)
            M[n_idx, view["jbasis"]] = c
            Ms.append(M)
    else:
        Ms = [
            sn.scipy.sparse.csc_matrix((c, (n_idx, view["jbasis"])), shape=(N, N))
            for c in coefs
        ]
    Phi = [np.zeros((N, N), dtype=np.complex128) for _ in range(3)]
    for keep, k_cls, med, eta_cls in (
        (~below, medium.k_p, None, s._fill_eta(medium.k_p, eta)),
        (below, medium.k_m, medium, s._medium_eta(medium)),
    ):
        idx = np.nonzero(keep)[0]
        for P, b in zip(
            Phi, s._class_block(geom, keep, idx, k_cls, med, eta_cls, cos_shape, plan)
        ):
            P[np.ix_(idx, idx)] = b
    c_all, t_all = geom["seg_centers"], geom["seg_tangents"]
    for src_keep, obs_below in () if crossing else ((below, False), (~below, True)):
        obs_keep = below if obs_below else ~below
        T = s._transmitted_tensor(
            geom, medium, plan, src_keep, obs_keep, obs_below, c_all, t_all,
            row_group=1, cos_shape=cos_shape,
        )  # fmt: skip
        r, c = np.nonzero(obs_keep)[0], np.nonzero(src_keep)[0]
        for P, t in zip(Phi, T):
            P[np.ix_(r, c)] += t[np.ix_(r, c)]
    return Phi, Ms, (view, medium, below, crossing)


def whole_phi_fill(s, geom):
    """The pre-fuse `_assemble_Z_mixed`: whole Φ tables, whole products."""
    Phi, Ms, (view, medium, below, crossing) = whole_phi_tables(s, geom)
    G = Phi[0] @ Ms[0]
    G += Phi[1] @ Ms[1]
    G += Phi[2] @ Ms[2]
    if crossing:
        s._crossing_point_rows(geom, view, medium, below, into=G)
    s._apply_loading(G, geom, view, None, medium=medium)
    return G


def _solver(name, swept_mem_mb):
    s = SinusoidalSolver(**DECKS[name](), swept_mem_mb=swept_mem_mb)
    return s, s._build_geometry()


@pytest.fixture(scope="module")
def reference():
    """The whole-Φ fill of each deck at each class budget, computed once."""
    out = {}
    for name in DECKS:
        for mb in (256, 1):
            s, geom = _solver(name, mb)
            out[name, mb] = whole_phi_fill(s, geom)
    return out


def _fused(name, mb, band, monkeypatch):
    """The production fill with the Z band forced to `band` rows (None: the
    budgeted one), and the row count of every band it reduced."""
    rows = []
    orig = sn._reduce_phi_band

    def spy(Phi, Ms, out):
        rows.append(out.shape[0])
        return orig(Phi, Ms, out)

    def whole(*a, **k):  # pragma: no cover - the failure being guarded
        raise AssertionError("the sparse regime reached the whole-table reduction")

    monkeypatch.setattr(sn, "_reduce_phi_band", spy)
    monkeypatch.setattr(sn, "_reduce_phi_rows_in_place", whole)
    if band is not None:
        monkeypatch.setattr(SinusoidalSolver, "_mixed_band_rows", lambda self, N: band)
    s, geom = _solver(name, mb)
    G, _view = s._assemble_Z_mixed(geom, s._fill_eta(s.k, None))
    return s, geom, G, rows


def _class_band_heights(s, geom):
    """The class bands' row counts, in the fill's order (above, then below)."""
    below = s._below_segments(geom)
    out = []
    for keep in (~below, below):
        idx = np.nonzero(keep)[0]
        n = idx.size
        chunk = max(1, int(s.swept_mem_mb * 1024 * 1024 // s._fill_row_bytes(n)))
        out += [
            min(b0 + chunk, e0) - b0
            for s0, e0 in _schedule.runs(idx)
            for b0 in range(s0, e0, chunk)
        ]
    return out


@pytest.mark.parametrize("name", list(DECKS))
@pytest.mark.parametrize(
    ("mb", "band"), [(256, 1), (256, 7), (256, 100), (256, None), (1, None), (1, 5)]
)
def test_fused_fill_is_the_whole_phi_fill(name, mb, band, reference, monkeypatch):
    s, geom, G, rows = _fused(name, mb, band, monkeypatch)
    assert s._is_crossing(geom) == (name == "invl")
    N = int(geom["n_segs"])
    assert N >= sn._DENSE_ASSEMBLY_THRESHOLD
    zband = band if band is not None else s._mixed_band_rows(N)
    # The bands the fill reduced ARE each class band cut into zband rows —
    # counted, so a fill that went whole (or skipped rows) cannot pass.
    want = [
        min(zband, h - z0)
        for h in _class_band_heights(s, geom)
        for z0 in range(0, h, zband)
    ]
    assert rows == want
    assert sum(rows) == N
    # Both regimes are exercised: Z bands finer than a class band, and not.
    heights = _class_band_heights(s, geom)
    if band in (1, 7, 100):
        assert len(rows) > len(heights)
    if mb == 256 and band is None:
        assert rows == heights  # one Z band per class band
    if mb == 1:
        assert len(heights) > 2  # the class bands themselves are split
    assert G.flags.c_contiguous
    assert _bytes_equal(G, reference[name, mb])


def _small_crossing_deck():
    """`test_sin_buried_hoist_1221`'s two-member crossing junction, N = 10."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
            np.array([(0.0, 0.0, 0.0), (0.5, 0.0, -1.5)]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "start"), (1, "start")]],
        wavelength=299792458.0 / 7e6,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


@pytest.mark.parametrize("crossing", [True, False], ids=["crossing", "detached"])
def test_dense_regime_is_the_old_fill_verbatim(crossing, monkeypatch):
    """Below the dense threshold the reduction is a zgemm, whose bits follow
    the row count, so the fill keeps the whole table and reduces it once:
    the old fill verbatim, even with the Z band forced to 1 row."""
    if crossing:
        deck = _small_crossing_deck()
    else:
        from test_sin_buried_below_1222 import detached

        deck = detached(1)
    s = SinusoidalSolver(**deck)
    geom = s._build_geometry()
    N = int(geom["n_segs"])
    assert N < sn._DENSE_ASSEMBLY_THRESHOLD
    assert s._is_crossing(geom) == crossing
    ref = whole_phi_fill(s, geom)
    calls = []
    orig = sn._reduce_phi_rows_in_place

    def spy(Phi, Ms, band):
        calls.append((Phi[0].shape, band, isinstance(Ms[0], np.ndarray)))
        return orig(Phi, Ms, band)

    monkeypatch.setattr(sn, "_reduce_phi_rows_in_place", spy)
    monkeypatch.setattr(SinusoidalSolver, "_mixed_band_rows", lambda self, N: 1)
    G, _view = s._assemble_Z_mixed(geom, s._fill_eta(s.k, None))
    assert calls == [((N, N), N, True)]
    assert _bytes_equal(G, ref)


@pytest.mark.parametrize("name", list(DECKS))
def test_z_is_the_only_square_array(name, reference, monkeypatch):
    """Traced peak of the fill's Φ phase at a 1 MB band budget: under 2 Z.
    Measured (momwire#1224): invl 1.27 Z and the detached deck 1.77 Z (Z
    plus its two compact transmitted blocks, ~0.47 Z at x4, and their
    evaluation transient); the whole-Φ fill measured 3.34 Z and 9.77 Z
    (the triple, and on the detached deck two (3, N, N) transmitted
    tensors on top).

    The phase runs from the serve plan's return to the crossing rows (their
    own unit) or, on a non-crossing deck, to the end of the fill. The plan
    is outside it on purpose: it runs before Z exists, so its transient
    never stacks on Z (on the detached deck it is ~9 Z at x4, the pair
    extents' own cost — a separate question from this one)."""
    # `reference` has filled both decks in this process, so the grid caches
    # are warm and their one-off build is not what the count measures.
    s, geom = _solver(name, 1)
    eta = s._fill_eta(s.k, None)
    N = int(geom["n_segs"])
    z_bytes = N * N * 16
    seen = {}
    plan = SinusoidalSolver._mixed_serve_plan
    cross = SinusoidalSolver._crossing_point_rows

    def after_plan(self, *a, **k):
        out = plan(self, *a, **k)
        seen["base"] = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        return out

    def at_crossing(self, *a, **k):
        seen["peak"] = tracemalloc.get_traced_memory()[1]
        return cross(self, *a, **k)

    monkeypatch.setattr(SinusoidalSolver, "_mixed_serve_plan", after_plan)
    monkeypatch.setattr(SinusoidalSolver, "_crossing_point_rows", at_crossing)
    tracemalloc.start()
    try:
        G, _view = s._assemble_Z_mixed(geom, eta)
        peak = seen.get("peak", tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()
    assert ("peak" in seen) == (name == "invl")
    assert G.shape == (N, N)
    assert peak - seen["base"] < 2 * z_bytes, (peak - seen["base"]) / z_bytes
