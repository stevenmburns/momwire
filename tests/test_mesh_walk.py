"""`momwire._mesh_walk` against the per-edge loops it replaces, bit for bit."""

import numpy as np
import pytest

from momwire import _mesh_walk as MW
from momwire._vec3 import norm3


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint8).tobytes(), a.shape, a.dtype


def _mesh(rng, n_wires=60):
    """Wires of 1-4 edges: random, axis-aligned (a zero step component, which
    takes numpy linspace's other branch), tiny and long, negative coordinates."""
    polylines, per_edge = [], []
    for w in range(n_wires):
        n_edges = int(rng.integers(1, 5))
        pts = [rng.normal(size=3) * 10.0 ** rng.integers(-4, 3)]
        for _ in range(n_edges):
            step = rng.normal(size=3)
            kind = w % 4
            if kind == 1:
                step[rng.integers(0, 3)] = 0.0
            elif kind == 2:
                step[:2] = 0.0
            elif kind == 3:
                step *= 1e-7
            pts.append(pts[-1] + step)
        polylines.append(np.array(pts))
        per_edge.append([int(rng.integers(1, 30)) for _ in range(n_edges)])
    return polylines, per_edge


def _ref_knots(polylines, per_edge, subdiv):
    out = []
    for polyline, npe in zip(polylines, per_edge):
        parts = []
        for i, n_e in enumerate(npe):
            seg = np.linspace(polyline[i], polyline[i + 1], n_e * subdiv + 1)
            parts.append(seg if i == 0 else seg[1:])
        out.append(np.vstack(parts))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("subdiv", [1, 3])
def test_wire_knots_and_arcs_are_the_loops_bits(seed, subdiv):
    polylines, per_edge = _mesh(np.random.default_rng(seed))
    ref = _ref_knots(polylines, per_edge, subdiv)
    knots, offs = MW.wire_knots(polylines, per_edge, subdiv)
    assert offs.tolist() == np.cumsum([0] + [k.shape[0] for k in ref]).tolist()
    assert _bits(knots) == _bits(np.concatenate(ref))
    arcs = MW.wire_arcs(knots, offs)
    ref_arcs = [
        np.concatenate([[0.0], np.cumsum(norm3(np.diff(k, axis=0)))]) for k in ref
    ]
    assert _bits(arcs) == _bits(np.concatenate(ref_arcs))


def test_segmented_cumsum_is_each_slices_cumsum():
    rng = np.random.default_rng(7)
    lengths = rng.integers(0, 40, size=200)
    offs = np.concatenate([[0], np.cumsum(lengths)])
    x = rng.normal(size=int(offs[-1])) * 10.0 ** rng.integers(-8, 8, size=int(offs[-1]))
    got = MW.segmented_cumsum(x, offs)
    ref = np.concatenate([np.cumsum(x[a:b]) for a, b in zip(offs[:-1], offs[1:])])
    assert _bits(got) == _bits(ref)


def _structure():
    """A solved multi-wire structure: a bent two-edge wire, a junction, and
    wires of 1 to 9 segments (momwire#1916 part D's all-wire readouts)."""
    from momwire.sinusoidal import SinusoidalSolver

    wires = [
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.31), (0.07, 0.02, 0.5)],
        [(0.0, 0.0, 0.0), (0.12, 0.0, -0.05)],
        [(0.3, 0.1, 0.2), (0.3, 0.1, 0.27)],
        [(0.0, 0.0, 0.0), (-0.2, 0.05, 0.0)],
    ]
    solver = SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[7, 4], [3], [1], [9]],
        wavelength=1.1,
        wire_radius=1e-3,
        feeds=[(0, None, 1.0)],
    )
    sol = solver.compute_port_solution()
    return solver, sol.coeffs[:, 0]


def test_all_wire_readouts_are_the_per_wire_spellings_bits():
    """`currents_at_knots`, `current_slopes` and `element_currents` walk
    every wire at once; each answer is the per-wire spelling's, bit for bit,
    here rebuilt from `_slope_eval_points` and the knot walk loop."""
    solver, alpha = _structure()
    geom = solver._build_geometry()
    seg_h = geom["seg_h"]
    view = solver._readout_view(geom)
    rng = np.random.default_rng(3)
    # Samples on knots, inside spans, at and past both ends.
    s_array = []
    for w in range(len(solver.wires_polylines)):
        f, last = geom["wire_first"][w], geom["wire_last"][w]
        arc = np.concatenate([[0.0], np.cumsum(seg_h[f : last + 1])])
        s_array.append(
            np.concatenate([arc, rng.uniform(-0.1, 1.1, 9) * arc[-1], arc[1:-1] * 0.5])
        )
    got = solver.current_slopes(alpha, s_array)
    for w, sv in enumerate(s_array):
        seg, xi = solver._slope_eval_points(geom, w, sv)
        want = solver._evaluate_basis_slope_at_points(view, seg, xi, alpha)
        assert _bits(got[w]) == _bits(np.asarray(want, dtype=np.complex128))
    default = solver.current_slopes(alpha)
    for w in range(len(solver.wires_polylines)):
        f, last = geom["wire_first"][w], geom["wire_last"][w]
        arc = np.concatenate([[0.0], np.cumsum(seg_h[f : last + 1])])
        seg, xi = solver._slope_eval_points(geom, w, arc)
        want = solver._evaluate_basis_slope_at_points(view, seg, xi, alpha)
        assert _bits(default[w]) == _bits(np.asarray(want, dtype=np.complex128))

    # currents_at_knots' knot path: per wire, left edges then right edges.
    knots_i = solver.currents_at_knots(alpha)
    for w in range(len(solver.wires_polylines)):
        f, last = geom["wire_first"][w], geom["wire_last"][w]
        m = last - f + 1
        segs = np.arange(f, last + 1)
        lw = np.full(m, 0.5)
        lw[0] = 1.0
        rw = np.full(m, 0.5)
        rw[-1] = 1.0
        ev = solver._evaluate_basis_at_points(
            view,
            np.concatenate([segs, segs]),
            np.concatenate([-0.5 * seg_h[segs], +0.5 * seg_h[segs]]),
            alpha,
        )
        want = np.zeros(m + 1, dtype=np.complex128)
        np.add.at(
            want,
            np.concatenate([np.arange(m), np.arange(1, m + 1)]),
            np.concatenate([lw, rw]) * ev,
        )
        assert _bits(knots_i[w]) == _bits(want)

    # element_currents: the per-wire walk it replaced.
    mid, moment, nodes, delta = solver.element_currents(alpha)
    ref = _ref_knots(solver.wires_polylines, solver.n_per_edge_per_wire, 1)
    r_mid, r_mom, r_delta = [], [], []
    zero = np.zeros(1, dtype=np.complex128)
    for knots, cur in zip(ref, knots_i):
        el = 0.5 * (cur[1:] + cur[:-1])
        r_mid.append(0.5 * (knots[1:] + knots[:-1]))
        r_mom.append(el[:, None] * (knots[1:] - knots[:-1]))
        r_delta.append(np.concatenate([zero, el]) - np.concatenate([el, zero]))
    assert _bits(mid) == _bits(np.concatenate(r_mid))
    assert _bits(moment) == _bits(np.concatenate(r_mom))
    assert _bits(nodes) == _bits(np.concatenate(ref))
    assert _bits(delta) == _bits(np.concatenate(r_delta))


def test_constructor_checks_at_once_are_the_loops():
    """`short_segment_wires`, `check_junction_coincidence` and the duplicate
    search's vertex keys decide over every wire at once (momwire#1916 part
    D); each agrees with its per-wire loop, which still answers a spec given
    as lists."""
    from momwire import _wire_spec as WS

    rng = np.random.default_rng(11)
    polylines, per_edge = _mesh(rng, n_wires=80)
    radius = rng.uniform(1e-4, 5e-2, size=len(polylines))
    radius[::7] = 0.0
    as_lists = [pl.tolist() for pl in polylines]
    got = WS.short_segment_wires(polylines, per_edge, radius)
    assert got, "the mesh should carry short segments"
    assert got == WS.short_segment_wires(as_lists, per_edge, radius)

    keys = WS._vertex_keys(polylines)
    assert keys == [WS._vertex_key(pl) for pl in polylines]

    # Junctions: every wire's start welded to the first wire's start, then
    # one member nudged past its tolerance.
    welded = [pl.copy() for pl in polylines[:6]]
    for pl in welded[1:]:
        pl[0] = welded[0][0]
    groups = [[(w, "start") for w in range(6)], [(2, "end")]]
    WS.check_junction_coincidence(welded, per_edge[:6], groups)
    assert WS._junctions_coincide(welded, per_edge[:6], groups)
    welded[4][0] = welded[0][0] + 1e-3
    with pytest.raises(ValueError, match="junction 0: members do not coincide"):
        WS.check_junction_coincidence(welded, per_edge[:6], groups)
    assert not WS._junctions_coincide(welded, per_edge[:6], groups)
