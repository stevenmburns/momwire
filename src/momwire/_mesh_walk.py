"""The mesh's knot walk, over every edge of every wire at once.

Several readouts walk a meshed structure the same way, one wire and one
edge at a time: ``np.linspace(p0, p1, n + 1)`` per edge, the edges of a
wire stacked with their shared knots taken once, then per-wire arcs as a
``np.cumsum`` of the steps. On a deck of a few thousand short wires (the
W8IO ELT whip's ground screen: 4,007 solved wires, 4,067 edges) that walk
is tens of thousands of small numpy calls per readout, per frequency.

These are the same walks as array expressions, BIT FOR BIT: every value is
produced by the same IEEE operations, in the same order, as the per-edge
spelling it replaces (`tests/test_mesh_walk.py` holds each against it).

* `edge_linspace` is numpy's ``linspace`` for array endpoints, including
  its branch on a zero step component (``y / div * delta`` instead of
  ``y * (delta / div)``), decided per edge as numpy decides it per call.
* `segmented_cumsum` is ``np.cumsum`` of each slice. An accumulate is a
  strictly sequential sum, so the slices are batched by length into 2-D
  blocks and accumulated along their rows: the same additions, in the same
  order, as one ``np.cumsum`` per slice.
"""

from __future__ import annotations

import numpy as np


def edge_counts(per_edge):
    """`(counts, edges_per_wire)`: every edge's count, flat, and each wire's
    number of edges."""
    edges_per_wire = np.fromiter((len(npe) for npe in per_edge), dtype=np.int64)
    if edges_per_wire.size and edges_per_wire.sum():
        counts = np.concatenate(
            [np.asarray(npe, dtype=np.int64).reshape(-1) for npe in per_edge]
        )
    else:
        counts = np.zeros(0, dtype=np.int64)
    return counts, edges_per_wire


def edge_endpoints(polylines, edges_per_wire):
    """`(p0, p1)`, (E, 3) each: every edge's start and end vertex, flat, in
    wire then edge order."""
    pls = [np.asarray(pl, dtype=np.float64) for pl in polylines]
    used = [pl[: int(n) + 1] for pl, n in zip(pls, edges_per_wire) if n > 0]
    if not used:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty
    p0 = np.concatenate([pl[:-1] for pl in used], axis=0)
    p1 = np.concatenate([pl[1:] for pl in used], axis=0)
    return p0, p1


def edge_linspace(p0, p1, num):
    """``np.linspace(p0[e], p1[e], num[e])`` for every edge e, concatenated:
    `(points, offsets)`, edge e's points at ``points[offsets[e]:offsets[e+1]]``.

    Every `num` must be at least 2 (an edge has two ends)."""
    num = np.asarray(num, dtype=np.int64)
    offsets = np.zeros(num.size + 1, dtype=np.int64)
    np.cumsum(num, out=offsets[1:])
    total = int(offsets[-1])
    if total == 0:
        return np.zeros((0, 3), dtype=np.float64), offsets
    edge = np.repeat(np.arange(num.size, dtype=np.int64), num)
    # numpy's y: arange(0, num) as float64, per edge.
    y = (np.arange(total, dtype=np.int64) - offsets[:-1][edge]).astype(np.float64)
    div = (num - 1).astype(np.float64)
    delta = p1 - p0
    step = delta / div[:, None]
    zero = (step == 0).any(axis=1)
    pts = np.empty((total, 3), dtype=np.float64)
    zp = zero[edge]
    if zp.any():
        # numpy: y /= div; y = y * delta
        pts[zp] = (y[zp] / div[edge[zp]])[:, None] * delta[edge[zp]]
    nz = ~zp
    if nz.any():
        # numpy: y = y * step
        pts[nz] = y[nz][:, None] * step[edge[nz]]
    pts += p0[edge]
    # numpy: y[-1, ...] = stop
    pts[offsets[1:] - 1] = p1
    return pts, offsets


def wire_knots(polylines, per_edge, subdiv=1):
    """Every wire's knots, `(knots, wire_offsets)`: wire w's at
    ``knots[wire_offsets[w]:wire_offsets[w+1]]``.

    The array spelling of, per wire::

        parts = []
        for i, n_e in enumerate(npe):
            seg = np.linspace(polyline[i], polyline[i + 1], n_e * subdiv + 1)
            parts.append(seg if i == 0 else seg[1:])
        knots = np.vstack(parts)

    A wire with no edges contributes nothing (that loop would fail on it)."""
    counts, edges_per_wire = edge_counts(per_edge)
    p0, p1 = edge_endpoints(polylines, edges_per_wire)
    pts, offs = edge_linspace(p0, p1, counts * int(subdiv) + 1)
    # Drop each edge's first point except on a wire's first edge.
    first_edge = np.zeros(counts.size, dtype=bool)
    starts = np.zeros(edges_per_wire.size, dtype=np.int64)
    np.cumsum(edges_per_wire[:-1], out=starts[1:])
    first_edge[starts[edges_per_wire > 0]] = True
    keep = np.ones(pts.shape[0], dtype=bool)
    keep[offs[:-1][~first_edge]] = False
    knots = pts[keep]
    per_wire = np.zeros(edges_per_wire.size, dtype=np.int64)
    if counts.size:
        wire_of_edge = np.repeat(
            np.arange(edges_per_wire.size, dtype=np.int64), edges_per_wire
        )
        np.add.at(per_wire, wire_of_edge, counts * int(subdiv))
    per_wire[edges_per_wire > 0] += 1
    wire_offsets = np.zeros(edges_per_wire.size + 1, dtype=np.int64)
    np.cumsum(per_wire, out=wire_offsets[1:])
    return knots, wire_offsets


def segmented_cumsum(x, offsets):
    """``np.cumsum(x[offsets[i]:offsets[i+1]])`` for every i, concatenated
    (an array shaped as `x`)."""
    x = np.asarray(x)
    offsets = np.asarray(offsets, dtype=np.int64)
    out = np.empty_like(x)
    lengths = np.diff(offsets)
    for length in np.unique(lengths):
        length = int(length)
        if length == 0:
            continue
        which = np.flatnonzero(lengths == length)
        idx = offsets[:-1][which][:, None] + np.arange(length, dtype=np.int64)
        out[idx] = np.cumsum(x[idx], axis=1)
    return out


def interior_pairs(offsets):
    """`(lo, hi)`: flat indices of every (k, k+1) pair inside one slice, for
    slices at `offsets` — the rows ``v[1:]`` and ``v[:-1]`` of each slice,
    concatenated in slice order."""
    offsets = np.asarray(offsets, dtype=np.int64)
    total = int(offsets[-1]) if offsets.size else 0
    last = np.zeros(total, dtype=bool)
    ends = offsets[1:][np.diff(offsets) > 0] - 1
    last[ends] = True
    lo = np.flatnonzero(~last)
    return lo, lo + 1


def wire_arcs(knots, wire_offsets):
    """Per wire, the arc along its chords at every knot,
    ``concatenate([[0.0], cumsum(norm3(diff(knots)))])``, flat at
    `wire_offsets`."""
    from ._vec3 import norm3

    lo, hi = interior_pairs(wire_offsets)
    step = norm3(knots[hi] - knots[lo])
    out = np.zeros(knots.shape[0], dtype=np.float64)
    # Each wire's steps land on its knots after the first.
    out[hi] = segmented_cumsum(step, _step_offsets(wire_offsets))
    return out


def _step_offsets(wire_offsets):
    """Offsets of each wire's steps (one fewer than its knots, none for an
    empty wire) in the flat step array `interior_pairs` orders."""
    n_knots = np.diff(wire_offsets)
    n_steps = np.maximum(n_knots - 1, 0)
    out = np.zeros(n_knots.size + 1, dtype=np.int64)
    np.cumsum(n_steps, out=out[1:])
    return out
