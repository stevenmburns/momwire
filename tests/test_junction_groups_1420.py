"""momwire#1420 step 2: junction grouping on a spatial grid, found once per
solver.

``_junction_rule.coincident_groups`` used to walk every group representative
for every point — O(P x groups) ``norm`` calls, millions on a deck of
thousands of wire ends — and ``RazorSolver._find_junctions`` ran that four
times per construction. Both are speed changes that must not move a bit, so
the gates are EXACT:

* the grid against a verbatim copy of the walk (``_walk`` below) on random,
  clustered, non-transitive, bin-boundary and large-wire point sets, and the
  cases where the grid must decline and the walk must answer;
* through the production seam (``_shell.render``) that the grid ran and the
  walk did not, and that a render detects junctions once per solver;
* a red control: a grid that picks the wrong candidate fails the gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _junction_rule
from momwire._junction_rule import JUNCTION_TOL, coincident_groups
from momwire.eznec._shell import render
from momwire.razor import RazorSolver


def _walk(points, tol=JUNCTION_TOL):
    """``coincident_groups`` as spelled before momwire#1420."""
    rep, reps = [], []
    for i, p in enumerate(points):
        p = np.asarray(p, dtype=float)
        for j, q in reps:
            if float(np.linalg.norm(p - q)) <= tol:
                rep.append(j)
                break
        else:
            rep.append(i)
            reps.append((i, p))
    return rep


def _clustered(rng, n, tol, spread=1.0):
    """Points that sit in clusters a few tol wide, so groups, near misses and
    non-transitive chains (A~B, B~C, A!~C) are all common."""
    centres = rng.uniform(-spread, spread, size=(max(1, n // 4), 3))
    pts = centres[rng.integers(0, len(centres), size=n)]
    jitter = rng.choice([0.0, 0.3, 0.6, 0.99, 1.01, 1.5], size=(n, 1)) * tol
    direction = rng.standard_normal((n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return pts + jitter * direction


@pytest.mark.parametrize("tol", [JUNCTION_TOL, 1e-6, 1e-3, 0.25])
@pytest.mark.parametrize("seed", range(6))
def test_grid_matches_the_walk_on_clustered_points(tol, seed):
    rng = np.random.default_rng(seed)
    for n in (0, 1, 2, 7, 40, 300):
        pts = _clustered(rng, n, tol)
        assert coincident_groups(list(pts), tol) == _walk(list(pts), tol)


def test_grid_matches_the_walk_on_a_non_transitive_chain():
    tol = JUNCTION_TOL
    for step in (0.4, 0.6, 0.99, 1.0, 1.01):
        pts = [np.array([k * step * tol, 0.0, 0.0]) for k in range(30)]
        assert coincident_groups(pts, tol) == _walk(pts, tol)
        rev = pts[::-1]
        assert coincident_groups(rev, tol) == _walk(rev, tol)


def test_grid_matches_the_walk_on_bin_boundaries():
    """Points exactly on, and one ulp either side of, the grid's bin edges
    (multiples of 2 tol), at distances straddling tol."""
    tol = JUNCTION_TOL
    h = 2.0 * tol
    pts = []
    for k in (-3, -1, 0, 1, 2, 1000, 10**6):
        edge = k * h
        for x in (np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)):
            for off in (0.0, tol, np.nextafter(tol, 0.0), np.nextafter(tol, 1.0)):
                pts.append(np.array([x, x + off, -x]))
                pts.append(np.array([x + off, x, -x - off]))
    assert coincident_groups(pts, tol) == _walk(pts, tol)


def test_the_grid_declines_where_it_cannot_promise_the_walk(monkeypatch):
    calls = {"walk": 0}
    walk = _junction_rule._first_match_walk

    def spy(pts, tol):
        calls["walk"] += 1
        return walk(pts, tol)

    monkeypatch.setattr(_junction_rule, "_first_match_walk", spy)
    nan = [np.array([0.0, 0.0, np.nan]), np.zeros(3), np.array([0.0, 0.0, np.nan])]
    far = [np.array([1e9, 0.0, 0.0]), np.array([1e9, 0.0, 0.0])]
    flat = [np.zeros(2), np.zeros(2)]
    cases = [
        (nan, JUNCTION_TOL),
        (far, JUNCTION_TOL),
        (flat, JUNCTION_TOL),
        ([np.zeros(3), np.zeros(3)], 0.0),
        ([np.zeros(3)], float("inf")),
    ]
    for pts, tol in cases:
        assert coincident_groups(pts, tol) == _walk(pts, tol)
    assert calls["walk"] == len(cases)


def _radial_field_ends(n_radials, rng):
    """A hub, ``n_radials`` radials, a ring joining their tips, and every
    tenth tip nudged by under a tol: thousands of ends, most coincident in
    twos and one hub group of ``n_radials``."""
    ang = np.linspace(0.0, 2 * np.pi, n_radials, endpoint=False)
    tips = np.stack([10 * np.cos(ang), 10 * np.sin(ang), np.full_like(ang, -0.05)], 1)
    tips[::10] += rng.uniform(-0.5, 0.5, size=(len(tips[::10]), 3)) * JUNCTION_TOL
    hub = np.array([0.0, 0.0, -0.05])
    ends = []
    for k in range(n_radials):
        ends += [hub, tips[k]]
    for k in range(n_radials):
        ends += [tips[k], tips[(k + 1) % n_radials]]
    return ends


def test_grid_matches_the_walk_on_a_large_wire_deck():
    ends = _radial_field_ends(240, np.random.default_rng(3))
    assert len(ends) == 960
    assert coincident_groups(ends) == _walk(ends)


@pytest.mark.slow
def test_grid_matches_the_walk_on_a_very_large_wire_deck():
    ends = _radial_field_ends(1500, np.random.default_rng(4))
    assert coincident_groups(ends) == _walk(ends)


MONOPOLE = (
    "CM radials on a perfect ground\nCE\nGW 1 10 0 0 0 0 0 5 1e-3\n"
    + "".join(
        f"GW {k + 2} 4 0 0 5 {float(2 * np.cos(a))!r} {float(2 * np.sin(a))!r} 5 1e-3\n"
        for k, a in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False))
    )
    + "GE 1\nGN 1\nFR 0 1 0 0 14\nEX 0 1 1 0 1 0\nXQ\nEN\n"
)


def test_the_seam_groups_on_the_grid_and_detects_once_per_solver(monkeypatch):
    """Through ``render``: the grid ran, the walk never did, and each razor
    solver detected its junctions once however often it asked."""
    calls = {"walk": 0, "cells": 0, "detect": 0, "find": 0, "solvers": 0}
    walk, cells = _junction_rule._first_match_walk, _junction_rule._cells
    detect, find = RazorSolver._detect_junctions, RazorSolver._find_junctions
    init = RazorSolver.__init__

    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)

        return wrapped

    monkeypatch.setattr(_junction_rule, "_first_match_walk", count("walk", walk))
    monkeypatch.setattr(_junction_rule, "_cells", count("cells", cells))
    monkeypatch.setattr(RazorSolver, "_detect_junctions", count("detect", detect))
    monkeypatch.setattr(RazorSolver, "_find_junctions", count("find", find))
    monkeypatch.setattr(RazorSolver, "__init__", count("solvers", init))
    out = render(MONOPOLE, basis="razor-2p")
    assert "NEC ERROR" not in out
    assert calls["walk"] == 0 and calls["cells"] >= 1
    assert calls["solvers"] >= 1
    assert calls["detect"] == calls["solvers"]
    assert calls["find"] > calls["detect"]


def test_kept_junctions_follow_a_new_geometry():
    """The kept answer is stamped with its inputs: a solver handed a new
    polyline list (the probe pattern tests use) detects again."""
    probe = RazorSolver.__new__(RazorSolver)
    probe.ground_z = None
    probe._declared_junctions = None
    probe.wires_polylines = [
        np.array([[0, 0, 0], [0, 0, 1.0]]),
        np.array([[0, 0, 1.0], [1, 0, 1.0]]),
    ]
    first = probe._find_junctions()
    assert first == [{"ends": [(0, "end"), (1, "start")], "grounded": False}]
    first[0]["ends"].clear()  # a caller mutating its copy changes nothing kept
    assert probe._find_junctions() == [
        {"ends": [(0, "end"), (1, "start")], "grounded": False}
    ]
    probe.wires_polylines = [
        np.array([[0, 0, 0], [0, 0, 1.0]]),
        np.array([[5, 0, 1.0], [6, 0, 1.0]]),
    ]
    assert probe._find_junctions() == []


def _seeded_last_match(points, tol=JUNCTION_TOL):
    """The grid with one seeded fault: candidates tried in DESCENDING order,
    so a point within tol of two groups joins the later one."""
    pts = [np.asarray(p, dtype=float) for p in points]
    rep, buckets = [], {}
    for i, (cx, cy, cz) in enumerate(_junction_rule._cells(pts, tol)):
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand += buckets.get((cx + dx, cy + dy, cz + dz), [])
        cand.sort(reverse=True)
        hit = next(
            (j for j in cand if float(np.linalg.norm(pts[i] - pts[j])) <= tol), -1
        )
        if hit < 0:
            rep.append(i)
            buckets.setdefault((cx, cy, cz), []).append(i)
        else:
            rep.append(hit)
    return rep


def test_a_seeded_wrong_candidate_fails_the_gate():
    """Red control: the clustered gate sees the seeded fault."""
    tol = JUNCTION_TOL
    two_groups = [np.zeros(3), np.array([1.2 * tol, 0, 0]), np.array([0.6 * tol, 0, 0])]
    assert _walk(two_groups) == [0, 1, 0]
    assert _seeded_last_match(two_groups) == [0, 1, 1]
    rng = np.random.default_rng(0)
    pts = list(_clustered(rng, 300, tol))
    assert _seeded_last_match(pts) != _walk(pts)
