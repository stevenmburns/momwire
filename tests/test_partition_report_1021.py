"""momwire#1021: the H-matrix partition is queryable after a build — the
clusters, the far/near blocks with their ranks, the Sommerfeld remainder's
state and the fragmentation guard's numbers, as data the solver produces."""

from __future__ import annotations

import json

import numpy as np

from momwire.hmatrix import HMatrixSolver

WL = 22.0


def _dipole(nsegs=64, degree=2):
    half = 0.962 * WL / 4
    wire = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    return HMatrixSolver(
        wires=[wire],
        degree=degree,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        wavelength=WL,
    )


def _junction(nsegs=48, degree=2):
    h = 0.962 * WL / 4
    w0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]])
    w1 = np.array([[0.0, 0.0, h], [0.0, h, h]])
    return HMatrixSolver(
        wires=[w0, w1],
        degree=degree,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        wavelength=WL,
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=0,
    )


def test_report_before_a_solve_describes_the_partition_exactly():
    s = _dipole()
    r = s.partition_report()
    n = r["summary"]["n"]
    assert sum(b["area"] for b in r["blocks"]) == n * n, (
        "the blocks cover the matrix exactly"
    )
    assert r["summary"]["n_far"] == sum(1 for b in r["blocks"] if b["kind"] == "far")
    assert r["summary"]["n_near"] == sum(1 for b in r["blocks"] if b["kind"] == "near")
    assert r["summary"]["built"] is False and not any("rank" in b for b in r["blocks"])
    assert r["clusters"][0]["depth"] == 0 and r["clusters"][0]["size"] == n
    assert all(0 <= b["s"] < len(r["clusters"]) for b in r["blocks"])
    assert r["summary"]["sommerfeld"]["rank"] is None  # free space
    assert r["summary"]["fragmentation"]["tripped"] is False
    json.dumps(r)  # plain data, nothing exotic


def test_report_after_a_solve_carries_every_far_block_rank():
    s = _junction()
    s.compute_impedance()
    r = s.partition_report()
    assert r["summary"]["built"] is True
    far = [b for b in r["blocks"] if b["kind"] == "far"]
    assert far, "the junction deck has admissible far blocks at this size"
    assert all(b.get("stored") in ("lowrank", "dense") for b in far)
    lowrank = [b for b in far if b["stored"] == "lowrank"]
    assert all(1 <= b["rank"] <= min(b["m"], b["n"]) for b in lowrank)
    assert r["summary"]["far_rank_max"] == max(b["rank"] for b in lowrank)
    assert r["summary"]["solve_iters"] is not None


def test_jsonl_is_produced_by_the_solver(tmp_path):
    s = _dipole(nsegs=32)
    s.compute_impedance()
    path = tmp_path / "partition.jsonl"
    r = s.partition_report_jsonl(path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 1 + len(r["clusters"]) + len(r["blocks"])
    assert "summary" in lines[0] and lines[0]["summary"]["n"] == r["summary"]["n"]
