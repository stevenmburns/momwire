"""M6 (momwire#182): the pins behind `docs/sinusoidal-galerkin-instrument-report.md`.

The report is a document, so its claims need to outlive the session that
measured them. These tests re-derive its three load-bearing statements from
the same harness the report was written from — `scripts/m6_residue_cluster.py`
and its checked-in geometry snapshot — so a regression in either the solver or
the harness turns the report red instead of merely stale.

What is pinned here:

1. **The feed-model axis.** With the feed model matched (a zero-width point
   gap on both), the sinusoidal Galerkin basis and the quadratic B-spline
   basis agree to ~4e-8 on the canonical dipole. The residual sin-Galerkin ↔
   bspline difference at the DEFAULT feed (1.5e-4) is therefore a FEED-MODEL
   difference, essentially all of it — not a basis difference. This is
   momwire#182 M5's finding, first-party and reproducible.

2. **The testing axis, on the geometry that motivated the exercise.**
   antennaknobs#521's top residue row (`specialty.hentenna`) is a TESTING
   effect: swapping only the testing scheme — same basis, same mesh, same feed
   — moves the sinusoidal answer onto the B-spline one.

3. **The harness is intact.** The geometry snapshot covers exactly the
   design ladder the report's tables were computed from.

Everything here is FREE SPACE. momwire#182 M4 pinned that a finite-ground
model over a ground-CONTACT wire is broken on both sinusoidal solvers (an
inherited #151 defect), and M5b scoped junction ports over any ground out, so
the report takes no ground reads at all and neither does this file.
"""

import json

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver, SinusoidalSolver

from scripts.m6_residue_cluster import (
    DESIGNS,
    SNAPSHOT,
    PointGapGalerkin,
    _kwargs_from_row,
    solve,
)


def _rel(a, b):
    """|a − b| / |b| — with b the NAMED reference. momwire#182 M3's rule:
    the delta-gap dipole has no mesh limit, so "the converged value" is not a
    number this model has and every error statement has to name what it is
    against."""
    return abs(a - b) / abs(b)


def _dipole_kwargs(n):
    hd = 0.962 * 22 / 4
    return dict(
        wires=[np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]])],
        n_per_edge_per_wire=[[n]],
        feed_wire_index=0,
        feed_arclength=hd,
        wavelength=22,
        wire_radius=0.0005,
    )


@pytest.mark.slow
def test_matched_feed_model_closes_the_dipole_basis_gap():
    """REPORT CLAIM 1 — the sin-Galerkin ↔ bspline gap is a feed model, not a
    basis.

    Three impedances on the same mesh, differing in exactly one thing at a
    time. `gal` and `ptgap` share the sinusoidal basis and the Galerkin
    testing and differ ONLY in the source model (NEC's segment-wide delta gap
    vs the zero-width point gap `BSplineSolver` drives); `ptgap` and `bs2`
    share the point gap and differ ONLY in the basis.

    Measured at N=321 (momwire#182 M5's numbers, reproduced):

        coll   69.631876 − 18.107822j
        gal    69.639094 − 18.056294j
        ptgap  69.633780 − 18.065312j
        bs2    69.633780 − 18.065315j

    so gal↔bs2 = 1.5e-4 while ptgap↔bs2 = 4.2e-8: matching the feed removes
    ~3500× of the difference, and what is left is four orders below the
    coarsest thing anyone would call a basis effect. The N=161 rung is
    included so the reading is a trend, not one lucky mesh (2.5e-4 → 1.5e-4
    for the mismatched feed; 2.8e-7 → 4.2e-8 for the matched one — the
    matched pair converges together, the mismatched pair converges to
    different feed models).

    ~9 s: four solves at each of two fine meshes.
    """
    seen = {}
    for n in (161, 321):
        kw = _dipole_kwargs(n)
        z = {c: solve(c, kw) for c in ("coll", "gal", "ptgap", "bs2")}
        seen[n] = z
        # The matched-feed pair is a different ORDER of agreement, not a
        # tighter version of the same one.
        assert _rel(z["ptgap"], z["bs2"]) < 1e-6, (n, z)
        assert _rel(z["gal"], z["bs2"]) > 1e-5, (n, z)
        assert _rel(z["ptgap"], z["bs2"]) < 0.01 * _rel(z["gal"], z["bs2"]), (n, z)
        # ...and collocation is the worst of the three, at every mesh.
        assert _rel(z["coll"], z["bs2"]) > _rel(z["gal"], z["bs2"]), (n, z)

    # The matched pair tightens with refinement; the mismatched pair is
    # converging toward two different answers, so its gap survives.
    assert _rel(seen[321]["ptgap"], seen[321]["bs2"]) < _rel(
        seen[161]["ptgap"], seen[161]["bs2"]
    )


def test_point_gap_drive_is_self_dual():
    """The matched-feed column's construction, checked rather than asserted.

    A point gap's Galerkin drive column IS the basis-evaluation vector at the
    gap, and the default `feed_readout="centre"` reads the current at exactly
    that point — so drive and readout are exact duals and the column's Y is
    machine-symmetric with no `feed_readout="variational"` opt-in and none of
    the M3 payoff traded. If that ever stops holding, the feed-model column of
    the report is measuring something other than a feed model.
    """
    kw = _dipole_kwargs(41)
    two_feeds = dict(
        kw,
        feeds=[(0, 0.962 * 22 / 4, 1 + 0j), (0, 0.962 * 22 / 4 * 0.5, 0 + 0j)],
    )
    two_feeds.pop("feed_wire_index")
    two_feeds.pop("feed_arclength")
    Y = PointGapGalerkin(**two_feeds).compute_y_matrix()
    asym = abs(Y - Y.T).max() / abs(Y).max()
    # The floor is the FILL's reciprocity floor (8.3e-12, momwire#182 M2), not
    # the port algebra's; the default delta gap sits at ~1e-5 here.
    assert asym < 1e-9, asym
    Y_gap = SinusoidalGalerkinSolver(**two_feeds).compute_y_matrix()
    assert abs(Y_gap - Y_gap.T).max() / abs(Y_gap).max() > asym


def test_hentenna_residue_is_a_testing_effect():
    """REPORT CLAIM 2 — antennaknobs#521's top row, from the snapshot.

    At the design's own catalog mesh the point-matched solver sits ~10 Ω away
    from the B-spline reactance while the Galerkin testing of the SAME basis
    on the SAME mesh lands on it. Read off the checked-in geometry so this
    tracks the report rather than the (mutable) antennaknobs catalog.

    Measured at nominal_nsegs=21 (119 segments, free space):

        coll   42.44 + 29.23j
        gal    43.01 + 38.90j
        ptgap  43.04 + 38.89j
        bs2    43.05 + 38.59j
    """
    rows = json.loads(SNAPSHOT.read_text())
    row = next(
        r for r in rows if r["design"] == "specialty.hentenna" and r["rung"] == 21
    )
    kw = _kwargs_from_row(row)
    z = {c: solve(c, kw) for c in ("coll", "gal", "ptgap", "bs2")}

    assert _rel(z["coll"], z["bs2"]) > 0.10, z
    assert _rel(z["gal"], z["bs2"]) < 0.02, z
    # The gap is a reactance gap — that is how #521 filed the whole cluster.
    assert abs(z["coll"].imag - z["bs2"].imag) > 5.0, z
    assert abs(z["gal"].imag - z["bs2"].imag) < 1.0, z


def test_all_four_columns_share_one_mesh():
    """The comparison's precondition, made explicit.

    A rung compares four SCHEMES only if they are handed one geometry. The
    snapshot stores a single mesh per rung and every column is constructed
    from it, so this is really a guard against a future solver growing its own
    mesh coercion — the exact failure `_parity_for_solver` exists to prevent on
    the antennaknobs side.
    """
    rows = json.loads(SNAPSHOT.read_text())
    row = next(
        r for r in rows if r["design"] == "specialty.hentenna" and r["rung"] == 21
    )
    kw = _kwargs_from_row(row)
    n = {}
    for cls in (SinusoidalSolver, SinusoidalGalerkinSolver, PointGapGalerkin):
        n[cls.__name__] = cls(**kw)._build_geometry()["n_segs"]
    assert len(set(n.values())) == 1, n
    assert set(n.values()) == {row["n_segs"]}
    # BSplineSolver counts BASES, not segments, so it gets the structural
    # check instead: same wires, same per-edge segment counts.
    bs = BSplineSolver(**kw, degree=2)
    assert [list(e) for e in bs.n_per_edge_per_wire] == row["n_per_edge_per_wire"]


def test_snapshot_covers_the_reported_design_ladder():
    """REPORT CLAIM 3 — the harness is intact.

    Every (design, rung) the report tabulates must be in the snapshot, and the
    snapshot must not have drifted to cover something else. A snapshot re-dump
    against a retuned catalog is a legitimate act; doing it without updating
    the report is not, and this is where that shows up.
    """
    rows = json.loads(SNAPSHOT.read_text())
    have = {(r["design"], r["knob"], r["rung"]) for r in rows}
    want = {
        (design, knob, rung)
        for design, (knob, rungs) in DESIGNS.items()
        for rung in rungs
    }
    assert have == want, (want - have, have - want)
    for r in rows:
        assert r["n_segs"] > 0
        assert r["feeds"], r["design"]
        assert len(r["feed_centre"]) == 3
