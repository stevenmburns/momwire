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
the report takes no ground reads at all and neither does this file. (#191 has
since lifted the port half of that for a PEC ground — report §14 — but the
contact defect and the finite grounds are untouched, so the scope stands.)
"""

import functools
import json

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver, SinusoidalSolver

from scripts.m6_residue_cluster import (
    COLUMNS,
    DESIGNS,
    M3_COARSE,
    M3_FEED_MATCHED_GEOMS,
    M3_REFS_QUOTED,
    M3_SEGMENT_BS2_REF,
    SNAPSHOT,
    PointGapCollocation,
    _FULLY_MATCHED_REFS,
    _kwargs_from_row,
    _MATCHED_REFS,
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


@pytest.mark.slow
def test_feed_model_option_reproduces_section_6():
    """§12 follow-up 5, landed: the `ptgap` column is now a solver OPTION
    (`feed_model="point"`, momwire#192), and it has to keep reading §6's table.

    §6 was measured off a `PointGapGalerkin` research subclass and §15
    re-read it after the #203 fold. Pinned here as VALUES, not just as a
    structural ordering, because the whole point of promoting the column is
    that a future default flip becomes a decision — and a decision needs a
    before.

    Measured (§15's post-fold numbers, reproduced through the option):

        N=161  ptgap 69.634327 − 18.112452j   ptgap↔bs2 2.806e-7
        N=321  ptgap 69.633779 − 18.065325j   ptgap↔bs2 1.303e-7

    The segment-gap default at the same meshes is 2.47e-4 / 1.45e-4, so the
    option buys 879× / 1116× — which is also the guard that it is not a no-op.

    ~7 s: three solves at each of two fine meshes.
    """
    pinned = {161: 2.806e-7, 321: 1.303e-7}
    seen = {}
    for n in (161, 321):
        kw = _dipole_kwargs(n)
        z_pt = complex(
            np.atleast_1d(
                SinusoidalGalerkinSolver(**kw, feed_model="point").compute_impedance()[
                    0
                ]
            )[0]
        )
        z_gal = complex(
            np.atleast_1d(
                SinusoidalGalerkinSolver(
                    **kw, feed_model="segment"
                ).compute_impedance()[0]
            )[0]
        )
        z_bs2 = complex(
            np.atleast_1d(BSplineSolver(**kw, degree=2).compute_impedance()[0])[0]
        )
        r_pt, r_gal = _rel(z_pt, z_bs2), _rel(z_gal, z_bs2)
        seen[n] = r_pt
        # 5 % on the pinned relative: enough to catch a changed feed model or
        # a lost #203 fold (both move it by ≥2×), loose enough to survive a
        # BLAS reassociation at the 1e-7 level these differences live at.
        assert abs(r_pt - pinned[n]) < 0.05 * pinned[n], (n, r_pt, pinned[n])
        assert r_pt < 0.01 * r_gal, (n, r_pt, r_gal)
    # The matched pair converges together; the mismatched pair does not.
    assert seen[321] < seen[161]


@pytest.mark.slow
def test_section_6_mirrors_with_the_oracle_on_the_segment_gap():
    """§19 (momwire#216) — §6 run the other way round: instead of moving the
    sinusoidal solvers onto `BSplineSolver`'s point gap, move the ORACLE onto
    NEC's segment gap (`BSplineSolver(feed_model="segment")`).

    The subtlety the mirror turns on is the READOUT. `BSplineSolver` reads
    Z = 1/(vᵀc), always its drive's dual, so under the segment gap it reads the
    gap-AVERAGED current — which is `SinusoidalGalerkinSolver`'s
    `feed_readout="variational"` functional, NOT its default centre readout.
    So the mirror has two pairings and only one of them is fully matched:

        N     gal(seg,centre)↔bs2(seg)   gal(seg,variational)↔bs2(seg)
        161            2.052e-4                    2.845e-7
        321            1.203e-4                    1.308e-7

    The fully matched pairing collapses to 2.845e-7 / 1.308e-7 — the same
    numbers §6/§15's point-gap pairing reads (2.806e-7 / 1.303e-7), to 1.4 %
    and 0.4 %. Matching the SOURCE alone (centre readout) buys only ~1.2×, so
    §6's residual is a source difference AND a functional difference, and the
    basis difference underneath is the same size under either feed model.

    ~7 s: five solves at each of two fine meshes.
    """
    pinned_matched = {161: 2.845e-7, 321: 1.308e-7}
    pinned_centre = {161: 2.052e-4, 321: 1.203e-4}
    matched, mirrored_point = {}, {}
    for n in (161, 321):
        kw = _dipole_kwargs(n)

        def _z(s):
            return complex(np.atleast_1d(s.compute_impedance()[0])[0])

        z_bs2_seg = _z(BSplineSolver(**kw, degree=2, feed_model="segment"))
        z_bs2_pt = _z(BSplineSolver(**kw, degree=2))
        z_gal_c = _z(SinusoidalGalerkinSolver(**kw, feed_model="segment"))
        z_gal_v = _z(
            SinusoidalGalerkinSolver(
                **kw, feed_model="segment", feed_readout="variational"
            )
        )
        z_gal_pt = _z(SinusoidalGalerkinSolver(**kw, feed_model="point"))

        r_matched = _rel(z_gal_v, z_bs2_seg)
        r_centre = _rel(z_gal_c, z_bs2_seg)
        r_mismatched = _rel(z_gal_c, z_bs2_pt)  # §6's own pairing
        matched[n] = r_matched
        mirrored_point[n] = _rel(z_gal_pt, z_bs2_pt)
        # Values, 5 % — the house tolerance on these relatives.
        assert abs(r_matched - pinned_matched[n]) < 0.05 * pinned_matched[n], (
            n,
            r_matched,
        )
        assert abs(r_centre - pinned_centre[n]) < 0.05 * pinned_centre[n], (n, r_centre)
        # The verdict, structurally: the FULLY matched pairing beats §6's
        # mismatched one by ≥100×...
        assert r_mismatched > 100 * r_matched, (n, r_mismatched, r_matched)
        # ...while matching the source alone does not — the readout is half
        # the story, which is what makes the two-pairing table necessary.
        assert r_centre > 0.1 * r_mismatched, (n, r_centre, r_mismatched)
        # The two feed models leave the SAME basis residual (§19's verdict:
        # §6 is now closed from both directions).
        assert abs(r_matched - mirrored_point[n]) < 0.05 * mirrored_point[n], (
            n,
            r_matched,
            mirrored_point[n],
        )
    # And, as in §6, the matched pair tightens with refinement.
    assert matched[321] < matched[161]


@pytest.mark.slow
def test_zero_width_gap_has_no_collocation_rhs():
    """§17 (momwire#212) — the refutation `SinusoidalSolver.feed_model="point"`
    is refused on, measured rather than argued.

    §16 wanted a feed-MATCHED payoff, which needs a zero-width gap on the
    point-matched solver. Collocation's drive is E_app sampled AT a match
    point, so the source has to be a function there; the zero-width gap is a
    delta. Regularize it to a nascent delta of width w and exactly two things
    can happen, both checked here on the canonical dipole:

      w ≪ h — take the only width the model owns, the wire radius (the b → a
      frill, `zero_width_gap_Ez`). The feed row sees the spike's PEAK, ≈ V/2a,
      and every other row sees ~0, so the drive is the segment-gap drive
      scaled by h/2a and Z = (2a/h)·Z_segment: 0.270 → 0.533 → 1.059 → 2.112 →
      4.218 Ω at N = 41…641, doubling every rung away from the answer instead
      of settling. Collocation cannot tell a narrow gap from a small voltage.

      w ≳ h — the smallest width the mesh resolves is one cell, and averaging
      the SAME zero-width field over each cell returns the segment gap to
      ~1e-7 of |Z|. A no-op, not an option.

    So the segment gap is not a rival source model here — it is the zero-width
    gap at the only resolution collocation has. Pinned as values because §17
    is a claim about magnitudes; 5 % on the relatives, matching
    `test_feed_model_option_reproduces_section_6`.

    ~20 s: three solves at each of four meshes.
    """
    a = 0.0005
    # Pinned: Z from point-sampling the zero-width field. Doubling per rung IS
    # the finding, so the values are pinned, not just their ordering.
    pinned_point = {161: 1.059402 - 0.276689j, 321: 2.112233 - 0.549288j}
    z_pt, z_seg, z_bs2, cell_rel, prop_dev = {}, {}, {}, {}, {}
    for n in (41, 81, 161, 321):
        kw = _dipole_kwargs(n)
        s = SinusoidalSolver(**kw)
        z_seg[n] = complex(np.atleast_1d(s.compute_impedance()[0])[0])
        geom = s._build_geometry()
        fi = geom["feed_segs"][0]
        h = float(geom["seg_h"][fi])
        pt = PointGapCollocation(**kw, sampled="point")
        z_pt[n] = complex(np.atleast_1d(pt.compute_impedance()[0])[0])
        v = pt.drive_vector
        v_seg = np.zeros_like(v)
        v_seg[fi] = -1.0 / h
        prop_dev[n] = float(
            np.abs(v - (v[fi] / v_seg[fi]) * v_seg).max() / np.abs(v).max()
        )
        z_cell = complex(
            np.atleast_1d(
                PointGapCollocation(**kw, sampled="cell").compute_impedance()[0]
            )[0]
        )
        cell_rel[n] = _rel(z_cell, z_seg[n])
        z_bs2[n] = complex(
            np.atleast_1d(BSplineSolver(**kw, degree=2).compute_impedance()[0])[0]
        )
        # The point-sampled drive carries no shape of its own: it is the
        # segment-gap drive rescaled, to parts in 1e5.
        assert prop_dev[n] < 1e-4, (n, prop_dev[n])
        assert _rel((2 * a / h) * z_seg[n], z_pt[n]) < 1e-4, (n, z_pt[n])
        # ...and the cell-averaged reading is the segment gap itself.
        assert cell_rel[n] < 1e-5, (n, cell_rel[n])

    for n, want in pinned_point.items():
        assert _rel(z_pt[n], want) < 0.05, (n, z_pt[n], want)
    # The two failure modes, structurally. The point-sampled solve converges to
    # nothing: its rung-to-rung STEP doubles where the segment gap's shrinks,
    # and it never comes within 10× of the oracle at any rung on the ladder.
    steps_pt = [abs(z_pt[b] - z_pt[a_]) for a_, b in zip((41, 81, 161), (81, 161, 321))]
    steps_seg = [
        abs(z_seg[b] - z_seg[a_]) for a_, b in zip((41, 81, 161), (81, 161, 321))
    ]
    assert all(b > 1.9 * a_ for a_, b in zip(steps_pt, steps_pt[1:])), steps_pt
    assert all(b < a_ for a_, b in zip(steps_seg, steps_seg[1:])), steps_seg
    assert all(abs(z_bs2[n]) > 10 * abs(z_pt[n]) for n in z_pt), z_pt
    # ... and the cell-averaged solve is the default by another name, so it is
    # not an alternative feed model either.
    assert cell_rel[321] < 0.01 * _rel(z_seg[321], z_bs2[321]), (
        cell_rel[321],
        _rel(z_seg[321], z_bs2[321]),
    )
    # The refusal the derivation buys.
    with pytest.raises(NotImplementedError, match="zero-width gap"):
        SinusoidalSolver(**_dipole_kwargs(41), feed_model="point")


@pytest.mark.slow
def test_feed_matched_payoff_is_the_pinned_m3_ratio():
    """§17's second half: what `errColl/errGal` reads with the feed model held
    fixed — which, given the refutation above, means the segment gap on both
    comparands, i.e. exactly what M3 already scores.

    The one place a feed model still varies inside that gate is the REFERENCE
    family: two of M3's six defensible references are `BSplineSolver` solves,
    which drive a zero-width gap. Dropping them is the only feed-matching left
    to do, and it moves the worst-case ratio by <0.2 %:

        geometry      worst over six   worst over the four matched
        dipole             2.062              2.062
        k2_junction        2.971              2.974

    So the M3 payoff verdict does not rest on the reference's feed model, and
    the confound §16 identified lives entirely on the comparand side — where
    it is unremovable, because the point-matched sibling cannot carry a point
    gap. Contrast the §16 flip reading (Galerkin on "point", collocation still
    on "segment"): 1.948 on the dipole, 1.774 on k2_junction, both below their
    pinned floors.

    ~25 s: two schemes × two geometries × three coarse meshes.
    """
    pinned = {"dipole": 2.062, "k2_junction": 2.971}
    for name, factory in M3_FEED_MATCHED_GEOMS.items():
        z = {
            (scheme, n): complex(
                np.atleast_1d(
                    (
                        SinusoidalSolver
                        if scheme == "coll"
                        else functools.partial(
                            SinusoidalGalerkinSolver, feed_model="segment"
                        )
                    )(**factory(n)).compute_impedance()[0]
                )[0]
            )
            for scheme in ("coll", "gal")
            for n in M3_COARSE
        }
        ratios = {
            ref_name: [
                _rel(z[("coll", n)], ref) / _rel(z[("gal", n)], ref) for n in M3_COARSE
            ]
            for ref_name, ref in M3_REFS_QUOTED[name].items()
        }
        w_all = min(min(v) for v in ratios.values())
        w_matched = min(min(ratios[r]) for r in _MATCHED_REFS)
        assert abs(w_all - pinned[name]) < 0.05 * pinned[name], (name, w_all)
        # Feed-matching the reference family does not move the verdict.
        assert abs(w_matched - w_all) < 0.01 * w_all, (name, w_all, w_matched)
        # Both readings clear the pinned M3 floor (2.0 / 2.9) they are the
        # feed-matched statement of.
        assert min(w_all, w_matched) > (2.0 if name == "dipole" else 2.9)


@pytest.mark.slow
def test_m3_payoff_is_unmoved_by_a_fully_feed_matched_reference_family():
    """§19's second half (momwire#216) — §17 could only feed-match M3's
    reference family by DROPPING its two `BSplineSolver` members. With
    `feed_model="segment"` on the oracle they can be kept instead: a seventh
    reference, `bs2_segment_321`, that is a B-spline solve on the comparands'
    own segment gap.

    Measured (`scripts/m6_residue_cluster.py --feed-matched`):

        geometry      bs2_segment_321 at N=11/15/21   worst over the matched five
        dipole              2.127 / 2.241 / 2.309              2.062
        k2_junction         5.450 / 4.181 / 3.501              2.974

    So the fully matched family reads exactly what §17's four-member matched
    family read — the new reference is never the worst — and §17's <0.2 %
    extrapolation lands at 0.0 %. The M3 payoff verdict is now feed-matched
    with nothing dropped.

    ~25 s: two schemes × two geometries × three coarse meshes.
    """
    pinned = {"dipole": 2.062, "k2_junction": 2.974}
    for name, factory in M3_FEED_MATCHED_GEOMS.items():
        z = {
            (scheme, n): complex(
                np.atleast_1d(
                    (
                        SinusoidalSolver
                        if scheme == "coll"
                        else functools.partial(
                            SinusoidalGalerkinSolver, feed_model="segment"
                        )
                    )(**factory(n)).compute_impedance()[0]
                )[0]
            )
            for scheme in ("coll", "gal")
            for n in M3_COARSE
        }
        refs = dict(M3_REFS_QUOTED[name])
        refs["bs2_segment_321"] = M3_SEGMENT_BS2_REF[name]
        ratios = {
            ref_name: [
                _rel(z[("coll", n)], ref) / _rel(z[("gal", n)], ref) for n in M3_COARSE
            ]
            for ref_name, ref in refs.items()
        }
        w_fully = min(min(ratios[r]) for r in _FULLY_MATCHED_REFS)
        w_matched = min(min(ratios[r]) for r in _MATCHED_REFS)
        assert abs(w_fully - pinned[name]) < 0.05 * pinned[name], (name, w_fully)
        # §17's extrapolated "<0.2 % movement", measured: it does not move.
        assert abs(w_fully - w_matched) < 0.002 * w_matched, (name, w_fully, w_matched)
        # And the new reference is a genuine seventh reading, not a duplicate
        # of the point-gap one it is the segment-gap twin of.
        assert (
            _rel(M3_SEGMENT_BS2_REF[name], M3_REFS_QUOTED[name]["bspline2_321"]) > 1e-5
        ), name
        assert w_fully > (2.0 if name == "dipole" else 2.9)


@pytest.mark.slow
@pytest.mark.parametrize(
    "design,gal_bs2,ptgap_bs2",
    [("wire.lazy_h", 8.100e-3, 7.087e-7), ("wire.vbeam", 5.951e-3, 4.980e-7)],
)
def test_near_open_collapse_survives_at_the_fine_rung(design, gal_bs2, ptgap_bs2):
    """§18 (momwire#213) — antennaknobs#478's class through the shipped option.

    §5 measured the near-open feed-model collapse at N=161 (55× / 74×) off a
    research subclass. It is a solver option now (#192), and the question #213
    asks is whether the collapse is a coarse-mesh coincidence. Pinned here at
    N=321 off the snapshot, where §13 read it:

        wire.lazy_h   gal↔bs2 0.813 %   ptgap↔bs2 0.0029 %      278×
        wire.vbeam    gal↔bs2 0.594 %   ptgap↔bs2 0.0008 %      704×

    Re-measured after the n_qp_pair split (momwire#743), which left the
    MISMATCHED column alone and shrank the matched one ~40x:

        wire.lazy_h   gal↔bs2 0.810 %   ptgap↔bs2 0.00007 %  11,429×
        wire.vbeam    gal↔bs2 0.595 %   ptgap↔bs2 0.00005 %  11,951×

    That asymmetry is itself the §5 claim restated: cross-edge under-
    integration was contributing to the RESIDUAL between two schemes that
    agree analytically once the feed model matches, so integrating properly
    removes it — while the gal↔bs2 gap, which is a real feed-model
    difference, does not move.

    N=641 (5130 / 5119 segments, ~2.5 min and 7 GiB) is the addendum's finest
    rung and lives in the harness (`--near-open`), not in CI; it read 992× and
    374× at the old default, so the ≥100× asserted here is a floor the deep
    rung cleared even before the split widened it.

    The last block is §5's caveat, kept honest: the collapse is a statement
    about the DISAGREEMENT between schemes, not about the error. Every scheme's
    own 161→321 step is still 0.5-1.8 %, larger than any cross-scheme gap at
    that rung and now ~10,000× the matched residual — so "expect the last
    percent to be physical, and budget fine mesh" is untouched by the feed
    model, and more clearly so than before.

    ~18 s per design (36 s for the pair): four solves at each of two fine
    meshes, 2570 / 1290 segments.
    """
    rows = json.loads(SNAPSHOT.read_text())
    z = {}
    for rung in (161, 321):
        row = next(r for r in rows if r["design"] == design and r["rung"] == rung)
        z[rung] = {c: solve(c, _kwargs_from_row(row)) for c in COLUMNS}
    fine, coarse = z[321], z[161]

    r_gal = _rel(fine["gal"], fine["bs2"])
    r_pt = _rel(fine["ptgap"], fine["bs2"])
    # 5 % on the mismatched column; 50 % on the matched one. The matched
    # column is a difference of two ~4 kΩ impedances and the n_qp_pair split
    # (momwire#743) shrank it ~40x, from the 1e-5 level to 1e-7 — a few
    # milliohms out of kilohms. The same absolute perturbation is therefore
    # ~40x larger in RELATIVE terms than when the 10 % bar here was ratified,
    # so keeping 10 % would be silently tightening the gate 40-fold on a
    # near-cancellation. 50 % preserves the margin that was actually agreed
    # (this tree's rule: >= 15-20 % against CI allocator variance) and still
    # catches the only failure that matters — a feed_model that stopped
    # taking effect lands ptgap on gal, four decades away. Both bars remain
    # far tighter than the 100x the verdict rests on.
    assert abs(r_gal - gal_bs2) < 0.05 * gal_bs2, (design, r_gal)
    assert abs(r_pt - ptgap_bs2) < 0.50 * ptgap_bs2, (design, r_pt)
    # The verdict, structurally: matching the feed model buys two decades or
    # better. (Also the no-op guard — a `feed_model` that stopped taking
    # effect would land `ptgap` on `gal` and read 1×.)
    assert r_gal > 100 * r_pt, (design, r_gal, r_pt)
    # Swapping the TESTING buys nothing on this class — §5's other half, and
    # what makes the feed model the whole story here.
    assert abs(_rel(fine["coll"], fine["bs2"]) - r_gal) < 0.25 * r_gal, fine

    # §5's caveat: the residual that is left is SHARED, and it dominates.
    steps = {c: _rel(coarse[c], fine[c]) for c in COLUMNS}
    assert min(steps.values()) > 100 * r_pt, (design, steps, r_pt)
    for c in ("coll", "gal"):
        assert steps[c] > _rel(fine[c], fine["bs2"]), (design, c, steps)


def test_point_gap_drive_is_self_dual():
    """The matched-feed column's construction, checked rather than asserted.

    A point gap's Galerkin drive column IS the basis-evaluation vector at the
    gap, and the default `feed_readout="centre"` reads the current at exactly
    that point — so drive and readout are exact duals and the column's Y is
    machine-symmetric with no `feed_readout="variational"` opt-in and none of
    the M3 payoff traded. If that ever stops holding, the feed-model column of
    the report is measuring something other than a feed model.

    (The construction itself is pinned by
    `test_point_gap_feed_y_is_symmetric_in_both_readouts` in
    `test_sinusoidal_galerkin.py`; this is the report's stake in it.)
    """
    kw = _dipole_kwargs(41)
    two_feeds = dict(
        kw,
        feeds=[(0, 0.962 * 22 / 4, 1 + 0j), (0, 0.962 * 22 / 4 * 0.5, 0 + 0j)],
    )
    two_feeds.pop("feed_wire_index")
    two_feeds.pop("feed_arclength")
    Y = SinusoidalGalerkinSolver(**two_feeds, feed_model="point").compute_y_matrix()
    asym = abs(Y - Y.T).max() / abs(Y).max()
    # The floor is the FILL's reciprocity floor (8.3e-12, momwire#182 M2), not
    # the port algebra's; the default delta gap sits at ~1e-5 here.
    assert asym < 1e-9, asym
    # The contrast is the SEGMENT gap, named rather than inherited: momwire#654
    # made the point gap this class's default, so a bare construction here
    # would compare the point gap against itself and the strict `>` below
    # would be measuring nothing but round-off.
    Y_gap = SinusoidalGalerkinSolver(
        **two_feeds, feed_model="segment"
    ).compute_y_matrix()
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
    for label, s in (
        ("coll", SinusoidalSolver(**kw)),
        ("gal", SinusoidalGalerkinSolver(**kw, feed_model="segment")),
        ("ptgap", SinusoidalGalerkinSolver(**kw, feed_model="point")),
    ):
        n[label] = s._build_geometry()["n_segs"]
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
