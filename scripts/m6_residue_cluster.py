"""M6 instrument sweep: the antennaknobs no-mutual residue cluster (#521) and
the near-open high-Q class (#478), read on all four cells of the basis ×
testing matrix plus the matched-feed column.

This is the reproduction harness for
``docs/sinusoidal-galerkin-instrument-report.md``. Every headline number in
that document comes out of this script.

Four columns per geometry, all on the SAME mesh at each rung (so a rung
compares four *schemes*, never four problems):

  ``coll``   ``SinusoidalSolver``               three-term basis, point-matched
  ``gal``    ``SinusoidalGalerkinSolver``       three-term basis, Galerkin
  ``ptgap``  ``…(feed_model="point")``          ditto, B-spline's feed model
  ``bs2``    ``BSplineSolver(degree=2)``        quadratic B-spline, Galerkin

``coll`` vs ``gal`` isolates the TESTING scheme (basis and feed held fixed).
``gal`` vs ``bs2`` is what the census called the sin↔bs2 basis gap — except it
also differs in the feed model, which is why ``ptgap`` exists: it is ``gal``
with the delta gap swapped for the zero-width (point) gap ``BSplineSolver``
uses, so ``ptgap`` vs ``bs2`` is the BASIS difference with everything else
matched. That swap was a research subclass in this file until momwire#192
made it the opt-in ``feed_model="point"`` (see the comment above `COLUMNS`);
the solver's default is still the segment-wide gap ``gal`` reads.

Geometry
--------
The geometries are antennaknobs catalog designs, snapshotted into
``m6_residue_cluster_geoms.json`` so this script — and therefore the report —
reproduces from a momwire checkout alone, and so a later retune of a catalog
design cannot silently move the report's numbers. Re-dump (needs antennaknobs
importable, i.e. run it from the antennaknobs superproject) with:

    python scripts/m6_residue_cluster.py --dump

The snapshot records, per (design, rung), exactly the solver kwargs
``antennaknobs.engines.momwire.MomwireEngine`` builds — polylines, per-edge
segment counts, junction groups, feeds, wavelength, wire radius — plus the
resolved 3-D position of the feed segment's centre, which is the guard for
M3's fixed-feed constraint: the sweep's closing section prints any rung-to-rung
feed drift, because a drifting delta gap is an O(h) perturbation of the PROBLEM
and would otherwise be read as a convergence rate.

Running
-------
    python scripts/m6_residue_cluster.py                    # every design
    python scripts/m6_residue_cluster.py --only specialty.hentenna
    python scripts/m6_residue_cluster.py --seg-cap 1000     # cheaper rungs
    python scripts/m6_residue_cluster.py --dipole-feed-model  # the M5 column
    python scripts/m6_residue_cluster.py --feed-matched       # §17 (momwire#212)
    python scripts/m6_residue_cluster.py --near-open          # §18 (momwire#213)

The near-open ladder runs deeper than the standard sweep — antennaknobs#478's
two designs down to N=641 (~5140 segments), past `DEFAULT_SEG_CAP`, which the
standard sweep therefore records as skipped so its cost stays where §13 quotes
it. Give the deep run ~24 GiB of address space.

Free space only, by construction: momwire#182 M4 pinned that a finite-ground
model on a ground-CONTACT wire is broken on BOTH sinusoidal solvers (an
inherited #151 defect), so no ground read is taken here at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from momwire import BSplineSolver, SinusoidalSolver, SinusoidalGalerkinSolver

SNAPSHOT = Path(__file__).resolve().parent / "m6_residue_cluster_geoms.json"

# antennaknobs#521's no-mutual residue cluster + its helix control, then
# antennaknobs#478's near-open high-Q class. Values are the mesh ladder.
# The helix control is `faceted_helix` on the standard nominal_nsegs ladder:
# antennaknobs#630 found the original helix builder hard-coded one segment
# per winding chord (why the census read it "byte-identical at every N" and
# why this harness once had to sweep `pts_per_turn` instead), and
# antennaknobs#636 split it into faceted/continuous variants that mesh at
# the design density like every other catalog design.
DESIGNS = {
    # --- antennaknobs#521, the T/X-junction residue cluster -----------------
    "specialty.hentenna": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    "specialty.hentenna_slant": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    "arrays.hentenna_array": ("nominal_nsegs", (21, 41, 81, 161)),
    "specialty.hourglass": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    "specialty.hourglass_slant": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    "arrays.hourglass_array": ("nominal_nsegs", (21, 41, 81, 161)),
    "broadband.discone": ("nominal_nsegs", (21, 41, 81, 161)),
    "specialty.bowtie": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    # --- antennaknobs#521's control: no junctions at all, curvature only ----
    "specialty.faceted_helix": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    # --- antennaknobs#478, near-open high-Q ---------------------------------
    "wire.lazy_h": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    "wire.vbeam": ("nominal_nsegs", (21, 41, 81, 161, 321)),
    # EXCLUDED, not overlooked: #478's third member `wire.expanded_lazy_h` and
    # `arrays.delta_looparray_with_tls` are NETWORK designs — their feeds sit
    # at 0 V and the driving-point impedance is produced by antennaknobs'
    # NetworkReducer composing a transmission-line network onto the solver's Y
    # matrix. That number is a property of the reduction, not of a bare
    # momwire solve, and this snapshot deliberately carries solver kwargs
    # only. Reading them here would give 0/0. See the report's exclusions
    # section. (delta_looparray_with_tls was in any case reclassified off
    # #478 as a PyNEC-on-TL-networks artifact: both momwire bases are flat.)
}

# Rungs whose mesh exceeds this are recorded as skipped rather than run.
#
# Once a MEMORY ceiling (2000: the unblocked Galerkin fill's O(N^2 * n_qp)
# workspace OOM-killed a 2379-segment rung on a 32 GiB box — the 2026-07-30
# report's §11), now a patience guard: #194 blocked the numpy fill and added
# the fused C++ far fill, so every rung antennaknobs#521 and #478 quoted
# (discone at N=161, hourglass* at N=321/161, lazy_h/vbeam at N=321, up to
# 2767 segments) runs in a few GiB and a few seconds per column.
DEFAULT_SEG_CAP = 4000


# ---------------------------------------------------------------------------
# the matched-feed column
# ---------------------------------------------------------------------------
# The ZERO-WIDTH (point) delta gap this column reads used to be a research
# subclass here. momwire#182 M5 found it while measuring the port duality:
# `SinusoidalSolver` inherits NEC's delta gap — E_app = V/Δ spread over the
# WHOLE feed segment — so refining the mesh shrinks the source, while
# `BSplineSolver` drives a point gap. Comparing the two bases therefore
# compares two feed models as well, and part of what M2/M3 filed as a sin↔bs2
# basis gap is a feed-model gap; the point gap is also exactly self-dual under
# the default `feed_readout="centre"`.
#
# momwire#192 graduated it to `SinusoidalGalerkinSolver(feed_model="point")` —
# opt-in, with `"segment"` still the default, so the report's M2-M4 numbers do
# not move and this column keeps reading §6/§15 unchanged.


COLUMNS = ("coll", "gal", "ptgap", "bs2")
COLUMN_LABEL = {
    "coll": "sin (collocation)",
    "gal": "sin-Galerkin",
    "ptgap": "sin-Gal, point gap",
    "bs2": "bspline d=2",
}


def solve(column: str, kw: dict) -> complex:
    """Driving-point impedance of one geometry on one column."""
    if column == "coll":
        z, _ = SinusoidalSolver(**kw).compute_impedance()
    elif column == "gal":
        z, _ = SinusoidalGalerkinSolver(**kw).compute_impedance()
    elif column == "ptgap":
        z, _ = SinusoidalGalerkinSolver(**kw, feed_model="point").compute_impedance()
    elif column == "bs2":
        z, _ = BSplineSolver(**kw, degree=2).compute_impedance()
    else:
        raise ValueError(f"unknown column {column!r}")
    return complex(np.atleast_1d(z)[0])


# ---------------------------------------------------------------------------
# snapshot I/O
# ---------------------------------------------------------------------------
def _kwargs_from_row(row: dict) -> dict:
    return dict(
        wires=[np.asarray(w, dtype=float) for w in row["wires"]],
        n_per_edge_per_wire=[list(e) for e in row["n_per_edge_per_wire"]],
        feeds=[(int(i), float(s), complex(*v)) for i, s, v in row["feeds"]],
        wavelength=float(row["wavelength"]),
        wire_radius=float(row["wire_radius"]),
        junctions=(
            [[(int(i), e) for i, e in grp] for grp in row["junctions"]]
            if row["junctions"]
            else None
        ),
    )


def dump_snapshot(path: Path) -> None:
    """Rebuild the geometry snapshot from the antennaknobs catalog.

    Needs `antennaknobs` importable — run from the superproject. Everything
    below reads only public builder/engine state; the engine is asked for the
    SinusoidalSolver mesh, which `_parity_for_solver` gives the sinusoidal
    pair and the quadratic B-spline alike (all three want odd), so one
    snapshot serves all four columns.
    """
    from antennaknobs.cli import resolve_class
    from antennaknobs.engines.momwire import MomwireEngine

    rows = []
    for design, (knob, rungs) in DESIGNS.items():
        cls = resolve_class(design)
        if cls is None:
            raise SystemExit(f"cannot resolve design {design!r}")
        for rung in rungs:
            b = cls()
            setattr(b, knob, rung)
            eng = MomwireEngine(b, solver=SinusoidalSolver, ground="free")
            wl = eng._wavelength_for(b.freq)
            feeds = eng._solver_feeds()
            # Where the delta gap actually LANDS: feed_arclength snaps to the
            # nearest segment centre, so this moves with the mesh unless the
            # request is a segment centre at every rung (M3's fixed-feed
            # constraint). Recorded so the report can state the drift instead
            # of quietly reading it as convergence.
            probe = SinusoidalSolver(
                wires=[np.asarray(w, float) for w in eng._polylines],
                n_per_edge_per_wire=eng._edge_segments,
                feeds=feeds,
                wavelength=wl,
                wire_radius=eng._wire_radius,
                junctions=eng._junctions or None,
            )
            geom = probe._build_geometry()
            fs = int(geom["feed_segs"][0])
            centre = np.asarray(geom["seg_centers"], float)[fs]
            rows.append(
                {
                    "design": design,
                    "knob": knob,
                    "rung": rung,
                    "wires": [np.asarray(w, float).tolist() for w in eng._polylines],
                    "n_per_edge_per_wire": [
                        list(map(int, e)) for e in eng._edge_segments
                    ],
                    "feeds": [
                        [int(i), float(s), [v.real, v.imag]] for i, s, v in feeds
                    ],
                    "wavelength": float(wl),
                    "wire_radius": float(eng._wire_radius),
                    "junctions": [
                        [[int(i), e] for i, e in grp] for grp in (eng._junctions or [])
                    ],
                    "n_segs": int(geom["n_segs"]),
                    "feed_centre": [float(x) for x in centre],
                }
            )
            print(f"dumped {design} {knob}={rung}  n_segs={geom['n_segs']}", flush=True)
    path.write_text(json.dumps(rows))
    print(f"\nwrote {path} ({path.stat().st_size / 1024:.0f} KiB, {len(rows)} rungs)")


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def rel(a: complex, b: complex) -> float:
    """|a − b| / |b| — every gap in the report is stated this way, against a
    NAMED reference b (M3's constraint: never 'the converged value')."""
    return abs(a - b) / abs(b)


def sweep(rows, only=None, seg_cap=DEFAULT_SEG_CAP, columns=COLUMNS):
    out = {}
    for row in rows:
        design = row["design"]
        if only and design not in only:
            continue
        if row["n_segs"] > seg_cap:
            print(
                f"{design:28} {row['knob']}={row['rung']:<4} SKIPPED "
                f"(n_segs {row['n_segs']} > cap {seg_cap})",
                flush=True,
            )
            continue
        kw = _kwargs_from_row(row)
        rec = {"rung": row["rung"], "n_segs": row["n_segs"], "z": {}, "t": {}}
        for col in columns:
            t0 = time.perf_counter()
            rec["z"][col] = solve(col, kw)
            rec["t"][col] = time.perf_counter() - t0
        out.setdefault(design, {"knob": row["knob"], "rungs": [], "feed": []})
        out[design]["rungs"].append(rec)
        out[design]["feed"].append(np.asarray(row["feed_centre"], float))
        cells = "  ".join(
            f"{col}={rec['z'][col].real:8.2f}{rec['z'][col].imag:+8.2f}j"
            for col in columns
        )
        times = "/".join("%.1fs" % rec["t"][c] for c in columns)
        print(
            f"{design:28} {row['knob']}={row['rung']:<4} n={row['n_segs']:<5} {cells}"
            f"   [{times}]",
            flush=True,
        )
    return out


def report(results, columns=COLUMNS):
    print("\n" + "=" * 100)
    print("M6 INSTRUMENT READING — free space, driving-point Z at feed 0")
    print("=" * 100)
    print(
        "\nEvery gap is |ΔZ| / |Z_ref| with the reference NAMED. 'finest' is the\n"
        "finest rung this run reached; 'last step' is that rung against the one\n"
        "below it — the per-scheme mesh-convergence check that has to be green\n"
        "before any gap is attributed to anything (momwire#182 M2's lesson)."
    )
    hdr = (
        f"\n{'design':28} {'finest':>8} {'coll↔bs2':>9} {'gal↔bs2':>9} "
        f"{'ptgap↔bs2':>10} {'coll step':>10} {'gal step':>9} {'bs2 step':>9}"
    )
    print(hdr)
    print("-" * len(hdr.strip("\n")))
    for design, d in results.items():
        rungs = d["rungs"]
        if len(rungs) < 2:
            print(f"{design:28} (fewer than two rungs — nothing to say)")
            continue
        fine, prev = rungs[-1], rungs[-2]
        zf = fine["z"]
        step = {c: rel(prev["z"][c], zf[c]) for c in columns}
        print(
            f"{design:28} {fine['rung']:>8} "
            f"{rel(zf['coll'], zf['bs2']) * 100:8.2f}% "
            f"{rel(zf['gal'], zf['bs2']) * 100:8.2f}% "
            f"{rel(zf['ptgap'], zf['bs2']) * 100:9.3f}% "
            f"{step['coll'] * 100:9.2f}% {step['gal'] * 100:8.2f}% "
            f"{step['bs2'] * 100:8.2f}%"
        )

    print("\nFEED PLACEMENT ACROSS THE LADDER (M3's fixed-feed guard)")
    print(
        "  The delta gap snaps to the nearest segment centre, so a mesh sweep can\n"
        "  translate the SOURCE by up to h/2. Cross-scheme gaps above are immune —\n"
        "  all four columns share one mesh and one feed segment at each rung — but\n"
        "  a per-scheme 'last step' is only a convergence rate where this is small."
    )
    for design, d in results.items():
        f = d["feed"]
        if len(f) < 2:
            continue
        drift = max(float(np.linalg.norm(f[i] - f[-1])) for i in range(len(f)))
        print(f"  {design:28} max |Δ feed centre| over the ladder = {drift:.4f} m")


# ---------------------------------------------------------------------------
# the near-open ladder, all the way down (momwire#213, report §18)
# ---------------------------------------------------------------------------
# §5 read antennaknobs#478's class at N=161 and found the `gal↔bs2` residual
# collapsing 55-74× under a matched feed; §13 re-read it at N=321 through the
# same research subclass's successor. #213 asks the question those two leave
# open: does the collapse SURVIVE at the finest mesh, or is it a coarse-mesh
# coincidence that the shared whole-structure residual §5 filed reasserts
# itself over? That needs a rung below the standard ladder, hence N=641.
#
# It is a separate entry point rather than a bigger default cap because the
# 641 rungs are ~5140 segments and cost more than the other ten designs put
# together; the standard sweep stays at ~3 minutes and records them skipped.
NEAR_OPEN = ("wire.lazy_h", "wire.vbeam")

# Enough for the 641 rungs (5140 / 5127 segments) and nothing beyond them.
NEAR_OPEN_SEG_CAP = 6000


def near_open_report(results, columns=COLUMNS):
    """Per-RUNG table: §5/§13 print the finest rung only, but the question
    here is a trend across the ladder, so every rung is shown with its own
    collapse factor `gal↔bs2 / ptgap↔bs2` — how much of the sinusoidal ↔
    B-spline disagreement the feed model accounts for at that depth."""
    print("\n" + "=" * 100)
    print("NEAR-OPEN LADDER (antennaknobs#478) — free space, driving-point Z at feed 0")
    print("=" * 100)
    print(
        "\n'collapse' is gal↔bs2 / ptgap↔bs2 at the SAME rung: the factor by which\n"
        "matching the feed model shrinks the residual. 'step' columns are each\n"
        "scheme against the rung below it — §5's shared whole-structure term, which\n"
        "no feed model explains and which is what the fine rungs are here to test."
    )
    for design, d in results.items():
        rungs = d["rungs"]
        print(f"\n  {design}")
        hdr = (
            f"  {'N':>5} {'segs':>6} {'coll↔bs2':>9} {'gal↔bs2':>9} "
            f"{'ptgap↔bs2':>10} {'collapse':>9} {'coll step':>10} {'gal step':>9} "
            f"{'ptgap step':>11} {'bs2 step':>9}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, rec in enumerate(rungs):
            z = rec["z"]
            g, p = rel(z["gal"], z["bs2"]), rel(z["ptgap"], z["bs2"])
            step = {c: rel(rungs[i - 1]["z"][c], z[c]) for c in columns} if i else None
            steps = (
                " ".join(f"{step[c] * 100:9.2f}%" for c in ("coll", "gal"))
                + f" {step['ptgap'] * 100:10.2f}% {step['bs2'] * 100:8.2f}%"
                if step
                else f"{'—':>10} {'—':>9} {'—':>11} {'—':>9}"
            )
            print(
                f"  {rec['rung']:>5} {rec['n_segs']:>6} "
                f"{rel(z['coll'], z['bs2']) * 100:8.2f}% {g * 100:8.2f}% "
                # four decimals, not §5/§13's three: the matched column reaches
                # 1e-3 % here and the collapse factor is a ratio OF it.
                f"{p * 100:9.4f}% {g / p:8.0f}× {steps}"
            )
        times = {c: sum(r["t"][c] for r in rungs) for c in columns}
        print(
            "    column wall clock over the ladder: "
            + "  ".join(f"{c} {times[c]:.1f}s" for c in columns)
        )


def near_open(rows, seg_cap=NEAR_OPEN_SEG_CAP):
    results = sweep(rows, only=set(NEAR_OPEN), seg_cap=seg_cap)
    near_open_report(results)
    return results


def dipole_feed_model(n_list=(81, 161, 321)):
    """The feed-model axis on the canonical dipole — momwire#182 M5's finding,
    first-party. Same geometry as the M1-M3 validation dipole."""
    hd = 0.962 * 22 / 4
    print("\n" + "=" * 100)
    print("FEED-MODEL COLUMN — 0.962λ/2 dipole, free space, a=0.5 mm")
    print("=" * 100)
    print(f"\n{'N':>5} {'coll':>26} {'gal':>26} {'ptgap':>26} {'bs2':>26}")
    for n in n_list:
        kw = dict(
            wires=[np.array([[0, 0, -hd], [0, 0, hd]], dtype=float)],
            n_per_edge_per_wire=[[n]],
            feed_wire_index=0,
            feed_arclength=hd,
            wavelength=22,
            wire_radius=0.0005,
        )
        z = {c: solve(c, kw) for c in COLUMNS}
        print(
            f"{n:>5} "
            + " ".join(f"{z[c].real:12.6f}{z[c].imag:+12.6f}j" for c in COLUMNS)
        )
        print(
            f"      gal↔bs2 {rel(z['gal'], z['bs2']):.3e}   "
            f"ptgap↔bs2 {rel(z['ptgap'], z['bs2']):.3e}   "
            f"coll↔bs2 {rel(z['coll'], z['bs2']):.3e}"
        )


# ---------------------------------------------------------------------------
# the feed-MATCHED payoff (momwire#212, report §17)
# ---------------------------------------------------------------------------
# §16 asked what `errColl/errGal` reads with the feed model genuinely held
# fixed. The answer needed a point gap on `SinusoidalSolver`, and there is
# none: a zero-width gap has no collocation RHS (derivation in
# `SinusoidalSolver._reject_point_feed_model`, which now refuses the option).
# What runs below is that refutation, measured, plus the payoff read on the
# only pairing that IS feed-matched — the segment gap on both sides, i.e. what
# M3 already scores.


def zero_width_gap_Ez(d, a, k):
    """E_z of the zero-width gap: the b → a limit of the magnetic frill, i.e.
    a delta ring of magnetic current at ρ = a, observed on the axis at
    arc-distance `d`. Normalized to ∫E_z dz = 1 so it is the segment gap's
    V/Δ with the width taken to zero, and parameter-free once a is fixed.

    Ring on the surface / observation on the axis reproduces the solver's own
    source-filament / BC-on-surface separation R = √(d² + a²), so this is
    evaluated in the fill's convention rather than beside it.
    """
    R = np.sqrt(d**2 + a**2)
    return (a**2 / 2) * (1 + 1j * k * R) * np.exp(-1j * k * R) / R**3


def _zero_width_cell_average(d0, d1, a, k, nq=200):
    """(1/(d1−d0))∫ E_z dz on z = a·sinh(u), which removes the width-a spike
    the raw variable cannot resolve when h ≫ a."""
    u0, u1 = np.arcsinh(d0 / a), np.arcsinh(d1 / a)
    gx, gw = np.polynomial.legendre.leggauss(nq)
    u = 0.5 * (u1 + u0) + 0.5 * (u1 - u0) * gx
    R = a * np.cosh(u)
    f = 0.5 * (1 + 1j * k * R) * np.exp(-1j * k * R) / np.cosh(u) ** 2
    return (f * (0.5 * (u1 - u0) * gw)).sum() / (d1 - d0)


class PointGapCollocation(SinusoidalSolver):
    """The REFUTED construction, kept so the refusal stays measured.

    `sampled="point"` is collocation done literally — the zero-width field
    evaluated AT each match point. `sampled="cell"` averages it over each
    match point's own cell, the smallest width the mesh resolves. Neither is
    a shippable point gap and `_reject_point_feed_model` says why; this class
    is how §17's table is produced.
    """

    def __init__(self, *, sampled="point", **kwargs):
        super().__init__(**kwargs)
        self.sampled = sampled

    def compute_impedance(self):
        import scipy.linalg

        geom = self._build_geometry()
        G, seg_view = self._assemble_Z(geom, self.k)
        fi = geom["feed_segs"][0]
        V = complex(self.feeds[0][2]) or 1.0 + 0j
        a = float(np.atleast_1d(self.wire_radius)[0])
        d = (geom["seg_centers"] - geom["seg_centers"][fi]) @ geom["seg_tangents"][fi]
        h = geom["seg_h"]
        if self.sampled == "point":
            e = zero_width_gap_Ez(d, a, self.k)
        else:
            e = np.array(
                [
                    _zero_width_cell_average(
                        d[m] - h[m] / 2, d[m] + h[m] / 2, a, self.k
                    )
                    for m in range(len(d))
                ]
            )
        self.drive_vector = -V * e
        alpha = scipy.linalg.solve(G, self.drive_vector)
        return V / self._feed_segment_current(alpha, seg_view, fi), alpha


def _m3_dipole_kwargs(n):
    hd = 0.962 * 22 / 4
    return dict(
        wires=[np.array([[0.0, -hd, 0.0], [0.0, hd, 0.0]])],
        n_per_edge_per_wire=[[n]],
        feed_wire_index=0,
        nsegs=n,
        wavelength=22,
    )


def _m3_k2_kwargs(n):
    a, b, c = (0.0, -5.0, 0.0), (0.0, 0.0, -2.0), (0.0, 5.0, 0.0)
    return dict(
        wires=[np.array([a, b]), np.array([b, c])],
        n_per_edge_per_wire=[[n], [n]],
        feed_wire_index=0,
        nsegs=n,
        junctions=[[(0, "end"), (1, "start")]],
        wavelength=22,
    )


# Quoted verbatim from `M3_REFS` in tests/test_sinusoidal_galerkin.py — the
# six defensible references M3's payoff gate takes its WORST ratio over. The
# `bspline2_*` pair is the only feed-MISMATCHED member (BSplineSolver drives a
# zero-width gap; both comparands drive NEC's segment gap), so splitting the
# family on that line is what "feed-matched payoff" can mean here.
M3_REFS_QUOTED = {
    "dipole": dict(
        sin_gal_321=69.639093 - 18.056307j,
        sin_coll_321=69.631876 - 18.107822j,
        bspline2_321=69.633780 - 18.065315j,
        rich_sin_gal=69.634796 - 18.008036j,
        rich_sin_coll=69.633551 - 18.018862j,
        rich_bspline2=69.633701 - 18.010747j,
    ),
    "k2_junction": dict(
        sin_gal_321=124.493250 + 0.392167j,
        sin_coll_321=124.479522 + 0.340718j,
        bspline2_321=124.492289 + 0.367751j,
        rich_sin_gal=124.513021 + 0.444108j,
        rich_sin_coll=124.513126 + 0.445724j,
        rich_bspline2=124.514726 + 0.445185j,
    ),
}
_MATCHED_REFS = ("sin_gal_321", "sin_coll_321", "rich_sin_gal", "rich_sin_coll")
M3_FEED_MATCHED_GEOMS = {"dipole": _m3_dipole_kwargs, "k2_junction": _m3_k2_kwargs}
M3_COARSE = (11, 15, 21)


def feed_matched(ladder=(41, 81, 161, 321, 641)):
    """§17: why coll-point does not exist, and what the payoff reads without it."""
    hd = 0.962 * 22 / 4
    print("\n" + "=" * 100)
    print("A. THE ZERO-WIDTH GAP UNDER COLLOCATION — 0.962λ/2 dipole, a=0.5 mm")
    print("=" * 100)
    print(
        f"\n{'N':>5} {'h':>9} {'a/h':>9} {'Z coll(segment)':>25} "
        f"{'Z point-sampled':>25} {'(2a/h)Zseg/Zpt−1':>17} {'‖v−c·v_seg‖':>13} "
        f"{'cell-avg ↔ segment':>19}"
    )
    for n in ladder:
        kw = dict(
            wires=[np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]])],
            n_per_edge_per_wire=[[n]],
            feed_wire_index=0,
            feed_arclength=hd,
            wavelength=22,
            wire_radius=0.0005,
        )
        s = SinusoidalSolver(**kw)
        z_seg = complex(np.atleast_1d(s.compute_impedance()[0])[0])
        geom = s._build_geometry()
        fi = geom["feed_segs"][0]
        h, a = float(geom["seg_h"][fi]), 0.0005
        pt = PointGapCollocation(**kw, sampled="point")
        z_pt = complex(np.atleast_1d(pt.compute_impedance()[0])[0])
        v = pt.drive_vector
        v_seg = np.zeros_like(v)
        v_seg[fi] = -1.0 / h
        dev = np.abs(v - (v[fi] / v_seg[fi]) * v_seg).max() / np.abs(v).max()
        z_cell = complex(
            np.atleast_1d(
                PointGapCollocation(**kw, sampled="cell").compute_impedance()[0]
            )[0]
        )
        print(
            f"{n:>5} {h:9.5f} {a / h:9.3e} {z_seg.real:11.6f}{z_seg.imag:+11.6f}j "
            f"{z_pt.real:11.6f}{z_pt.imag:+11.6f}j "
            f"{abs((2 * a / h) * z_seg / z_pt - 1):17.3e} {dev:13.3e} "
            f"{rel(z_cell, z_seg):19.3e}"
        )
    print(
        "\n  Point-sampled: the RHS is the segment-gap RHS times h/2a, so Z comes\n"
        "  out (2a/h)·Z_segment and DOUBLES every rung — collocation reads the\n"
        "  source's width off the feed row's amplitude. Cell-averaged: the\n"
        "  segment gap, to ~1e-7. There is no third reading."
    )

    print("\n" + "=" * 100)
    print("B. errColl/errGal WITH THE FEED MODEL HELD FIXED (segment on both)")
    print("=" * 100)
    for name, factory in M3_FEED_MATCHED_GEOMS.items():
        z = {
            (scheme, n): complex(
                np.atleast_1d(
                    (
                        SinusoidalSolver
                        if scheme == "coll"
                        else SinusoidalGalerkinSolver
                    )(**factory(n)).compute_impedance()[0]
                )[0]
            )
            for scheme in ("coll", "gal")
            for n in M3_COARSE
        }
        print(f"\n  {name}")
        print(
            "  "
            + f"{'reference':>16} "
            + " ".join(f"{'N=%d' % n:>9}" for n in M3_COARSE)
            + "   reference feed"
        )
        per_ref = {}
        for ref_name, ref in M3_REFS_QUOTED[name].items():
            per_ref[ref_name] = [
                rel(z[("coll", n)], ref) / rel(z[("gal", n)], ref) for n in M3_COARSE
            ]
            tag = "segment" if ref_name in _MATCHED_REFS else "POINT (bspline)"
            print(
                "  "
                + f"{ref_name:>16} "
                + " ".join(f"{v:9.3f}" for v in per_ref[ref_name])
                + f"   {tag}"
            )
        w_all = min(min(v) for v in per_ref.values())
        w_m = min(min(per_ref[r]) for r in _MATCHED_REFS)
        print(
            f"    worst over all six references      {w_all:.3f}\n"
            f"    worst over feed-matched references {w_m:.3f}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", action="store_true", help="rebuild the geometry snapshot")
    ap.add_argument("--only", nargs="+", help="restrict to these dotted design names")
    # Unset means "this run's own cap": DEFAULT_SEG_CAP for the standard sweep,
    # the higher NEAR_OPEN_SEG_CAP for --near-open, which exists to reach past it.
    ap.add_argument("--seg-cap", type=int, default=None)
    ap.add_argument("--dipole-feed-model", action="store_true")
    ap.add_argument(
        "--feed-matched",
        action="store_true",
        help="report §17: the zero-width gap under collocation, and the "
        "feed-matched payoff",
    )
    ap.add_argument(
        "--near-open",
        action="store_true",
        help="report §18: antennaknobs#478's two designs at every rung "
        "including N=641, with the per-rung feed-model collapse factor",
    )
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = ap.parse_args(argv)

    if args.dump:
        dump_snapshot(args.snapshot)
        return 0
    if args.dipole_feed_model:
        dipole_feed_model()
        return 0
    if args.feed_matched:
        feed_matched()
        return 0
    if not args.snapshot.exists():
        raise SystemExit(f"{args.snapshot} missing — run with --dump first")
    rows = json.loads(args.snapshot.read_text())
    if args.near_open:
        near_open(
            rows,
            seg_cap=NEAR_OPEN_SEG_CAP if args.seg_cap is None else args.seg_cap,
        )
        return 0
    results = sweep(
        rows,
        only=set(args.only) if args.only else None,
        seg_cap=DEFAULT_SEG_CAP if args.seg_cap is None else args.seg_cap,
    )
    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
