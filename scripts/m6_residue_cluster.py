"""M6 instrument sweep: the antennaknobs no-mutual residue cluster (#521) and
the near-open high-Q class (#478), read on all four cells of the basis ×
testing matrix plus the matched-feed column.

This is the reproduction harness for
``docs/sinusoidal-galerkin-instrument-report.md``. Every headline number in
that document comes out of this script.

Four columns per geometry, all on the SAME mesh at each rung (so a rung
compares four *schemes*, never four problems):

  ``coll``   ``SinusoidalSolver``            three-term basis, point-matched
  ``gal``    ``SinusoidalGalerkinSolver``    three-term basis, Galerkin
  ``ptgap``  ``PointGapGalerkin`` (below)    ditto, with B-spline's feed model
  ``bs2``    ``BSplineSolver(degree=2)``     quadratic B-spline, Galerkin

``coll`` vs ``gal`` isolates the TESTING scheme (basis and feed held fixed).
``gal`` vs ``bs2`` is what the census called the sin↔bs2 basis gap — except it
also differs in the feed model, which is why ``ptgap`` exists: it is ``gal``
with the delta gap swapped for the zero-width (point) gap ``BSplineSolver``
uses, so ``ptgap`` vs ``bs2`` is the BASIS difference with everything else
matched. See `PointGapGalerkin` for why that is a ten-line subclass and not a
solver change.

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
# antennaknobs#478's near-open high-Q class. Values are the mesh ladder; the
# helix's mesh knob is `pts_per_turn`, not `nominal_nsegs`, because its
# builder hard-codes one segment per chord (which is exactly why the census
# ladder read it "byte-identical at every N").
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
    "specialty.helix": ("pts_per_turn", (8, 12, 16, 24, 32)),
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
# 2000 is a MEMORY ceiling, not a patience one. The Galerkin fill's near-pair
# quadrature workspace is O(N^2 * n_qp) complex, so a rung that solves in
# ~60 s at 1900 segments is OOM-killed at 2379 on a 32 GiB box. Raising it is
# not a matter of waiting longer; it needs the fill blocked or accelerated
# (deliberately out of scope while momwire#182's Python-first rule stands).
# Consequence, stated in the report: a few of the rungs antennaknobs#521 and
# #478 quoted (discone at N=161, lazy_h/vbeam at N=321) are out of reach here,
# and are reported as skipped rather than substituted for.
DEFAULT_SEG_CAP = 2000


# ---------------------------------------------------------------------------
# the matched-feed column
# ---------------------------------------------------------------------------
class PointGapGalerkin(SinusoidalGalerkinSolver):
    """`SinusoidalGalerkinSolver` with a ZERO-WIDTH (point) delta gap.

    momwire#182 M5 found a third instrument axis while measuring the port
    duality: the source model. `SinusoidalSolver` inherits NEC's delta gap —
    E_app = V/Δ spread over the WHOLE feed segment — so refining the mesh
    shrinks the source, which is why the dipole reactance walks logarithmically
    with no limit (M3). `BSplineSolver` drives a point gap instead. Comparing
    the two bases therefore compares two feed models as well, and part of what
    M2/M3 filed as a sin↔bs2 basis gap is a feed-model gap.

    The point gap is E_app = V·δ(s − s0), so the Galerkin test integral
    collapses on the delta and the drive column is just −V·f_i(s0): the same
    basis-evaluation vector the CENTRE readout already uses (`sigma·(A + C)`,
    since sin(k·0) = 0 kills B). Putting s0 at the feed segment's centre makes
    drive and readout exact duals of each other, so this column's Y is
    machine-symmetric with the DEFAULT `feed_readout="centre"` — no knob, no
    payoff traded.

    Deliberately a research subclass in a script, not a solver option: adopting
    it would re-litigate every M2-M4 number, and its job here is to MEASURE the
    feed-model contribution, not to become a fifth cell.
    """

    def _drive_columns(self, geom, seg_view, k):
        U = super()._drive_columns(geom, seg_view, k)
        starts = seg_view["starts"]
        for j, fseg in enumerate(geom["feed_segs"]):
            s, e = starts[fseg], starts[fseg + 1]
            U[:, j] = 0.0
            np.add.at(
                U[:, j],
                seg_view["jbasis"][s:e],
                -(seg_view["sigma"][s:e] * (seg_view["A"][s:e] + seg_view["C"][s:e])),
            )
        return U


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
        z, _ = PointGapGalerkin(**kw).compute_impedance()
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", action="store_true", help="rebuild the geometry snapshot")
    ap.add_argument("--only", nargs="+", help="restrict to these dotted design names")
    ap.add_argument("--seg-cap", type=int, default=DEFAULT_SEG_CAP)
    ap.add_argument("--dipole-feed-model", action="store_true")
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = ap.parse_args(argv)

    if args.dump:
        dump_snapshot(args.snapshot)
        return 0
    if args.dipole_feed_model:
        dipole_feed_model()
        return 0
    if not args.snapshot.exists():
        raise SystemExit(f"{args.snapshot} missing — run with --dump first")
    rows = json.loads(args.snapshot.read_text())
    results = sweep(
        rows, only=set(args.only) if args.only else None, seg_cap=args.seg_cap
    )
    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
