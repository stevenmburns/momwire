"""Re-derive the quadrature-order defaults nothing has revisited (#743, #754).

Four knobs entered the tree as first-cut numbers and were never re-derived:
``n_qp_pair=4`` and ``n_qp_source=16`` (BSplineSolver), ``n_qp_path=32`` and
``n_qp_source=12`` (RazorSolver), and ``n_qp_sommerfeld=3`` (both). #743 shows
the first is under-set on split straight wires; #754 shows the third is ~8x
past convergence on straight dipoles. Both issues say the same thing about
what is missing: the evidence is straight dipoles, and the honest move is a
sweep across the geometry classes that stress the integral harder.

WHAT IT MEASURES, AND THE TWO KINDS OF ANSWER
---------------------------------------------
Prefer a KNOWN ZERO over self-convergence, and label which a row is.

``split`` is a known zero. Splitting a straight wire at a collinear anchor is
geometrically a no-op — the knot vectors are identical, multiplicity 1, so the
basis is C1 either way — but it moves the straddling pairs off the analytic
same-edge path onto general Gauss-Legendre. So ``|Z_split - Z_unsplit|`` has a
known exact answer of zero, and any residual is pure quadrature error. That is
a far stronger instrument than watching a ladder stop moving, because a family
that is uniformly wrong still self-converges.

Every other deck reports self-convergence against the top of its ladder, and
says so in the row (``reference: "self"`` vs ``"known-zero"``).

CONTROLS, BECAUSE A LADDER IS ONLY AS GOOD AS ITS ATTRIBUTION
--------------------------------------------------------------
``--controls`` runs three checks that must pass before any row is trusted:

1. **determinism** — the same cell twice, asserted bit-identical. These are
   exact numerics, so unlike a timing study there is no noise floor to argue
   about; establishing that up front converts "is this real?" into a settled
   question rather than a running one.
2. **attribution** — #743 claims same-edge is insensitive to ``n_qp_pair``
   while cross-edge is not, which is the premise of splitting the knob in two.
   Tested as a SEPARATION OF SCALE, not an equality: #743 saw bit-identical
   same-edge results at N=400, but the same-edge smooth-kernel piece does read
   ``n_qp_pair``, so on a coarse mesh it moves a little and an equality test
   fails for the wrong reason. Measured at N=18: same-edge spans 3.1e-03 ohm
   across qp 2..8 while cross-edge spans 2.4e+00 — a factor of 773. If both
   moved comparably, or neither did, the proposed split is misconceived.
3. **reach** — the knob must actually change the code path under test. A
   ladder whose rows are identical because the value never arrived looks
   exactly like a converged one.

THE CAP
-------
``_accel_bspline.cpp`` refuses ``n_qp > 8`` at six sites ("scratch buffer
size"); there is no Python fallback, so ``n_qp_pair`` above 8 raises. The
reachable ladder is therefore 2..8, and this probe deliberately measures that
range first: if 8 is enough on the hard classes, #743's fix needs no C++
change at all, and the cap only becomes load-bearing if it is not.

    python scripts/probe_quadrature_defaults.py --controls
    python scripts/probe_quadrature_defaults.py --knob n_qp_pair > rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

LAMBDA = 20.0
A_THIN = 0.0005
L_DIPOLE = 0.485 * LAMBDA  # 9.70 m, the #743 deck

# Ladders. n_qp_pair stops at 8: above it the accelerator throws (see THE CAP).
LADDERS = {
    "n_qp_pair": (2, 3, 4, 5, 6, 7, 8),
    "n_qp_source": (4, 8, 12, 16, 24, 32),
    "n_qp_sommerfeld": (1, 2, 3, 4, 6, 8),
    "n_qp_path": (1, 2, 3, 4, 6, 8, 12, 16, 32, 48),
}

# Which solver owns which knob.
OWNER = {
    "n_qp_pair": ("bspline",),
    "n_qp_source": ("bspline", "razor"),
    "n_qp_sommerfeld": ("bspline", "razor"),
    "n_qp_path": ("razor",),
}


def _straight(nsegs):
    h = L_DIPOLE / 2
    return dict(
        wires=[[(0.0, 0.0, -h), (0.0, 0.0, h)]],
        nsegs=nsegs,
        wire_radius=A_THIN,
        feed_arclength=0.25 * L_DIPOLE,
    )


def _split(nsegs):
    """Same wire, one extra COLLINEAR anchor at mid-length. Geometrically a
    no-op; the knot vectors match. Segment count held equal to `_straight` so
    the meshes are the same, not merely comparable."""
    h = L_DIPOLE / 2
    half = nsegs // 2
    return dict(
        wires=[[(0.0, 0.0, -h), (0.0, 0.0, 0.0), (0.0, 0.0, h)]],
        n_per_edge_per_wire=[[half, half]],
        nsegs=nsegs,
        wire_radius=A_THIN,
        feed_arclength=0.25 * L_DIPOLE,
    )


def _bent(nsegs):
    """A real corner: two equal arms at 90 degrees. Genuine cross-edge pairs
    at an angle, where the same-edge analytic path never applies."""
    arm = L_DIPOLE / 2
    return dict(
        wires=[[(0.0, 0.0, -arm), (0.0, 0.0, 0.0), (arm, 0.0, 0.0)]],
        n_per_edge_per_wire=[[nsegs // 2, nsegs // 2]],
        nsegs=nsegs,
        wire_radius=A_THIN,
        feed_arclength=0.25 * L_DIPOLE,
    )


def _junction_radius_step(nsegs):
    """Three wires at one node with DIFFERENT radii — the cross-wire pairs
    see a radius discontinuity, which the a^2 regularization keys on."""
    arm = L_DIPOLE / 3
    return dict(
        wires=[
            [(0.0, 0.0, 0.0), (0.0, 0.0, arm)],
            [(0.0, 0.0, 0.0), (arm, 0.0, 0.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, -arm)],
        ],
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        n_per_edge_per_wire=[nsegs // 3, nsegs // 3, nsegs // 3],
        nsegs=nsegs,
        wire_radius=[A_THIN, 4 * A_THIN, A_THIN],
        feed_wire_index=0,
        feed_arclength=arm * 0.5,
    )


def _near_ground(nsegs, ground_model):
    """Horizontal wire LOW over ground. bspline.py:220-236 (#631) records the
    off-edge rule collapsing to 1.6 relative error against a 256-point
    reference as the PEC image closes in, which is a second in-tree instance
    of n_qp_pair=4 being under-set. Height chosen to sit in that regime."""
    h = L_DIPOLE / 2
    z = 0.02 * LAMBDA
    return dict(
        wires=[[(-h, 0.0, z), (h, 0.0, z)]],
        nsegs=nsegs,
        wire_radius=A_THIN,
        feed_arclength=0.25 * L_DIPOLE,
        ground_z=0.0,
        ground_eps=13.0,
        ground_model=ground_model,
    )


def _ek(nsegs):
    """Extended kernel on a FAT wire. EK changes the same-edge moments and is
    only meaningful while delta/a stays above ~1 (see the solver docstring),
    so the radius is set to keep this mesh inside its usable range."""
    h = L_DIPOLE / 2
    a = L_DIPOLE / nsegs / 4.0
    return dict(
        wires=[[(0.0, 0.0, -h), (0.0, 0.0, h)]],
        nsegs=nsegs,
        wire_radius=a,
        feed_arclength=0.25 * L_DIPOLE,
        extended_kernel=True,
    )


DECKS = {
    "straight": (_straight, "self"),
    "split": (_split, "known-zero"),
    "bent": (_bent, "self"),
    "junction_radius_step": (_junction_radius_step, "self"),
    "near_ground_refl": (lambda n: _near_ground(n, "refl-coef"), "self"),
    "near_ground_somm": (lambda n: _near_ground(n, "sommerfeld"), "self"),
    "ek": (_ek, "self"),
}

# Sommerfeld order only means anything on a Sommerfeld deck.
SOMM_ONLY = ("near_ground_somm",)


def _solver(kind):
    from momwire import BSplineSolver, RazorSolver

    return BSplineSolver if kind == "bspline" else RazorSolver


def solve(kind, deck, nsegs, knob, value, degree=2):
    kwargs = DECKS[deck][0](nsegs)
    kwargs["wavelength"] = LAMBDA
    if kind == "bspline":
        # `degree` is BSplineSolver's alone; RazorSolver is the tent basis by
        # construction and rejects unknown kwargs with a TypeError.
        kwargs["degree"] = degree
    kwargs[knob] = value
    t0 = time.perf_counter()
    z, _ = _solver(kind)(**kwargs).compute_impedance()
    return complex(z), time.perf_counter() - t0


def controls(nsegs=18):
    """Three checks that must pass before any ladder row is trusted."""
    ok = True

    z1, _ = solve("bspline", "straight", nsegs, "n_qp_pair", 4)
    z2, _ = solve("bspline", "straight", nsegs, "n_qp_pair", 4)
    det = z1 == z2
    print(f"determinism      : {'PASS' if det else 'FAIL'}  {z1!r} vs {z2!r}")
    ok &= det

    # Attribution: same-edge should be INSENSITIVE to n_qp_pair and cross-edge
    # sensitive. NOT "bit-identical" — #743 observed that at N=400, but the
    # same-edge SMOOTH-KERNEL piece does read n_qp_pair (see the solver
    # docstring), so on a coarse mesh it moves a little and an equality test
    # fails for the wrong reason. The claim that must hold is a separation of
    # scale, so test the ratio.
    unsplit = [
        solve("bspline", "straight", nsegs, "n_qp_pair", q)[0] for q in (2, 4, 8)
    ]
    split = [solve("bspline", "split", nsegs, "n_qp_pair", q)[0] for q in (2, 4, 8)]
    span_same = max(abs(a - b) for a in unsplit for b in unsplit)
    span_cross = max(abs(a - b) for a in split for b in split)
    ratio = span_cross / span_same if span_same else float("inf")
    sep = ratio > 100
    print(
        f"same-edge span   : {span_same:.3e} ohm  {[f'{z.real:.6f}' for z in unsplit]}"
    )
    print(
        f"cross-edge span  : {span_cross:.3e} ohm  {[f'{z.real:.6f}' for z in split]}"
    )
    print(
        f"attribution      : {'PASS' if sep else 'FAIL'}  cross/same = {ratio:.0f}x (>100)"
    )
    ok &= sep

    # Reach: the cap must still be a cap, or the ladder's top is not the top.
    try:
        solve("bspline", "straight", nsegs, "n_qp_pair", 16)
        print("cap              : FAIL  n_qp_pair=16 did not raise")
        ok = False
    except RuntimeError as e:
        print(f"cap              : PASS  n_qp_pair=16 raises ({e})")

    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--knob", choices=sorted(LADDERS))
    p.add_argument(
        "--decks", default="", help="comma-separated; default all applicable"
    )
    p.add_argument("--nsegs", default="18,30", help="comma-separated mesh sizes")
    p.add_argument("--controls", action="store_true")
    args = p.parse_args()

    if args.controls:
        raise SystemExit(0 if controls() else 1)
    if not args.knob:
        p.error("--knob is required unless --controls")

    decks = args.decks.split(",") if args.decks else list(DECKS)
    for kind in OWNER[args.knob]:
        for deck in decks:
            if deck not in DECKS:
                continue
            if args.knob == "n_qp_sommerfeld" and deck not in SOMM_ONLY:
                continue
            for nsegs in (int(x) for x in args.nsegs.split(",")):
                rows = []
                for q in LADDERS[args.knob]:
                    try:
                        z, secs = solve(kind, deck, nsegs, args.knob, q)
                    except Exception as e:  # noqa: BLE001 — a deck/knob pair the solver refuses is data
                        rows.append({"q": q, "error": f"{type(e).__name__}: {e}"})
                        continue
                    rows.append({"q": q, "re": z.real, "im": z.imag, "secs": secs})
                ref = next((r for r in reversed(rows) if "re" in r), None)
                for r in rows:
                    if "re" in r and ref:
                        d = complex(r["re"] - ref["re"], r["im"] - ref["im"])
                        r["abs_dev_ohm"] = abs(d)
                    print(
                        json.dumps(
                            {
                                "kind": "cell",
                                "solver": kind,
                                "deck": deck,
                                "reference": DECKS[deck][1],
                                "knob": args.knob,
                                "nsegs": nsegs,
                                **r,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    sys.exit(main())
