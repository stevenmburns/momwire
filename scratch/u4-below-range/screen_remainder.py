"""U4 screen: the below/below remainder at and past momwire's 4 λ_m cap.

MEASUREMENTS.md (registered before the run). Evaluates the #524 phase-0
prototype's regime-2 field, direct + A_m·image + remainder, at image distances
R1 ∈ {0.1 … 8} λ_m and at the pair's self scale ρ = Δ, for the LPDA's
soil, frequency and depths and for the SPEC soils. Reports M1 (the plan's
literal |E_rem|/|E_dir| at 4 λ_m), M2 (|E_rem(R1)| / |E_tot(Δ)|) and M3 (the
log-log slope of |E_rem| over 4 → 8 λ_m), under guards G-a (the ε̃ = 1 collapse
at range) and G-b (the prototype's own quadrature estimate).

Run from the momwire repo root with antennaknobs checked out alongside:
  python scratch/u4-below-range/screen_remainder.py [--proto DIR] [--out F]
    [--guard-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_PROTO = Path.home() / "stevenmburns/antennaknobs/scratch/524-phase0/proto"
R1_LAMBDA = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
KINDS = (("HED", 0.0), ("HED", 90.0), ("VED", 0.0))
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}


def cases():
    out = [
        dict(
            name="LPDA-shallow",
            soil=(13.0, 0.005),
            f=3.5e6,
            ds=0.1524,
            do=0.1524,
            delta=0.6096,
        ),
        dict(
            name="LPDA-deep",
            soil=(13.0, 0.005),
            f=3.5e6,
            ds=0.6858,
            do=0.6858,
            delta=0.6096,
        ),
        dict(
            name="LPDA-cross",
            soil=(13.0, 0.005),
            f=3.5e6,
            ds=0.1524,
            do=0.6858,
            delta=0.6096,
        ),
    ]
    for sid, soil in SOILS.items():
        for f in (7e6, 21e6):
            for d in (0.02, 0.15):
                out.append(
                    dict(
                        name=f"SPEC-{sid}-{f / 1e6:g}-{d:g}",
                        soil=soil,
                        f=f,
                        ds=d,
                        do=d,
                        delta=None,
                    )
                )
    return out


def _proto(proto_dir):
    sys.path.insert(0, str(proto_dir))
    import buried_proto as bp

    return bp


def _point(args):
    proto_dir, soil, f, ds, do, rho, kind, phi_deg = args
    bp = _proto(proto_dir)
    hs = bp.HalfSpace(f, soil[0], soil[1])
    hs.assert_decay()
    phi = math.radians(phi_deg)
    obs = (rho * math.cos(phi), rho * math.sin(phi), -do)
    t0 = time.time()
    tot, rel, _q, parts = bp.field_in_medium(hs, obs, -ds, kind, parts=True)
    return dict(
        rho=rho,
        kind=kind,
        phi=phi_deg,
        direct=float(np.linalg.norm(parts["direct"])),
        image=float(np.linalg.norm(parts["A_m"] * parts["image"])),
        rem=float(np.linalg.norm(parts["rem"])),
        total=float(np.linalg.norm(tot)),
        rel=float(rel),
        seconds=time.time() - t0,
    )


def lam_m(soil, f, proto_dir):
    bp = _proto(proto_dir)
    return 2.0 * math.pi / abs(bp.HalfSpace(f, soil[0], soil[1]).km)


def guard_jobs(proto_dir):
    """G-c: small-contrast linearity at range. The remainder is first order in
    (eps~ - 1), so eps_r 1.02 against 1.01 (sigma = 0) doubles it. That makes
    the far-range integrator do real work, unlike eps~ = 1 exactly, which the
    prototype short-circuits (`k_s == k_o`)."""
    jobs = []
    lam0 = 299792458.0 / 7e6
    h = 0.15 + 0.15
    for r1 in (4.0, 8.0):
        for kind, phi in (("HED", 0.0), ("VED", 0.0)):
            rho = math.sqrt((r1 * lam0) ** 2 - h * h)
            for er in (1.01, 1.02):
                jobs.append((proto_dir, (er, 0.0), 7e6, 0.15, 0.15, rho, kind, phi))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", type=Path, default=DEFAULT_PROTO)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--guard-only", action="store_true")
    args = ap.parse_args()
    proto_file = args.proto / "buried_proto.py"
    meta = dict(
        proto=str(proto_file),
        proto_sha256=hashlib.sha256(proto_file.read_bytes()).hexdigest(),
        integrator=dict(n=16, rtol=1e-11, depth=14, max_panels=6000, detour=2.0),
    )
    print(meta, flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        guard = list(ex.map(_point, guard_jobs(args.proto)))
        ratios = [
            (guard[k + 1]["rem"] / guard[k]["rem"], guard[k])
            for k in range(0, len(guard), 2)
        ]
        for ratio, g in ratios:
            print(
                f"G-c {g['kind']} phi={g['phi']:g} rho={g['rho']:.1f}: rem(1.02)/rem(1.01) "
                f"{ratio:.4f}  rel {g['rel']:.1e}  {g['seconds']:.0f}s",
                flush=True,
            )
        if not all(1.9 <= r <= 2.1 for r, _g in ratios):
            raise SystemExit(
                "G-c missed: the far-range remainder is not linear in contrast; stop"
            )
        if args.guard_only:
            return

        rows = []
        for c in cases():
            lm = lam_m(c["soil"], c["f"], args.proto)
            delta = c["delta"] if c["delta"] is not None else lm / 26.0
            h = c["ds"] + c["do"]
            jobs, labels = [], []
            for kind, phi in KINDS:
                jobs.append(
                    (args.proto, c["soil"], c["f"], c["ds"], c["do"], delta, kind, phi)
                )
                labels.append((kind, phi, "self"))
                for r1 in R1_LAMBDA:
                    rho = math.sqrt(max((r1 * lm) ** 2 - h * h, 0.0))
                    jobs.append(
                        (
                            args.proto,
                            c["soil"],
                            c["f"],
                            c["ds"],
                            c["do"],
                            rho,
                            kind,
                            phi,
                        )
                    )
                    labels.append((kind, phi, r1))
            res = list(ex.map(_point, jobs))
            for (kind, phi, tag), r in zip(labels, res):
                rows.append(
                    dict(case=c["name"], lam_m=lm, delta=delta, r1_lambda=tag, **r)
                )
            for kind, phi in KINDS:
                sub = {
                    r["r1_lambda"]: r
                    for r in rows
                    if r["case"] == c["name"] and r["kind"] == kind and r["phi"] == phi
                }
                selfv = sub["self"]["total"]
                m1 = sub[4.0]["rem"] / sub[4.0]["direct"]
                m2 = {x: sub[x]["rem"] / selfv for x in (4.0, 5.0, 6.0, 8.0)}
                m3 = math.log(sub[8.0]["rem"] / sub[4.0]["rem"]) / math.log(2.0)
                worst_rel = max(r["rel"] for k, r in sub.items())
                print(
                    f"{c['name']:18s} {kind} phi={phi:>4g}  lam_m={lm:7.3f}  M1={m1:9.3e}  "
                    f"M2@4,5,6,8={m2[4.0]:.2e},{m2[5.0]:.2e},{m2[6.0]:.2e},{m2[8.0]:.2e}  "
                    f"M3={m3:+.3f}  worst rel {worst_rel:.1e}",
                    flush=True,
                )
    if args.out is not None:
        args.out.write_text(
            json.dumps(dict(meta=meta, guard=guard, rows=rows), indent=1)
        )
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
