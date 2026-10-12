"""Time one solve per deck under each accelerator variant (momwire#1370).

Every timed solve is ONE cold `compute_impedance()` in its OWN interpreter,
with MOMWIRE_FORCE_VARIANT set: two builds of one extension never share a
process (see tests/test_accel_cpu_guard_1032.py), and a warm process is not
the measurement. The per-process memos (the Sommerfeld grids above all) make
a second solve of the same deck in one process up to ~6x faster (bs2
Sommerfeld, Skylake: 0.07 s warm against 0.41 s cold), and that is exactly
the work whose codegen the variants differ in. So the import and the solver's
construction are excluded and the solve is timed cold; `--warmup N` adds N
untimed in-process solves first, for a warm figure on purpose.

`--repeats` children per (deck, variant), run round-robin across the
variants so drift on the box lands on all of them alike; the median is the
figure. Thread pools are pinned to `--threads` (OMP, OpenBLAS, MKL) and, with
`--cpus`, each child is bound to those cores. Sommerfeld's disk cache is off
in the children, so no solve reads tables an earlier one wrote.

The variants timed are those this install carries and this CPU can run:
`avx512` only where `momwire._accel`'s AVX-512 check passes (forcing it
elsewhere loads nothing, by design), then `avx2` and `sse2`.

    python scripts/time_variants_1370.py --threads 4 --out variants.json
    python scripts/time_variants_1370.py --threads 1 --decks bs2_free_2000

Prints the CPU model and flags first, then a table (median, min, speed
relative to avx2, |dZ|/|Z| against avx2), and writes all of it as JSON.
Uses only the installed momwire: run it from a checkout or anywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

C0 = 299792458.0
WL = C0 / 7.1e6
WL7 = C0 / 7e6
A_WIRE = 0.001
SOIL = (13.0, 0.005)


# ---------------------------------------------------------------------------
# Decks. hub/invl are the shapes of tests/test_crossing_serve_524.py's
# `hub_deck` and `invl_deck`, copied so the script needs no test tree.
# ---------------------------------------------------------------------------


def _dipole(n, z0=0.0, **kw):
    import numpy as np

    half = 0.48 * WL / 2
    return dict(
        wires=[np.array([(-half, 0.0, z0), (half, 0.0, z0)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, half, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        **kw,
    )


def _hub(n_radials=16, depth=0.15):
    import numpy as np

    dirs = [
        (np.cos(2 * np.pi * i / n_radials), np.sin(2 * np.pi * i / n_radials))
        for i in range(n_radials)
    ]
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -depth), (0.0, 0.0, -depth)]) for dx, dy in dirs
    ]
    npe = [[10] for _ in dirs]
    rise_i = len(wires)
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    npe.append([2])
    mono_i = rise_i + 1
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    npe.append([15])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "end") for i in range(n_radials)] + [(rise_i, "start")],
            [(rise_i, "end"), (mono_i, "end")],
        ],
        feeds=[(mono_i, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )


def _invl(n_radials=16, x=1):
    import numpy as np

    d = _hub(n_radials=n_radials)
    mono_i = n_radials + 1
    top = (0.0, 0.0, 10.0)
    d["wires"][mono_i] = np.array([top, (0.0, 0.0, 0.0)])
    d["wires"].append(np.array([top, (5.0, 0.0, 10.0)]))
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[n * x for n in e] for e in npe[:n_radials]] + [
        npe[n_radials],
        [npe[mono_i][0] * x],
        [8 * x],
    ]
    d["junctions"].append([(mono_i, "start"), (mono_i + 1, "start")])
    return d


def _make(deck):
    from momwire.bspline import BSplineSolver
    from momwire.razor import RazorSolver
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    if deck == "bs2_free_2000":
        return BSplineSolver(**_dipole(2000))
    if deck == "bs2_sommerfeld":
        return BSplineSolver(
            **_dipole(
                301, z0=10.0, ground_z=0.0, ground_eps=SOIL, ground_model="sommerfeld"
            )
        )
    if deck == "sg_buried_hub16":
        return SinusoidalGalerkinSolver(**_hub(16))
    if deck == "razor_invl16x4":
        return RazorSolver(**_invl(16, x=4), nec5_quadrature=True)
    raise SystemExit(f"unknown deck {deck!r}")


DECKS = ("bs2_free_2000", "bs2_sommerfeld", "sg_buried_hub16", "razor_invl16x4")


# ---------------------------------------------------------------------------
# The child: one variant, one deck
# ---------------------------------------------------------------------------


def _child(args) -> int:
    if args.cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, _parse_cpus(args.cpus))
    import warnings

    import numpy as np

    import momwire

    want = os.environ.get("MOMWIRE_FORCE_VARIANT")
    if momwire.accelerator_variant != want:
        print(
            f"variant {want!r} was forced but {momwire.accelerator_variant!r} loaded",
            file=sys.stderr,
        )
        return 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(args.warmup):
            _make(args.deck).compute_impedance()
        solver = _make(args.deck)
        spent = _time_the_dense_solves()
        t0 = time.perf_counter()
        out = solver.compute_impedance()
        dt = time.perf_counter() - t0
    z = complex(np.asarray(out[0] if isinstance(out, tuple) else out).ravel()[0])
    print(
        json.dumps(
            {
                "variant": want,
                "time": dt,
                "solve": spent["s"],
                "solve_calls": spent["calls"],
                "file": momwire.__file__,
                "z": [z.real, z.imag],
            }
        )
    )
    return 0


def _time_the_dense_solves() -> dict:
    """Wrap the dense LU entry points the solvers reach (`bspline._lu_solve`
    and SG's `_solve_in_place` call `scipy.linalg.lu_factor`/`lu_solve`; the
    constraint and port paths call `scipy.linalg.solve` / `np.linalg.solve`),
    so the parent can split a solve into the LAPACK part (OpenBLAS in every
    variant) and the rest (the fills, which the variants compile). Installed
    after the warmups. `calls` is reported so a split of 0 s reads as "the
    wrappers were never reached", not as "the solve was free"."""
    import numpy as np
    import scipy.linalg

    spent = {"s": 0.0, "calls": 0}

    def wrap(owner, name):
        fn = getattr(owner, name)

        def timed(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                spent["s"] += time.perf_counter() - t
                spent["calls"] += 1

        setattr(owner, name, timed)

    for name in ("lu_factor", "lu_solve", "solve"):
        wrap(scipy.linalg, name)
    wrap(np.linalg, "solve")
    return spent


def _parse_cpus(spec):
    cpus = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------


def _cpu() -> dict:
    info = {"machine": platform.machine(), "platform": sys.platform, "model": "?"}
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                key, _, val = line.partition(":")
                key = key.strip()
                if key == "model name" and info["model"] == "?":
                    info["model"] = val.strip()
                if key == "flags":
                    info["flags"] = val.split()
                    break
    except OSError:
        info["model"] = platform.processor() or "?"
    info["logical_cpus"] = os.cpu_count()
    return info


def _variants() -> tuple[list[str], dict]:
    from momwire import _accel

    notes = {
        "avx512_missing": _accel._avx512_missing(),
        "chosen_by_default": _accel.VARIANT,
    }
    out = []
    for label, suffix in (_accel._AVX512, _accel._AVX2, _accel._SSE2):
        if not _accel._extension_built(suffix):
            notes[label] = "not built in this install"
        elif label == "avx512" and _accel._cpu_supports_avx512() is not True:
            notes[label] = "built, but this CPU cannot run it"
        else:
            out.append(label)
    return out, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpus", help="bind each child to these cores, e.g. 0-3")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--decks", default=",".join(DECKS))
    ap.add_argument("--variants", help="comma list; default: every runnable one")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="LABEL=VARIANT@SRC",
        help="also time VARIANT from another build, with SRC (a checkout's "
        "src/) first on the children's PYTHONPATH, labelled LABEL and "
        "interleaved with the rest; e.g. avx512-s1=avx512@/home/me/s1/src",
    )
    ap.add_argument("--out", default="time_variants_1370.json")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--deck", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(args)

    import momwire

    try:
        from importlib.metadata import version

        mw_version = version("momwire")
    except Exception:  # noqa: BLE001 — a version we cannot read is not a failure
        mw_version = "?"
    cpu = _cpu()
    variants, notes = _variants()
    if args.variants:
        variants = [v for v in args.variants.split(",") if v]
    # (label, forced variant, PYTHONPATH prefix or None)
    arms = [(v, v, None) for v in variants]
    for spec in args.arm:
        label, _, rest = spec.partition("=")
        variant, _, src = rest.partition("@")
        arms.append((label, variant, os.path.abspath(src)))
    print(f"cpu      {cpu['model']}  ({cpu['machine']}, {cpu['logical_cpus']} logical)")
    print(f"flags    {' '.join(cpu.get('flags', [])) or '(no /proc/cpuinfo)'}")
    print(f"momwire  {mw_version}  {momwire.__file__}")
    print(f"python   {sys.version.split()[0]}")
    print(f"threads  {args.threads}  cpus {args.cpus or 'unbound'}")
    print(f"variants {variants}  {notes}")
    for label, variant, src in arms[len(variants) :]:
        print(f"arm      {label}: {variant} from {src}")
    print()

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(args.threads)
    env["MOMWIRE_SOMM_CACHE"] = "0"
    env.pop("MOMWIRE_REQUIRE_ACCEL", None)

    rows = []
    for deck in [d for d in args.decks.split(",") if d]:
        got = {
            a[0]: {"times": [], "solve": [], "z": None, "file": None, "error": None}
            for a in arms
        }
        for _ in range(args.repeats):
            for label, variant, src in arms:
                if got[label]["error"]:
                    continue
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--child",
                    "--deck",
                    deck,
                    "--warmup",
                    str(args.warmup),
                ]
                if args.cpus:
                    cmd += ["--cpus", args.cpus]
                child_env = {**env, "MOMWIRE_FORCE_VARIANT": variant}
                if src:
                    child_env["PYTHONPATH"] = os.pathsep.join(
                        p for p in (src, env.get("PYTHONPATH", "")) if p
                    )
                res = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
                if res.returncode != 0:
                    got[label]["error"] = res.stderr[-2000:]
                    print(f"{deck} {label}: FAILED\n{res.stderr[-2000:]}")
                    continue
                one = json.loads(res.stdout.strip().splitlines()[-1])
                if not one["solve_calls"]:
                    got[label]["error"] = "the dense-solve wrappers were never reached"
                    print(f"{deck} {label}: FAILED (no dense solve was timed)")
                    continue
                got[label]["times"].append(one["time"])
                got[label]["solve"].append(one["solve"])
                got[label]["z"] = one["z"]
                got[label]["file"] = one["file"]
        for label, _variant, _src in arms:
            g = got[label]
            if g["error"]:
                rows.append({"deck": deck, "variant": label, "error": g["error"]})
                continue
            fills = [t - s for t, s in zip(g["times"], g["solve"], strict=True)]
            row = {
                "deck": deck,
                "variant": label,
                "times": g["times"],
                "solve_times": g["solve"],
                "median": statistics.median(g["times"]),
                "min": min(g["times"]),
                "median_solve": statistics.median(g["solve"]),
                "median_rest": statistics.median(fills),
                "z": g["z"],
                "file": g["file"],
            }
            rows.append(row)
            print(
                f"  {deck:18s} {label:9s} median {row['median']:.3f} s "
                f"(LAPACK {row['median_solve']:.3f}, rest {row['median_rest']:.3f})"
                f"  {row['file']}",
                flush=True,
            )

    print()
    print(f"{'deck':18s} {'variant':9s} {'median s':>9s} {'min s':>8s} "
          f"{'rest s':>7s} {'vs avx2':>8s} {'rest vs':>8s} "
          f"{'|dZ|/|Z| vs avx2':>17s}")  # fmt: skip
    ref = {r["deck"]: r for r in rows if r["variant"] == "avx2" and "median" in r}
    for r in rows:
        if "median" not in r:
            print(f"{r['deck']:18s} {r['variant']:9s} {'FAILED':>9s}")
            continue
        base = ref.get(r["deck"])
        speed = rest = rel = None
        if base is not None:
            speed = base["median"] / r["median"]
            rest = base["median_rest"] / r["median_rest"]
            zb = complex(*base["z"])
            rel = abs(complex(*r["z"]) - zb) / abs(zb)
            r["speedup_vs_avx2"] = speed
            r["rest_speedup_vs_avx2"] = rest
            r["rel_dz_vs_avx2"] = rel
        print(
            f"{r['deck']:18s} {r['variant']:9s} {r['median']:9.3f} {r['min']:8.3f} "
            f"{r['median_rest']:7.3f} "
            f"{'' if speed is None else f'{speed:7.3f}x':>8s} "
            f"{'' if rest is None else f'{rest:7.3f}x':>8s} "
            f"{'' if rel is None else f'{rel:.2e}':>17s}"
        )

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cpu": cpu,
                "momwire": mw_version,
                "python": sys.version,
                "threads": args.threads,
                "cpus": args.cpus,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "variants": variants,
                "arms": arms,
                "notes": notes,
                "rows": rows,
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {args.out}")
    return 0 if all("median" in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
