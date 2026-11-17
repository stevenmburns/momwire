"""PD2-measure (arch report 11-17): SG-mp's buried pair order against
bspline's #1752 S2/H2 ladder, Python and the C++ fill core alike.

Needs the scratch branch's MEASUREMENT PATCH (env overrides read by both
sides). One subprocess per (deck, mesh, setting, side), because the C++
core reads its env once per process.

    python sg_buried_orders.py worker SIDE DECK MESH        (prints one JSON line)
    python sg_buried_orders.py drive DECK_DIR OUT.jsonl      (runs the grid)
    python sg_buried_orders.py table OUT.jsonl               (the report table)

SIDE: py | cxx | bs2. MESH: segment-count multiplier on every GW line.
Timings are single runs on a shared box: indicative only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings

HERE = pathlib.Path(__file__).resolve().parent
CROSS = "MOMWIRE_MEASURE_SG_CROSSING_BELOW"
WHOLE = "MOMWIRE_MEASURE_SG_WHOLLY_BURIED"

# setting -> (crossing-below spec, wholly-buried spec); None = shipped
SHIPPED_LADDER = "2:8,16:4"
BS_LADDER = "0.1:32,1.5:16,2:8,4:4"  # bspline BURIED_PAIR_ORDER_LADDER
SETTINGS = {
    "cur": (None, None),
    # liveness probe: a deliberately starved order must move Z, or the knob
    # never reached the kernel on that deck's route
    "q4": ("4|", "4|"),
    "q32": (None, f"32|{SHIPPED_LADDER}"),
    "q64": (f"64|{SHIPPED_LADDER}", f"64|{SHIPPED_LADDER}"),
    "q128": (f"128|{SHIPPED_LADDER}", f"128|{SHIPPED_LADDER}"),
    "bsl": (f"128|{BS_LADDER}", f"128|{BS_LADDER}"),
    # the quadrature reference: every tier raised; C++ tables stop at 192
    "ref": ("256|2:32,16:16", "256|2:32,16:16"),
    "ref_cxx": ("192|2:32,16:16", "192|2:32,16:16"),
}


def refine(text: str, m: int) -> str:
    if m == 1:
        return text
    out = []
    for ln in text.splitlines():
        if ln.startswith("GW"):
            f = re.split(r"[,\s]+", ln.strip())
            f[2] = str(int(f[2]) * m)
            ln = " ".join(f)
        elif ln.startswith("EX"):
            # the feed segment at the same place: segment s of n becomes the
            # middle one of its m children (m odd)
            f = re.split(r"[,\s]+", ln.strip())
            if int(f[3]) > 0:
                f[3] = str((int(f[3]) - 1) * m + (m + 1) // 2)
            ln = " ".join(f)
        out.append(ln)
    return "\n".join(out) + "\n"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker(side: str, deck: pathlib.Path, mesh: int) -> dict:
    import numpy as np

    repo = pathlib.Path(os.environ["SGQ_REPO"])
    gfc = _load("gfc", repo / "scripts/core_split/sinusoidal_galerkin_fill_compare.py")
    text = refine(deck.read_text(encoding="latin-1"), mesh)
    rec = {"deck": deck.stem, "mesh": mesh, "side": side}
    if side in ("py", "bs2"):
        # The PRODUCTION route: the NEC-5 dialect's in-process render (what
        # momwire-nec5-sinusoidal-galerkin / -bspline serve), with the
        # solver class instrumented. In-process, so the env reaches it (a
        # resident server would not see it).
        from momwire.bspline import BSplineSolver
        from momwire.dropin.nec5 import _shell
        from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

        cls = SinusoidalGalerkinSolver if side == "py" else BSplineSolver
        seen = {"y": [], "fill": [], "routes": [], "n": []}
        orig_cps = cls.compute_port_solution
        orig_az = cls._assemble_Z

        def cps(self, *a, **k):
            sol = orig_cps(self, *a, **k)
            seen["y"].append(np.atleast_2d(np.asarray(sol.y)).copy())
            return sol

        def az(self, *a, **k):
            t0 = time.perf_counter()
            r = orig_az(self, *a, **k)
            seen["fill"].append(time.perf_counter() - t0)
            seen["n"].append(int(np.asarray(r[0]).shape[0]))
            return r

        cls.compute_port_solution = cps
        cls._assemble_Z = az
        if side == "py":
            for name in ("_assemble_Z_mp_mixed", "_assemble_Z_mp"):
                orig = getattr(cls, name)

                def wrap(self, *a, _o=orig, _n=name, **k):
                    tag = _n
                    if _n == "_assemble_Z_mp" and self._active_medium is not None:
                        tag += "@k_m"
                    seen["routes"].append(tag)
                    return _o(self, *a, **k)

                setattr(cls, name, wrap)
        basis = "sinusoidal-galerkin" if side == "py" else "bspline"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            out = _shell.render(text, basis=basis)
            tot = time.perf_counter() - t0
        if not seen["y"]:
            rec.update(error="no port solution: " + _refusal(out))
            return rec
        z = complex(np.linalg.inv(seen["y"][0])[0, 0])
        rec.update(
            z=[z.real, z.imag],
            z_printout=_printout_z(out),
            n_basis=seen["n"][0] if seen["n"] else None,
            routes=sorted(set(seen["routes"])),
            fill_s=seen["fill"][0] if seen["fill"] else None,
            total_s=tot,
        )
        return rec
    # the C++ fill core through its dump (front, fill, port solve)
    acc = gfc.dc.accel()
    blas = gfc.blas_addresses()
    t0 = time.perf_counter()
    dump = acc.sinusoidal_galerkin_fill_dump(text.encode("latin-1"), blas, 64)
    tot = time.perf_counter() - t0
    if gfc.classify(dump) != "fills":
        why = next((ln for ln in dump.splitlines() if "?" in ln), "")[:300]
        rec.update(declined=why)
        return rec
    y = {}
    pat = re.compile(
        r"^fill\.structures\[0\]\.points\[0\]\.y\[(\d+)\]\[(\d+)\] = complex (\S+) (\S+)$"
    )
    for ln in dump.splitlines():
        mm = pat.match(ln)
        if mm:
            y[(int(mm[1]), int(mm[2]))] = complex(_num(mm[3]), _num(mm[4]))
    n = 1 + max(i for i, _j in y)
    Y = np.array([[y[(i, j)] for j in range(n)] for i in range(n)])
    z = complex(np.linalg.inv(Y)[0, 0])
    rec.update(z=[z.real, z.imag], total_s=tot)
    return rec


def _refusal(out: str) -> str:
    for ln in out.splitlines():
        if "REFUS" in ln.upper() or "ERROR" in ln.upper():
            return ln.strip()[:300]
    return out.strip().splitlines()[-1][:300] if out.strip() else "empty printout"


def _printout_z(out: str):
    """The first feed's printed impedance (five figures), as a cross-check
    that the captured Y is the one the printout reports."""
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in ln:
            for row in lines[i + 1 : i + 8]:
                f = row.split()
                if len(f) >= 11 and f[0].isdigit():
                    try:
                        return [float(f[-5]), float(f[-4])]
                    except ValueError:
                        return None
    return None


def _num(s: str) -> float:
    """`dc._f`'s spelling: the float64's big-endian bytes in hex."""
    import struct

    return struct.unpack(">d", bytes.fromhex(s))[0]


def drive(deck_dir: pathlib.Path, out: pathlib.Path) -> None:
    decks = sorted(deck_dir.glob("*.nec"))
    done = set()
    if out.exists():
        for ln in out.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["deck"], r["mesh"], r["side"], r["setting"]))
    jobs = []
    for d in decks:
        for st in ("cur", "q4", "q32", "q64", "q128", "bsl", "ref"):
            jobs.append((d, 1, "py", st))
        for st in ("cur", "q4", "q32", "q64", "q128", "bsl", "ref_cxx"):
            jobs.append((d, 1, "cxx", st))
        jobs.append((d, 1, "bs2", "cur"))
    for d in decks:  # the mesh reference after the quadrature grid
        jobs.append((d, 3, "py", "bsl"))
        jobs.append((d, 3, "bs2", "cur"))
    for d, mesh, side, st in jobs:
        if (d.stem, mesh, side, st) in done:
            continue
        env = dict(os.environ)
        env.pop(CROSS, None)
        env.pop(WHOLE, None)
        cs, ws = SETTINGS[st]
        if side != "bs2":
            if cs:
                env[CROSS] = cs
            if ws:
                env[WHOLE] = ws
        t0 = time.time()
        try:
            p = subprocess.run(
                [sys.executable, __file__, "worker", side, str(d), str(mesh)],
                env=env,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("SGQ_TIMEOUT", "1500")),
            )
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            rec = (
                json.loads(line)
                if line.startswith("{")
                else {
                    "deck": d.stem,
                    "mesh": mesh,
                    "side": side,
                    "error": (p.stderr.strip().splitlines() or ["?"])[-1][:300],
                    "rc": p.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            rec = {"deck": d.stem, "mesh": mesh, "side": side, "error": "timeout"}
        rec["setting"] = st
        rec["wall_s"] = time.time() - t0
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            d.stem,
            mesh,
            side,
            st,
            rec.get("z"),
            rec.get("error") or rec.get("declined") or "",
            flush=True,
        )


def table(path: pathlib.Path) -> None:
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    by = {(r["deck"], r["mesh"], r["side"], r["setting"]): r for r in rows}
    decks = sorted({r["deck"] for r in rows})

    def z(key):
        r = by.get(key)
        return complex(*r["z"]) if r and "z" in r else None

    print(
        "| deck | N | route | side | setting | Z (ohm) | dZ vs q-ref | dZ vs cur | fill/total t ratio vs cur |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for d in decks:
        cur_py = by.get((d, 1, "py", "cur"), {})
        route = ",".join(sorted(set(cur_py.get("routes", [])))) or "?"
        n = cur_py.get("n_basis")
        for side, ref_st in (("py", "ref"), ("cxx", "ref_cxx")):
            zr, zc = z((d, 1, side, ref_st)), z((d, 1, side, "cur"))
            tc = by.get((d, 1, side, "cur"), {}).get(
                "fill_s" if side == "py" else "total_s"
            )
            for st in ("cur", "q4", "q32", "q64", "q128", "bsl", ref_st):
                r = by.get((d, 1, side, st), {})
                zz = z((d, 1, side, st))
                if zz is None:
                    msg = r.get("declined") or r.get("error") or "missing"
                    if r.get("declined") == "":
                        msg = (
                            "not served: the C++ dump's NEC-4.2 front records no point"
                        )
                    print(f"| {d} | {n} | {route} | {side} | {st} | {msg[:80]} | | | |")
                    continue
                t = r.get("fill_s" if side == "py" else "total_s")
                print(
                    f"| {d} | {n} | {route} | {side} | {st} | {zz.real:.5f}{zz.imag:+.5f}j | "
                    f"{abs(zz - zr) if zr is not None else float('nan'):.2e} | "
                    f"{abs(zz - zc) if zc is not None else float('nan'):.2e} | "
                    f"{(t / tc) if (t and tc) else float('nan'):.2f} |"
                )
        z1, z3 = z((d, 1, "py", "bsl")), z((d, 3, "py", "bsl"))
        b1, b3 = z((d, 1, "bs2", "cur")), z((d, 3, "bs2", "cur"))
        fmt = lambda v: "n/a" if v is None else f"{v.real:.4f}{v.imag:+.4f}j"  # noqa: E731
        print(
            f"| {d} | mesh | | SG m1/m3, bs2 m1/m3 | | {fmt(z1)} / {fmt(z3)} ; {fmt(b1)} / {fmt(b3)} | "
            f"SG mesh err {abs(z1 - z3) if z1 is not None and z3 is not None else float('nan'):.2e} | "
            f"bs2 m3 vs SG m3 {abs(b3 - z3) if b3 is not None and z3 is not None else float('nan'):.2e} | |"
        )


def summary(path: pathlib.Path) -> None:
    """One row per deck: the largest quadrature move over the candidate
    settings, the q4 liveness move, the mesh error, and SG vs bs2."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    by = {(r["deck"], r["mesh"], r["side"], r["setting"]): r for r in rows}

    def z(*key):
        r = by.get(key)
        return complex(*r["z"]) if r and "z" in r else None

    def mx(d, side, ref):
        zr = z(d, 1, side, ref)
        if zr is None:
            return None
        vals = [z(d, 1, side, st) for st in ("cur", "q32", "q64", "q128", "bsl")]
        return max(abs(v - zr) for v in vals if v is not None)

    f = lambda v: "-" if v is None else f"{v:.1e}"  # noqa: E731
    print(
        "| deck | route (py) | N | py: max dZ vs ref (cur..bsl) | py: cur->bsl | py: q4 move "
        "| cxx: max dZ vs ref | cxx: cur->bsl | SG mesh err m1-m3 | SG m3 - bs2 m3 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted({r["deck"] for r in rows}):
        cur = by.get((d, 1, "py", "cur"), {})
        route = ",".join(cur.get("routes", [])) or "neither patched route"

        def dd(a, b):
            return None if a is None or b is None else abs(a - b)

        print(
            f"| {d} | {route} | {cur.get('n_basis')} | {f(mx(d, 'py', 'ref'))} | "
            f"{f(dd(z(d, 1, 'py', 'bsl'), z(d, 1, 'py', 'cur')))} | "
            f"{f(dd(z(d, 1, 'py', 'q4'), z(d, 1, 'py', 'cur')))} | "
            f"{f(mx(d, 'cxx', 'ref_cxx'))} | "
            f"{f(dd(z(d, 1, 'cxx', 'bsl'), z(d, 1, 'cxx', 'cur')))} | "
            f"{f(dd(z(d, 1, 'py', 'bsl'), z(d, 3, 'py', 'bsl')))} | "
            f"{f(dd(z(d, 3, 'py', 'bsl'), z(d, 3, 'bs2', 'cur')))} |"
        )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "worker":
        print(
            json.dumps(worker(sys.argv[2], pathlib.Path(sys.argv[3]), int(sys.argv[4])))
        )
    elif cmd == "drive":
        drive(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif cmd == "summary":
        summary(pathlib.Path(sys.argv[2]))
    elif cmd == "table":
        table(pathlib.Path(sys.argv[2]))
