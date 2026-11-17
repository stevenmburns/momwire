"""PD2-measure (arch report 11-17): SG-mp's buried pair order against
bspline's #1752 S2/H2 ladder, Python and the C++ fill core alike.

Needs the scratch branch's MEASUREMENT PATCH (env overrides read by both
sides). One subprocess per (deck, mesh, setting, side), because the C++
core reads its env once per process and `_sinusoidal_mp` binds its near
rules at import.

Two families of knob, because SG has two quadrature regimes:

- FAR (ratio >= NEAR_RATIO = 2.0): the WindowFill ladder.
  `MOMWIRE_MEASURE_SG_CROSSING_BELOW` / `_WHOLLY_BURIED` = `<base>|<ladder>`.
  The BASE order is structurally dead: every pair below the first ladder
  threshold (2.0) is also below NEAR_RATIO (2.0), so `near_pair_moments`
  overwrites it (`WindowFill.accumulate`, `J[:, :, ii, jj] = Jn`). q32/q64/
  q128 therefore cannot move Z, and do not (0.0 exactly).
- NEAR (ratio < 2.0): `_sinusoidal_mp.near_pair_moments`.
  `MOMWIRE_MEASURE_SG_NEAR` = `<graded>|<near_n_qp>|<n_t>|<n_xi>` (shipped
  8|32|24|8): the graded rule on touching non-parallel pairs, the product
  rule on near non-touching ones, the parallel reduction. This is the
  regime bs2's base order carried in #1752.

    python sg_buried_orders.py worker SIDE DECK MESH        (prints one JSON line)
    python sg_buried_orders.py drive DECK_DIR OUT.jsonl      (runs the grid)
    python sg_buried_orders.py table OUT.jsonl               (the report table)
    python sg_buried_orders.py summary OUT.jsonl             (one row per deck)
    python sg_buried_orders.py ladder OUT.jsonl              (slant: near error vs mesh)
    python sg_buried_orders.py identity BASE_PY BASE_REPO DECK_DIR OUT.jsonl
                                         (env unset, unpatched build vs `cur`)

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
NEAR = "MOMWIRE_MEASURE_SG_NEAR"

# setting -> (crossing-below spec, wholly-buried spec, near spec); None = shipped
SHIPPED_LADDER = "2:8,16:4"
BS_LADDER = "0.1:32,1.5:16,2:8,4:4"  # bspline BURIED_PAIR_ORDER_LADDER
FAR_REF = "256|2:32,16:16"
FAR_REF_CXX = "192|2:32,16:16"  # the C++ Gauss tables stop at 192
NEAR_REF = "48|192|144|48"  # every near rule at 6x
# The C++ two-rule batch declines a rule wider than 256 nodes, which a
# graded 48 per panel exceeds; its near reference is 3x (Python's g24 and
# g48 differ by 2.6e-3 ohm on slant, 2e-12 elsewhere)
NEAR_REF_CXX = "24|96|72|24"
SETTINGS = {
    "cur": (None, None, None),
    # run-to-run repeats of `cur` (the reproducibility floor)
    "rep2": (None, None, None),
    "rep3": (None, None, None),
    # far liveness probe: a starved order must move Z, or the knob never
    # reached the kernel on that deck's route. "4|" also deletes the far
    # tiers, which is the only reason it moves Z (the base order is dead)
    "q4": ("4|", "4|", None),
    "q32": (None, f"32|{SHIPPED_LADDER}", None),
    "q64": (f"64|{SHIPPED_LADDER}", f"64|{SHIPPED_LADDER}", None),
    "q128": (f"128|{SHIPPED_LADDER}", f"128|{SHIPPED_LADDER}", None),
    "bsl": (f"128|{BS_LADDER}", f"128|{BS_LADDER}", None),
    # far tiers raised, near rules shipped (the first run's "reference")
    "ref": (FAR_REF, FAR_REF, None),
    "ref_cxx": (FAR_REF_CXX, FAR_REF_CXX, None),
    # near liveness probe: a starved graded rule must move Z on any deck
    # with a touching non-parallel pair
    "g2": (None, None, "2"),
    # the near rules, one at a time
    "g16": (None, None, "16"),
    "g24": (None, None, "24"),
    "g48": (None, None, "48"),
    "nq96": (None, None, "|96"),
    "par3": (None, None, "||72|24"),
    # the quadrature reference: far AND near raised
    "qref": (FAR_REF, FAR_REF, NEAR_REF),
    "qref_cxx": (FAR_REF_CXX, FAR_REF_CXX, NEAR_REF_CXX),
}
PY_GRID = ("cur", "rep2", "rep3", "q4", "q32", "q64", "q128", "bsl", "ref")
PY_GRID += ("g2", "g16", "g24", "g48", "nq96", "par3", "qref")
CXX_GRID = tuple(
    {"ref": "ref_cxx", "qref": "qref_cxx", "par3": None}.get(st, st) for st in PY_GRID
)
CXX_GRID = tuple(st for st in CXX_GRID if st)


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
            # which near rules this process runs, and whether the graded rule
            # ran at all (a touching non-parallel pair exists)
            import momwire._sinusoidal_mp as smp

            graded = {"calls": 0}
            orig_g = smp.graded_endpoint_rule01

            def gcount(*a, **k):
                graded["calls"] += 1
                return orig_g(*a, **k)

            smp.graded_endpoint_rule01 = gcount
            seen["graded"] = graded
            rec["near_rules"] = [
                smp.GRADED_N_PER_PANEL,
                smp.NEAR_N_QP,
                smp.PARALLEL_N_T,
                smp.PARALLEL_N_XI,
            ]
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
        if "graded" in seen:
            rec["graded_calls"] = seen["graded"]["calls"]
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
        for st in PY_GRID:
            jobs.append((d, 1, "py", st))
        for st in CXX_GRID:
            jobs.append((d, 1, "cxx", st))
        jobs.append((d, 1, "bs2", "cur"))
    for d in decks:  # the mesh reference after the quadrature grid
        jobs.append((d, 3, "py", "bsl"))
        jobs.append((d, 3, "py", "cur"))
        jobs.append((d, 3, "py", "qref"))
        jobs.append((d, 3, "bs2", "cur"))
    slant = [d for d in decks if d.stem == "fx_slant_base_source"]
    for d in slant:  # the deck with the near-rule error: a longer ladder
        for m in (5, 9):
            for st in ("cur", "g48", "qref"):
                jobs.append((d, m, "py", st))
            jobs.append((d, m, "bs2", "cur"))
    for d, mesh, side, st in jobs:
        if (d.stem, mesh, side, st) in done:
            continue
        env = dict(os.environ)
        env.pop(CROSS, None)
        env.pop(WHOLE, None)
        env.pop(NEAR, None)
        cs, ws, ns = SETTINGS[st]
        if side != "bs2":
            if cs:
                env[CROSS] = cs
            if ws:
                env[WHOLE] = ws
            if ns:
                env[NEAR] = ns
        t0 = time.time()
        try:
            p = subprocess.run(
                [sys.executable, __file__, "worker", side, str(d), str(mesh)],
                env=env,
                check=False,
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


def _index(path: pathlib.Path):
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    by = {(r["deck"], r["mesh"], r["side"], r["setting"]): r for r in rows}

    def z(*key):
        r = by.get(key)
        return complex(*r["z"]) if r and "z" in r else None

    return rows, by, z


def _d(a, b):
    return None if a is None or b is None else abs(a - b)


def _f(v):
    return "-" if v is None else f"{v:.1e}"


def _zs(v):
    return "n/a" if v is None else f"{v.real:.5f}{v.imag:+.5f}j"


def noise(by, z, d, side):
    """The spread of the shipped setting over its repeats (cur, rep2, rep3):
    the reproducibility floor. A delta at or below it is not a measurement."""
    vals = [z(d, 1, side, st) for st in ("cur", "rep2", "rep3")]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    return max(abs(a - b) for a in vals for b in vals)


def table(path: pathlib.Path) -> None:
    rows, by, z = _index(path)
    decks = sorted({r["deck"] for r in rows})
    print(
        "| deck | N | route | side | setting | Z (ohm) | dZ vs qref | dZ vs cur "
        "| fill/total t ratio vs q32 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for d in decks:
        cur_py = by.get((d, 1, "py", "cur"), {})
        route = ",".join(sorted(set(cur_py.get("routes", [])))) or "?"
        n = cur_py.get("n_basis")
        for side, grid, qref in (
            ("py", PY_GRID, "qref"),
            ("cxx", CXX_GRID, "qref_cxx"),
        ):
            zr, zc = z(d, 1, side, qref), z(d, 1, side, "cur")
            # q32 is warm (cur is each deck's cold first run)
            tc = by.get((d, 1, side, "q32"), {}).get(
                "fill_s" if side == "py" else "total_s"
            )
            for st in grid:
                r = by.get((d, 1, side, st), {})
                zz = z(d, 1, side, st)
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
                    f"| {d} | {n} | {route} | {side} | {st} | {_zs(zz)} | "
                    f"{_f(_d(zz, zr))} | {_f(_d(zz, zc))} | "
                    f"{(t / tc) if (t and tc) else float('nan'):.2f} |"
                )
        for m in sorted({r["mesh"] for r in rows if r["deck"] == d}):
            print(
                f"| {d} | mesh m={m} | | SG py cur / qref ; bs2 | | "
                f"{_zs(z(d, m, 'py', 'cur'))} / {_zs(z(d, m, 'py', 'qref'))} ; "
                f"{_zs(z(d, m, 'bs2', 'cur'))} | | | |"
            )


def summary(path: pathlib.Path) -> None:
    """One row per deck, in ohms: the noise floor, the total quadrature
    error at the shipped rules split into its far and near parts, the
    #1752 ladder's move, the two liveness moves, and the mesh terms."""
    rows, by, z = _index(path)
    print(
        "| deck | route (py) | graded calls | N | py noise | py cur-qref | far: cur-ref "
        "| near: cur-g48 | cur->bsl | q4 move | g2 move | cxx noise | cxx cur-qref "
        "| cxx cur-g24 | SG qref m1-m3 | SG qref m3 - bs2 m3 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted({r["deck"] for r in rows}):
        cur = by.get((d, 1, "py", "cur"), {})
        route = ",".join(cur.get("routes", [])) or "neither patched route"
        # counted on a run of the instrumented worker
        gcalls = by.get((d, 1, "py", "g48"), {}).get("graded_calls", "-")
        zc = z(d, 1, "py", "cur")
        xc = z(d, 1, "cxx", "cur")
        print(
            f"| {d} | {route} | {gcalls} | {cur.get('n_basis')} | "
            f"{_f(noise(by, z, d, 'py'))} | {_f(_d(zc, z(d, 1, 'py', 'qref')))} | "
            f"{_f(_d(zc, z(d, 1, 'py', 'ref')))} | {_f(_d(zc, z(d, 1, 'py', 'g48')))} | "
            f"{_f(_d(zc, z(d, 1, 'py', 'bsl')))} | {_f(_d(zc, z(d, 1, 'py', 'q4')))} | "
            f"{_f(_d(zc, z(d, 1, 'py', 'g2')))} | {_f(noise(by, z, d, 'cxx'))} | "
            f"{_f(_d(xc, z(d, 1, 'cxx', 'qref_cxx')))} | "
            f"{_f(_d(xc, z(d, 1, 'cxx', 'g24')))} | "
            f"{_f(_d(z(d, 1, 'py', 'qref'), z(d, 3, 'py', 'qref')))} | "
            f"{_f(_d(z(d, 3, 'py', 'qref'), z(d, 3, 'bs2', 'cur')))} |"
        )


def ladder(path: pathlib.Path, deck: str = "fx_slant_base_source") -> None:
    """The near-rule error against mesh on one deck: |cur - qref| at each m,
    and SG qref against bs2 at the same m."""
    rows, _by, z = _index(path)
    print("| m | SG cur | SG g48 | SG qref | cur-qref | bs2 | SG qref - bs2 |")
    print("|---|---|---|---|---|---|---|")
    for m in sorted({r["mesh"] for r in rows if r["deck"] == deck}):
        c, g, q, b = (
            z(deck, m, "py", "cur"),
            z(deck, m, "py", "g48"),
            z(deck, m, "py", "qref"),
            z(deck, m, "bs2", "cur"),
        )
        print(
            f"| {m} | {_zs(c)} | {_zs(g)} | {_zs(q)} | {_f(_d(c, q))} | {_zs(b)} | "
            f"{_f(_d(q, b))} |"
        )


def identity(
    base_python: str, base_repo: str, deck_dir: pathlib.Path, out: pathlib.Path
):
    """The env-unset run on an UNPATCHED build (the branch's merge base,
    built on the same box) against this branch's `cur` rows: the patch must
    be value-identical when no override is set. Prints one row per
    (deck, side) and exits 1 on any difference."""
    _rows, by, z = _index(out)
    env = dict(os.environ, SGQ_REPO=base_repo)
    for k in (CROSS, WHOLE, NEAR):
        env.pop(k, None)
    bad = 0
    for d in sorted(deck_dir.glob("*.nec")):
        for side in ("py", "cxx"):
            p = subprocess.run(
                [base_python, __file__, "worker", side, str(d), "1"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            r = json.loads(line) if line.startswith("{") else {}
            zb = complex(*r["z"]) if "z" in r else None
            zc = z(d.stem, 1, side, "cur")
            same = zb == zc
            bad += not same
            print(
                f"| {d.stem} | {side} | {_zs(zb)} | {_zs(zc)} | {'identical' if same else 'DIFFERS'} |"
            )
    sys.exit(1 if bad else 0)


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
    elif cmd == "ladder":
        ladder(pathlib.Path(sys.argv[2]))
    elif cmd == "identity":
        identity(
            sys.argv[2],
            sys.argv[3],
            pathlib.Path(sys.argv[4]),
            pathlib.Path(sys.argv[5]),
        )
