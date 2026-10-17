"""The C++ Sommerfeld grid (`_somm_grid_core.cpp`, momwire#1479 Sommerfeld
S2) against `_sommerfeld.get_grid`, call by call, bit for bit, and its `far_cpp_pack` (the momwire#1258 continuation's
constants the remainder kernels take).

    python scripts/core_split/somm_grid_compare.py CALLS.jsonl [--random N]

CALLS.jsonl holds `get_grid` argument tuples, one JSON list per line
`[eps_re, eps_im, k2, r1_max, omega, mu]` (record them off real renders with
`--record OUT.jsonl DECK_OR_DIR...`). `--random N` adds N synthetic calls over
soils, frequencies and extents. Each grid is compared on every scalar the
grid carries (eps, lattice eps, k2, omega, mu, r1_max, r_break, r_near, the
r0 fill) and, region by region, its lattice and its value table's bytes.
Python fills with the disk cache off and both in-process caches cleared per
call. Exit 1 on any mismatch.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

os.environ["MOMWIRE_SOMM_CACHE"] = "0"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "momwire"
SPLIT = ROOT / "scripts" / "core_split"
SOURCES = ("_somm_kernel_core.cpp", "_somm_grid_core.cpp")


def build_driver(out: Path) -> Path:
    sys.path.insert(0, str(SPLIT))
    import printout_compare as pc  # noqa: PLC0415 - its Linux flags

    from momwire import _accel  # noqa: PLC0415

    t = pc._standalone_test_module()
    compile_args, link_args, avx2_only = t._linux_flags()
    # The loaded variant's flags, as printout_compare's driver takes them: an
    # sse2 extension is compared with an sse2 driver, never an avx2 one.
    if _accel.acc is not None and _accel.acc.__name__.endswith("_sse2"):
        compile_args = [a for a in compile_args if a not in avx2_only]
    exe = out / "somm_grid_driver"
    cmd = [
        *shlex.split(os.environ.get("CXX", "g++")),
        "-fvisibility=hidden",
        "-g0",
        *compile_args,
        "-I",
        str(PKG),
        *[str(PKG / s) for s in SOURCES],
        str(SPLIT / "somm_grid_driver.cpp"),
        *link_args,
        "-o",
        str(exe),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        raise SystemExit(f"{shlex.join(cmd)}\n{run.stderr}")
    return exe


def record(out: Path, decks: list[str]) -> None:
    from momwire import _sommerfeld as s  # noqa: PLC0415
    from momwire.eznec import _shell  # noqa: PLC0415

    calls = set()
    orig = s.get_grid

    def rec(eps_t, k2, r1_max, omega, mu=s._MU0, cancel_flag=0):
        e = complex(eps_t)
        calls.add((e.real, e.imag, float(k2), float(r1_max), float(omega), float(mu)))
        return orig(eps_t, k2, r1_max, omega, mu, cancel_flag)

    s.get_grid = rec
    paths = []
    for d in decks:
        p = Path(d)
        paths += sorted(p.rglob("*.nec")) if p.is_dir() else [p]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in paths:
            try:
                _shell.render(p.read_text(encoding="latin-1"), basis="razor-2p")
            except Exception:  # noqa: BLE001 - a deck that raises still recorded its calls
                pass
    out.write_text("".join(json.dumps(list(c)) + "\n" for c in sorted(calls)))
    print(f"decks {len(paths)}: distinct get_grid calls {len(calls)}")


def random_calls(n: int) -> list[tuple]:
    rng = np.random.default_rng(1479)
    c = 299792458.0
    out = []
    for _ in range(n):
        f_hz = float(10.0 ** rng.uniform(5.5, 8.5))
        eps_r = float(rng.choice([1.5, 3.0, 5.0, 13.0, 15.0, 20.0, 81.0]))
        sigma = float(10.0 ** rng.uniform(-4.0, 0.7))
        omega = 2.0 * math.pi * f_hz
        eps = complex(eps_r, -sigma / (omega * 8.854187817e-12))
        k2 = omega / c
        lam = 2.0 * math.pi / k2
        r1 = float(lam * 10.0 ** rng.uniform(-1.5, 1.4))
        out.append((eps.real, eps.imag, k2, r1, omega, 4e-7 * math.pi))
    return out


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--record":
        record(Path(args[1]), args[2:])
        return 0
    calls = []
    if "--random" in args:
        k = args.index("--random")
        calls += random_calls(int(args[k + 1]))
        del args[k : k + 2]
    if args:
        calls += [
            tuple(json.loads(line)) for line in Path(args[0]).read_text().splitlines()
        ]
    from momwire import _sommerfeld as s  # noqa: PLC0415

    work = Path(tempfile.mkdtemp(prefix="somm_grid_"))
    exe = build_driver(work)
    lines = "".join(" ".join(float(x).hex() for x in c) + "\n" for c in calls)
    run = subprocess.run(
        [str(exe), str(work)], input=lines, text=True, capture_output=True
    )
    if run.returncode:
        raise SystemExit(run.stderr)
    bad = 0
    for k, c in enumerate(calls):
        s._GRID_CACHE.clear()
        s._NORM_CACHE.clear()
        g = s.get_grid(complex(c[0], c[1]), c[2], c[3], c[4], c[5])
        txt = (work / f"{k}.txt").read_text().split("\n")
        head = [float.fromhex(x) for x in txt[0].split()]
        want = [
            g.eps_t.real,
            g.eps_t.imag,
            g.lattice_eps.real,
            g.lattice_eps.imag,
            g.k2,
            g.omega,
            g.mu,
            g.r1_max,
            g.r_break,
            g.r_near,
            g._r0_fill,
        ]
        problems = [
            i for i, (a, b) in enumerate(zip(head, want)) if a.hex() != float(b).hex()
        ]
        blob = (work / f"{k}.bin").read_bytes()
        off = 0
        if len(txt) - 2 != len(g._regions):
            problems.append("region count")
        for r, reg in enumerate(g._regions):
            f = txt[1 + r].split()
            lat = [
                float.fromhex(f[0]),
                float.fromhex(f[1]),
                int(f[2]),
                float.fromhex(f[3]),
                float.fromhex(f[4]),
                int(f[5]),
            ]
            want_lat = [
                reg["r0"],
                reg["dr"],
                reg["n_r"],
                float(reg["th0"]),
                float(reg["dth"]),
                reg["n_th"],
            ]
            if [repr(x) for x in lat] != [
                repr(float(x)) if isinstance(x, float | np.floating) else repr(x)
                for x in want_lat
            ]:
                problems.append(f"region {r} lattice")
            raw = np.ascontiguousarray(reg["vals"], dtype=np.complex128).tobytes()
            if blob[off : off + len(raw)] != raw:
                problems.append(f"region {r} values")
            off += len(raw)
        far = np.ascontiguousarray(s.far_cpp_pack(g), dtype=np.complex128).tobytes()
        if (work / f"{k}.far").read_bytes() != far:
            problems.append("far pack")
        if problems:
            bad += 1
            print(f"call {k} {c}: {problems}")
    print(f"grids {len(calls)}, mismatches {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
