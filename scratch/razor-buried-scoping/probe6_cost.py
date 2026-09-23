"""Probe 6: cost, razor-2p vs bspline d2, on the hub-spelled radial screen.

Each (solver, deck) runs in a FRESH subprocess (no shared grid/table caches),
timing a cold solve and then a warm re-solve on a new instance. Reports Z,
both times and peak RSS. Usage: probe6_cost.py [child <solver> <n_rad> <x>]
"""

import json
import pathlib
import resource
import subprocess
import sys
import time
import warnings

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable


def child(solver, n_rad, x):
    MW = pathlib.Path("/home/smburns/antennas/antennaknobs/momwire")
    sys.path.insert(0, str(MW / "tests"))
    import momwire
    from momwire import razor as _razor
    from momwire.bspline import BSplineSolver
    from momwire.razor import RazorSolver
    from test_crossing_serve_524 import hub_deck

    assert str(MW) in momwire.__file__
    warnings.filterwarnings("ignore")
    _razor._SERVE_BELOW_PLANE = True
    _razor._SERVE_CROSSING = True
    d = hub_deck(n_radials=n_rad)
    npe = d["n_per_edge_per_wire"]
    # far mesh = radials and the monopole; the 2-segment rise is the node's
    d["n_per_edge_per_wire"] = [[n * x for n in e] for e in npe[:n_rad]] + [
        npe[n_rad],
        [npe[n_rad + 1][0] * x],
    ]
    N = sum(sum(e) for e in d["n_per_edge_per_wire"])
    cls, kw = (
        (RazorSolver, {"nec5_quadrature": True})
        if solver == "razor_2p"
        else (BSplineSolver, {})
    )
    if cls is RazorSolver:
        d.pop("junctions")
    t0 = time.perf_counter()
    z, _ = cls(**d, **kw).compute_impedance()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    z2, _ = cls(**d, **kw).compute_impedance()
    warm = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            dict(
                solver=solver,
                n_rad=n_rad,
                x=x,
                N=N,
                z=[complex(z).real, complex(z).imag],
                cold=round(cold, 2),
                warm=round(warm, 2),
                rss_mb=round(rss),
            )
        )
    )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    sys.exit(0)

cases = [(4, 1), (4, 2), (4, 4), (12, 1), (24, 1)]
if len(sys.argv) > 1:
    cases = [tuple(int(v) for v in a.split(",")) for a in sys.argv[1:]]
rows = []
for n_rad, x in cases:
    for solver in ("bspline_d2", "razor_2p"):
        r = subprocess.run(
            [PY, __file__, "child", solver, str(n_rad), str(x)],
            capture_output=True,
            text=True,
            timeout=1500,
        )
        if r.returncode != 0:
            print(f"{solver} n={n_rad} x{x} FAILED: {r.stderr[-600:]}", flush=True)
            continue
        row = json.loads(r.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(
            f"n_rad={n_rad:3d} x{x} N={row['N']:5d} {solver:10s} "
            f"Z {row['z'][0]:9.3f}{row['z'][1]:+9.3f}j cold {row['cold']:7.2f} s "
            f"warm {row['warm']:7.2f} s rss {row['rss_mb']} MB",
            flush=True,
        )
tag = "_".join(f"{a}x{b}" for a, b in cases)
(HERE / f"probe6_cost_{tag}.json").write_text(json.dumps(rows, indent=1))
