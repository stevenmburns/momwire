"""momwire#883 — does degree 3 earn its place on the basis axis?

The issue makes this the gate: add `bspline-3` to `AXIS_VALUES["basis"]` only
if it solves the standard decks AND converges at least as fast as degree 2;
otherwise leave the refusal and put the measurement on the issue.

"Converges as fast" is measured, not asserted, as the rate at which Z_in stops
moving under mesh refinement. Each degree is compared against ITS OWN finest
mesh rather than against a shared reference: the three degrees are three
different discretisations of the same operator and they need not agree in the
limit to the last digit, so a shared reference would score the degrees on
their bias rather than on their convergence. The observed ORDER is the slope
of log|dZ| against log N.

Decks: a centre-fed dipole near resonance, a bent (inverted-vee) wire so the
junction machinery is exercised, and a two-wire deck so off-edge pairs are
not a special case. All in free space; ground would add a second convergence
story on top of the one being measured.

Reported per degree per deck:
  * Z at each mesh, so a deck that does not solve is visible rather than
    silently absent
  * |Z(N) - Z(N_max)| and the fitted order
  * wall time, because degree 3 falls to the numpy path (the C++ same-edge
    dispatch is a 9-case switch), which is a real cost and belongs next to
    the accuracy it buys
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from momwire.bspline import BSplineSolver  # noqa: E402

LAM = 8.0


def dipole(n):
    return dict(
        wires=[np.array([(0.0, 0.0, -1.9), (0.0, 0.0, 1.9)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 1.9, 1 + 0j)],
    )


def invvee(n):
    return dict(
        wires=[np.array([(-1.6, 0.0, 0.6), (0.0, 0.0, 1.9), (1.6, 0.0, 0.6)])],
        n_per_edge_per_wire=[[n, n]],
        feeds=[(0, 2.06, 1 + 0j)],
    )


def two_wire(n):
    return dict(
        wires=[
            np.array([(0.0, 0.0, -1.9), (0.0, 0.0, 1.9)]),
            np.array([(1.2, 0.0, -1.9), (1.2, 0.0, 1.9)]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.9, 1 + 0j)],
    )


DECKS = [("dipole", dipole), ("inv-vee", invvee), ("two-wire", two_wire)]
MESHES = [10, 20, 40, 80]


def solve(deck_fn, n, degree, feed_model="point"):
    t0 = time.perf_counter()
    s = BSplineSolver(
        wavelength=LAM,
        wire_radius=0.01,
        degree=degree,
        feed_model=feed_model,
        **deck_fn(n),
    )
    z = complex(s.compute_impedance()[0])
    return z, time.perf_counter() - t0


def run(feed_model):
    print(f"=== feed_model = {feed_model!r}\n")
    for name, fn in DECKS:
        print(f"  {name}")
        for degree in (1, 2, 3):
            zs, ts = [], []
            for n in MESHES:
                try:
                    z, t = solve(fn, n, degree, feed_model)
                except Exception as exc:  # noqa: BLE001 — a probe: report and go on
                    print(f"    d={degree} N={n}: FAILED {type(exc).__name__}: {exc}")
                    zs, ts = [], []
                    break
                zs.append(z)
                ts.append(t)
            if not zs:
                continue
            ref = zs[-1]
            errs = [abs(z - ref) for z in zs[:-1]]
            ns = MESHES[:-1]
            assert all(e > 0 for e in errs), (
                f"d={degree} {name}: a zero error means a mesh returned the "
                f"reference exactly, so the slope below is meaningless"
            )
            order = np.polyfit(np.log(ns), np.log(errs), 1)[0]
            print(
                f"    d={degree}  "
                + "  ".join(
                    f"N{n}:{z.real:7.3f}{z.imag:+7.3f}j"
                    for n, z in zip(MESHES, zs, strict=True)
                )
            )
            print(
                "          |dZ| vs own finest: "
                + " ".join(f"{e:8.2e}" for e in errs)
                + f"   order {-order:5.2f}   {ts[-1] * 1e3:7.1f} ms at N={MESHES[-1]}"
            )
        print()


def main():
    print(f"momwire#883 — bspline degree 1 / 2 / 3, lambda = {LAM} m, a = 0.01 m")
    print("Each degree is scored against its OWN finest mesh (see the docstring).")
    print()
    print("BOTH feed models are run because the point (delta-gap) drive has a")
    print("convergence story of its own — the same log divergence as Delta -> 0")
    print("that momwire#887 ran into — and if it dominates, every degree scores")
    print("the FEED and the basis comparison is blind. Compare the two blocks:")
    print("orders that are the same across degrees in both blocks are the feed's.")
    print()
    for fm in ("point", "segment"):
        run(fm)


if __name__ == "__main__":
    main()
