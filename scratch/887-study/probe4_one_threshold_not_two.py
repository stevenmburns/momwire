"""momwire#887, part 4 — the load-bearing claim: ONE threshold, in Delta/a_kernel.

probe3 pushed every arm down a long ladder and found that BARE collapses too,
at lambda/640 and beyond. So "the bare control stays flat" is true only over
the four rungs the issue tested. That reframes the whole report: the jacket may
not be creating a failure mode at all, only ARRIVING at an existing one sooner,
because a' is 1.85x a.

The claim that would settle it: cond(G) is governed by Delta / a_kernel alone,
with a_kernel = a bare and a' coated, and the two curves SUPERIMPOSE when
plotted against that ratio rather than against the rung.

That is a real prediction and it is falsifiable two ways:

  * if the coated curve sits at a systematically different height at matched
    Delta/a_kernel, the jacket does contribute something of its own;
  * if either curve's knee moves when a is changed at fixed Delta/a_kernel,
    the ratio is not the governing variable and some other length is.

So this probe holds Delta/a_kernel fixed and varies a over 4x, which no rung
ladder can do. A single ladder cannot distinguish "the ratio governs" from
"the absolute segment length governs"; two conductor radii can.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from momwire._wire_loading import equivalent_radius  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402

C0 = 299792458.0
LAM = C0 / 600e6
A = 3.175e-3
B = 6.35e-3
H = 0.20 * LAM


def cond_for(a_conductor, ratio, jacket_eps=None):
    """cond(G) at a given Delta/a_kernel, for a bare or coated wire.

    `ratio` is Delta / a_kernel, so the segment length is DERIVED from it —
    that is the point: the mesh is chosen to hit the ratio rather than the
    ratio being read off a chosen mesh.
    """
    if jacket_eps is None:
        a_kernel = a_conductor
        kwargs = {"wire_radius": a_conductor}
    else:
        b = 2.0 * a_conductor  # b/a = 2, the issue's geometry
        a_kernel = equivalent_radius(a_conductor, b, jacket_eps)
        kwargs = {
            "wire_radius": a_conductor,
            "insulation_radius": b,
            "insulation_eps_r": jacket_eps,
        }
    seg = ratio * a_kernel
    n = max(4, int(round(2.0 * H / seg)))
    n += n % 2
    s = SinusoidalSolver(
        wires=[np.array([(0.0, 0.0, -H), (0.0, 0.0, H)])],
        n_per_edge_per_wire=[[n]],
        wavelength=LAM,
        feeds=[(0, H, 1 + 0j)],
        **kwargs,
    )
    assert abs(float(s._radius_per_wire[0]) / a_kernel - 1.0) < 1e-12, (
        "the deck's kernel radius is not the a_kernel this row is indexed by"
    )
    G, _ = s._assemble_Z(s._build_geometry(), s.k)
    return np.linalg.cond(np.asarray(G)), n, a_kernel


def main():
    ratios = [1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    cases = [
        ("bare a=3.175mm", A, None),
        ("bare a=1.0mm", 1.0e-3, None),
        ("bare a=6.0mm", 6.0e-3, None),
        ("coated eps=3.5", A, 3.5),
        ("coated eps=9", A, 9.0),
        ("coated eps=15", A, 15.0),
    ]
    print("cond(G) against Delta / a_kernel, h/lam = 0.20, 600 MHz")
    print("If the ratio governs, every row knees in the same COLUMN — including")
    print("the two rows whose conductor radius differs by 6x at matched ratio.\n")
    print(f"    {'case':16s}" + "".join(f" {r:8.2f}" for r in ratios))
    for name, a_c, eps in cases:
        row, ns = [], []
        for r in ratios:
            c, n, a_k = cond_for(a_c, r, eps)
            row.append(c)
            ns.append(n)
        print(f"    {name:16s}" + "".join(f" {v:8.1e}" for v in row))
        print(
            f"    {'  a_k, nsegs':16s}"
            + f" {a_k * 1e3:6.3f}mm"
            + "".join(f" {n:8d}" for n in ns[:9])
        )
    print()
    print("First ratio at which cond exceeds 1e4 (the knee), per case:")
    for name, a_c, eps in cases:
        knee = None
        for r in ratios:
            if cond_for(a_c, r, eps)[0] > 1e4:
                knee = r
                break
        print(f"    {name:16s} {knee if knee is not None else 'never in range'}")


if __name__ == "__main__":
    main()
