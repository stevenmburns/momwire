"""momwire#887, part 3 — the falsifiable prediction, and the confound in part 2.

probe2 measured cond(G) at a fixed h across the rung ladder:

    bare     2.19e2 -> 2.95e2    1.3x     flat
    L-only   3.13e2 -> 4.11e2    1.3x     flat
    a-only   9.97e1 -> 2.46e3   24.7x     degrading
    a'+L     1.04e2 -> 1.83e4  177.0x     degrading, 7x faster

So the conditioning says the EQUIVALENT RADIUS is the cause and the
inductance is an amplifier, not a second cause — which is the issue's stated
suspicion after all. probe1's peak-location table appeared to exonerate
`a-only` only because that observable is a LOCATION: it is insensitive until
the solve is bad enough to move where the maximum sits.

That reading makes a falsifiable prediction: **`a-only` must break too, just
further down the ladder.** If it stays clean at lambda/1000 the reading is
wrong and the pair really is irreducible.

(It also explains why probe2's |Z| column must NOT be read as convergence.
`bare` drifts 73.1 -> 45.0 over the same rungs, and bare is the control. That
is the delta-gap feed's own log divergence as Delta -> 0, present in every arm
and nothing to do with the jacket. The conductance-peak LOCATION is the
issue's observable precisely because it survives that; cond(G) survives it
because it never looks at the RHS at all.)

Part (a): cond(G) for `a-only` and `a'+L` over a long ladder, against Delta/a'.
Part (b): the peak-location observable for `a-only` pushed to lambda/1000.
Part (c): the same two, at eps_r 3.5 (catalog) — where a'/a = 1.42 rather
          than 1.85 — to check the threshold moves with a' and not with eps_r
          as such.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from momwire._wire_loading import equivalent_radius  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402

C0 = 299792458.0
FREQ = 600e6
LAM = C0 / FREQ
A = 3.175e-3
B = 6.35e-3


def build(arm, h, n_seg, eps_r):
    a_eq = equivalent_radius(A, B, eps_r)
    radius, jacket, restore = {
        "bare": (A, {}, False),
        "a-only": (a_eq, {}, False),
        "L-only": (A, {"insulation_radius": B, "insulation_eps_r": eps_r}, True),
        "a'+L": (A, {"insulation_radius": B, "insulation_eps_r": eps_r}, False),
    }[arm]
    s = SinusoidalSolver(
        wires=[np.array([(0.0, 0.0, -h), (0.0, 0.0, h)])],
        n_per_edge_per_wire=[[n_seg]],
        wavelength=LAM,
        wire_radius=radius,
        feeds=[(0, h, 1 + 0j)],
        **jacket,
    )
    if restore:
        s._radius_per_wire = np.full_like(s._radius_per_wire, A)
        s._uniform_radius = float(A)
    return s


def nseg_for(h, seg_len):
    n = max(4, int(round(2.0 * h / seg_len)))
    return n + (n % 2)


def cond_of(arm, h, seg_len, eps_r):
    s = build(arm, h, nseg_for(h, seg_len), eps_r)
    G, _ = s._assemble_Z(s._build_geometry(), s.k)
    return np.linalg.cond(np.asarray(G))


def peak_of(arm, seg_len, eps_r, hs):
    vals = [
        (
            1.0
            / complex(build(arm, h, nseg_for(h, seg_len), eps_r).compute_impedance()[0])
        ).real
        for h in hs
    ]
    return hs[int(np.argmax(vals))] / LAM


def part_a(eps_r, rungs):
    a_eq = equivalent_radius(A, B, eps_r)
    print(f"(a) cond(G) at h/lam = 0.20, eps_r = {eps_r}, a'/a = {a_eq / A:.4f}")
    print("    rung     " + "".join(f" {r:9d}" for r in rungs))
    print("    Delta/a' " + "".join(f" {LAM / r / a_eq:9.3f}" for r in rungs))
    for arm in ("bare", "L-only", "a-only", "a'+L"):
        row = [cond_of(arm, 0.20 * LAM, LAM / r, eps_r) for r in rungs]
        print(
            f"    {arm:8s} "
            + "".join(f" {v:9.2e}" for v in row)
            + f"   x{row[-1] / row[0]:.0f}"
        )
    print()


def part_b(eps_r, rungs):
    print(f"(b) conductance-peak h/lam pushed down the ladder, eps_r = {eps_r}")
    hs = np.linspace(0.06, 0.16, 101)
    print("    rung     " + "".join(f" {r:6d}" for r in rungs))
    for arm in ("bare", "L-only", "a-only", "a'+L"):
        row = [peak_of(arm, LAM / r, eps_r, hs) for r in rungs]
        print(f"    {arm:8s} " + "".join(f" {v:.4f}" for v in row))
    print()


if __name__ == "__main__":
    part_a(9.0, [120, 160, 240, 320, 500, 640, 800, 1000])
    part_b(9.0, [240, 320, 500, 640, 800, 1000])
    part_a(3.5, [120, 160, 240, 320, 500, 640, 800, 1000])
    part_b(3.5, [240, 320, 500, 640, 800, 1000])
