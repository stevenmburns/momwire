"""momwire#887, part 2 — what the lambda/320 jump actually IS.

probe1 bisected the failure onto the PAIR: `a-only` (kernel radius a', no L)
and `L-only` (radius a, with L) are both flat across lambda/120..lambda/320,
and only `a'+L` moves. So the issue's stated suspicion -- that Delta/a'
degrades the thin-wire ratio -- is not sufficient on its own: `a-only` sees
exactly the same Delta/a' at every rung and does not move.

This probe asks what the jump is, rather than where it comes from:

  (a) the whole G(h) curve at the last two rungs, so a MOVED peak can be told
      apart from a SPURIOUS one appearing elsewhere;
  (b) the system matrix's condition number per rung and arm;
  (c) a finer rung ladder, to find whether it is a threshold or a drift.

Read (a) first. If the physical peak stays put and a new spike outgrows it,
the answer is conditioning, not physics.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from momwire._wire_loading import equivalent_radius  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402

C0 = 299792458.0
FREQ = 600e6
LAM = C0 / FREQ
A = 3.175e-3
B = 6.35e-3


def build(cls, arm, h, n_seg, eps_r):
    a_eq = equivalent_radius(A, B, eps_r)
    radius, jacket, restore = {
        "bare": (A, {}, False),
        "a-only": (a_eq, {}, False),
        "L-only": (A, {"insulation_radius": B, "insulation_eps_r": eps_r}, True),
        "a'+L": (A, {"insulation_radius": B, "insulation_eps_r": eps_r}, False),
    }[arm]
    s = cls(
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


def g_of(cls, arm, h, seg_len, eps_r):
    s = build(cls, arm, h, nseg_for(h, seg_len), eps_r)
    return (1.0 / complex(s.compute_impedance()[0])).real


def part_a(cls, eps_r):
    print(f"(a) G(h) across the peak, {cls.__name__}, arm a'+L, eps_r={eps_r}")
    hs = np.linspace(0.085, 0.145, 25)
    cols = [240, 280, 320]
    print("    h/lam  " + "".join(f"    lam/{r:<4d}" for r in cols))
    for h in hs:
        row = [g_of(cls, "a'+L", h, LAM / r, eps_r) * 1e3 for r in cols]
        star = " <" if row[-1] == max(row[-1:]) else ""
        print(f"    {h / LAM:.4f} " + "".join(f"  {v:9.4f}" for v in row) + star)
    for r in cols:
        vals = [g_of(cls, "a'+L", h, LAM / r, eps_r) for h in hs]
        i = int(np.argmax(vals))
        print(
            f"      lam/{r}: peak at h/lam = {hs[i] / LAM:.4f}, G = {vals[i] * 1e3:.4f} mS"
        )
    print()


def _system_matrix(s):
    """The dense collocation matrix `SinusoidalSolver` factors.

    It is not stored — `compute_impedance` factors it in place and keeps only
    the LU — so it is re-assembled through the solver's own entry point.
    SINUSOIDAL ONLY: `BSplineSolver._assemble_Z` takes a different argument
    list and stashes no factors, and probe1 already showed both classes carry
    the same symptom, so one class is enough to say what the symptom IS.
    """
    geom = s._build_geometry()
    G, _ = s._assemble_Z(geom, s.k)
    return np.asarray(G)


def part_b(cls, eps_r):
    print(f"(b) cond(G) and |Z_in| at h/lam = 0.20, {cls.__name__}, eps_r={eps_r}")
    h = 0.20 * LAM
    rungs = (120, 160, 240, 280, 320)
    print("    arm      " + "".join(f"    lam/{r:<5d}" for r in rungs))
    for arm in ("bare", "a-only", "L-only", "a'+L"):
        conds, zs = [], []
        for r in rungs:
            s = build(cls, arm, h, nseg_for(h, LAM / r), eps_r)
            try:
                conds.append(np.linalg.cond(_system_matrix(s)))
            except Exception as exc:  # noqa: BLE001 — a probe: report and go on
                print(f"      [{arm} lam/{r}] cond unavailable: {exc}")
                conds.append(np.nan)
            zs.append(abs(complex(s.compute_impedance()[0])))
        print(f"    {arm:8s} " + "".join(f"  {v:10.3e}" for v in conds))
        print("      |Z|    " + "".join(f"  {v:10.3f}" for v in zs))
    print()


def part_c(cls, eps_r):
    print(f"(c) fine rung ladder, {cls.__name__}, eps_r={eps_r}")
    rungs = [160, 200, 240, 260, 280, 300, 320, 360, 400]
    hs = np.linspace(0.06, 0.16, 101)
    for arm in ("a-only", "L-only", "a'+L"):
        row = []
        for r in rungs:
            vals = [g_of(cls, arm, h, LAM / r, eps_r) for h in hs]
            row.append(hs[int(np.argmax(vals))] / LAM)
        print(f"    {arm:8s} " + "".join(f" {v:.4f}" for v in row))
    print("    rung     " + "".join(f" {r:6d}" for r in rungs))
    print(
        "    Delta/a' "
        + "".join(f" {LAM / r / equivalent_radius(A, B, eps_r):6.3f}" for r in rungs)
    )
    print()


if __name__ == "__main__":
    eps_r = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
    for cls in (SinusoidalSolver, BSplineSolver):
        part_a(cls, eps_r)
        if cls is SinusoidalSolver:
            part_b(cls, eps_r)
        part_c(cls, eps_r)
